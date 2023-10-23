import polars as pl
import pyarrow as pa
import duckdb
import pandas as pd

# import pyarrow.csv as pc
# import pyarrow.feather as pf
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from fsspec import filesystem as fsspec_filesystem

from ..schema import (
    replace_schema,
    shrink_large_string,
    convert_timestamp,
)
from .misc import get_partitions_from_path
from .polars_ext import pl
import os
import datetime as dt
import uuid


def read_table(
    path: str,
    schema: pa.Schema | None = None,
    filesystem: AbstractFileSystem | None = None,
    partitioning: str | list[str] | None = None,
) -> pa.Table:
    """Loads the data for the given file path into a pyarrow table.

    Args:
        path (str): File path.
        schema (pa.Schema | None, optional): Pyarrow schema. Defaults to None.
        filesystem (AbstractFileSystem | None, optional): Filesystem. Defaults to None.
        partitioning (str | list[str] | None, optional): Partitioning of the file. Could be 'hive' for
            hive style partitioning or a list of column names. Defaults to None.

    Returns:
        pa.Table: Pyarrow table.
    """
    # sourcery skip: avoid-builtin-shadow
    if filesystem is None:
        filesystem = fsspec_filesystem("file")

    filesystem.invalidate_cache()

    table = pq.read_table(pa.BufferReader(filesystem.read_bytes(path)), schema=schema)

    if partitioning is not None:
        partitions = get_partitions_from_path(path, partitioning=partitioning)

        for key, values in partitions:
            if key in table.column_names:
                table = table.drop(key)
            table = table.append_column(
                field_=key, column=pa.array([values] * len(table))
            )
    if schema is not None:
        return table.cast(schema)

    return table


def write_table(
    table: pa.Table,
    path: str,
    filesystem: AbstractFileSystem | None = None,
    row_group_size: int | None = None,
    compression: str = "zstd",
    **kwargs,
) -> tuple[str, pq.FileMetaData]:
    """Write pyarrow table to the given file path.

    Args:
        df (pl.DataFrame): Polars DataFrame.
        path (str): File path.
        schema (pa.Schema | None, optional): Pyarrow schwma. Defaults to None.
        filesystem (AbstractFileSystem | None, optional): _description_. Defaults to None.
        row_group_size (int | None, optional): Row group size of the parquet or arrow
            file. Defaults to None.
        compression (str, optional): Compression of the parquet or arrow file. Defaults
            to "zstd".
        sort_by (str | list[str] | list[tuple[str,str]] | None, optional): Name of the
            column to use to sort (ascending) or a list of multiple sorting conditions
            where each entry is a tuple with column name and sorting order ("ascending"
            or "descending").  Defaults to None.
        distinct (bool, optional): Wheter to write a table with distinct rows. When the
            value is a str or list[str], than only this columns are taken into account.
            Defaults to false.
    """
    if filesystem is None:
        filesystem = fsspec_filesystem("file")

    filesystem.invalidate_cache()

    metadata = []
    pq.write_table(
        table,
        path,
        filesystem=filesystem,
        row_group_size=row_group_size,
        compression=compression,
        metadata_collector=metadata,
        **kwargs,
    )
    metadata = metadata[0]
    # metadata.set_file_path(path)
    return path, metadata


class Writer:
    def __init__(
        self,
        data: pa.Table
        | pl.DataFrame
        | pl.LazyFrame
        | pd.DataFrame
        | duckdb.DuckDBPyRelation,
        path: str,
        schema: pa.Schema | None,
        filesystem: AbstractFileSystem | None = None,
    ):
        self.schema = schema
        self.data = data
        self.base_path = path
        self.path = None
        self.filesystem = filesystem

    def _to_polars(self):
        if isinstance(self.data, pa.Table):
            self.data = pl.from_arrow(self.data)
        elif isinstance(self.data, pd.DataFrame):
            self.data = pl.from_pandas(self.data)
        elif isinstance(self.data, duckdb.DuckDBPyRelation):
            self.data = self.data.pl()

    def _to_arrow(self):
        if isinstance(self.data, pl.DataFrame):
            self.data = self.data.to_arrow()
        elif isinstance(self.data, pl.LazyFrame):
            self.data = self.data.collect().to_arrow()
        elif isinstance(self.data, pd.DataFrame):
            self.data = pa.Table.from_pandas(self.data)
        elif isinstance(self.data, duckdb.DuckDBPyRelation):
            self.data = self.data.to_arrow()

    def _set_schema(self):
        self._to_arrow()
        self.schema = self.schema or self.data.schema

    def sort_data(self, by: str | list[str] | list[tuple[str, str]] | None = None):
        if by is not None:
            self._to_arrow()
            self.data = self.data.sort_by(by)

    def unique(self, columns: bool | str | list[str] = False):
        if columns is not None:
            self._to_polars()
            if isinstance(columns, bool):
                columns = None
            self.data = self.data.unique(columns, maintain_order=True)

    def cast_schema(
        self,
        use_large_string: bool = False,
        tz: str = None,
        ts_unit: str = None,
        remove_tz: bool = False,
    ):
        self._to_arrow()
        self._set_schema()
        self._use_large_string = use_large_string
        if not use_large_string:
            self.schema = shrink_large_string(self.schema)

        if tz is not None or ts_unit is not None or remove_tz:
            self.schema = convert_timestamp(
                self.schema,
                tz=tz,
                ts_unit=ts_unit,
                remove_tz=remove_tz,
            )

        self.data = replace_schema(self.data, self.schema)

    def delta(
        self,
        other: pl.DataFrame | pl.LazyFrame,
        subset: str | list[str] | None = None,
    ):
        self._to_polars()
        self.data = self.data.delta(other, subset=subset)

    def partition_by(
        self,
        columns: str | list[str] | None = None,
        num_rows: int | None = None,
        strftime: str | list[str] | None = None,
        timedelta: str | list[str] | None = None,
    ):
        self._to_polars()
        self.data = self.data.partition_by_ext(
            columns=columns,
            strftime=strftime,
            timedelta=timedelta,
            num_rows=num_rows,
        )

    def set_path(self, base_name: str | None = None):
        if base_name is None:
            base_name = f"data-{dt.datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]}"
        self.path = [
            os.path.join(
                self.base_path,
                "/".join(
                    (
                        "=".join([k, str(v).lstrip("0")])
                        for k, v in partition[0].items()
                        if k != "row_nr"
                    )
                ),
                f"{base_name}-{num}-{uuid.uuid4().hex[:16]}.parquet",
            )
            for num, partition in enumerate(self.data)
        ]

    def write(
        self, row_group_size: int | None = None, compression: str = "zstd", **kwargs
    ):
        if self.path is None:
            raise ValueError("No path set. Call set_path() first.")
        if not isinstance(self, list):
            self._to_arrow()
            self.data = [self.data]
        file_metadata = []
        for path, part in zip(self.path, self.data):
            part = part[1].to_arrow()
            if self._use_large_string:
                part = part.cast(shrink_large_string(part.schema))

            metadata = write_table(
                table=part,
                path=path,
                filesystem=self.filesystem,
                row_group_size=row_group_size,
                compression=compression,
                **kwargs,
            )
            file_metadata.append(metadata)

        file_metadata = dict(file_metadata)
        for f in file_metadata:
            file_metadata[f].set_file_path(f.split(self.base_path)[-1].lstrip("/"))

        return file_metadata

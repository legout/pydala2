import polars as pl
import pyarrow as pa
import duckdb
import pandas as pd

import polars.selectors as cs
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from fsspec import filesystem as fsspec_filesystem

from ..schema import (
    replace_schema,
    shrink_large_string,
    convert_timestamp,
)
from .misc import get_partitions_from_path, run_parallel
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
    """
    Writes a PyArrow table to Parquet format.

    Args:
        table (pa.Table): The PyArrow table to write.
        path (str): The path to write the Parquet file to.
        filesystem (AbstractFileSystem | None, optional): The filesystem to use for writing the file. Defaults to None.
        row_group_size (int | None, optional): The size of each row group in the Parquet file. Defaults to None.
        compression (str, optional): The compression algorithm to use. Defaults to "zstd".
        **kwargs: Additional keyword arguments to pass to `pq.write_table`.

    Returns:
        tuple[str, pq.FileMetaData]: A tuple containing the file path and the metadata of the written Parquet file.
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
        """
        Initialize the object with the given data, path, schema, and filesystem.

        Parameters:
            data (pa.Table | pl.DataFrame | pl.LazyFrame | pd.DataFrame | duckdb.DuckDBPyRelation):
                The input data, which can be one of the following types: pa.Table, pl.DataFrame, pl.LazyFrame,
                pd.DataFrame, duckdb.DuckDBPyRelation.
            path (str): The path of the data.
            schema (pa.Schema | None): The schema of the data, if available.
            filesystem (AbstractFileSystem | None, optional): The filesystem to use, defaults to None.

        Returns:
            None
        """
        self.schema = schema
        self.data = data
        self.base_path = path
        self.path = None
        self.filesystem = filesystem

    def _to_polars(self):
        """
        Convert the data attribute to a Polars DataFrame.

        This function checks the type of the data attribute and converts it to a Polars DataFrame if it is not
        already one.
        It supports conversion from Arrow tables, Pandas DataFrames, and DuckDBPyRelations.

        """
        if isinstance(self.data, pa.Table):
            self.data = pl.from_arrow(self.data)
        elif isinstance(self.data, pd.DataFrame):
            self.data = pl.from_pandas(self.data)
        elif isinstance(self.data, duckdb.DuckDBPyRelation):
            self.data = self.data.pl()

    def _to_arrow(self):
        """
        Convert the data in the DataFrame to Arrow format.

        This method checks the type of the data and converts it to Arrow format accordingly.
        It supports conversion from Polars DataFrames, Polars LazyFrames, Pandas DataFrames, and DuckDBPyRelations.
        """
        if isinstance(self.data, pl.DataFrame):
            self.data = self.data.to_arrow()
        elif isinstance(self.data, pl.LazyFrame):
            self.data = self.data.collect().to_arrow()
        elif isinstance(self.data, pd.DataFrame):
            self.data = pa.Table.from_pandas(self.data)
        elif isinstance(self.data, duckdb.DuckDBPyRelation):
            self.data = self.data.to_arrow()

    def _set_schema(self):
        """
        Sets the schema of the DataFrame.

        This private method is called internally to set the schema of the DataFrame. It first converts the DataFrame
        to an Arrow table using the `_to_arrow()` method. Then, it checks if a schema has already been set for the
        DataFrame.
        If not, it assigns the schema of the DataFrame's underlying data to the `schema` attribute.

        """
        if self.schema is None:
            self._to_polars()
            self.data = self.data.opt_dtype()

        self._to_arrow()
        self.schema = self.schema or self.data.schema

    def sort_data(self, by: str | list[str] | list[tuple[str, str]] | None = None):
        """
        Sorts the data based on the given criteria.

        Args:
            by (str | list[str] | list[tuple[str, str]] | None): The criteria for sorting the data.
                - If a single string is provided, the data will be sorted by that column name.
                - If a list of strings is provided, the data will be sorted by multiple columns in the order they are
                    listed.
                - If a list of tuples is provided, each tuple should contain a column name and a sort order
                    ('ascending' or 'descending').
            If no criteria is provided, the data will not be sorted.

        """
        if by is not None:
            self._to_arrow()
            if isinstance(by, list | tuple):
                if isinstance(by[0], str):
                    by = [(col, "ascending") for col in by]
            self.data = self.data.sort_by(by)

    def unique(self, columns: bool | str | list[str] = False):
        """
        Generates a unique subset of the DataFrame based on the specified columns.

        Args:
            columns (bool | str | list[str], optional): The columns to use for determining uniqueness.
                If set to False, uniqueness is determined based on all columns.
                If set to a string, uniqueness is determined based on the specified column.
                If set to a list of strings, uniqueness is determined based on the specified columns.
                Defaults to False.

        """
        if columns is not None:
            self._to_polars()
            self.data = self.data.with_columns(cs.by_dtype(pl.Null()).cast(pl.Int64()))
            if isinstance(columns, bool):
                columns = None
            self.data = self.data.unique(columns, maintain_order=True)

    def add_datepart_columns(self, timestamp_column: str|None=None):
        """
        Adds datepart columns to the data.

        Args:
            timestamp_column (str): The name of the timestamp column.

        Returns:
            None
        """
        if timestamp_column is not None:
            
            self._set_schema()
            self._to_polars()
            datepart_columns = {
                col: True
                for col in self.schema.names
                if col in ["year", "month", "week", "yearday", "monthday", "weekday"]
            }
            self.data.with_datepart_columns(
                timestamp_column=timestamp_column, **datepart_columns
            )

    def cast_schema(
        self,
        use_large_string: bool = False,
        tz: str = None,
        ts_unit: str = None,
        remove_tz: bool = False,
    ):
        """
        Casts the schema of the current Table to self.schema.

        Args:
            use_large_string (bool, optional): Whether to use large string format. Defaults to False.
            tz (str, optional): Timezone to use for timestamp conversion. Defaults to None.
            ts_unit (str, optional): Unit to use for timestamp conversion. Defaults to None.
            remove_tz (bool, optional): Whether to remove the timezone from timestamps. Defaults to False.


        """
        self._to_arrow()
        self._set_schema()
        self._use_large_string = use_large_string
        if not use_large_string:
            self.schema = shrink_large_string(self.schema)

        if tz is not None or ts_unit is not None or remove_tz:
            self.schema = convert_timestamp(
                self.schema,
                tz=tz,
                unit=ts_unit,
                remove_tz=remove_tz,
            )

        self.data = replace_schema(self.data, self.schema, exact=True)

    def delta(
        self,
        other: pl.DataFrame | pl.LazyFrame,
        subset: str | list[str] | None = None,
    ):
        """
        Computes the difference between the current DataFrame and another DataFrame or LazyFrame.

        Parameters:
            other (DataFrame | LazyFrame): The DataFrame or LazyFrame to compute the difference with.
            subset (str | list[str] | None, optional): The column(s) to compute the difference on. If `None`,
                the difference is computed on all columns. Defaults to `None`.

        """
        self._to_polars()
        self.data = self.data.delta(other, subset=subset)

    def partition_by(
        self,
        columns: str | list[str] | None = None,
        num_rows: int | None = None,
        strftime: str | list[str] | None = None,
        timedelta: str | list[str] | None = None,
    ):
        """
        Partitions the data by the specified columns, number of rows, strftime format, and timedelta.

        Parameters:
            columns (str | list[str] | None): The column or columns to partition the data by. If None, the data
                will not be partitioned.
            num_rows (int | None): The maximum number of rows in each partition. If None, the data will not be
                partitioned.
            strftime (str | list[str] | None): The strftime format string or list of format strings to use for
                partitioning the data by date or time. If None, the data will not be partitioned by date or time.
            timedelta (str | list[str] | None): The timedelta string or list of timedelta strings to use for
                partitioning the data by time interval. If None, the data will not be partitioned by time interval.


        """
        self._to_polars()
        self.data = self.data.partition_by_ext(
            columns=columns,
            strftime=strftime,
            timedelta=timedelta,
            num_rows=num_rows,
        )

    def set_path(self, base_name: str | None = None):
        """
        Set the path for the data files.

        Args:
            base_name (str | None, optional): The base name for the data files. Defaults to None.

        """
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
        """
        Writes the data to Parquet files.

        Args:
            row_group_size (int | None, optional): The number of rows in each row group. Defaults to None.
            compression (str, optional): The compression algorithm to use. Defaults to "zstd".
            **kwargs: Additional keyword arguments to pass to the underlying Parquet writer.

        Returns:
            dict: A dictionary mapping each file path to its corresponding Parquet metadata.
        """
        if self.path is None:
            raise ValueError("No path set. Call set_path() first.")
        if not isinstance(self.data, list):
            self._to_arrow()
            self.data = [self.data]

        file_metadata = []
        for path, part in zip(self.path, self.data):
            part = part[1].to_arrow()
            if not self._use_large_string:
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

        # def _write(path_part, fs):
        #     path, part = path_part
        #     part = part[1].to_arrow()

        #     if not self._use_large_string:
        #         part = part.cast(shrink_large_string(part.schema))

        #     metadata = write_table(
        #         table=part,
        #         path=path,
        #         filesystem=fs,
        #         row_group_size=row_group_size,
        #         compression=compression,
        #         **kwargs,
        #     )
        #     return metadata

        #     # file_metadata.append(metadata)

        # if len(self.data) == 1:
        #     metadata = _write(list(zip(self.path[0], self.data[0])), self.filesystem)
        #     file_metadata = [metadata]

        # else:
        #     file_metadata = run_parallel(
        #         _write, list(zip(self.path, self.data)), fs=self.filesystem
        #     )

        file_metadata = dict(file_metadata)
        for f in file_metadata:
            file_metadata[f].set_file_path(f.split(self.base_path)[-1].lstrip("/"))

        return file_metadata

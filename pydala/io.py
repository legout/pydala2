import os
import re

import polars as pl
import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.feather as pf
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from fsspec import filesystem as fsspec_filesystem

from .helpers import get_partitions_from_path
from .schema import convert_timestamp, shrink_large_string, unify_schemas


def read_table(
    path: str,
    schema: pa.Schema | None = None,
    # format: str | None = None,
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

    # format = format or os.path.splitext(path)[-1]

    # if re.sub("\.", "", format) == "parquet":
    table = pq.read_table(pa.BufferReader(filesystem.read_bytes(path)), schema=schema)

    # elif re.sub("\.", "", format) == "csv":
    #     table = pc.read_csv(pa.BufferReader(filesystem.read_bytes(path)), schema=schema)

    # elif re.sub("\.", "", format) in ["arrow", "ipc", "feather"]:
    #     table = pf.read_table(
    #         pa.BufferReader(filesystem.read_bytes(path))
    #     )  # , schema=schema
    #     # )

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
    df: pl.DataFrame,
    path: str,
    schema: pa.Schema | None = None,
    # format: str | None = None,
    filesystem: AbstractFileSystem | None = None,
    row_group_size: int | None = None,
    compression: str = "zstd",
    sort_by: str | list[str] | list[tuple[str, str]] | None = None,
    distinct: bool | str | list[str] = False,
    tz: str = "UTC",
    ts_unit: str = "us",
    sort_schema:bool|list[str]=False,
    use_large_string: bool = False,
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
    # format = format or os.path.splitext(path)[-1]

    if distinct:
        if isinstance(distinct, str | list):
            df = df.unique(distinct)
        else:
            df = df.unique()

    table = df.to_arrow()
    for col in schema.names:
        if col not in table.column_names:
            #print(col)
            table = table.append_column(col, pa.nulls(table.shape[0]))
    

    if schema is not None:
        table = table.select(schema.names)
        schema, _ = unify_schemas(
            [schema, table.schema],
            use_large_string=use_large_string,
            ts_unit=ts_unit,
            tz=tz,
            sort=sort_schema
        )

        table = table.cast(schema)

    if sort_by is not None:
        if isinstance(sort_by, list):
            if isinstance(sort_by[0], str):
                sort_by = [(c, "descending") for c in sort_by]
        table = table.sort_by(sorting=sort_by)

    # if re.sub("\.", "", format) == "parquet":
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
    # elif re.sub("\.", "", format) == "csv":
    #     with filesystem.open_output_stream(path) as f:
    #         table = pa.table.from_batches(table.to_batches(), schema=schema)
    #         pc.write_csv(table, f, **kwargs)

    # elif re.sub("\.", "", format) in ["arrow", "ipc", "feather"]:
    #     with filesystem.open_output_scream(path) as f:
    #         table = pa.table.from_batches(table.to_batches(), schema=schema)
    #         pf.write_feather(
    #             f, compression=compression, chunksize=row_group_size, **kwargs
    #         )

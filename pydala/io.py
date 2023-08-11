import os
import re

import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.feather as pf
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from fsspec import filesystem as fsspec_filesystem

from .helpers import get_partitions_from_path


def read_table(
    path: str,
    schema: pa.Schema | None = None,
    format: str | None = None,
    filesystem: AbstractFileSystem | None = None,
    partitioning: str | list[str] | None = None,
) -> pa.Table:
    """Loads the data for the given file path into a pyarrow table.

    Args:
        path (str): File path.
        schema (pa.Schema | None, optional): Pyarrow schema. Defaults to None.
        format (str | None, optional): Format of the file. Could be 'parquet', 'csv' or 'arrow/feather/ipc'.
            Defaults to None.
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

    format = format or os.path.splitext(path)[-1]

    if re.sub("\.", "", format) == "parquet":
        table = pq.read_table(
            pa.BufferReader(filesystem.read_bytes(path)), schema=schema
        )

    elif re.sub("\.", "", format) == "csv":
        table = pc.read_csv(pa.BufferReader(filesystem.read_bytes(path)), schema=schema)

    elif re.sub("\.", "", format) in ["arrow", "ipc", "feather"]:
        table = pf.read_table(
            pa.BufferReader(filesystem.read_bytes(path)), schema=schema
        )

    if partitioning is not None:
        partitions = get_partitions_from_path(path, partitioning=partitioning)

        for key, values in partitions:
            table = table.append_column(
                field_=key, column=pa.array([values] * len(table))
            )

    return table


def write_table(
    table: pa.Table,
    path: str,
    schema: pa.Schema | None = None,
    format: str | None = None,
    filesystem: AbstractFileSystem | None = None,
    row_group_size: int | None = None,
    compression: str = "zstd",
    **kwargs,
):
    """Write pyarrow table to the given file path.

    Args:
        table (pa.Table): Pyarrow table.
        path (str): File path.
        schema (pa.Schema | None, optional): Pyarrow schwma. Defaults to None.
        format (str | None, optional): Format of the file. Could be 'parquet', 'csv' or 'arrow/feather/ipc'.
            Defaults to None.
        filesystem (AbstractFileSystem | None, optional): _description_. Defaults to None.
        row_group_size (int | None, optional): Row group size of the parquet or arrow file. Defaults to None.
        compression (str, optional): Compression of the parquet or arrow file. Defaults to "zstd".
    """
    if filesystem is None:
        filesystem = fsspec_filesystem("file")

    filesystem.invalidate_cache()
    format = format or os.path.splitext(path)[-1]
    schema = kwargs.pop(schema, None) or schema or table.schema

    if re.sub("\.", "", format) == "parquet":
        pq.write_table(
            table,
            path,
            filesystem=filesystem,
            row_group_size=row_group_size,
            compression=compression,
            **kwargs,
        )

    elif re.sub("\.", "", format) == "csv":
        with filesystem.open_output_stream(path) as f:
            table = pa.table.from_batches(table.to_batches(), schema=schema)
            pc.write_csv(table, f, **kwargs)

    elif re.sub("\.", "", format) in ["arrow", "ipc", "feather"]:
        with filesystem.open_output_scream(path) as f:
            table = pa.table.from_batches(table.to_batches(), schema=schema)
            pf.write_feather(
                f, compression=compression, chunksize=row_group_size, **kwargs
            )

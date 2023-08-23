import os
import re

import polars as pl
import pyarrow as pa
import pyarrow.fs as pfs
import pyarrow.parquet as pq
import tqdm
from fsspec import AbstractFileSystem
from joblib import Parallel, delayed

# from .schema import shrink_large_string, convert_timestamp
# from .polars_ext import pl


def get_timedelta_str(timedelta: str, to: str = "polars") -> str:
    polars_timedelta_units = [
        "ns",
        "us",
        "ms",
        "s",
        "m",
        "h",
        "d",
        "w",
        "mo",
        "y",
    ]
    duckdb_timedelta_units = [
        "nanosecond",
        "microsecond",
        "millisecond",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
    ]

    unit = re.sub("[0-9]", "", timedelta).strip()
    val = timedelta.replace(unit, "").strip()
    if to == "polars":
        return (
            timedelta
            if unit in polars_timedelta_units
            else val
            + dict(zip(duckdb_timedelta_units, polars_timedelta_units))[
                re.sub("s$", "", unit)
            ]
        )

    if unit in polars_timedelta_units:
        return (
            f"{val} " + dict(zip(polars_timedelta_units, duckdb_timedelta_units))[unit]
        )

    return f"{val} " + re.sub("s$", "", unit)


def get_timestamp_column(df: pl.DataFrame) -> str | list[str]:
    # return [
    #    col.name
    #    for col in df.to_arrow().schema
    #    if col.type
    #    in [
    #        pa.timestamp("s"),
    #        pa.timestamp("ms"),
    #        pa.timestamp("us"),
    #        pa.timestamp("ns"),
    #        pa.date32(),
    #        pa.date64(),
    #    ]
    # ]
    return [
        name
        for name, type_ in df.schema.items()
        if type_ in [pl.Time(), pl.Datetime(), pl.Date()]
    ]


def run_parallel(
    func: callable,
    func_params: list[any],
    *args,
    n_jobs: int = -1,
    backend: str = "threading",
    verbose: bool = True,
    **kwargs,
) -> list[any]:
    """Runs a function for a list of parameters in parallel.

    Args:
        func (Callable): function to run in Parallelallel.
        func_params (list[any]): parameters for the function
        n_jobs (int, optional): Number of joblib workers. Defaults to -1.
        backend (str, optional): joblib backend. Valid options are
        `loky`,`threading`, `mutliprocessing` or `sequential`.  Defaults to "threading".

    Returns:
        list[any]: Function output.
    """
    if verbose:
        return Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(func)(fp, *args, **kwargs) for fp in tqdm.tqdm(func_params)
        )

    else:
        return Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(func)(fp, *args, **kwargs) for fp in func_params
        )


def collect_file_schemas(
    files: list[str] | str,
    filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
    n_jobs: int = -1,
    backend: str = "threading",
    verbose: bool = True,
) -> dict[str, pa.Schema]:
    """Collect all pyarrow schemas for the given parquet files.

    Args:
        files (list[str] | str): Parquet files.
        filesystem (AbstractFileSystem | pfs.FileSystem | None, optional): Filesystem. Defaults to None.
        n_jobs (int, optional): n_jobs parameter of joblib.Parallel. Defaults to -1.
        backend (str, optional): backend parameter of joblib.Parallel. Defaults to "threading".
        verbose (bool, optional): Wheter to show the task progress using tqdm or not. Defaults to True.

    Returns:
        dict[str, pa.Schema]: Pyarrow schemas of the given files.
    """

    def get_schema(f, filesystem):
        return {f: pq.read_schema(f, filesystem=filesystem)}

    schemas = run_parallel(
        get_schema,
        files,
        filesystem=filesystem,
        backend=backend,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return {key: value for d in schemas for key, value in d.items()}


def collect_metadata(
    files: list[str] | str,
    filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
    n_jobs: int = -1,
    backend: str = "threading",
    verbose: bool = True,
) -> dict[str, pq.FileMetaData]:
    """Collect all metadata information of the given parqoet files.

    Args:
        files (list[str] | str): Parquet files.
        filesystem (AbstractFileSystem | pfs.FileSystem | None, optional): Filesystem. Defaults to None.
        n_jobs (int, optional): n_jobs parameter of joblib.Parallel. Defaults to -1.
        backend (str, optional): backend parameter of joblib.Parallel. Defaults to "threading".
        verbose (bool, optional): Wheter to show the task progress using tqdm or not. Defaults to True.

    Returns:
        dict[str, pq.FileMetaData]: Parquet metadata of the given files.
    """

    def get_metadata(f, filesystem):
        return {f: pq.read_metadata(f, filesystem=filesystem)}

    metadata = run_parallel(
        get_metadata,
        files,
        filesystem=filesystem,
        backend=backend,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return {key: value for d in metadata for key, value in d.items()}


def get_partitions_from_path(
    path: str, partitioning: str | list[str] | None = None
) -> list[tuple]:
    """Get the dataset partitions from the file path.

    Args:
        path (str): File path.
        partitioning (str | list[str] | None, optional): Partitioning type. Defaults to None.

    Returns:
        list[tuple]: Partitions.
    """
    if "." in path:
        path = os.path.dirname(path)

    parts = path.split("/")

    if isinstance(partitioning, str):
        if partitioning == "hive":
            return [tuple(p.split("=")) for p in parts if "=" in p]

        else:
            return [
                (partitioning, parts[0]),
            ]
    else:
        return list(zip(partitioning, parts[-len(partitioning) :]))


def get_row_group_stats(
    row_group: pq.RowGroupMetaData, partitioning: None | str | list[str] = None
):
    stats = {}
    file_path = row_group.column(0).file_path
    stats["file_path"] = file_path
    if "=" in file_path:
        partitioning = partitioning or "hive"
    if partitioning is not None:
        partitions = get_partitions_from_path(file_path, partitioning=partitioning)
        stats.update(dict(partitions))

    stats["num_columns"] = row_group.num_columns
    stats["num_rows"] = row_group.num_rows
    stats["total_byte_size"] = row_group.total_byte_size
    stats["compression"] = row_group.column(0).compression

    for i in range(row_group.num_columns):
        name = row_group.column(i).path_in_schema
        stats[name + "_total_compressed_size"] = row_group.column(
            i
        ).total_compressed_size
        stats[name + "_total_uncompressed_size"] = row_group.column(
            i
        ).total_uncompressed_size

        stats[name + "_physical_type"] = row_group.column(i).physical_type
        if row_group.column(i).is_stats_set:
            stats[name + "_min"] = row_group.column(i).statistics.min
            stats[name + "_max"] = row_group.column(i).statistics.max
            stats[name + "_null_count"] = row_group.column(i).statistics.null_count
            stats[name + "_distinct_count"]: row_group.column(
                i
            ).statistics.distinct_count

    return stats


def partition_by(
    df: pl.DataFrame | pl.LazyFrame,
    timestamp_column: str | None = None,
    columns: str | list[str] | None = None,
    strftime: str | list[str] | None = None,
    timedelta: str | list[str] | None = None,
    num_rows: int | None = None,
):
    if timestamp_column is None:
        timestamp_column = get_timestamp_column(df)[0]

    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]

    else:
        columns = []

    drop_columns = columns.copy()

    if strftime is not None:
        if isinstance(strftime, str):
            strftime = [strftime]

        df = df.with_striftime_columns(
            timestamp_column=timestamp_column, strftime=strftime
        )
        strftime_columns = [
            f"_strftime_{strftime_.replaace('%', '')}_" for strftime_ in strftime
        ]
        columns += strftime_columns
        drop_columns += strftime_columns

    if timedelta:
        if isinstance(timedelta, str):
            timedelta = [timedelta]

        df = df.with_duration_columns(
            timestamp_column=timestamp_column, timedelta=timedelta
        )
        timedelta_columns = [f"_timedelta_{timedelta_}_" for timedelta_ in timedelta]
        columns += timedelta_columns
        drop_columns += timedelta_columns

    datetime_columns = {
        col: col in [col.lower() for col in columns]
        for col in [
            "year",
            "month",
            "week",
            "yearday",
            "monthday",
            "weekday",
            "strftime",
        ]
        if col not in [table_col.lower() for table_col in df.columns]
    }
    if len(datetime_columns):
        df = df.with_datepart_columns(
            timestamp_column=timestamp_column, **datetime_columns
        )

    if num_rows:
        df = df.with_row_count_ext(over=columns).with_columns(
            (pl.col("row_nr") - 1) // num_rows
        )
        columns += ["row_nr"]
        drop_columns += ["row_nr"]

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    partitions = [
        (p.select(columns).unique().to_dicts()[0], p.drop(drop_columns))
        for p in df.partition_by(by=columns, as_dict=False)
    ]

    return partitions

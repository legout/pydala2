import os
import re

import polars as pl
import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.feather as pf
import pyarrow.fs as pfs
import pyarrow.parquet as pq
import tqdm
from joblib import Parallel, delayed
from fsspec.implementations import dirfs, cached as cachedfs
from fsspec import filesystem as fsspec_filesystem, AbstractFileSystem


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
    return [
        col.name
        for col in df.to_arrow().schema
        if col.type
        in [
            pa.timestamp("s"),
            pa.timestamp("ms"),
            pa.timestamp("us"),
            pa.timestamp("ns"),
            pa.date32(),
            pa.date64(),
        ]
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


def set_timezone(
    schema: pa.Schema, timestamp_fields: str | list[str] | None = None, tz: None = None
) -> pa.Schema:
    """Set timezone for every timestamp in a pyarrow schema.

    Args:
        schema (pa.Schema): Pyarrow schema.
        timestamp_fields (str | list[str] | None, optional): timestamp fields. Defaults to None.
        tz (None, optional): timezone. Defaults to None.

    Returns:
        pa.Schema: Pyarrow schema with give timezone.
    """
    if isinstance(timestamp_fields, str):
        timestamp_fields = [timestamp_fields]

    for timestamp_field in timestamp_fields:
        timestamp_field_idx = schema.get_field_index(timestamp_field)
        unit = schema.field(timestamp_field).type.unit
        schema = schema.remove(timestamp_field_idx).insert(
            timestamp_field_idx,
            pa.field(timestamp_field, pa.timestamp(unit=unit, tz=tz)),
        )

    return schema


def sort_schema(schema: pa.Schema) -> pa.Schema:
    """Sort fields of a pyarrow schema in alphabetical order.

    Args:
        schema (pa.Schema): Pyarrow schema.

    Returns:
        pa.Schema: Sorted pyarrow schema.
    """
    return pa.schema(
        [
            pa.field(name, type_)
            for name, type_ in sorted(zip(schema.names, schema.types))
        ]
    )


def _unify_schemas(
    schema1: pa.Schema, schema2: pa.Schema, tz: str | None = "UTC"
) -> tuple[dict, bool]:
    """Returns a unified pyarrow schema.

    Args:
        schema1 (pa.Schema): Pyarrow schema 1.
        schema2 (pa.Schema): Pyarrow schema 2.
        tz (str): timezone for timestamp fields. Defaults to "UTC".

    Returns:
        Tuple[dict, bool]: Unified pyarrow schema, bool value if schemas were equal.
    """

    dtype_rank = [
        pa.null(),
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
        pa.float16(),
        pa.float32(),
        pa.float64(),
        pa.string(),
    ]
    timestamp_rank = ["ns", "us", "ms", "s"]

    # check for equal columns and column order
    if schema1.names == schema2.names:
        if schema1.types == schema2.types:
            return schema1, True

        all_names = schema1.names

    elif sorted(schema1.names) == sorted(schema2.names):
        all_names = sorted(schema1.names)

    else:
        all_names = sorted(set(schema1.names + schema2.names))

    file_schemas_equal = True
    schema = []
    for name in all_names:
        if name in schema1.names:
            type1 = schema1.field(name).type
        else:
            type1 = schema2.field(name).type
        if name in schema2.names:
            type2 = schema2.field(name).type
        else:
            type2 = schema1.field(name).type

        if type1 != type2:
            file_schemas_equal = False

            if isinstance(type1, pa.TimestampType):
                rank1 = timestamp_rank.index(type1.unit)
                rank2 = timestamp_rank.index(type2.unit)
                if type1.tz != tz:
                    type1 = pa.timestamp(type1.unit, tz=tz)
                if type2.tz != tz:
                    type2 = pa.timestamp(type2.unit, tz=tz)
            else:
                rank1 = dtype_rank.index(type1) if type1 in dtype_rank else 0
                rank2 = dtype_rank.index(type2) if type2 in dtype_rank else 0

            schema.append(pa.field(name, type1 if rank1 > rank2 else type2))

        else:
            schema.append(pa.field(name, type1))

    return pa.schema(schema), file_schemas_equal


def unify_schemas(schemas: list[pa.Schema]) -> tuple[pa.Schema, bool]:
    """Get the unified pyarrow schema for a list of schemas.

    Args:
        schemas (list[pa.Schema]): Pyarrow schemas.

    Returns:
        tuple[pa.Schema, bool]: Unified pyarrow schema.
    """

    schemas_equal = True
    unified_schema = schemas[0]
    for schema in schemas[1:]:
        unified_schema, schemas_equal_ = _unify_schemas(unified_schema, schema)

        schemas_equal *= schemas_equal_

    return unified_schema, schemas_equal


def repair_schema(
    files: list[str],
    schema: pa.Schema | None = None,
    filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
    n_jobs: int = -1,
    backend: str = "threading",
    verbose: bool = True,
    **kwargs,
):
    """Repairs the pyarrow schema of a parquet or arrow dataset

    Args:
        files (list[str]): Files of the dataset.
        schema (pa.Schema | None, optional): Unified pyarrow schema.
            If None, the unified schema is generated from the given files of the dataset. Defaults to None.
        filesystem (AbstractFileSystem | pfs.FileSystem | None, optional): Filesystem. Defaults to None.
        n_jobs (int, optional): n_jobs parameter of joblib.Parallel. Defaults to -1.
        backend (str, optional): backend parameter of joblib.Parallel. Defaults to "threading".
        verbose (bool, optional): Wheter to show the task progress using tqdm or not. Defaults to True.
    """
    if schema is None:
        schemas = collect_file_schemas(
            files=files,
            filesystem=filesystem,
            n_jobs=n_jobs,
            backend=backend,
            verbose=verbose,
        )
        schema, schemas_equal = unify_schemas(schemas=schemas)

    def _repair_schema(f, schema, filesystem):
        filesystem.invalidate_cache()
        # file_schema = pq.read_metadata(f, filesystem=filesystem)
        # if file_schema!=schema:
        table = read_table(f, schema, filesystem=filesystem)
        write_table(table, f, filesystem=filesystem, **kwargs)

    _ = run_parallel(
        _repair_schema,
        files,
        schema=schema,
        filesystem=filesystem,
        backend=backend,
        n_jobs=n_jobs,
        verbose=verbose,
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
        filesystem.invalidate_cache()
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
        filesystem.invalidate_cache()
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


def get_filesystem(
    bucket: str | None = None, fs=AbstractFileSystem | None, cached: bool = False
):
    if fs is None:
        fs = fsspec_filesystem("file")

    if bucket is not None:
        fs = dirfs.DirFileSystem(path=bucket, fs=fs)

    if cached:
        fs = cachedfs.SimpleCacheFileSystem(
            cache_storage=os.path.expanduser("~/.tmp"),
            check_files=True,
            cache_check=100,
            expire_time=24 * 60 * 60,
            fs=fs,
            same_names=True,
        )
    return fs


# Polars utils


def unnest_all(df: pl.DataFrame, seperator="_", fields: list[str] | None = None):
    def _unnest_all(struct_columns):
        return (
            df.with_columns(
                [
                    pl.col(col).struct.rename_fields(
                        [
                            f"{col}{seperator}{field_name}"
                            for field_name in df[col].struct.fields
                        ]
                    )
                    for col in struct_columns
                ]
            )
            .unnest(struct_columns)
            .select(
                list(set(df.columns) - set(struct_columns))
                + sorted(
                    [
                        f"{col}{seperator}{field_name}"
                        for field_name in fields
                        for col in struct_columns
                    ]
                )
            )
        )

    struct_columns = [
        col for col in df.columns if df[col].dtype == pl.Struct()
    ]  # noqa: F821
    while len(struct_columns):
        df = _unnest_all(struct_columns=struct_columns)
        struct_columns = [col for col in df.columns if df[col].dtype == pl.Struct()]
    return df


def _opt_dtype(s: pl.Series) -> pl.Series:
    if (s.str.contains("^[0-9,\.-]{1,}$") | s.is_null()).all():
        s = s.str.replace_all(",", ".").str.replace_all(".0$", "")
        if s.str.contains("\.").any():
            s = s.cast(pl.Float64()).shrink_dtype()
        else:
            if (s.str.lengths() > 0).all():
                s = s.cast(pl.Int64()).shrink_dtype()
    elif s.str.contains("^[T,t]rue|[F,f]alse$").all():
        s = s.str.contains("^[T,t]rue$")

    return s


def opt_dtype(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.all().map(_opt_dtype))


def explode_all(df: pl.DataFrame | pl.LazyFrame):
    list_columns = [col for col in df.columns if df[col].dtype == pl.List()]
    for col in list_columns:
        df = df.explode(col)
    return df


pl.DataFrame.unnest_all = unnest_all
pl.DataFrame.explode_all = explode_all


def with_strftime_columns(
    df: pl.DataFrame | pl.LazyFrame,
    timestamp_column: str,
    strftime: str | list[str],
    column_names: str | list[str] | None = None,
):
    if isinstance(strftime, str):
        strftime = [strftime]
    if isinstance(column_names, str):
        column_names = [column_names]

    if column_names is None:
        column_names = [
            f"_strftime_{strftime_.replace('%', '').replace('-', '_')}_"
            for strftime_ in strftime
        ]
    return df.with_columns(
        [
            pl.col(timestamp_column).dt.strftime(strftime_).alias(column_name)
            for strftime_, column_name in zip(strftime, column_names)
        ]
    )


def with_duration_columns(
    df: pl.DataFrame | pl.LazyFrame,
    timestamp_column: str,
    timedelta: str | list[str],
    column_names: str | list[str] | None = None,
):
    if isinstance(timedelta, str):
        timedelta = [timedelta]

    if isinstance(column_names, str):
        column_names = [column_names]

    if column_names is None:
        column_names = [
            f"_timebucket_{timedelta_.replace(' ', '_')}_" for timedelta_ in timedelta
        ]

    timedelta = [get_timedelta_str(timedelta_, to="polars") for timedelta_ in timedelta]
    return df.with_columns(
        [
            pl.col(timestamp_column).dt.truncate(timedelta_).alias(column_name)
            for timedelta_, column_name in zip(timedelta, column_names)
        ]
    )


def with_datepart_columns(
    df: pl.DataFrame | pl.LazyFrame,
    timestamp_column: str | None = None,
    year: bool = False,
    month: bool = False,
    week: bool = False,
    yearday: bool = False,
    monthday: bool = False,
    weekday: bool = False,
    strftime: str | None = None,
):
    if not timestamp_column:
        timestamp_column = get_timestamp_column(df)

    if strftime:
        if isinstance(strftime, str):
            strftime = [strftime]
        column_names = [
            f"_strftime_{strftime_.replace('%', '').replace('-', '_')}_"
            for strftime_ in strftime
        ]
    else:
        strftime = []
        column_names = []

    if year:
        strftime.append("%Y")
        column_names.append("year")
    if month:
        strftime.append("%m")
        column_names.append("month")
    if week:
        strftime.append("%W")
        column_names.append("week")
    if yearday:
        strftime.append("%j")
        column_names.append("year_day")
    if monthday:
        strftime.append("%d")
        column_names.append("month_day")
    if weekday:
        strftime.append("%a")
        column_names.append("week_day")

    return with_strftime_columns(
        df=df,
        timestamp_column=timestamp_column,
        strftime=strftime,
        column_names=column_names,
    )


def with_row_count(
    df: pl.DataFrame | pl.LazyFrame,
    over: str | list[str] | None = None,
):
    if over:
        if len(over) == 0:
            over = None

    if isinstance(over, str):
        over = [over]

    if over:
        return df.with_columns(pl.lit(1).alias("row_nr")).with_columns(
            pl.col("row_nr").cumsum().over(over)
        )
    else:
        return df.with_columns(pl.lit(1).alias("row_nr")).with_columns(
            pl.col("row_nr").cumsum()
        )


pl.DataFrame.unnest_all = unnest_all
pl.DataFrame.explode_all = explode_all
pl.DataFrame.opt_dtype = opt_dtype
pl.DataFrame.with_row_count_ext = with_row_count
pl.DataFrame.with_datepart_columns = with_datepart_columns
pl.DataFrame.with_duration_columns = with_duration_columns
pl.DataFrame.with_striftime_columns = with_strftime_columns

pl.LazyFrame.unnest_all = unnest_all
pl.LazyFrame.explode_all = explode_all
pl.LazyFrame.opt_dtype = opt_dtype
pl.LazyFrame.with_row_count_ext = with_row_count
pl.LazyFrame.with_datepart_columns = with_datepart_columns
pl.LazyFrame.with_duration_columns = with_duration_columns
pl.LazyFrame.with_striftime_columns = with_strftime_columns


def partition_by(
    df: pl.DataFrame | pl.LazyFrame,
    timestamp_column: str | None = None,
    columns: str | list[str] | None = None,
    strftime: str | list[str] | None = None,
    timedelta: str | list[str] | None = None,
    num_rows: int | None = None,
):
    drop_columns = columns.copy()

    if timestamp_column is None:
        timestamp_column = get_timestamp_column(df)[0]

    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]

    else:
        columns = []

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
            pl.col("row_nr") // (num_rows + 1)
        )
        columns += ["row_nr"]
        drop_columns = ["row_nr"]

    return df, columns, drop_columns

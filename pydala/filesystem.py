import datetime as dt
import inspect
import os
import posixpath
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

import duckdb as ddb
import orjson
import pandas as pd
import polars as pl
import psutil
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.fs as pfs
import pyarrow.parquet as pq
import s3fs
from fsspec import AbstractFileSystem, filesystem
from fsspec.implementations.cache_mapper import AbstractCacheMapper
from fsspec.implementations.cached import SimpleCacheFileSystem

# from fsspec.implementations import cached as cachedfs
from fsspec.implementations.dirfs import DirFileSystem
from loguru import logger

from .helpers.misc import read_table, run_parallel
from .schema import convert_large_types_to_normal


def get_credentials_from_fssspec(fs: AbstractFileSystem) -> dict[str, str]:
    if "s3" in fs.protocol:
        credendials = fs.s3._get_credentials()
        return {
            "access_key": credendials.access_key,
            "secret_key": credendials.secret_key,
            "session_token": credendials.token,
            "endpoint_override": fs.s3._endpoint.host,
        }


def get_total_directory_size(directory: str):
    return sum(f.stat().st_size for f in Path(directory).glob("**/*") if f.is_file())


class FileNameCacheMapper(AbstractCacheMapper):
    def __init__(self, directory):
        self.directory = directory

    def __call__(self, path: str) -> str:
        os.makedirs(
            posixpath.dirname(posixpath.join(self.directory, path)), exist_ok=True
        )
        return path


class throttle(object):
    """
    Decorator that prevents a function from being called more than once every
    time period.
    To create a function that cannot be called more than once a minute:
        @throttle(minutes=1)
        def my_fun():
            pass
    """

    def __init__(self, seconds=0, minutes=0, hours=0):
        self.throttle_period = timedelta(seconds=seconds, minutes=minutes, hours=hours)
        self.time_of_last_call = datetime.min

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            now = datetime.now()
            time_since_last_call = now - self.time_of_last_call

            if time_since_last_call > self.throttle_period:
                self.time_of_last_call = now
                return fn(*args, **kwargs)

        return wrapper


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


last_free = None
first_free = None


def get_friendly_disk_usage(storage: str) -> str:
    global last_free
    global first_free
    usage = psutil.disk_usage(storage)
    if first_free is None:
        first_free = usage.free
    current_usage = get_total_directory_size(storage)
    message = f"{sizeof_fmt(current_usage)} used {sizeof_fmt(usage.free)} available"
    if last_free is not None:
        downloaded_recently = last_free - usage.free
        if downloaded_recently > 10_000_000:
            downloaded_since_start = first_free - usage.free
            if downloaded_recently != downloaded_since_start:
                message += f" delta: {sizeof_fmt(downloaded_recently)}"
            message += f" delta since start: {sizeof_fmt(downloaded_since_start)}"

    last_free = usage.free
    return message


#############################################################################################################
# This was originally implemented by Burak Emre Kabakcı (@buremba on github) in its universql project
# project: https://github.com/buremba/universql
# original code: https://github.com/buremba/universql/blob/main/universql/lake/fsspec_util.py
#############################################################################################################


class MonitoredSimpleCacheFileSystem(SimpleCacheFileSystem):
    def __init__(self, **kwargs):
        # kwargs["cache_storage"] = posixpath.join(
        #    kwargs.get("cache_storage"), kwargs.get("fs").protocol[0]
        # )
        self._verbose = kwargs.get("verbose", False)
        super().__init__(**kwargs)
        self._mapper = FileNameCacheMapper(kwargs.get("cache_storage"))

    def _check_file(self, path):
        self._check_cache()
        cache_path = self._mapper(path)
        for storage in self.storage:
            fn = posixpath.join(storage, cache_path)
            if posixpath.exists(fn):
                return fn
            if self._verbose:
                logger.info(f"Downloading {self.protocol[0]}://{path}")

    # def glob(self, path):
    #    return [self._strip_protocol(path)]

    def size(self, path):
        cached_file = self._check_file(self._strip_protocol(path))
        if cached_file is None:
            return self.fs.size(path)
        else:
            return posixpath.getsize(cached_file)

    # def make_dirs(self, path, exist_ok=True):
    #     if self.fs.exists(path) and :
    #         return

    #     return self.fs.makedirs(path, exist_ok=exist_ok)

    def __getattribute__(self, item):
        if item in {
            # new items
            "size",
            "glob",
            # previous
            "load_cache",
            "_open",
            "save_cache",
            "close_and_update",
            "__init__",
            "__getattribute__",
            "__reduce__",
            "_make_local_details",
            "open",
            "cat",
            "cat_file",
            "cat_ranges",
            "get",
            "read_block",
            "tail",
            "head",
            "info",
            "ls",
            "exists",
            "isfile",
            "isdir",
            "_check_file",
            "_check_cache",
            "_mkcache",
            "clear_cache",
            "clear_expired_cache",
            "pop_from_cache",
            "local_file",
            "_paths_from_path",
            "get_mapper",
            "open_many",
            "commit_many",
            "hash_name",
            "__hash__",
            "__eq__",
            "to_json",
            "to_dict",
            "cache_size",
            "pipe_file",
            "pipe",
            "start_transaction",
            "end_transaction",
        }:
            # all the methods defined in this class. Note `open` here, since
            # it calls `_open`, but is actually in superclass
            return lambda *args, **kw: getattr(type(self), item).__get__(self)(
                *args, **kw
            )
        if item in ["__reduce_ex__"]:
            raise AttributeError
        if item in ["transaction"]:
            # property
            return type(self).transaction.__get__(self)
        if item in ["_cache", "transaction_type"]:
            # class attributes
            return getattr(type(self), item)
        if item == "__class__":
            return type(self)
        d = object.__getattribute__(self, "__dict__")
        fs = d.get("fs", None)  # fs is not immediately defined
        if item in d:
            return d[item]
        elif fs is not None:
            if item in fs.__dict__:
                # attribute of instance
                return fs.__dict__[item]
            # attributed belonging to the target filesystem
            cls = type(fs)
            m = getattr(cls, item)
            if (inspect.isfunction(m) or inspect.isdatadescriptor(m)) and (
                not hasattr(m, "__self__") or m.__self__ is None
            ):
                # instance method
                return m.__get__(fs, cls)
            return m  # class method or attribute
        else:
            # attributes of the superclass, while target is being set up
            return super().__getattribute__(item)


def get_new_file_names(src: list[str], dst: list[str]) -> list[str]:
    """
    Returns a list of new file names that are not in the destination list

    Parameters
    ----------
    src : list[str]
        List of source file paths
    dst : list[str]
        List of destination file paths

    Returns
    -------
    list[str]
        List of new file names that are not in the destination list
    """
    if len(dst) == 0:
        return src
    src_file_names = [posixpath.basename(f).split(".")[0] for f in src]
    src_ext = posixpath.basename(src[0]).split(".")[1]
    dst_file_names = [posixpath.basename(f).split(".")[0] for f in dst]
    # dst_ext = posixpath.basename(dst[0]).split(".")[1]

    return [
        posixpath.join(posixpath.dirname(src), f, src_ext)
        for f in src_file_names
        if f not in dst_file_names
    ]


def read_parquet(
    self, path: str, filename: bool = False, **kwargs
) -> dict[str, pl.DataFrame] | pl.DataFrame:
    data = pl.from_arrow(read_table(path, filesystem=self, **kwargs))

    if filename:
        return {path: data}

    return data


def read_parquet_schema(self, path: str, **kwargs) -> pa.Schema:
    return pq.read_schema(path, filesystem=self, **kwargs)


def read_paruet_metadata(self, path: str, **kwargs) -> pq.FileMetaData:
    return pq.read_metadata(path, filesystem=self, **kwargs)


def read_json(
    self,
    path: str,
    filename: bool = False,
    as_dataframe: bool = True,
    flatten: bool = True,
) -> dict | pl.DataFrame:
    with self.open(path) as f:
        # data = msgspec.json.decode(f.read())
        data = orjson.loads(f.read())

    if as_dataframe:
        data = pl.from_dicts(data)
        if flatten:
            data = data.explode_all().unnest_all()

    if filename:
        return {path: data}

    return data


def read_csv(
    self, path: str, filename: bool = False, **kwargs
) -> dict[str, pl.DataFrame] | pl.DataFrame:
    with self.open(path) as f:
        data = pl.from_csv(f, **kwargs)

    if filename:
        return {path: data}

    return data


def read_parquet_dataset(
    self, path: str | list[str], concat: bool = True, filename: bool = False, **kwargs
) -> (
    dict[str, pl.DataFrame]
    | pl.DataFrame
    | list[dict[str, pl.DataFrame]]
    | list[pl.DataFrame]
):
    if filename:
        concat = False

    if isinstance(path, str):
        files = self.glob(posixpath.join(path, "*.parquet"))
    else:
        files = path
    if isinstance(files, str):
        files = [files]

    dfs = run_parallel(self.read_parquet, files, filename, **kwargs)

    dfs = pl.concat(dfs, how="diagonal_relaxed") if concat else dfs

    dfs = dfs[0] if len(dfs) == 1 else dfs

    return dfs


def read_json_dataset(
    self,
    path: str | list[str],
    filename: bool = False,
    as_dataframe: bool = True,
    flatten: bool = True,
    concat: bool = True,
) -> (
    dict[str, pl.DataFrame]
    | pl.DataFrame
    | list[dict[str, pl.DataFrame]]
    | list[pl.DataFrame]
):
    if filename:
        concat = False

    if isinstance(path, str):
        files = self.glob(posixpath.join(path, "*.json"))
    else:
        files = path
    if isinstance(files, str):
        files = [files]

    data = run_parallel(
        self.read_json,
        files,
        filename=filename,
        as_dataframe=as_dataframe,
        flatten=flatten,
    )
    if as_dataframe and concat:
        data = pl.concat(data, how="diagonal_relaxed")

    data = data[0] if len(data) == 1 else data

    return data


def read_csv_dataset(
    self, path: str | list[str], concat: bool = True, filename: bool = False, **kwargs
) -> (
    dict[str, pl.DataFrame]
    | pl.DataFrame
    | list[dict[str, pl.DataFrame]]
    | list[pl.DataFrame]
):
    if filename:
        concat = False

    if isinstance(path, str):
        files = self.glob(posixpath.join(path, "*.csv"))
    else:
        files = path
    if isinstance(files, str):
        files = [files]
    dfs = run_parallel(self.read_csv, files, self=self, **kwargs)

    dfs = pl.concat(dfs, how="diagonal_relaxed") if concat else dfs
    dfs = dfs[0] if len(dfs) == 1 else dfs

    return dfs


def pyarrow_dataset(
    self,
    path: str,
    format="parquet",
    schema: pa.Schema | None = None,
    partitioning: str | list[str] | pds.Partitioning = None,
    **kwargs,
) -> pds.Dataset:
    return pds.dataset(
        self.glob(posixpath.join(path, f"*.{format}")),
        filesystem=self,
        partitioning=partitioning,
        schema=schema,
        **kwargs,
    )


def pyarrow_parquet_dataset(
    self,
    path: str,
    schema: pa.Schema | None = None,
    partitioning: str | list[str] | pds.Partitioning = None,
    **kwargs,
) -> pds.FileSystemDataset:
    return pds.dataset(
        posixpath.join(path, "_metadata"),
        filesystem=self,
        partitioning=partitioning,
        schema=schema,
        **kwargs,
    )


def write_parquet(
    self,
    data: pl.DataFrame | pa.Table | pd.DataFrame | ddb.DuckDBPyRelation,
    path: str,
    **kwargs,
) -> None:
    if isinstance(data, pl.DataFrame):
        data = data.to_arrow()
        data = data.cast(convert_large_types_to_normal(data.schema))
    elif isinstance(data, pd.DataFrame):
        data = pa.Table.from_pandas(data, preserve_index=False)
    elif isinstance(data, ddb.DuckDBPyRelation):
        data = data.arrow()

    pq.write_table(data, path, filesystem=self, **kwargs)


def write_json(
    self,
    data: dict | pl.DataFrame | pa.Table | pd.DataFrame | ddb.DuckDBPyRelation,
    path: str,
) -> None:
    if isinstance(data, pl.DataFrame):
        data = data.to_arrow()
        data = data.cast(convert_large_types_to_normal(data.schema)).to_pydict()
    elif isinstance(data, pd.DataFrame):
        data = pa.Table.from_pandas(data, preserve_index=False).to_pydict()
    elif isinstance(data, ddb.DuckDBPyRelation):
        data = data.arrow().to_pydict()

    with self.open(path, "w") as f:
        # f.write(msgspec.json.encode(data))
        f.write(orjson.dumps(data))


def write_csv(
    self,
    data: pl.DataFrame | pa.Table | pd.DataFrame | ddb.DuckDBPyRelation,
    path: str,
    **kwargs,
) -> None:
    if isinstance(data, pa.Table):
        data = pl.from_arrow(data)
    elif isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)
    elif isinstance(data, ddb.DuckDBPyRelation):
        data = data.pl()

    with self.open(path, "w") as f:
        data.write_csv(f, **kwargs)


def write_to_pyarrow_dataset(
    self,
    data: (
        pl.DataFrame
        | pa.Table
        | pd.DataFrame
        | ddb.DuckDBPyRelation
        | list[pl.DataFrame]
        | list[pa.Table]
        | list[pd.DataFrame]
        | list[ddb.DuckDBPyRelation]
    ),
    path: str,
    basename: str | None = None,
    concat: bool = True,
    schema: pa.Schema | None = None,
    partitioning: str | list[str] | pds.Partitioning | None = None,
    partitioning_flavor: str = "hive",
    mode: str = "append",
    format: str | None = "parquet",
    **kwargs,
) -> None:
    if not isinstance(data, list):
        data = [data]

    if isinstance(data[0], pl.DataFrame):
        data = [dd.to_arrow() for dd in data]
        data = [dd.cast(convert_large_types_to_normal(dd.schema)) for dd in data]

    elif isinstance(data[0], pd.DataFrame):
        data = [pa.Table.from_pandas(dd, preserve_index=False) for dd in data]

    elif isinstance(data, ddb.DuckDBPyRelation):
        data = [dd.arrow() for dd in data]

    if concat:
        data = pa.concat_tables(data, promote=True)

    if mode == "delete_matching":
        existing_data_behavior = "delete_matching"
    elif mode == "append":
        existing_data_behavior = "overwrite_or_ignore"
    elif mode == "overwrite":
        self.rm(path, recursive=True)
        existing_data_behavior = "overwrite_or_ignore"
    else:
        existing_data_behavior = mode

    if basename is None:
        basename = (
            f"data-{dt.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]}"
            + "-{i}"
            + f".{format}"
        )

    pds.write_dataset(
        data=data,
        base_dir=path,
        basename_template=basename,
        partitioning=partitioning,
        partitioning_flavor=partitioning_flavor,
        filesystem=self,
        existing_data_behavior=existing_data_behavior,
        schema=schema,
        format=format,
        **kwargs,
    )


def _json_to_parquet(
    self,
    src: str,
    dst: str,
    flatten: bool = True,
    distinct: bool = True,
    sort_by: str | list[str] | None = None,
    auto_optimize_dtypes: bool = True,
    **kwargs,
):
    data = self.read_json(src, as_dataframe=True, flatten=flatten)

    if auto_optimize_dtypes:
        data = data.opt_dtype()

    if distinct:
        data = data.unique(maintain_order=True)

    if sort_by is not None:
        data = data.sort(sort_by)

    if ".parquet" not in dst:
        dst = posixpath.join(
            dst, f"{posixpath.basename(src).replace('.json', '.parquet')}"
        )

    self.write_parquet(data, dst, **kwargs)


def _csv_to_parquet(
    self,
    src: str,
    dst: str,
    distinct: bool = True,
    sort_by: str | list[str] | None = None,
    auto_optimize_dtypes: bool = True,
    **kwargs,
):
    data = self.read_csv(src, self, **kwargs)

    if auto_optimize_dtypes:
        data = data.opt_dtype()

    if distinct:
        data = data.unique(maintain_order=True)

    if sort_by is not None:
        data = data.sort(sort_by)

    if ".parquet" not in dst:
        dst = posixpath.join(
            dst, f"{posixpath.basename(src).replace('.csv', '.parquet')})"
        )

    self.write_parquet(data, dst, **kwargs)


def json_to_parquet(
    self,
    src: str,
    dst: str,
    flatten: bool = True,
    sync: bool = True,
    parallel: bool = True,
    distinct: bool = True,
    sort_by: str | list[str] | None = None,
    auto_optimize_dtypes: bool = True,
    **kwargs,
):
    if sync:
        src_files = self.glob(posixpath.join(src, "*.json"))
        dst_files = self.glob(posixpath.join(dst, "*.parquet"))
        new_src_files = get_new_file_names(src_files, dst_files)

    else:
        new_src_files = self.glob(posixpath.join(src, "*.json"))

    kwargs.pop("backend", None)
    if len(new_src_files) == 1:
        parallel = False

    if len(new_src_files) > 0:
        run_parallel(
            self._json_to_parquet,
            new_src_files,
            dst=dst,
            flatten=flatten,
            backend="threading" if parallel else "sequential",
            distinct=distinct,
            sort_by=sort_by,
            auto_optimize_dtypes=auto_optimize_dtypes,
            **kwargs,
        )


def csv_to_parquet(
    self,
    src: str,
    dst: str,
    sync: bool = True,
    parallel: bool = True,
    distinct: bool = True,
    sort_by: str | list[str] | None = None,
    auto_optimize_dtypes: bool = True,
    **kwargs,
):
    if sync:
        src_files = self.glob(posixpath.join(src, "*.csv"))
        dst_files = self.glob(posixpath.join(dst, "*.parquet"))
        new_src_files = get_new_file_names(src_files, dst_files)

    else:
        new_src_files = self.glob(posixpath.join(src, "*.csv"))

    kwargs.pop("backend", None)
    if len(new_src_files) == 1:
        parallel = False
    if len(new_src_files) > 0:
        run_parallel(
            self._csv_to_parquet,
            new_src_files,
            dst=dst,
            backend="threading" if parallel else "sequential",
            distinct=distinct,
            sort_by=sort_by,
            auto_optimize_dtypes=auto_optimize_dtypes,
            **kwargs,
        )


def ls(
    self,
    path,
    recursive: bool = False,
    maxdepth: int | None = None,
    detail: bool = False,
    files_only: bool = False,
    dirs_only: bool = False,
):
    if detail:
        return self.listdir(path)

    if isinstance(path, str):
        if self.isfile(path):
            path = [path]
        else:
            path = path.rstrip("*").rstrip("/")
            if self.exists(path):
                if not recursive:
                    path = self.glob(posixpath.join(path, "*"))
                else:
                    if maxdepth is not None:
                        path = self.glob(posixpath.join(path, *(["*"] * maxdepth)))
                    else:
                        path = self.glob(posixpath.join(path, "**"))
            else:
                path = self.glob(path)

    if files_only:
        files = [f for f in path if "." in posixpath.basename(f)]
        files += [f for f in sorted(set(path) - set(files)) if self.isfile(f)]
        return files

    if dirs_only:
        dirs = [f for f in path if "." not in posixpath.basename(f)]
        dirs += [f for f in sorted(set(path) - set(dirs)) if self.isdir(f)]
        return dirs

    return path


def sync_folder(
    self, src: str, dst: str, recursive: bool = False, maxdepth: int | None = None
):
    src_ = self.lss(src, recursive=recursive, maxdepth=maxdepth, files_only=True)
    dst_ = self.lss(dst, recursive=recursive, maxdepth=maxdepth, files_only=True)

    if len(src) == 0:
        return

    src_names = [posixpath.basename(f) for f in src_]
    dst_names = [posixpath.basename(f) for f in dst_]

    new_src = [
        posixpath.join(posixpath.dirname(src[0]), f)
        for f in sorted(set(src_names) - set(dst_names))
    ]

    if len(new_src):
        self.cp(new_src, dst)


# NOTE: This is not working properly due to some event loop issues

# def list_files_recursive(self, path: str, format: str = ""):
#     bucket, prefix = path.split("/", maxsplit=1)
#     return [
#         f["Key"]
#         for f in asyncio.run(self.s3.list_objects_v2(Bucket=bucket, Prefix=prefix))[
#             "Contents"
#         ]
#         if f["Key"].endswith(format)
#     ]


# async def _list_files_recursive(
#     self, path: str, format: str = "", max_items: int = 10000
# ):
#     bucket, prefix = path.split("/", maxsplit=1)
#     continuation_token = None
#     files = []

#     while True:
#         if continuation_token:
#             response = await self.s3.list_objects_v2(
#                 Bucket=bucket,
#                 Prefix=prefix,
#                 ContinuationToken=continuation_token,
#                 MaxKeys=max_items,
#             )
#         else:
#             response = await self.s3.list_objects_v2(
#                 Bucket=bucket, Prefix=prefix, MaxKeys=max_items
#             )

#         if "Contents" in response:
#             files.extend(
#                 [f["Key"] for f in response["Contents"] if f["Key"].endswith(format)]
#             )

#         if response.get("IsTruncated"):  # Check if there are more objects to retrieve
#             continuation_token = response.get("NextContinuationToken")
#         else:
#             break

#     return files


# def list_files_recursive(self, path: str, format: str = "", max_items: int = 10000):
#     loop = asyncio.get_event_loop()
#     if loop.is_closed():
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#     return loop.run_until_complete(_list_files_recursive(self, path, format, max_items))


AbstractFileSystem.read_parquet = read_parquet
AbstractFileSystem.read_parquet_dataset = read_parquet_dataset
AbstractFileSystem.write_parquet = write_parquet
# DirFileSystem.write_to_parquet_dataset = write_to_pydala_dataset
DirFileSystem.write_to_dataset = write_to_pyarrow_dataset
AbstractFileSystem.read_parquet_schema = read_parquet_schema
AbstractFileSystem.read_parquet_metadata = read_paruet_metadata

AbstractFileSystem.read_csv = read_csv
AbstractFileSystem.read_csv_dataset = read_csv_dataset
AbstractFileSystem.write_csv = write_csv
AbstractFileSystem._csv_to_parquet = _csv_to_parquet
AbstractFileSystem.csv_to_parquet = csv_to_parquet

AbstractFileSystem.read_json = read_json
AbstractFileSystem.read_json_dataset = read_json_dataset
AbstractFileSystem.write_json = write_json
AbstractFileSystem._json_to_parquet = _json_to_parquet
AbstractFileSystem.json_to_parquet = json_to_parquet

AbstractFileSystem.pyarrow_dataset = pyarrow_dataset
AbstractFileSystem.pyarrow_parquet_dataset = pyarrow_parquet_dataset

AbstractFileSystem.lss = ls
AbstractFileSystem.ls2 = ls

# AbstractFileSystem.parallel_cp = parallel_cp
# AbstractFileSystem.parallel_mv = parallel_mv
# AbstractFileSystem.parallel_rm = parallel_rm
AbstractFileSystem.sync_folder = sync_folder
# AbstractFileSystem.list_files_recursive = list_files_recursive


def FileSystem(
    bucket: str | None = None,
    fs: AbstractFileSystem | None = None,
    profile: str | None = None,
    key: str | None = None,
    endpoint_url: str | None = None,
    secret: str | None = None,
    token: str | None = None,
    protocol: str | None = None,
    cached: bool = False,
    cache_storage="~/.tmp",
    check_files: bool = False,
    cache_check: int = 120,
    expire_time: int = 24 * 60 * 60,
    same_names: bool = False,
    **kwargs,
) -> AbstractFileSystem:
    if protocol is None:
        protocol = "file"
        if bucket is None:
            bucket = "."

    if all([fs, profile, key, endpoint_url, secret, token, protocol]):
        fs = filesystem("file", use_listings_cache=False)

    elif fs is None:
        if "client_kwargs" in kwargs:
            fs = s3fs.S3FileSystem(
                profile=profile,
                key=key,
                endpoint_url=endpoint_url,
                secret=secret,
                token=token,
                **kwargs,
            )
        else:
            fs = filesystem(
                protocol=protocol,
                profile=profile,
                key=key,
                endpoint_url=endpoint_url,
                secret=secret,
                token=token,
                use_listings_cache=False,
            )

    if bucket is not None:
        if protocol in ["file", "local"]:
            bucket = posixpath.abspath(bucket)

        fs = DirFileSystem(path=bucket, fs=fs)

    if cached:
        if "~" in cache_storage:
            cache_storage = posixpath.expanduser(cache_storage)

        return MonitoredSimpleCacheFileSystem(
            cache_storage=cache_storage,
            check_files=check_files,
            cache_check=cache_check,
            expire_time=expire_time,
            fs=fs,
            same_names=same_names,
            **kwargs,
        )

    return fs


def PyArrowFileSystem(
    bucket: str | None = None,
    fs: AbstractFileSystem | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    session_token: str | None = None,
    endpoint_override: str | None = None,
    protocol: str | None = None,
) -> pfs.FileSystem:
    credentials = None
    if fs is not None:
        protocol = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol

        if protocol == "dir":
            bucket = fs.path
            fs = fs.fs
            protocol = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol

        if protocol == "s3":
            credentials = get_credentials_from_fssspec(fs)

    if credentials is None:
        credentials = {
            "access_key": access_key,
            "secret_key": secret_key,
            "session_token": session_token,
            "endpoint_override": endpoint_override,
        }
    if protocol == "s3":
        fs = pfs.S3FileSystem(
            **credentials,
        )
    elif protocol in ("file", "local", None):
        fs = pfs.LocalFileSystem()

    else:
        fs = pfs.LocalFileSystem()

    if bucket is not None:
        if protocol in ["file", "local", "None"]:
            bucket = posixpath.abspath(bucket)

        fs = pfs.SubTreeFileSystem(base_fs=fs, base_path=bucket)

    return fs


# class FileSystem:
#     def __init__(
#         self,
#         bucket: str | None = None,
#         fs: AbstractFileSystem | None = None,
#         profile: str | None = None,
#         key: str | None = None,
#         endpoint_url: str | None = None,
#         secret: str | None = None,
#         token: str | None = None,
#         protocol: str | None = None,
#         cached: bool = False,
#         cache_storage="~/.tmp",
#         check_files: bool = False,
#         cache_check: int = 120,
#         expire_time: int = 24 * 60 * 60,
#         same_names: bool = False,
#         **kwargs,
#     ):
#         self._fsspec_fs = FsSpecFileSystem(
#             bucket=bucket,
#             fs=fs,
#             profile=profile,
#             key=key,
#             endpoint_url=endpoint_url,
#             secret=secret,
#             token=token,
#             protocol=protocol,
#             cached=cached,
#             cache_storage=cache_storage,
#             check_files=check_files,
#             cache_check=cache_check,
#             expire_time=expire_time,
#             same_names=same_names,
#             **kwargs,
#         )
#         self._pfs_fs = PyArrowFileSystem(
#             bucket=bucket,
#             fs=fs,
#             access_key=key,
#             secret_key=secret,
#             session_token=token,
#             endpoint_override=endpoint_url,
#             protocol=protocol,
#         )


def clear_cache(fs: AbstractFileSystem | None):
    if hasattr(fs, "dir_cache"):
        if fs is not None:
            if hasattr(fs, "clear_cache"):
                fs.clear_cache()
            fs.invalidate_cache()
            fs.clear_instance_cache()
            if hasattr(fs, "fs"):
                fs.fs.invalidate_cache()
                fs.fs.clear_instance_cache()

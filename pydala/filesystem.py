import datetime as dt
import inspect
import os
import posixpath
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Union

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
from .helpers.security import safe_join, validate_path
from .schema import convert_large_types_to_normal


def get_credentials_from_fssspec(fs: AbstractFileSystem, redact_secrets: bool = True) -> dict[str, str]:
    """
    Safely extract credentials from fsspec filesystem.
    
    Args:
        fs: The fsspec filesystem object
        redact_secrets: If True, redact sensitive values from returned dict
        
    Returns:
        Dictionary with credential information (secrets redacted if redact_secrets=True)
    """
    if "s3" in fs.protocol:
        credentials = fs.s3._get_credentials()
        
        if redact_secrets:
            # For logging and display purposes, redact actual secrets
            return {
                "access_key": f"REDACTED({len(credentials.access_key) if credentials.access_key else 0} chars)",
                "secret_key": f"REDACTED({len(credentials.secret_key) if credentials.secret_key else 0} chars)",
                "session_token": f"REDACTED({len(credentials.token) if credentials.token else 0} chars)" if credentials.token else None,
                "endpoint_override": fs.s3._endpoint.host if fs.s3._endpoint else None,
            }
        else:
            # Only return actual credentials when explicitly requested
            return {
                "access_key": credentials.access_key,
                "secret_key": credentials.secret_key,
                "session_token": credentials.token,
                "endpoint_override": fs.s3._endpoint.host if fs.s3._endpoint else None,
            }
    
    return {}


def get_total_directory_size(directory: str):
    return sum(f.stat().st_size for f in Path(directory).glob("**/*") if f.is_file())


class FileNameCacheMapper(AbstractCacheMapper):
    def __init__(self, directory):
        # Validate and normalize the directory path
        self.directory = validate_path(directory)

    def __call__(self, path: str) -> str:
        # Validate the input path to prevent traversal
        validated_path = validate_path(path)
        
        # Safely join paths
        full_path = safe_join(self.directory, validated_path)
        
        # Create parent directory if needed
        parent_dir = posixpath.dirname(full_path)
        os.makedirs(parent_dir, exist_ok=True)
        
        # Return relative path from directory
        return validated_path


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


class DiskUsageTracker:
    """Thread-safe disk usage tracking."""
    
    def __init__(self):
        self._last_free = None
        self._first_free = None
        self._lock = threading.Lock()
    
    def get_usage_message(self, storage: str) -> str:
        """Get friendly disk usage message with deltas."""
        with self._lock:
            usage = psutil.disk_usage(storage)
            if self._first_free is None:
                self._first_free = usage.free
            current_usage = get_total_directory_size(storage)
            message = f"{sizeof_fmt(current_usage)} used {sizeof_fmt(usage.free)} available"
            if self._last_free is not None:
                downloaded_recently = self._last_free - usage.free
                if downloaded_recently > 10_000_000:
                    downloaded_since_start = self._first_free - usage.free
                    if downloaded_recently != downloaded_since_start:
                        message += f" delta: {sizeof_fmt(downloaded_recently)}"
                    message += f" delta since start: {sizeof_fmt(downloaded_since_start)}"

            self._last_free = usage.free
            return message
    
    def reset(self):
        """Reset tracking state."""
        with self._lock:
            self._first_free = None
            self._last_free = None


# Global instance for backward compatibility
_disk_tracker = DiskUsageTracker()


def get_friendly_disk_usage(storage: str) -> str:
    """Get friendly disk usage message (legacy function)."""
    return _disk_tracker.get_usage_message(storage)


#############################################################################################################
# This was originally implemented by Burak Emre Kabakcı (@buremba on github) in its universql project
# project: https://github.com/buremba/universql
# original code: https://github.com/buremba/universql/blob/main/universql/lake/fsspec_util.py
#############################################################################################################


class CacheMixin:
    """Mixin for cache filesystem operations"""

    def __init__(self, **kwargs):
        self._verbose = kwargs.get("verbose", False)
        self.cache_storage = kwargs.get("cache_storage", "~/.tmp")
        self.fs = kwargs.get("fs")
        self._check_cache()  # Ensure cache is initialized
        self._mapper = FileNameCacheMapper(self.cache_storage)

    def _check_file_in_cache(self, path: str) -> Optional[str]:
        """Check if file exists in cache storage"""
        self._check_cache()
        cache_path = self._mapper(path)

        for storage in self.storage:
            cache_file = posixpath.join(storage, cache_path)
            if posixpath.exists(cache_file):
                return cache_file

            if self._verbose:
                logger.info(f"Downloading {getattr(self.fs, 'protocol', ['file'])[0]}://{path}")

        return None

    def get_cached_file_size(self, path: str) -> int:
        """Get size of cached file"""
        stripped_path = self._strip_protocol(path)
        cached_file = self._check_file_in_cache(stripped_path)
        if cached_file is None:
            # Check if file exists directly
            if hasattr(self.fs, 'exists') and self.fs.exists(path):
                return self.fs.size(path)
            # For local files, check if path exists
            elif os.path.exists(stripped_path):
                return os.path.getsize(stripped_path)
            else:
                # Return 0 as fallback
                return 0
        else:
            return os.path.getsize(cached_file)

    def _strip_protocol(self, path: str) -> str:
        """Strip protocol prefix from path"""
        protocol = getattr(self.fs, 'protocol', 'file')
        if isinstance(protocol, (list, tuple)):
            protocol = protocol[0]

        protocol_map = {
            's3': 's3://',
            'file': 'file://',
            'local': 'local://',
        }
        prefix = protocol_map.get(protocol, '')

        if prefix and path.startswith(prefix):
            return path[len(prefix):]
        return path


class MonitoredSimpleCacheFileSystem(SimpleCacheFileSystem, CacheMixin):
    """Enhanced cache filesystem with monitoring and validation"""

    def __init__(self, **kwargs):
        CacheMixin.__init__(self, **kwargs)
        SimpleCacheFileSystem.__init__(self, **kwargs)

    def size(self, path):
        """Get file size with cache support"""
        return self.get_cached_file_size(path)

    # Delegate attribute access to wrapped filesystem
    def __getattribute__(self, item):
        # Handle special attributes
        if item in ["__reduce_ex__"]:
            raise AttributeError

        # Handle cache mixin's attributes
        if hasattr(CacheMixin, item) and not item.startswith('_'):
            return CacheMixin.__getattribute__(self, item)

        # Handle this class's attributes
        self_methods = {"size", "_check_file_in_cache", "get_cached_file_size"}
        if item in self_methods:
            return object.__getattribute__(self, item)

        # Handle implemented methods
        impl_methods = {
            "_check_file", "_check_cache", "_mkcache", "clear_cache",
            "clear_expired_cache", "pop_from_cache", "local_file"
        }
        if item in impl_methods:
            return lambda *args, **kw: getattr(type(self), item).__get__(self)(*args, **kw)

        # Handle properties
        if item in ["transaction"]:
            return type(self).transaction.__get__(self)

        # Handle class attributes
        if item in ["_cache", "transaction_type"]:
            return getattr(type(self), item)

        if item == "__class__":
            return type(self)

        # Try instance dict first
        d = object.__getattribute__(self, "__dict__")
        if item in d:
            return d[item]

        # If we have a wrapped fs, delegate to it
        wrapped_fs = d.get("fs", None)
        if wrapped_fs is not None:
            # Check filesystem's dict first
            if hasattr(wrapped_fs, item):
                return getattr(wrapped_fs, item)

            # Check filesystem's class
            fs_cls = type(wrapped_fs)
            if hasattr(fs_cls, item):
                m = getattr(fs_cls, item)
                if (inspect.isfunction(m) or inspect.isdatadescriptor(m)) and \
                   (not hasattr(m, "__self__") or m.__self__ is None):
                    return m.__get__(wrapped_fs, fs_cls)
                return m

        # Fall back to superclass
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


# Configuration Classes to reduce parameter counts
@dataclass
class FileSystemConfig:
    """Base configuration for filesystems"""
    path: Optional[str] = None
    protocol: str = "file"
    cached: bool = False
    cache_storage: str = "~/.tmp"
    check_files: bool = False
    cache_check: int = 120
    expire_time: int = 24 * 60 * 60
    same_names: bool = False


@dataclass
class S3Config(FileSystemConfig):
    """S3-specific configuration"""
    profile: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    session_token: Optional[str] = None
    endpoint_url: Optional[str] = None
    region: Optional[str] = None


# Strategy Pattern for Authentication
class AuthenticationStrategy:
    """Abstract base class for authentication strategies"""

    def apply_credentials(self, **kwargs) -> Dict[str, Any]:
        """Apply authentication credentials to filesystem kwargs"""
        pass


class S3Authentication(AuthenticationStrategy):
    """S3 authentication strategy"""

    def __init__(self, profile=None, key=None, secret=None, token=None, endpoint_url=None):
        self.profile = profile
        self.key = key
        self.secret = secret
        self.token = token
        self.endpoint_url = endpoint_url

    def apply_credentials(self, **kwargs) -> Dict[str, Any]:
        """Apply S3 credentials to kwargs"""
        if self.profile:
            kwargs['profile'] = self.profile
        if self.key:
            kwargs['key'] = self.key
        if self.secret:
            kwargs['secret'] = self.secret
        if self.token:
            kwargs['token'] = self.token
        if self.endpoint_url:
            kwargs['endpoint_url'] = self.endpoint_url
        return kwargs


class FileSystemBuilder:
    """Builder class for filesystem creation"""

    def __init__(self):
        self.protocol = "file"
        self.bucket = None
        self.cached = False
        self.cache_storage = "~/.tmp"
        self.check_files = False
        self.cache_check = 120
        self.expire_time = 24 * 60 * 60
        self.same_names = False
        self.auth_strategy = None
        self.existing_fs = None
        self.kwargs = {}

    def with_protocol(self, protocol: str):
        """Set filesystem protocol"""
        self.protocol = protocol
        return self

    def with_bucket(self, bucket: str):
        """Set bucket/directory path"""
        self.bucket = bucket
        return self

    def with_cache(self, enabled: bool = True, storage: str = "~/.tmp",
                   check_files: bool = False, cache_check: int = 120,
                   expire_time: int = 24 * 60 * 60, same_names: bool = False):
        """Enable caching with configuration"""
        self.cached = enabled
        self.cache_storage = storage
        self.check_files = check_files
        self.cache_check = cache_check
        self.expire_time = expire_time
        self.same_names = same_names
        return self

    def with_s3_auth(self, profile=None, key=None, endpoint_url=None, secret=None, token=None):
        """Configure S3 authentication"""
        self.auth_strategy = S3Authentication(
            profile=profile,
            key=key,
            secret=secret,
            token=token,
            endpoint_url=endpoint_url
        )
        return self

    def with_existing_fs(self, fs: AbstractFileSystem):
        """Use existing filesystem"""
        self.existing_fs = fs
        return self

    def with_kwargs(self, **kwargs):
        """Add additional kwargs"""
        self.kwargs.update(kwargs)
        return self

    def _create_base_filesystem(self) -> AbstractFileSystem:
        """Create the base filesystem"""
        if self.existing_fs:
            return self.existing_fs

        if self.protocol == "s3":
            return self._create_s3_filesystem()
        else:
            return self._create_local_filesystem()

    def _create_s3_filesystem(self) -> AbstractFileSystem:
        """Create S3 filesystem"""
        from s3fs import S3FileSystem

        kwargs = {'use_listings_cache': False}
        if self.auth_strategy:
            kwargs = self.auth_strategy.apply_credentials(**kwargs)

        if "client_kwargs" in self.kwargs:
            kwargs.update(self.kwargs)
            return S3FileSystem(**kwargs)
        else:
            kwargs.update(self.kwargs)
            from fsspec import filesystem
            return filesystem(
                protocol="s3",
                use_listings_cache=False,
                **kwargs
            )

    def _create_local_filesystem(self) -> AbstractFileSystem:
        """Create local filesystem"""
        from fsspec import filesystem
        return filesystem("file", use_listings_cache=False)

    def _wrap_with_directory(self, fs: AbstractFileSystem) -> AbstractFileSystem:
        """Wrap filesystem with directory wrapper"""
        if not self.bucket:
            return fs

        if self.protocol in ["file", "local"]:
            self.bucket = posixpath.abspath(self.bucket)

        return DirFileSystem(path=self.bucket, fs=fs)

    def _wrap_with_cache(self, fs: AbstractFileSystem) -> AbstractFileSystem:
        """Wrap filesystem with cache if enabled"""
        if not self.cached:
            return fs

        cache_storage = self.cache_storage
        if "~" in cache_storage:
            cache_storage = posixpath.expanduser(cache_storage)

        return MonitoredSimpleCacheFileSystem(
            cache_storage=cache_storage,
            check_files=self.check_files,
            cache_check=self.cache_check,
            expire_time=self.expire_time,
            fs=fs,
            same_names=self.same_names,
            **self.kwargs
        )

    def build(self) -> AbstractFileSystem:
        """Build the final filesystem"""
        fs = self._create_base_filesystem()
        fs = self._wrap_with_directory(fs)
        fs = self._wrap_with_cache(fs)
        return fs


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
    """Simplified FileSystem factory using builder pattern"""
    # Validate inputs
    if protocol is None and fs is None:
        protocol = "file"

    # Create builder
    builder = FileSystemBuilder()

    # Build filesystem based on parameters
    builder.with_protocol(protocol or "file")

    if bucket:
        builder.with_bucket(bucket)

    if cached:
        builder.with_cache(
            enabled=cached,
            storage=cache_storage,
            check_files=check_files,
            cache_check=cache_check,
            expire_time=expire_time,
            same_names=same_names
        )

    if fs:
        builder.with_existing_fs(fs)
    elif protocol == "s3":
        builder.with_s3_auth(
            profile=profile,
            key=key,
            secret=secret,
            token=token,
            endpoint_url=endpoint_url
        )

    builder.with_kwargs(**kwargs)

    # Special case: if all auth params provided but no fs, use file
    if all([fs is None, profile, key, endpoint_url, secret, token, protocol]):
        builder.with_existing_fs(filesystem("file", use_listings_cache=False))

    return builder.build()


class PyArrowFileSystemBuilder:
    """Builder for PyArrow filesystem"""

    def __init__(self):
        self.bucket = None
        self.fs = None
        self.credentials = {
            "access_key": None,
            "secret_key": None,
            "session_token": None,
            "endpoint_override": None,
        }
        self.protocol = None

    def with_bucket(self, bucket: str):
        """Set bucket/directory"""
        self.bucket = bucket
        return self

    def with_fs(self, fs: AbstractFileSystem):
        """Set existing filesystem"""
        self.fs = fs
        return self

    def with_credentials(self, access_key=None, secret_key=None,
                        session_token=None, endpoint_override=None):
        """Set credentials"""
        self.credentials.update({
            "access_key": access_key,
            "secret_key": secret_key,
            "session_token": session_token,
            "endpoint_override": endpoint_override,
        })
        return self

    def _extract_protocol_and_credentials(self) -> tuple[str, dict]:
        """Extract protocol and credentials from existing filesystem if provided"""
        if self.fs is not None:
            protocol = self.fs.protocol
            if isinstance(protocol, tuple):
                protocol = protocol[0]

            if protocol == "dir":
                self.bucket = self.fs.path
                self.fs = self.fs.fs
                protocol = self.fs.protocol
                if isinstance(protocol, tuple):
                    protocol = protocol[0]

            if protocol == "s3":
                return protocol, get_credentials_from_fssspec(self.fs, redact_secrets=False)

        return self.protocol or "file", self.credentials

    def _create_pyarrow_fs(self, protocol: str, credentials: dict) -> pfs.FileSystem:
        """Create PyArrow filesystem based on protocol"""
        if protocol == "s3":
            return pfs.S3FileSystem(**credentials)
        else:
            return pfs.LocalFileSystem()

    def _wrap_with_bucket(self, pa_fs: pfs.FileSystem) -> pfs.FileSystem:
        """Wrap with bucket/directory if specified"""
        if self.bucket is not None:
            if self.protocol in ["file", "local", "None"]:
                bucket = posixpath.abspath(self.bucket)
            else:
                bucket = self.bucket

            return pfs.SubTreeFileSystem(base_fs=pa_fs, base_path=bucket)
        return pa_fs

    def build(self) -> pfs.FileSystem:
        """Build the PyArrow filesystem"""
        protocol, credentials = self._extract_protocol_and_credentials()
        pa_fs = self._create_pyarrow_fs(protocol, credentials)
        return self._wrap_with_bucket(pa_fs)


def PyArrowFileSystem(
    bucket: str | None = None,
    fs: AbstractFileSystem | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    session_token: str | None = None,
    endpoint_override: str | None = None,
    protocol: str | None = None,
) -> pfs.FileSystem:
    """Simplified PyArrow filesystem factory using builder pattern"""
    builder = PyArrowFileSystemBuilder()

    if bucket:
        builder.with_bucket(bucket)

    if fs:
        builder.with_fs(fs)
    else:
        builder.protocol = protocol
        builder.with_credentials(
            access_key=access_key,
            secret_key=secret_key,
            session_token=session_token,
            endpoint_override=endpoint_override
        )

    return builder.build()


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

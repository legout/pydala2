"""Enhanced filesystem module leveraging fsspec-utils with pydala's custom security and monitoring features."""

import datetime as dt
import inspect
import os
import posixpath
import threading
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Union

import duckdb as ddb
import orjson
import pandas as pd
import polars as pl
import psutil
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.fs as pfs
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from fsspec.implementations.cached import SimpleCacheFileSystem
from fsspec.implementations.dirfs import DirFileSystem
from fsspec_utils import filesystem as fsspec_utils_filesystem
from fsspec_utils.storage_options import (
    BaseStorageOptions,
    LocalStorageOptions,
    AwsStorageOptions,
    AzureStorageOptions,
    GcsStorageOptions,
    StorageOptions,
    from_env as storage_options_from_env,
)
from loguru import logger

from .helpers.misc import run_parallel
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


class FileNameCacheMapper:
    """Enhanced cache mapper that validates paths for security."""

    def __init__(self, directory):
        # For cache directories, allow absolute paths but still validate for traversal
        if not directory:
            raise ValueError("Cache directory cannot be empty")

        # Remove null bytes and other dangerous characters
        directory = directory.replace('\0', '').strip()

        # Normalize the path
        normalized = os.path.normpath(directory)

        # Check for traversal attempts (but allow absolute paths for cache)
        if '..' in normalized:
            raise ValueError(f"Path traversal attempt detected in cache directory: {directory}")

        self.directory = normalized

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


class throttle:
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


class MonitoredSimpleCacheFileSystem(SimpleCacheFileSystem):
    """Enhanced caching filesystem with monitoring and security."""

    def __init__(self, **kwargs):
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

    def size(self, path):
        cached_file = self._check_file(self._strip_protocol(path))
        if cached_file is None:
            return self.fs.size(path)
        else:
            return posixpath.getsize(cached_file)

    def __getattribute__(self, item):
        if item in {
            "size",
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
            return lambda *args, **kw: getattr(type(self), item).__get__(self)(
                *args, **kw
            )
        if item in ["__reduce_ex__"]:
            raise AttributeError
        if item in ["transaction"]:
            return type(self).transaction.__get__(self)
        if item in ["_cache", "transaction_type"]:
            return getattr(type(self), item)
        if item == "__class__":
            return type(self)
        d = object.__getattribute__(self, "__dict__")
        fs = d.get("fs", None)
        if item in d:
            return d[item]
        elif fs is not None:
            if item in fs.__dict__:
                return fs.__dict__[item]
            cls = type(fs)
            m = getattr(cls, item)
            if (inspect.isfunction(m) or inspect.isdatadescriptor(m)) and (
                not hasattr(m, "__self__") or m.__self__ is None
            ):
                return m.__get__(fs, cls)
            return m
        else:
            return super().__getattribute__(item)


def ls(
    self,
    path,
    recursive: bool = False,
    maxdepth: int | None = None,
    detail: bool = False,
    files_only: bool = False,
    dirs_only: bool = False,
):
    """Enhanced ls method with additional options."""
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
    """Sync files from source to destination."""
    src_ = self.ls(src, recursive=recursive, maxdepth=maxdepth, files_only=True)
    dst_ = self.ls(dst, recursive=recursive, maxdepth=maxdepth, files_only=True)

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


class PydalaFileSystem:
    """
    Enhanced filesystem wrapper that combines fsspec-utils capabilities
    with pydala's custom security and monitoring features.
    """

    def __init__(
        self,
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
        storage_options: StorageOptions | None = None,
        **kwargs,
    ):
        # Use fsspec-utils StorageOptions if provided
        if storage_options is not None:
            self._fs = storage_options.to_filesystem()
            self._storage_options = storage_options
        else:
            # Create StorageOptions from parameters using fsspec-utils
            if protocol is None and fs is None:
                protocol = "file"

            if protocol == "s3" or (endpoint_url and "s3" in endpoint_url.lower()):
                self._storage_options = AwsStorageOptions(
                    profile=profile,
                    access_key_id=key,
                    secret_access_key=secret,
                    session_token=token,
                    endpoint_url=endpoint_url,
                    **kwargs,
                )
            elif protocol == "az" or (endpoint_url and "azure" in endpoint_url.lower()):
                self._storage_options = AzureStorageOptions(
                    account_name=profile,
                    connection_string=kwargs.get("connection_string"),
                    **kwargs,
                )
            elif protocol == "gs" or (endpoint_url and "googleapis" in endpoint_url.lower()):
                self._storage_options = GcsStorageOptions(
                    project=profile,
                    token=token,
                    **kwargs,
                )
            else:
                # Default to local storage
                self._storage_options = LocalStorageOptions(
                    auto_mkdir=kwargs.get("auto_mkdir", True),
                    **kwargs,
                )

            self._fs = self._storage_options.to_filesystem()

        # Apply bucket/directory if specified
        if bucket is not None:
            if isinstance(self._fs.protocol, (list, tuple)):
                fs_protocol = self._fs.protocol[0]
            else:
                fs_protocol = self._fs.protocol

            if fs_protocol in ["file", "local"]:
                bucket = posixpath.abspath(bucket)

            self._fs = DirFileSystem(path=bucket, fs=self._fs)

        # Apply caching if requested
        if cached:
            if "~" in cache_storage:
                cache_storage = posixpath.expanduser(cache_storage)

            self._fs = MonitoredSimpleCacheFileSystem(
                cache_storage=cache_storage,
                check_files=check_files,
                cache_check=cache_check,
                expire_time=expire_time,
                fs=self._fs,
                same_names=same_names,
                **kwargs,
            )

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying filesystem."""
        return getattr(self._fs, name)

    @property
    def fs(self):
        """Access the underlying filesystem."""
        return self._fs

    @property
    def storage_options(self):
        """Access the storage options."""
        return self._storage_options


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
    storage_options: StorageOptions | None = None,
    **kwargs,
) -> PydalaFileSystem:
    """
    Create a filesystem instance using fsspec-utils with pydala enhancements.
    """
    return PydalaFileSystem(
        bucket=bucket,
        fs=fs,
        profile=profile,
        key=key,
        endpoint_url=endpoint_url,
        secret=secret,
        token=token,
        protocol=protocol,
        cached=cached,
        cache_storage=cache_storage,
        check_files=check_files,
        cache_check=cache_check,
        expire_time=expire_time,
        same_names=same_names,
        storage_options=storage_options,
        **kwargs,
    )


def FileSystemFromOptions(
    storage_options: StorageOptions,
    bucket: str | None = None,
    cached: bool = False,
    cache_storage="~/.tmp",
    **kwargs,
) -> PydalaFileSystem:
    """
    Create a PydalaFileSystem from fsspec-utils StorageOptions.
    """
    return PydalaFileSystem(
        storage_options=storage_options,
        bucket=bucket,
        cached=cached,
        cache_storage=cache_storage,
        **kwargs,
    )


def PyArrowFileSystem(
    bucket: str | None = None,
    fs: AbstractFileSystem | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    session_token: str | None = None,
    endpoint_override: str | None = None,
    protocol: str | None = None,
) -> pfs.FileSystem:
    """Create a PyArrow filesystem from fsspec credentials."""
    credentials = None
    if fs is not None:
        protocol = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol

        if protocol == "dir":
            bucket = fs.path
            fs = fs.fs
            protocol = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol

        if protocol == "s3":
            credentials = get_credentials_from_fssspec(fs, redact_secrets=False)

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


def clear_cache(fs: AbstractFileSystem | None):
    """Clear cache from filesystem."""
    if hasattr(fs, "dir_cache"):
        if fs is not None:
            if hasattr(fs, "clear_cache"):
                fs.clear_cache()
            fs.invalidate_cache()
            fs.clear_instance_cache()
            if hasattr(fs, "fs"):
                fs.fs.invalidate_cache()
                fs.fs.clear_instance_cache()


# Monkey patch additional methods that fsspec-utils doesn't provide
# Note: fsspec-utils already provides read_parquet, write_parquet, etc.
# We only need to add the ones that are unique to pydala

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

    return [
        posixpath.join(posixpath.dirname(src), f, src_ext)
        for f in src_file_names
        if f not in dst_file_names
    ]


# Add the ls method to AbstractFileSystem
AbstractFileSystem.ls = ls
AbstractFileSystem.lss = ls
AbstractFileSystem.ls2 = ls
AbstractFileSystem.sync_folder = sync_folder
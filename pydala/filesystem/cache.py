"""
Caching module for filesystem operations.
"""
import os
import posixpath
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional

from fsspec import AbstractFileSystem
from fsspec.implementations.cache_mapper import AbstractCacheMapper
from fsspec.implementations.cached import SimpleCacheFileSystem
from loguru import logger

from ..helpers.security import safe_join, validate_path


class FileNameCacheMapper(AbstractCacheMapper):
    """Maps remote paths to local cache paths with security validation."""

    def __init__(self, directory: str):
        """Initialize with cache directory.

        Args:
            directory: Base cache directory
        """
        self.directory = validate_path(directory)

    def __call__(self, path: str) -> str:
        """Map remote path to local cache path.

        Args:
            path: Remote path to cache

        Returns:
            Local cache path
        """
        validated_path = validate_path(path)
        full_path = safe_join(self.directory, validated_path)

        # Create parent directory if needed
        parent_dir = posixpath.dirname(full_path)
        os.makedirs(parent_dir, exist_ok=True)

        return validated_path


class throttle:
    """Decorator that prevents a function from being called more than once every time period."""

    def __init__(self, seconds: int = 0, minutes: int = 0, hours: int = 0):
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


class MonitoredSimpleCacheFileSystem(SimpleCacheFileSystem):
    """Enhanced cached filesystem with monitoring capabilities."""

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
            "size", "glob", "load_cache", "_open", "save_cache",
            "close_and_update", "__init__", "__getattribute__", "__reduce__",
            "_make_local_details", "open", "cat", "cat_file", "cat_ranges",
            "get", "read_block", "tail", "head", "info", "ls", "exists",
            "isfile", "isdir", "_check_file", "_check_cache", "_mkcache",
            "clear_cache", "clear_expired_cache", "pop_from_cache",
            "local_file", "_paths_from_path", "get_mapper", "open_many",
            "commit_many", "hash_name", "__hash__", "__eq__", "to_json",
            "to_dict", "cache_size", "pipe_file", "pipe",
            "start_transaction", "end_transaction",
        }:
            return lambda *args, **kw: getattr(type(self), item).__get__(self)(*args, **kw)
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
            if m is not None and (not hasattr(m, "__self__") or m.__self__ is None):
                return m.__get__(fs, cls)
            return m
        else:
            return super().__getattribute__(item)


class CacheManager:
    """Manages filesystem caching operations."""

    def __init__(self,
                 cache_storage: str = "~/.tmp",
                 check_files: bool = False,
                 cache_check: int = 120,
                 expire_time: int = 24 * 60 * 60,
                 same_names: bool = False,
                 verbose: bool = False):
        """Initialize cache manager.

        Args:
            cache_storage: Directory for cache storage
            check_files: Whether to check file existence
            cache_check: Cache check interval in seconds
            expire_time: Cache expiration time in seconds
            same_names: Whether to use same names for cached files
            verbose: Whether to enable verbose logging
        """
        self.cache_storage = cache_storage
        self.check_files = check_files
        self.cache_check = cache_check
        self.expire_time = expire_time
        self.same_names = same_names
        self.verbose = verbose

        # Expand user path if needed
        if "~" in self.cache_storage:
            self.cache_storage = posixpath.expanduser(self.cache_storage)

    def create_cached_filesystem(self, fs: AbstractFileSystem) -> MonitoredSimpleCacheFileSystem:
        """Create a cached filesystem wrapper.

        Args:
            fs: Base filesystem to wrap

        Returns:
            Cached filesystem instance
        """
        return MonitoredSimpleCacheFileSystem(
            cache_storage=self.cache_storage,
            check_files=self.check_files,
            cache_check=self.cache_check,
            expire_time=self.expire_time,
            fs=fs,
            same_names=self.same_names,
            verbose=self.verbose,
        )

    @staticmethod
    def clear_cache(fs: Optional[AbstractFileSystem] = None) -> None:
        """Clear filesystem cache.

        Args:
            fs: Filesystem instance to clear cache for
        """
        if fs is None:
            return

        if hasattr(fs, "clear_cache"):
            fs.clear_cache()
        if hasattr(fs, "invalidate_cache"):
            fs.invalidate_cache()
        if hasattr(fs, "clear_instance_cache"):
            fs.clear_instance_cache()
        if hasattr(fs, "fs"):
            if hasattr(fs.fs, "invalidate_cache"):
                fs.fs.invalidate_cache()
            if hasattr(fs.fs, "clear_instance_cache"):
                fs.fs.clear_instance_cache()
import posixpath
import os
from typing import Any

from fsspec.implementations.cache_mapper import AbstractCacheMapper
from fsspec.implementations.cached import SimpleCacheFileSystem
from loguru import logger

from .helpers.security import safe_join, validate_path


class FileNameCacheMapper(AbstractCacheMapper):
    def __init__(self, directory: str):
        self.directory = validate_path(directory)

    def __call__(self, path: str) -> str:
        validated_path = validate_path(path)
        full_path = safe_join(self.directory, validated_path)
        parent_dir = posixpath.dirname(full_path)
        os.makedirs(parent_dir, exist_ok=True)
        return validated_path


class MonitoredSimpleCacheFileSystem(SimpleCacheFileSystem):
    def __init__(self, verbose: bool = False, **kwargs):
        self._verbose = verbose
        super().__init__(**kwargs)
        self._mapper = FileNameCacheMapper(kwargs.get("cache_storage", "~/.tmp"))

    def _check_file(self, path: str):
        self._check_cache()
        cache_path = self._mapper(path)
        for storage in self.storage:
            fn = posixpath.join(storage, cache_path)
            if posixpath.exists(fn):
                return fn
            if self._verbose:
                logger.info(f"Downloading {self.protocol[0]}://{path}")

    def size(self, path: str):
        cached_file = self._check_file(self._strip_protocol(path))
        if cached_file is None:
            return self.fs.size(path)
        return posixpath.getsize(cached_file)

    def __getattribute__(self, item: str) -> Any:
        if item in self._delegated_methods:
            return lambda *args, **kwargs: getattr(type(self), item).__get__(self, type(self))(*args, **kwargs)
        if item in {"__reduce_ex__", "__reduce__"}:
            raise AttributeError(item)
        if item == "transaction":
            return type(self).transaction.__get__(self, type(self))
        if item in {"_cache", "transaction_type"}:
            return getattr(type(self), item)
        if item == "__class__":
            return type(self)
        return self._delegate_to_fs(item)

    def _delegate_to_fs(self, item: str) -> Any:
        d = object.__getattribute__(self, "__dict__")
        fs = d.get("fs")
        if item in d:
            return d[item]
        if fs is None:
            return super().__getattribute__(item)
        if item in fs.__dict__:
            return fs.__dict__[item]
        cls = type(fs)
        m = getattr(cls, item, None)
        if m is None:
            raise AttributeError(f"'{item}' not found in underlying fs")
        if callable(m) and not hasattr(m, "__self__") or m.__self__ is None:
            return m.__get__(fs, cls)
        return m

    _delegated_methods = {
        "size", "glob", "load_cache", "_open", "save_cache", "close_and_update",
        "__init__", "__getattribute__", "__reduce__", "_make_local_details", "open",
        "cat", "cat_file", "cat_ranges", "get", "read_block", "tail", "head", "info",
        "ls", "exists", "isfile", "isdir", "_check_file", "_check_cache", "_mkcache",
        "clear_cache", "clear_expired_cache", "pop_from_cache", "local_file",
        "_paths_from_path", "get_mapper", "open_many", "commit_many", "hash_name",
        "__hash__", "__eq__", "to_json", "to_dict", "cache_size", "pipe_file", "pipe",
        "start_transaction", "end_transaction"
    }

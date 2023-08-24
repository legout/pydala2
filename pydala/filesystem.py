import os

from fsspec import AbstractFileSystem
from fsspec import filesystem as fsspec_filesystem
from fsspec.implementations import cached as cachedfs
from fsspec.implementations import dirfs


def get_filesystem(
    bucket: str | None = None,
    fs=AbstractFileSystem | None,
    cached: bool = False,
    cache_storage="~/.tmp",
    check_files: bool = False,
    cache_check: int = 120,
    expire_time: int = 24 * 60 * 60,
    same_names: bool = False,
    **kwargs,
):
    if fs is None:
        fs = fsspec_filesystem("file")

    if bucket is not None:
        fs = dirfs.DirFileSystem(path=bucket, fs=fs)

    if cached:
        if "~" in cache_storage:
            cache_storage = os.path.expanduser(cache_storage)

        return cachedfs.SimpleCacheFileSystem(
            cache_storage=cache_storage,
            check_files=check_files,
            cache_check=cache_check,
            expire_time=expire_time,
            fs=fs,
            same_names=same_names,
            **kwargs,
        )
    return fs


def clear_cache(fs: AbstractFileSystem):
    if hasattr(fs, "clear_cache"):
        fs.clear_cache()
    fs.invalidate_cache()
    fs.clear_instance_cache()
    if hasattr(fs, "fs"):
        fs.invalidate_cache()
        fs.clear_instance_cache()

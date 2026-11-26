# Filesystem

`FileSystem(...)` is a factory that returns an fsspec-compatible filesystem. Use the result directly with fsspec methods such as `open()`, `exists()`, and `glob()`. It is not a PyDala2 wrapper with a guaranteed `.fs` property.

## Local files

With no arguments, the factory returns a local filesystem. To scope paths to a directory, pass it as `bucket`:

```python
from pathlib import Path

from pydala import FileSystem

root = Path("local_storage").resolve()
root.mkdir()
local = FileSystem()
assert local.exists(str(root))

fs = FileSystem(bucket=str(root))
with fs.open("example.txt", "w") as stream:
    stream.write("hello")
assert fs.exists("example.txt")
```

Pass a filesystem as `filesystem=fs` when opening a dataset. For a rooted filesystem, the dataset path is relative to that root.

## Remote storage and caching

Install `pydala2[s3]` for S3-compatible storage or `pydala2[sftp]` for SSH/SFTP. Other protocols need their corresponding fsspec backend. Configure the remote root with `protocol` and `bucket`; configure read caching with `cached=True` and `cache_storage`. See the [deployment example](../advanced/deployment.md#s3-configuration).

`dataset.clear_cache()` invalidates filesystem caches used by the dataset. It does not rebuild Parquet sidecars; see [Metadata Management](../user-guide/metadata.md) for that operation.

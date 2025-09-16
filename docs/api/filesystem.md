# Filesystem

This document describes the `FileSystem` class which provides advanced filesystem operations with caching capabilities.

## FileSystem

```python
class FileSystem
```

The `FileSystem` class wraps fsspec filesystems with added caching and monitoring capabilities. It provides a unified interface for working with different storage backends including local filesystem, S3, GCS, and others.

### Constructor

```python
FileSystem(
    fs: AbstractFileSystem | None = None,
    bucket: str | None = None,
    cached: bool = False,
    cache_storage: str | None = None,
    **kwargs
) -> None
```

**Parameters:**
- `fs` (AbstractFileSystem, optional): Existing fsspec filesystem instance
- `bucket` (str, optional): Bucket name for cloud storage
- `cached` (bool): Whether to enable caching
- `cache_storage` (str, optional): Path to cache storage directory
- `**kwargs`: Additional filesystem configuration

**Example:**
```python
from pydala import FileSystem

# Local filesystem with caching
fs = FileSystem(cached=True, cache_storage="/tmp/pydala2_cache")

# S3 filesystem
fs = FileSystem(
    protocol="s3",
    bucket="my-bucket",
    key="access_key",
    secret="secret_key",
    cached=True
)

# GCS filesystem
fs = FileSystem(
    protocol="gcs",
    token="/path/to/service-account.json",
    cached=True
)
```

### Properties

#### fs
```python
@property
def fs(self) -> AbstractFileSystem
```
Get the underlying fsspec filesystem.

**Returns:**
- `AbstractFileSystem`: The underlying filesystem instance

### Methods

#### open
```python
def open(
    self,
    path: str,
    mode: str = "rb",
    block_size: int | None = None,
    cache_options: dict | None = None,
    compression: str | None = None,
    **kwargs
) -> fsspec.core.OpenFile
```
Open a file for reading or writing.

**Parameters:**
- `path` (str): Path to the file
- `mode` (str): File mode ('r', 'w', 'rb', 'wb', etc.)
- `block_size` (int, optional): Block size for reading/writing
- `cache_options` (dict, optional): Cache options
- `compression` (str, optional): Compression type
- `**kwargs`: Additional arguments

**Returns:**
- `fsspec.core.OpenFile`: File-like object

**Example:**
```python
# Read file
with fs.open("data/file.parquet", "rb") as f:
    data = f.read()

# Write file
with fs.open("data/output.parquet", "wb") as f:
    f.write(data)
```

#### glob
```python
def glob(self, path: str, **kwargs) -> list[str]
```
Find files matching a pattern.

**Parameters:**
- `path` (str): Glob pattern
- `**kwargs`: Additional glob options

**Returns:**
- `list[str]`: List of matching file paths

**Example:**
```python
# Find all Parquet files
files = fs.glob("data/**/*.parquet")

# Find files with date pattern
files = fs.glob("data/sales/2023-*.parquet")
```

#### exists
```python
def exists(self, path: str) -> bool
```
Check if a path exists.

**Parameters:**
- `path` (str): Path to check

**Returns:**
- `bool`: True if path exists, False otherwise

**Example:**
```python
if fs.exists("data/dataset"):
    print("Dataset exists")
```

#### ls
```python
def ls(self, path: str, detail: bool = False, **kwargs) -> list | dict
```
List contents of a directory.

**Parameters:**
- `path` (str): Directory path
- `detail` (bool): Whether to return detailed information
- `**kwargs`: Additional options

**Returns:**
- `list | dict`: List of file names or detailed info

**Example:**
```python
# Simple listing
contents = fs.ls("data/")

# Detailed listing
details = fs.ls("data/", detail=True)
for item in details:
    print(f"{item['name']}: {item['size']} bytes")
```

#### mkdirs
```python
def mkdirs(self, path: str, exist_ok: bool = True) -> None
```
Create directories recursively.

**Parameters:**
- `path` (str): Directory path to create
- `exist_ok` (bool): Whether to ignore if directory exists

**Example:**
```python
# Create directory structure
fs.mkdirs("data/sales/2023/01")
```

#### rm
```python
def rm(self, path: str, recursive: bool = False) -> None
```
Remove files or directories.

**Parameters:**
- `path` (str): Path to remove
- `recursive` (bool): Whether to remove recursively

**Example:**
```python
# Remove single file
fs.rm("data/old_file.parquet")

# Remove directory
fs.rm("data/old_dataset", recursive=True)
```

#### mv
```python
def mv(self, path1: str, path2: str, **kwargs) -> None
```
Move/rename files or directories.

**Parameters:**
- `path1` (str): Source path
- `path2` (str): Destination path
- `**kwargs`: Additional options

**Example:**
```python
# Rename file
fs.mv("data/old_name.parquet", "data/new_name.parquet")

# Move directory
fs.mv("data/temp", "data/archive")
```

#### cp
```python
def cp(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None
```
Copy files or directories.

**Parameters:**
- `path1` (str): Source path
- `path2` (str): Destination path
- `recursive` (bool): Whether to copy recursively
- `**kwargs`: Additional options

**Example:**
```python
# Copy file
fs.cp("data/source.parquet", "data/backup.parquet")

# Copy directory
fs.cp("data/dataset", "data/backup", recursive=True)
```

#### touch
```python
def touch(self, path: str, truncate: bool = True, **kwargs) -> None
```
Create an empty file or update timestamp.

**Parameters:**
- `path` (str): File path
- `truncate` (bool): Whether to truncate if file exists
- `**kwargs`: Additional options

**Example:**
```python
# Create empty file
fs.touch("data/placeholder.txt")
```

#### info
```python
def info(self, path: str, **kwargs) -> dict
```
Get file information.

**Parameters:**
- `path` (str): File path
- `**kwargs`: Additional options

**Returns:**
- `dict`: File information dictionary

**Example:**
```python
info = fs.info("data/file.parquet")
print(f"Size: {info['size']} bytes")
print(f"Modified: {info['modified']}")
```

#### size
```python
def size(self, path: str) -> int
```
Get file size in bytes.

**Parameters:**
- `path` (str): File path

**Returns:**
- `int`: File size in bytes

**Example:**
```python
size = fs.size("data/large_file.parquet")
print(f"File size: {size / 1024 / 1024:.1f} MB")
```

#### cat
```python
def cat(self, path: str, **kwargs) -> bytes
```
Read entire file contents.

**Parameters:**
- `path` (str): File path
- `**kwargs`: Additional options

**Returns:**
- `bytes`: File contents

**Example:**
```python
contents = fs.cat("data/metadata.json")
metadata = json.loads(contents)
```

#### cat_file
```python
def cat_file(self, path, start=None, end=None, **kwargs)
```
Read a range of bytes from a file.

**Parameters:**
- `path` (str): File path
- `start` (int, optional): Start byte offset
- `end` (int, optional): End byte offset
- `**kwargs`: Additional options

**Returns:**
- `bytes`: File contents in range

**Example:**
```python
# Read first 1KB
header = fs.cat_file("data/file.parquet", end=1024)
```

#### put
```python
def put(self, lpath, rpath, recursive=False, **kwargs)
```
Upload local files to remote filesystem.

**Parameters:**
- `lpath` (str): Local path
- `rpath` (str): Remote path
- `recursive` (bool): Whether to upload recursively
- `**kwargs`: Additional options

**Example:**
```python
# Upload single file
fs.put("local/data.parquet", "remote/data.parquet")

# Upload directory
fs.put("local/dataset", "remote/dataset", recursive=True)
```

#### get
```python
def get(self, rpath, lpath, recursive=False, **kwargs)
```
Download files from remote filesystem.

**Parameters:**
- `rpath` (str): Remote path
- `lpath` (str): Local path
- `recursive` (bool): Whether to download recursively
- `**kwargs`: Additional options

**Example:**
```python
# Download single file
fs.get("remote/data.parquet", "local/data.parquet")

# Download directory
fs.get("remote/dataset", "local/dataset", recursive=True)
```

#### expand_path
```python
def expand_path(self, path, **kwargs)
```
Expand a path with wildcards.

**Parameters:**
- `path` (str): Path with possible wildcards
- `**kwargs`: Additional options

**Returns:**
- `list`: Expanded paths

**Example:**
```python
paths = fs.expand_path("data/sales/*.parquet")
```

### Caching

The FileSystem class supports caching for improved performance:

```python
# Enable caching
fs = FileSystem(
    protocol="s3",
    bucket="my-bucket",
    cached=True,
    cache_storage="/tmp/pydala2_cache",
    cache_options={
        'expiry_time': 3600,  # 1 hour
        'cache_check': False  # Don't check if cached file is stale
    }
)

# Subsequent reads will be faster
data1 = fs.cat("large_file.parquet")  # First read - from remote
data2 = fs.cat("large_file.parquet")  # Second read - from cache
```

### Cloud Storage Examples

#### S3 Configuration

```python
# S3 with credentials
fs = FileSystem(
    protocol="s3",
    bucket="my-bucket",
    key="your-access-key",
    secret="your-secret-key",
    client_kwargs={
        "region_name": "us-east-1"
    },
    cached=True
)

# S3 with IAM role
fs = FileSystem(
    protocol="s3",
    bucket="my-bucket",
    client_kwargs={
        "region_name": "us-east-1"
    }
)
```

#### GCS Configuration

```python
# GCS with service account
fs = FileSystem(
    protocol="gcs",
    token="/path/to/service-account.json",
    bucket="my-bucket",
    cached=True
)

# GCS with default credentials
fs = FileSystem(
    protocol="gcs",
    bucket="my-bucket"
)
```

#### Azure Blob Storage

```python
# Azure with connection string
fs = FileSystem(
    protocol="abfs",
    account_name="myaccount",
    account_key="mykey",
    bucket="mycontainer"
)
```

### Performance Monitoring

The FileSystem class includes basic performance monitoring:

```python
import time

# Time operations
start = time.time()
fs.cat("large_file.parquet")
duration = time.time() - start
print(f"Read took {duration:.2f} seconds")

# Check cache statistics
if hasattr(fs, 'cache_stats'):
    print(f"Cache hits: {fs.cache_stats['hits']}")
    print(f"Cache misses: {fs.cache_stats['misses']}")
```

### Error Handling

```python
from fsspec.exceptions import FileNotFoundError

try:
    data = fs.cat("nonexistent.file")
except FileNotFoundError:
    print("File not found")

try:
    fs.open("/protected/file.txt", "w")
except PermissionError:
    print("Permission denied")
```

### Best Practices

1. **Enable caching for frequently accessed data**
2. **Use appropriate block sizes for large files**
3. **Batch operations when possible**
4. **Handle connection errors gracefully**
5. **Monitor cache usage and clean up periodically**

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `cached` | bool | False | Enable filesystem caching |
| `cache_storage` | str | None | Cache storage directory |
| `cache_options` | dict | {} | Cache configuration |
| `block_size` | int | None | Default block size |
| `client_kwargs` | dict | {} | Client configuration |
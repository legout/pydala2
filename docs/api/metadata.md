# Metadata Management

This document describes the metadata classes in PyDala2 that handle collection, storage, and management of dataset metadata.

## Overview

PyDala2 implements a sophisticated two-tier metadata system that leverages Apache Arrow's metadata capabilities:

1. **ParquetDatasetMetadata**: Base class that manages `_metadata` and `_file_metadata` files
2. **PydalaDatasetMetadata**: Extends the base class with DuckDB integration and advanced scanning capabilities

The metadata system creates consolidated metadata files that contain schema and row group information, enabling efficient dataset operations without directory crawling. PyArrow's `parquet_dataset()` function uses the `_metadata` file to instantly create datasets without scanning individual files.

## ParquetDatasetMetadata

```python
class ParquetDatasetMetadata
```

The `ParquetDatasetMetadata` class is the foundation of PyDala2's metadata system, managing consolidated and individual file metadata for Parquet datasets.

### Architecture

The class maintains two types of metadata files:

- **`_metadata`**: Consolidated metadata from all files, used by PyArrow's `parquet_dataset()` for efficient dataset loading
- **`_file_metadata`**: Detailed metadata for individual files, stored with brotli compression

### Constructor

```python
ParquetDatasetMetadata(
    path: str,
    filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
    use_cache: bool = True,
    cache_key: str | None = None,
    partitioning: str | list[str] | None = None,
    **kwargs
) -> None
```

**Parameters:**
- `path` (str): Path to the Parquet dataset
- `filesystem` (AbstractFileSystem | pfs.FileSystem, optional): Filesystem for data access
- `use_cache` (bool): Whether to cache metadata (default: True)
- `cache_key` (str, optional): Key for caching
- `partitioning` (str | list[str], optional): Partitioning scheme
- `**kwargs`: Additional arguments

### Properties

#### path
```python
@property
def path(self) -> str
```
Get the dataset path.

**Returns:**
- `str`: Dataset path

#### filesystem
```python
@property
def filesystem(self) -> AbstractFileSystem | pfs.FileSystem
```
Get the filesystem instance.

**Returns:**
- `AbstractFileSystem | pfs.FileSystem`: Filesystem instance

#### metadata_file
```python
@property
def metadata_file(self) -> str
```
Get the path to the _metadata file.

**Returns:**
- `str`: Path to _metadata file

#### file_metadata_file
```python
@property
def file_metadata_file(self) -> str
```
Get the path to the _file_metadata file.

**Returns:**
- `str`: Path to _file_metadata file

#### has_metadata_file
```python
@property
def has_metadata_file(self) -> bool
```
Check if _metadata file exists.

**Returns:**
- `bool`: True if _metadata file exists

#### has_file_metadata_file
```python
@property
def has_file_metadata_file(self) -> bool
```
Check if _file_metadata file exists.

**Returns:**
- `bool`: True if _file_metadata file exists

#### metadata
```python
@property
def metadata(self) -> pa.Table
```
Get the consolidated metadata as a PyArrow Table.

**Returns:**
- `pa.Table`: Metadata table

#### file_metadata
```python
@property
def file_metadata(self) -> list[pyarrow.parquet.FileMetaData]
```
Get individual file metadata objects.

**Returns:**
- `list[pyarrow.parquet.FileMetaData]`: List of file metadata objects

#### files
```python
@property
def files(self) -> list[str]
```
Get list of Parquet files in the dataset.

**Returns:**
- `list[str]`: List of file paths

#### num_files
```python
@property
def num_files(self) -> int
```
Get the number of files in the dataset.

**Returns:**
- `int`: Number of files

#### num_row_groups
```python
@property
def num_row_groups(self) -> int
```
Get the total number of row groups.

**Returns:**
- `int`: Number of row groups

#### schema
```python
@property
def schema(self) -> pa.Schema
```
Get the unified schema of the dataset.

**Returns:**
- `pa.Schema`: Dataset schema

#### columns
```python
@property
def columns(self) -> list[str]
```
Get the list of column names.

**Returns:**
- `list[str]`: Column names

#### size_bytes
```python
@property
def size_bytes(self) -> int
```
Get the total size of the dataset in bytes.

**Returns:**
- `int`: Size in bytes

#### num_rows
```python
@property
def num_rows(self) -> int
```
Get the total number of rows in the dataset.

**Returns:**
- `int`: Total row count

### Methods

#### update
```python
def update(
    self,
    reload: bool = False,
    verbose: bool = False,
    update_file_metadata: bool = True,
    update_metadata: bool = True,
    repair: bool = True,
    **kwargs
) -> None
```
Update metadata files. This is the main method for metadata management.

**Parameters:**
- `reload` (bool): Whether to reload existing metadata
- `verbose` (bool): Whether to print verbose output
- `update_file_metadata` (bool): Whether to update _file_metadata
- `update_metadata` (bool): Whether to update _metadata
- `repair` (bool): Whether to repair file schemas
- `**kwargs`: Additional arguments

**Example:**
```python
# Update all metadata
metadata.update(verbose=True)

# Update only consolidated metadata
metadata.update(update_file_metadata=False)
```

#### update_file_metadata
```python
def update_file_metadata(
    self,
    files: list[str] | None = None,
    verbose: bool = False,
    **kwargs
) -> None
```
Update the _file_metadata file with individual file metadata.

**Parameters:**
- `files` (list[str], optional): Specific files to update. If None, updates all files
- `verbose` (bool): Whether to print verbose output

**Example:**
```python
# Update file metadata for specific files
metadata.update_file_metadata(files=["file1.parquet", "file2.parquet"])
```

#### _update_metadata
```python
def _update_metadata(self, **kwargs) -> None
```
Internal method to update the consolidated _metadata file.

**Parameters:**
- `**kwargs`: Additional arguments

#### _read_metadata
```python
def _read_metadata(self) -> pa.Table
```
Internal method to read metadata from files.

**Returns:**
- `pa.Table`: Metadata table

#### _repair_file_schemas
```python
def _repair_file_schemas(self, verbose: bool = False, **kwargs) -> None
```
Internal method to repair inconsistent file schemas.

**Parameters:**
- `verbose` (bool): Whether to print verbose output
- `**kwargs`: Additional arguments

## PydalaDatasetMetadata

```python
class PydalaDatasetMetadata(ParquetDatasetMetadata)
```

The `PydalaDatasetMetadata` class extends `ParquetDatasetMetadata` with DuckDB integration and advanced scanning capabilities for metadata-based filtering.

### Constructor

```python
PydalaDatasetMetadata(
    path: str,
    filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
    use_cache: bool = True,
    cache_key: str | None = None,
    partitioning: str | list[str] | None = None,
    ddb_con: duckdb.DuckDBPyConnection | None = None,
    **kwargs
) -> None
```

**Parameters:**
- `path` (str): Path to the Parquet dataset
- `filesystem` (AbstractFileSystem | pfs.FileSystem, optional): Filesystem for data access
- `use_cache` (bool): Whether to cache metadata (default: True)
- `cache_key` (str, optional): Key for caching
- `partitioning` (str | list[str], optional): Partitioning scheme
- `ddb_con` (duckdb.DuckDBPyConnection, optional): DuckDB connection. If None, creates a new connection
- `**kwargs`: Additional arguments

### Properties

#### ddb_con
```python
@property
def ddb_con(self) -> duckdb.DuckDBPyConnection
```
Get the DuckDB connection instance.

**Returns:**
- `duckdb.DuckDBPyConnection`: DuckDB connection

#### metadata_table
```python
@property
def metadata_table(self) -> duckdb.DuckDBPyRelation
```
Get the metadata as a DuckDB table.

**Returns:**
- `duckdb.DuckDBPyRelation`: Metadata table relation

#### scan_table
```python
@property
def scan_table(self) -> duckdb.DuckDBPyRelation
```
Get the scan results as a DuckDB table.

**Returns:**
- `duckdb.DuckDBPyRelation`: Scan results table

### Methods

#### reset_scan
```python
def reset_scan(self) -> None
```
Reset scan results.

**Example:**
```python
# Reset scan to start fresh
metadata.reset_scan()
```

#### scan
```python
def scan(
    self,
    filters: str | list[str] | dict | None = None,
    files: list[str] | None = None,
    columns: list[str] | None = None,
    verbose: bool = False,
    return_table: bool = False,
    **kwargs
) -> list[str] | pa.Table
```
Scan files based on metadata statistics. This is a powerful method for filtering files without reading their contents.

**Parameters:**
- `filters` (str | list[str] | dict, optional): Filter conditions
- `files` (list[str], optional): Specific files to scan
- `columns` (list[str], optional): Columns to consider in filters
- `verbose` (bool): Whether to print verbose output
- `return_table` (bool): Whether to return results as a table
- `**kwargs`: Additional arguments

**Returns:**
- `list[str] | pa.Table`: List of matching files or results table

**Example:**
```python
# Scan files based on date range
matching_files = metadata.scan(
    filters="date >= '2023-01-01' AND date <= '2023-12-31'",
    verbose=True
)

# Scan using dictionary filters
matching_files = metadata.scan(
    filters={
        'amount': {'min': 100, 'max': 1000},
        'category': ['A', 'B', 'C']
    }
)

# Get results as a table
results = metadata.scan(
    filters="status = 'completed'",
    return_table=True
)
```

#### update_metadata_table
```python
def update_metadata_table(self, update_parquet_file_metadata: bool = False) -> None
```
Update the DuckDB metadata table with current metadata.

**Parameters:**
- `update_parquet_file_metadata` (bool): Whether to update from _file_metadata

**Example:**
```python
# Update metadata table
metadata.update_metadata_table()
```

#### query_metadata
```python
def query_metadata(self, query: str, **kwargs) -> duckdb.DuckDBPyRelation
```
Execute SQL queries on the metadata table.

**Parameters:**
- `query` (str): SQL query
- `**kwargs`: Additional arguments

**Returns:**
- `duckdb.DuckDBPyRelation`: Query result

**Example:**
```python
# Query metadata statistics
result = metadata.query_metadata("""
    SELECT column_name, min_value, max_value, null_count
    FROM metadata_table
    WHERE column_name IN ('amount', 'date')
""")
```

## The _metadata File

The `_metadata` file is a special Parquet file that consolidates metadata from all files in the dataset. It contains:
- Schema information unified across all files
- Row group metadata from all files
- Column statistics (min, max, null counts)
- File paths and sizes
- Compression information

This file enables:
- **Instant dataset loading**: PyArrow's `parquet_dataset()` function uses this file to create datasets without directory crawling
- **Efficient query planning**: Statistics enable predicate pushdown
- **File pruning**: Skip files that don't contain relevant data
- **Schema evolution**: Handle schema changes across files

### Two-Tier Metadata System

PyDala2 maintains two metadata files:

1. **`_metadata`**: Consolidated metadata used by PyArrow for efficient dataset loading
2. **`_file_metadata`**: Detailed individual file metadata stored with brotli compression for advanced operations

### Automatic Metadata Management

```python
from pydala import ParquetDataset

# Create dataset - metadata is automatically managed
dataset = ParquetDataset("data/sales")

# Write data - metadata files are automatically created/updated
dataset.write_to_dataset(data, partition_cols=['year', 'month'])

# The _metadata file is created automatically
# PyArrow can now load the dataset instantly
```

### Manual Metadata Updates

```python
# Update metadata when files are added/modified
dataset.update(verbose=True)

# Update only file metadata
dataset.update(update_metadata=False)

# Update only consolidated metadata
dataset.update(update_file_metadata=False)
```

## Metadata Operations

### Metadata-Based File Filtering

```python
# Use the scan method to filter files without reading data
matching_files = dataset.scan(
    filters="date >= '2023-01-01' AND amount > 1000",
    verbose=True
)

# Only process files that contain matching data
for file_path in matching_files:
    # These files are guaranteed to contain relevant data
    process_file(file_path)
```

### SQL Queries on Metadata

```python
# Query metadata statistics using SQL
stats = dataset.query_metadata("""
    SELECT
        column_name,
        min_value,
        max_value,
        null_count,
        SUM(num_rows) as total_rows
    FROM metadata_table
    WHERE column_name IN ('amount', 'date', 'category')
    GROUP BY column_name
""").to_arrow()

print(stats)
```

### Advanced Filtering with DuckDB

```python
# Use DuckDB for complex metadata queries
result = dataset.ddb_con.sql("""
    SELECT file_path, num_rows
    FROM dataset_metadata
    WHERE
        date >= '2023-01-01' AND
        amount > 1000 AND
        category IN ('A', 'B', 'C')
""").to_arrow()
```

## Performance Benefits

Using the metadata system provides several performance benefits:

1. **Instant Dataset Loading**: The `_metadata` file enables PyArrow to create datasets without directory scanning
2. **File Pruning**: The `scan()` method skips files that don't contain relevant data based on statistics
3. **Query Optimization**: Statistics enable predicate pushdown to individual files
4. **Schema Evolution**: Automatic schema repair handles schema differences across files
5. **Efficient Storage**: Brotli compression for `_file_metadata` reduces storage overhead
6. **SQL Access**: DuckDB integration enables SQL queries on metadata

## Example: Metadata-Enabled Workflow

```python
from pydala import ParquetDataset

# Create or load dataset
dataset = ParquetDataset("data/sales")

# Write data - metadata is automatically created
dataset.write_to_dataset(data, partition_cols=['year', 'month'])

# Dataset automatically uses _metadata for efficient loading
print(f"Dataset info: {dataset.num_rows} rows in {dataset.num_files} files")
print(f"Total size: {dataset.size_bytes / 1024 / 1024:.1f} MB")

# Use metadata-based filtering
# First, scan to find relevant files
matching_files = dataset.scan(
    filters="year = 2023 AND month IN (1, 2, 3) AND amount > 1000"
)

print(f"Found {len(matching_files)} files with matching data")

# Then read only relevant data
result = dataset.table.to_polars()  # Uses _metadata for fast loading
```

## Metadata Serialization

PyDala2 uses efficient serialization for metadata storage:

```python
# Metadata is automatically serialized/deserialized
# _file_metadata uses brotli compression
# _metadata is in PyArrow's native format

# The serialization/deserialization is handled automatically:
import brotli
import base64
import pyarrow as pa

# These functions are used internally:
def serialize_metadata(metadata: pa.Table) -> bytes:
    """Serialize metadata table to compressed bytes."""
    return base64.b64encode(
        brotli.compress(
            metadata.serialize().to_pybytes()
        )
    )

def deserialize_metadata(data: bytes) -> pa.Table:
    """Deserialize metadata from compressed bytes."""
    return pa.Table.deserialize(
        brotli.decompress(
            base64.b64decode(data)
        )
    )
```
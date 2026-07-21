# Core Classes

This document describes the core classes that form the foundation of PyDala2.

## BaseDataset

```python
class BaseDataset
```

The `BaseDataset` class provides the foundation for all dataset operations in PyDala2. It implements common functionality for working with datasets across different file formats using a dual-engine architecture (PyArrow + DuckDB).

### Constructor

```python
BaseDataset(
    path: str,
    name: str | None = None,
    schema: pa.Schema | None = None,
    filesystem: AbstractFileSystem | None = None,
    bucket: str | None = None,
    partitioning: str | list[str] | None = None,
    format: str | None = "parquet",
    cached: bool = False,
    timestamp_column: str | None = None,
    ddb_con: duckdb.DuckDBPyConnection | None = None,
    **fs_kwargs
) -> None
```

**Parameters:**
- `path` (str): The path to the dataset
- `name` (str, optional): The name of the dataset. If None, uses basename of path
- `schema` (pa.Schema, optional): The schema for the dataset. If None, inferred from data
- `filesystem` (AbstractFileSystem, optional): The filesystem to use for data access
- `bucket` (str, optional): The bucket name for cloud storage
- `partitioning` (str | list[str], optional): The partitioning scheme ('hive', 'ignore', or list of columns)
- `format` (str, optional): The file format ('parquet', 'csv', 'json')
- `cached` (bool): Whether to use caching for filesystem operations
- `timestamp_column` (str, optional): The column to use as timestamp for time operations
- `ddb_con` (duckdb.DuckDBPyConnection, optional): Existing DuckDB connection
- `**fs_kwargs`: Additional arguments for filesystem configuration

**Example:**
```python
from pydala import BaseDataset
import pyarrow as pa

# Define schema
schema = pa.schema([
    ('id', pa.int64()),
    ('name', pa.string()),
    ('value', pa.float64())
])

# Create dataset
dataset = BaseDataset(
    path="data/my_dataset",
    schema=schema,
    partitioning=['year', 'month'],
    cached=True
)
```

### Properties

#### path
```python
@property
def path(self) -> str
```
Get the path of the dataset.

**Returns:**
- `str`: The path to the dataset

#### name
```python
@property
def name(self) -> str
```
Get the name of the dataset.

**Returns:**
- `str`: The dataset name

#### table
```python
@property
def table(self) -> PydalaTable
```
Get the PydalaTable for data operations.

**Returns:**
- `PydalaTable`: The table interface for data export and operations

**Example:**
```python
# Access the table interface
table = dataset.table

# Export to Polars
df = table.to_polars()

# Use shortcut notation
df = dataset.t.to_polars()  # Same as above
```

#### t
```python
@property
def t(self) -> PydalaTable
```
Shortcut property for accessing the table interface.

**Returns:**
- `PydalaTable`: The table interface

#### ddb_con
```python
@property
def ddb_con(self) -> duckdb.DuckDBPyConnection
```
Get the DuckDB connection for SQL operations.

**Returns:**
- `duckdb.DuckDBPyConnection`: The DuckDB connection

#### schema
```python
@property
def schema(self) -> pa.Schema
```
Get the schema of the dataset.

**Returns:**
- `pa.Schema`: The schema of the dataset

#### files
```python
@property
def files(self) -> list[str]
```
Get a list of files in the dataset.

**Returns:**
- `list[str]`: List of file paths

#### columns
```python
@property
def columns(self) -> list[str]
```
Get a list of column names.

**Returns:**
- `list[str]`: List of column names

#### partitions
```python
@property
def partitions(self) -> list[str]
```
Get partition information for the dataset.

**Returns:**
- `list[str]`: List of partition values

#### metadata_file
```python
@property
def metadata_file(self) -> str | None
```
Get the path to the metadata file if it exists.

**Returns:**
- `str | None`: Path to the _metadata file or None

### Methods

#### load
```python
def load(self) -> None
```
Load the dataset from the specified path and initialize the PydalaTable.

This method creates a PyArrow dataset from the files at the specified path, initializes a PydalaTable for data operations, and registers the dataset with DuckDB for SQL queries.

**Example:**
```python
dataset = BaseDataset("data/existing_dataset")
dataset.load()
```

#### exists
```python
def exists(self) -> bool
```
Check if the dataset exists.

**Returns:**
- `bool`: True if the dataset exists, False otherwise

**Example:**
```python
if dataset.exists():
    print(f"Dataset found with {dataset.count_rows()} rows")
```

#### count_rows
```python
def count_rows(self) -> int
```
Returns the number of rows in the dataset.

**Returns:**
- `int`: The number of rows in the dataset

#### filter
```python
def filter(self, filters: str) -> BaseDataset
```
Filter the dataset using the specified filters.

**Parameters:**
- `filters` (str): Filter expression

**Returns:**
- `BaseDataset`: A new dataset instance with the filters applied

**Example:**
```python
# Filter dataset
filtered = dataset.filter("date > '2023-01-01' AND category IN ('A', 'B')")

# Export filtered data
result = filtered.table.to_polars()
```

#### scan
```python
def scan(self, filters: str | None = None) -> pa.dataset.Scanner
```
Create a scanner for efficient data access.

**Parameters:**
- `filters` (str, optional): Filter expression

**Returns:**
- `pa.dataset.Scanner`: A PyArrow dataset scanner

**Example:**
```python
# Create scanner
scanner = dataset.scan("value > 100")

# Read data in batches
for batch in scanner.to_batches():
    process_batch(batch)
```

#### clear_cache
```python
def clear_cache(self) -> None
```
Clear the cache for the dataset.

This method clears the cache for both the filesystem and base filesystem.

#### delete_files
```python
def delete_files(self, files: str | list[str]) -> None
```
Delete specified files from the dataset.

**Parameters:**
- `files` (str | list[str]): The file(s) to be deleted

**Example:**
```python
# Delete specific files
dataset.delete_files(["data/file1.parquet", "data/file2.parquet"])

# Delete all files
dataset.delete_files(dataset.files)
```

#### vacuum
```python
def vacuum(self) -> None
```
Delete all files in the dataset.

This method removes all files from the dataset directory but keeps the directory structure intact.

## Optimize

```python
class Optimize
```

The `Optimize` class provides dataset optimization and compaction operations.

### Methods

#### compact_partitions
```python
def compact_partitions(self, target_file_size: int | None = None) -> None
```
Compact partitions within the dataset.

**Parameters:**
- `target_file_size` (int, optional): Target size for compacted files in bytes

**Example:**
```python
# Compact partitions with default size
dataset.optimize.compact_partitions()

# Compact with specific target size
dataset.optimize.compact_partitions(target_file_size=100 * 1024 * 1024)  # 100MB
```

#### compact_by_timeperiod
```python
def compact_by_timeperiod(
    self,
    timestamp_column: str,
    timeperiod: str = "day",
    target_file_size: int | None = None
) -> None
```
Compact data by time period.

**Parameters:**
- `timestamp_column` (str): Column containing timestamps
- `timeperiod` (str): Time period for compaction ('day', 'week', 'month', 'year')
- `target_file_size` (int, optional): Target size for compacted files

**Example:**
```python
# Compact by month
dataset.optimize.compact_by_timeperiod(
    timestamp_column="date",
    timeperiod="month"
)
```

#### compact_by_rows
```python
def compact_by_rows(self, target_rows: int) -> None
```
Compact data to achieve target row count per file.

**Parameters:**
- `target_rows` (int): Target number of rows per file

**Example:**
```python
# Compact to 1M rows per file
dataset.optimize.compact_by_rows(target_rows=1000000)
```

#### repartition
```python
def repartition(self, partition_cols: list[str]) -> None
```
Repartition the dataset.

**Parameters:**
- `partition_cols` (list[str]): New partitioning columns

**Example:**
```python
# Change partitioning
dataset.optimize.repartition(partition_cols=['year', 'month', 'day'])
```

#### optimize_dtypes
```python
def optimize_dtypes(self) -> None
```
Optimize data types to reduce memory usage.

This method analyzes the data and converts columns to more efficient data types where possible.

**Example:**
```python
# Optimize data types
dataset.optimize.optimize_dtypes()
```

## Writer

```python
class Writer
```

The `Writer` class provides data writing and transformation utilities.

### Methods

#### write_to_dataset
```python
@staticmethod
def write_to_dataset(
    data: pa.Table | pd.DataFrame | pl.DataFrame,
    dataset: BaseDataset,
    partition_cols: list[str] | None = None,
    basename_template: str | None = None,
    **kwargs
) -> None
```
Write data to a dataset.

**Parameters:**
- `data` (pa.Table | pd.DataFrame | pl.DataFrame): Data to write
- `dataset` (BaseDataset): Target dataset
- `partition_cols` (list[str], optional): Columns to partition by
- `basename_template` (str, optional): Template for file names
- `**kwargs`: Additional arguments for writing

**Example:**
```python
# Write data to dataset
Writer.write_to_dataset(
    data=dataframe,
    dataset=dataset,
    partition_cols=['category'],
    basename_template="data-{i}.parquet"
)
```

## Config

```python
class Config
```

The `Config` class manages configuration settings for PyDala2 operations.

### Class Attributes

#### default_config
```python
default_config: dict = {
    'default_backend': 'polars',
    'cache_enabled': True,
    'cache_max_size': 1024 * 1024 * 1024,  # 1GB
    'cache_ttl': 3600,  # 1 hour
    'log_level': 'INFO',
    'max_memory': 4 * 1024 * 1024 * 1024,  # 4GB
    'n_workers': 4,
    'compression': 'zstd',
    'compression_level': 3,
    'row_group_size': 1000000,
    'auto_discover_partitions': True,
    'validate_schema': True,
    'enable_profiling': False
}
```

Default configuration values for PyDala2.

### Methods

#### get
```python
@classmethod
def get(cls, key: str, default=None)
```
Get a configuration value.

**Parameters:**
- `key` (str): The configuration key
- `default`: Default value if key not found

**Returns:**
- The configuration value

**Example:**
```python
# Get default backend
backend = Config.get('default_backend')

# Get with default
cache_size = Config.get('cache_max_size', 512 * 1024 * 1024)
```

#### set
```python
@classmethod
def set(cls, key: str, value) -> None
```
Set a configuration value.

**Parameters:**
- `key` (str): The configuration key
- `value`: The value to set

**Example:**
```python
# Set default backend
Config.set('default_backend', 'duckdb')

# Enable caching
Config.set('cache_enabled', True)
```

#### update
```python
@classmethod
def update(cls, config: dict) -> None
```
Update multiple configuration values.

**Parameters:**
- `config` (dict): Dictionary of configuration values

**Example:**
```python
Config.update({
    'default_backend': 'polars',
    'cache_enabled': True,
    'cache_max_size': 2 * 1024 * 1024 * 1024,
    'log_level': 'DEBUG'
})
```

#### reset
```python
@classmethod
def reset(cls) -> None
```
Reset all configuration values to defaults.

**Example:**
```python
# Reset to defaults
Config.reset()
```

## Configuration Management

PyDala2 provides several ways to manage configuration:

### Environment Variables

Configuration can be set via environment variables:

```bash
export PYDALA2_DEFAULT_BACKEND=polars
export PYDALA2_CACHE_ENABLED=true
export PYDALA2_CACHE_MAX_SIZE=1073741824
export PYDALA2_LOG_LEVEL=INFO
```

### Configuration File

Configuration can be loaded from a YAML file:

```python
from pydala import load_config

# Load from file
config = load_config("config.yaml")
Config.update(config)
```

### Programmatic Configuration

```python
from pydala import Config, set_config

# Set individual values
Config.set('default_backend', 'polars')

# Update multiple values
set_config({
    'cache_enabled': True,
    'cache_max_size': 1024 * 1024 * 1024,
    'log_level': 'INFO'
})
```

### Context Manager

Configuration can be temporarily changed using a context manager:

```python
from pydala import config_context

with config_context({'default_backend': 'duckdb'}):
    # Use DuckDB for operations in this block
    result = dataset.read(backend=None)  # Uses DuckDB

# Back to previous configuration
```

## Configuration Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_backend` | str | 'polars' | Default backend for operations |
| `cache_enabled` | bool | True | Enable filesystem caching |
| `cache_max_size` | int | 1073741824 | Maximum cache size in bytes |
| `cache_ttl` | int | 3600 | Cache time-to-live in seconds |
| `log_level` | str | 'INFO' | Logging level |
| `max_memory` | int | 4294967296 | Maximum memory usage in bytes |
| `n_workers` | int | 4 | Number of worker threads |
| `compression` | str | 'zstd' | Default compression codec |
| `compression_level` | int | 3 | Default compression level |
| `row_group_size` | int | 1000000 | Default row group size |
| `auto_discover_partitions` | bool | True | Auto-discover partitioning |
| `validate_schema` | bool | True | Validate schema on operations |
| `enable_profiling` | bool | False | Enable performance profiling |
# Dataset Classes

This document describes the dataset classes that provide file format-specific operations.

## ParquetDataset

```python
class ParquetDataset(PydalaDatasetMetadata, BaseDataset)
```

The `ParquetDataset` class inherits from both `PydalaDatasetMetadata` (for metadata management) and `BaseDataset` (for dataset operations). It provides comprehensive Parquet file operations with advanced metadata support, optimization features, and efficient querying.

### Constructor

```python
ParquetDataset(
    path: str,
    name: str | None = None,
    filesystem: AbstractFileSystem | None = None,
    bucket: str | None = None,
    partitioning: str | list[str] | None = None,
    cached: bool = False,
    timestamp_column: str | None = None,
    ddb_con: duckdb.DuckDBPyConnection | None = None,
    **fs_kwargs
) -> None
```

**Parameters:**
- `path` (str): The path to the Parquet dataset
- `name` (str, optional): The name of the dataset
- `filesystem` (AbstractFileSystem, optional): Filesystem for data access
- `bucket` (str, optional): Bucket name for cloud storage
- `partitioning` (str | list[str], optional): Partitioning scheme
- `cached` (bool): Whether to use caching
- `timestamp_column` (str, optional): Timestamp column for time operations
- `ddb_con` (duckdb.DuckDBPyConnection, optional): DuckDB connection
- `**fs_kwargs`: Additional filesystem arguments

**Example:**
```python
from pydala import ParquetDataset

# Create dataset
dataset = ParquetDataset(
    path="data/sales",
    partitioning=['year', 'month'],
    cached=True
)

# Write data
dataset.write_to_dataset(
    dataframe,
    partition_cols=['year', 'month'],
    basename_template="data-{i}.parquet"
)

# Export data with filters
result = dataset.filter(
    "year = 2023 AND month IN (1, 2, 3)"
).table.to_polars()
```

### Methods

#### write_to_dataset
```python
def write_to_dataset(
    self,
    data: pa.Table | pd.DataFrame | pl.DataFrame,
    partition_cols: list[str] | None = None,
    basename_template: str | None = None,
    max_rows_per_file: int | None = None,
    min_rows_per_group: int | None = None,
    max_rows_per_group: int | None = None,
    compression: str | None = None,
    compression_level: int | None = None,
    use_threads: bool | None = None,
    **kwargs
) -> None
```
Write data to the Parquet dataset.

**Parameters:**
- `data` (pa.Table | pd.DataFrame | pl.DataFrame): Data to write
- `partition_cols` (list[str], optional): Columns to partition by
- `basename_template` (str, optional): Template for file names
- `max_rows_per_file` (int, optional): Maximum rows per output file
- `min_rows_per_group` (int, optional): Minimum rows per row group
- `max_rows_per_group` (int, optional): Maximum rows per row group
- `compression` (str, optional): Compression codec
- `compression_level` (int, optional): Compression level
- `use_threads` (bool, optional): Whether to use threading
- `**kwargs`: Additional write options

**Example:**
```python
# Write with partitioning
dataset.write_to_dataset(
    data=df,
    partition_cols=['category', 'date'],
    basename_template="data-{i}.parquet",
    max_rows_per_file=1000000,
    compression='zstd',
    compression_level=3
)
```

#### load
```python
def load(
    self,
    update_metadata: bool = False,
    reload_metadata: bool = False,
    **kwargs
) -> None
```
Load the dataset using PyArrow's `parquet_dataset()` function with the `_metadata` file for efficient loading.

**Parameters:**
- `update_metadata` (bool): Whether to update metadata before loading
- `reload_metadata` (bool): Whether to reload metadata from disk
- `**kwargs`: Additional arguments

**Example:**
```python
# Load dataset (uses _metadata file for instant loading)
dataset.load()

# Load with metadata update
dataset.load(update_metadata=True)

# Access the loaded Arrow dataset
arrow_dataset = dataset._arrow_dataset
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
Scan files based on metadata statistics without reading their contents.

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
matching_files = dataset.scan(
    filters="date >= '2023-01-01' AND date <= '2023-12-31'",
    verbose=True
)

# Get results as a table
results = dataset.scan(
    filters="status = 'completed'",
    return_table=True
)
```

#### optimize_dtypes
```python
def optimize_dtypes(self) -> None
```
Optimize data types to reduce memory usage.

This method is accessed through the optimize attribute:

**Example:**
```python
# Optimize data types
dataset.optimize.optimize_dtypes()
```

#### compact_partitions
```python
def compact_partitions(self, target_file_size: int | None = None) -> None
```
Compact partitions within the dataset.

This method is accessed through the optimize attribute:

**Parameters:**
- `target_file_size` (int, optional): Target size for compacted files in bytes

**Example:**
```python
# Compact partitions
dataset.optimize.compact_partitions(target_file_size=100 * 1024 * 1024)
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

This method is accessed through the optimize attribute:

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

This method is accessed through the optimize attribute:

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

This method is accessed through the optimize attribute:

**Parameters:**
- `partition_cols` (list[str]): New partitioning columns

**Example:**
```python
# Change partitioning
dataset.optimize.repartition(partition_cols=['year', 'month', 'day'])
```

## PyarrowDataset

```python
class PyarrowDataset(BaseDataset)
```

Dataset implementation using PyArrow's native capabilities.

### Constructor

```python
PyarrowDataset(
    path: str,
    name: str | None = None,
    filesystem: AbstractFileSystem | None = None,
    bucket: str | None = None,
    partitioning: str | list[str] | None = None,
    cached: bool = False,
    timestamp_column: str | None = None,
    ddb_con: duckdb.DuckDBPyConnection | None = None,
    **fs_kwargs
) -> None
```

### Methods

#### load
```python
def load(self) -> None
```
Load the dataset using PyArrow's dataset reading capabilities.

**Example:**
```python
from pydala import PyarrowDataset

dataset = PyarrowDataset("data/dataset")
dataset.load()

# Access as Arrow dataset
arrow_dataset = dataset.to_arrow_dataset()
```

## CSVDataset

```python
class CSVDataset(BaseDataset)
```

Dataset implementation for CSV files with support for various CSV formats and options.

### Constructor

```python
CSVDataset(
    path: str,
    name: str | None = None,
    filesystem: AbstractFileSystem | None = None,
    bucket: str | None = None,
    partitioning: str | list[str] | None = None,
    cached: bool = False,
    timestamp_column: str | None = None,
    ddb_con: duckdb.DuckDBPyConnection | None = None,
    **fs_kwargs
) -> None
```

### Methods

#### load
```python
def load(self) -> None
```
Load CSV files using PyArrow's CSV reading capabilities.

**Example:**
```python
from pydala import CSVDataset

dataset = CSVDataset("data/sales.csv")
dataset.load()

# Read with CSV options
data = dataset.read(
    parse_options={'delimiter': ',', 'quote_char': '"'},
    convert_options={'column_types': {'id': pa.int64()}}
)
```

## JSONDataset

```python
class JSONDataset(BaseDataset)
```

Dataset implementation for JSON files with support for nested JSON structures.

### Constructor

```python
JSONDataset(
    path: str,
    name: str | None = None,
    filesystem: AbstractFileSystem | None = None,
    bucket: str | None = None,
    partitioning: str | list[str] | None = None,
    cached: bool = False,
    timestamp_column: str | None = None,
    ddb_con: duckdb.DuckDBPyConnection | None = None,
    **fs_kwargs
) -> None
```

### Methods

#### load
```python
def load(self) -> None
```
Load JSON files using PyArrow's JSON reading capabilities.

**Example:**
```python
from pydala import JSONDataset

dataset = JSONDataset("data/events")
dataset.load()

# Read JSON data
data = dataset.read(
    read_options={'block_size': 4096},
    parse_options={'explicit_schema': schema}
)
```

## Common Dataset Operations

### Writing Data

All dataset classes support writing data using the `write_to_dataset` method:

```python
# Write with partitioning
dataset.write_to_dataset(
    data=dataframe,
    partition_cols=['year', 'month'],
    basename_template="data-{i}.parquet"
)

# Write with compression
dataset.write_to_dataset(
    data=dataframe,
    compression='zstd',
    compression_level=3
)
```

### Reading/Exporting Data

Data is accessed through the `table` property or its shortcut `t`:

```python
# Export to different formats
df_polars = dataset.table.to_polars()  # LazyFrame
df_pandas = dataset.table.df  # Pandas DataFrame
table_arrow = dataset.table.arrow  # PyArrow Table

# Use shortcut notation
df_polars = dataset.t.to_polars()  # Same as above
```

### Filtering Data

```python
# Filter with automatic backend selection
filtered = dataset.filter("date > '2023-01-01' AND category = 'premium'")

# Export filtered data
result = filtered.table.to_polars()
result = filtered.t.df  # Using shortcut
```

### Using SQL with DuckDB

```python
# Execute SQL queries
result = dataset.ddb_con.sql("""
    SELECT category, AVG(amount) as avg_amount
    FROM dataset
    WHERE date > '2023-01-01'
    GROUP BY category
""").to_arrow()
```

### Schema Operations

```python
# View schema
print(dataset.schema)

# View columns
print(dataset.columns)

# Check if dataset exists
if dataset.exists():
    print(f"Dataset has {dataset.count_rows()} rows")
```

### Performance Optimization

```python
# Compact partitions
dataset.optimize.compact_partitions(target_file_size=100 * 1024 * 1024)

# Compact by time period
dataset.optimize.compact_by_timeperiod(
    timestamp_column="date",
    timeperiod="month"
)

# Optimize data types
dataset.optimize.optimize_dtypes()

# Repartition dataset
dataset.optimize.repartition(partition_cols=['year', 'month', 'day'])
```

## Dataset Selection Guide

| Dataset Type | Use Case | Features |
|--------------|----------|-----------|
| `ParquetDataset` | Columnar data, analytics | Compression, partitioning, metadata |
| `PyarrowDataset` | Arrow-native operations | Maximum Arrow compatibility |
| `CSVDataset` | CSV data interchange | Flexible CSV parsing options |
| `JSONDataset` | Nested/semi-structured data | JSON schema support |

Choose the appropriate dataset type based on your data format and use case requirements.
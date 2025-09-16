# Basic Usage

This guide covers the fundamental operations you'll perform with PyDala2 datasets.

## Creating Datasets

### Basic Dataset Creation

```python
from pydala import ParquetDataset

# Create a new dataset (directory auto-created)
dataset = ParquetDataset("data/my_dataset")

# The dataset is now ready with:
# - Automatic DuckDB connection
# - Object caching enabled
# - Partitioning inference
```

### Dataset with Configuration

```python
# Configure dataset options
dataset = ParquetDataset(
    "data/configured_dataset",
    name="sales_data",           # Custom name (defaults to directory name)
    partitioning="hive",         # Auto-detect hive partitioning
    cached=True,                 # Enable filesystem caching
    timestamp_column="created_at" # For time-based operations
)
```

### Cloud Storage Dataset

```python
# S3 dataset
s3_dataset = ParquetDataset(
    "s3://my-bucket/sales-data",
    bucket="my-bucket",
    key="your-access-key",
    secret="your-secret-key",
    cached=True,
    cache_storage="/tmp/s3_cache"
)

# The dataset handles S3 authentication and caching automatically
```

## Writing Data

### Basic Write

```python
import polars as pl

data = pl.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'value': [100, 200, 300]
})

# Write with default settings
dataset.write_to_dataset(data)
```

### Optimized Write

```python
# Write with optimization options
dataset.write_to_dataset(
    data,
    mode="append",                    # append, delta, overwrite
    partition_by=["category"],        # Hive-style partitioning
    max_rows_per_file=1000000,       # Control file size
    row_group_size=250000,           # Parquet row group size
    compression="zstd",              # Compression algorithm
    sort_by="id DESC",               # Sort before writing
    unique=True,                     # Remove duplicates
    update_metadata=True             # Update metadata after write
)
```

### Partitioned Writes

```python
# Partition by multiple columns
data = pl.DataFrame({
    'id': range(100),
    'date': pl.date_range(start=2023-01-01, periods=100, interval='1d', eager=True),
    'region': ['US', 'EU', 'APAC'] * 33 + ['US'],
    'category': ['A', 'B'] * 50,
    'value': range(100)
})

dataset.write_to_dataset(
    data,
    partition_by=["year", "month", "region"],  # Creates nested directories
    max_rows_per_file=50000
)

# Creates structure:
# data/my_dataset/
#   ├── year=2023/
#   │   ├── month=01/
#   │   │   ├── region=US/
#   │   │   │   └── data-0.parquet
#   │   │   └── region=EU/
#   │   │       └── data-0.parquet
```

### Delta Updates

```python
# Efficiently merge new data
new_data = pl.DataFrame({
    'id': [101, 102],
    'category': ['A', 'B'],
    'value': [100, 200],
    'updated_at': [datetime.now(), datetime.now()]
})

# Delta mode checks for existing data
dataset.write_to_dataset(
    new_data,
    mode="delta",
    delta_subset=["id"],             # Columns to check for duplicates
    partition_by=["category"]
)
```

## Reading Data

### Basic Reading

```python
# Read entire dataset as Polars DataFrame
df = dataset.table.pl.collect()

# Or as PyArrow Table
table = dataset.table.arrow()
```

### Filtering with Automatic Backend Selection

```python
# Simple filters use PyArrow
simple_filtered = dataset.filter("category = 'A'")

# Complex filters automatically use DuckDB
complex_filtered = dataset.filter("""
    category IN ('A', 'B')
    AND value > 100
    AND name LIKE '%test%'
    OR (category = 'C' AND value < 50)
""")

# Get results
result = complex_filtered.collect()  # Returns Polars DataFrame
```

### Using DuckDB Directly

```python
# Dataset is automatically registered in DuckDB
sql_result = dataset.ddb_con.sql("""
    SELECT
        category,
        COUNT(*) as count,
        AVG(value) as avg_value,
        MIN(value) as min_value,
        MAX(value) as max_value
    FROM dataset
    GROUP BY category
    ORDER BY count DESC
""").pl()  # Convert to Polars
```

### Partition Pruning

```python
# For partitioned datasets, PyDala2 automatically prunes partitions
# when filtering on partition columns
result = dataset.filter("year = 2023 AND month = 01")
# Only reads files in year=2023/month=01/ directory
```

## Dataset Information

### Basic Properties

```python
# Dataset information
print(f"Dataset name: {dataset.name}")
print(f"Dataset path: {dataset.path}")
print(f"Exists: {dataset.exists()}")

# Partition information
print(f"Partition columns: {dataset.partition_names}")
print(f"Partition values: {dataset.partition_values}")
```

### Schema Information

```python
# Get schema
schema = dataset.schema
print(f"Schema: {schema}")

# Check if column exists
if 'value' in schema.names:
    print("Value column exists")

# Get column type
value_type = schema.field('value').type
print(f"Value type: {value_type}")
```

### Row Count and Statistics

```python
# Count rows (efficient - uses metadata)
row_count = dataset.count_rows()
print(f"Total rows: {row_count}")

# Get partition information
if dataset.partitions is not None:
    print(f"Partitions: {dataset.partitions}")
```

## Working with Partitions

### Hive Partitioning

```python
# Dataset with hive partitioning
hive_dataset = ParquetDataset(
    "data/hive_data",
    partitioning="hive"  # Auto-detects partition columns from paths
)

# Partition columns are automatically extracted
print(f"Auto-detected partitions: {hive_dataset.partition_names}")

# Access partition values
print(f"Available values: {hive_dataset.partition_values}")
```

### Manual Partitioning

```python
# Specify partition columns manually
manual_dataset = ParquetDataset(
    "data/manual",
    partitioning=["year", "month", "day"]
)
```

## Metadata Management

### Understanding PyDala2 Metadata

PyDala2 automatically maintains two metadata files:

1. **_metadata**: Combined Parquet metadata for all files
2. **_file_metadata**: Per-file statistics (JSON, brotli compressed)

### Working with Metadata

```python
# Metadata is automatically managed
# After writing, files are created:
# - data/_metadata
# - data/_file_metadata

# Update metadata if files changed manually
dataset.update_metadata()

# Check metadata version
print(f"Metadata version: {dataset.metadata_version}")
```

### Custom Metadata

```python
# Add custom metadata to dataset
# This is stored in the _file_metadata
dataset.write_to_dataset(
    data,
    custom_metadata={
        'description': 'Sales data for 2023',
        'owner': 'sales_team',
        'update_frequency': 'daily'
    }
)
```

## Common Patterns

### Time Series Data

```python
# Time-based dataset
time_dataset = ParquetDataset(
    "metrics/daily",
    timestamp_column="timestamp",
    partitioning=["year", "month", "day"]
)

# Write time series data
time_dataset.write_to_dataset(
    time_data,
    partition_by=["year", "month", "day"],
    sort_by="timestamp"
)

# Time-based compaction
time_dataset.compact_by_timeperiod(
    interval="1 month",
    timestamp_column="timestamp"
)
```

### Schema Evolution

```python
# Add new column to existing dataset
new_data_with_col = data.with_columns(
    pl.col("value").alias("double_value") * 2
)

# Allow schema changes
dataset.write_to_dataset(
    new_data_with_col,
    mode="append",
    alter_schema=True
)
```

### Data Type Optimization

```python
# Optimize data types to reduce storage
dataset.optimize_dtypes(
    exclude=["id"],  # Don't optimize ID columns
    ts_unit="ms"    # Convert timestamps to milliseconds
)

# This converts:
# - int64 to int32/int16 where possible
# - float64 to float32 where precision allows
# - string to categorical for low cardinality
```

### Working with Large Datasets

```python
# Process in batches using DuckDB
batch_size = 100000
offset = 0

while True:
    batch = dataset.ddb_con.sql(f"""
        SELECT * FROM dataset
        ORDER BY id
        LIMIT {batch_size} OFFSET {offset}
    """).pl()

    if len(batch) == 0:
        break

    process_batch(batch)
    offset += batch_size
```

## Error Handling

```python
# PyDala2 handles many errors gracefully
try:
    # This will work even if directory doesn't exist
    dataset = ParquetDataset("new/dataset")
    dataset.write_to_dataset(data)
except Exception as e:
    print(f"Error: {e}")

# Check dataset before operations
if not dataset.exists():
    print("Dataset is empty - first write will create it")
```

## Best Practices

1. **Use appropriate partitioning**:
   - Partition on columns frequently used in filters
   - Keep cardinality moderate (100-1000 values per partition)
   - Avoid over-partitioning (too many small files)

2. **Optimize file sizes**:
   - Target 100MB-1GB per file
   - Use `max_rows_per_file` to control size
   - Compact small files periodically

3. **Leverage automatic features**:
   - Let PyDala2 choose between PyArrow/DuckDB
   - Use `filter()` for automatic partition pruning
   - Enable caching for remote filesystems

4. **Monitor and maintain**:
   - Update metadata after manual file operations
   - Use `optimize_dtypes()` to reduce storage
   - Compact partitions when files become too small

5. **Use appropriate write modes**:
   - `append` for adding new data
   - `delta` for merging with change detection
   - `overwrite` for complete replacement
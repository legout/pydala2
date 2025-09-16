# Metadata Management

This guide explains PyDala2's sophisticated metadata system that enables efficient dataset management and query optimization through automatic metadata collection and intelligent file-based operations.

## Overview

PyDala2 implements a two-tier metadata system:

1. **ParquetDatasetMetadata**: Base class managing metadata collection and storage
2. **PydalaDatasetMetadata**: Extended class with DuckDB integration and advanced scanning capabilities

The system maintains two key files:
- **`_metadata`**: Consolidated metadata used by PyArrow for instant dataset loading
- **`_file_metadata`**: Detailed individual file metadata with brotli compression

Key benefits:
- **Instant dataset loading** via PyArrow's `parquet_dataset()` function
- **Metadata-based filtering** to skip irrelevant files
- **Automatic schema evolution** and repair
- **SQL access** to metadata statistics through DuckDB

## The Two-Tier Metadata System

### The _metadata File

The `_metadata` file is a special Parquet file that consolidates metadata from all files:
- Unified schema information across all files
- Row group metadata from all files
- Column statistics (min, max, null counts)
- File paths and sizes
- Compression information

This file is used directly by PyArrow's `parquet_dataset()` function to create datasets instantly without directory scanning.

### The _file_metadata File

The `_file_metadata` file stores detailed individual file metadata:
- Complete `pyarrow.parquet.FileMetaData` objects for each file
- Serialized with brotli compression for efficient storage
- Used for advanced operations and schema repair

### Key Benefits

1. **Instant Dataset Loading**: PyArrow loads datasets using `_metadata` without scanning
2. **File Pruning**: Skip files that don't contain relevant data based on statistics
3. **Query Optimization**: Statistics enable predicate pushdown to individual files
4. **Schema Evolution**: Automatic detection and repair of schema differences
5. **SQL Integration**: DuckDB provides SQL access to metadata statistics

## Working with Metadata

### Automatic Metadata Management

PyDala2 automatically manages metadata files when you work with datasets:

```python
from pydala import ParquetDataset
import pandas as pd

# Create dataset - metadata is automatically managed
dataset = ParquetDataset("data/sales")

# Write data - metadata files are automatically created/updated
dataset.write_to_dataset(
    data=df,
    partition_cols=['year', 'month']
)

# Both _metadata and _file_metadata are created automatically
```

### Manual Metadata Updates

```python
# Update metadata when needed
dataset.update(verbose=True)

# Update only consolidated metadata
dataset.update(update_file_metadata=False)

# Update only file metadata
dataset.update(update_metadata=False)
```

### Accessing Metadata Information

```python
# Access dataset properties (uses metadata)
print(f"Total rows: {dataset.num_rows:,}")
print(f"Number of files: {dataset.num_files}")
print(f"Dataset size: {dataset.size_bytes / 1024 / 1024:.1f} MB")
print(f"Schema: {dataset.schema}")

# Access metadata properties
print(f"Has _metadata: {dataset.has_metadata_file}")
print(f"Has _file_metadata: {dataset.has_file_metadata_file}")

# Access metadata table
metadata_table = dataset.metadata
```

## Metadata-Based Operations

### File Scanning

The `scan()` method is a powerful feature that filters files based on metadata statistics without reading their contents:

```python
# Scan files based on date range
matching_files = dataset.scan(
    filters="date >= '2023-01-01' AND date <= '2023-12-31'",
    verbose=True
)

# Use dictionary filters for complex conditions
matching_files = dataset.scan(
    filters={
        'amount': {'min': 100, 'max': 1000},
        'category': ['A', 'B', 'C']
    }
)

# Get results as a table with statistics
results = dataset.scan(
    filters="status = 'completed'",
    return_table=True
)
```

### SQL Queries on Metadata

PyDala2 integrates with DuckDB to enable SQL queries on metadata:

```python
# Query metadata statistics
stats = dataset.query_metadata("""
    SELECT
        column_name,
        min_value,
        max_value,
        null_count,
        SUM(num_rows) as total_rows
    FROM metadata_table
    WHERE column_name IN ('amount', 'date')
    GROUP BY column_name
""").to_arrow()

# Get file information
files_info = dataset.ddb_con.sql("""
    SELECT file_path, num_rows, size_bytes
    FROM dataset_metadata
    WHERE date >= '2023-01-01'
    ORDER BY num_rows DESC
""").to_arrow()
```

### Query Optimization

PyDala2 automatically uses metadata for query optimization:

```python
# Filters automatically use metadata statistics
filtered = dataset.filter("date > '2023-01-01' AND amount > 1000")

# The query engine:
# 1. Uses metadata to skip irrelevant files
# 2. Applies predicate pushdown to remaining files
# 3. Only reads necessary columns

# Column selection also optimized
result = dataset.table.to_polars(columns=['id', 'date', 'amount'])
# Only these columns are read from disk
```

## Advanced Metadata Usage

### Working with Scan Results

```python
# Reset scan to start fresh
dataset.reset_scan()

# Perform complex scans
matching_files = dataset.scan(
    filters=[
        "date >= '2023-01-01'",
        "amount > 1000",
        "category IN ('A', 'B')"
    ],
    columns=['date', 'amount', 'category'],
    verbose=True
)

# Use the matching files list
for file_path in matching_files[:5]:  # Process first 5 files
    print(f"Processing: {file_path}")
```

### Metadata Table Operations

```python
# Update the metadata table explicitly
dataset.update_metadata_table()

# Access the DuckDB metadata table
metadata_relation = dataset.metadata_table

# Perform complex metadata queries
result = metadata_relation.filter("num_rows > 10000").project("file_path, num_rows")
```

### Schema Management

```python
# Access unified schema
schema = dataset.schema

# Check schema consistency
print(f"Schema fields: {len(schema)}")
for field in schema:
    print(f"  {field.name}: {field.type}")

# Schema repair happens automatically during update
dataset.update(repair=True)
```

## PyArrow Integration

PyDala2 leverages PyArrow's native metadata support:

```python
import pyarrow.dataset as pds

# PyDala2 uses _metadata file internally
# When you access the dataset, it loads instantly:
dataset = ParquetDataset("data/sales")

# This is equivalent to:
# ds = pds.parquet_dataset("data/sales/_metadata")

# The _metadata file contains all necessary information
# No directory scanning required
```

## Performance Considerations

### When to Update Metadata

Update metadata using `dataset.update()` when:
- After writing new data to the dataset
- After compaction operations (`compact_partitions`, `compact_by_timeperiod`)
- After manually adding/removing files
- When query performance seems degraded
- After schema changes in source data

### Metadata File Size

The metadata files size depends on:
- Number of files in the dataset
- Complexity of the schema
- Number of row groups across all files

For very large datasets (1000+ files):
- Use partitioning to organize data logically
- Consider periodic compaction to reduce file count
- The `_file_metadata` uses brotli compression for efficiency

### Storage Efficiency

```python
# Metadata storage is optimized:
# - _metadata: PyArrow native format
# - _file_metadata: brotli compressed + base64 encoded

# Check metadata file sizes
import os

metadata_size = os.path.getsize(dataset.metadata_file)
file_metadata_size = os.path.getsize(dataset.file_metadata_file)

print(f"_metadata: {metadata_size / 1024:.1f} KB")
print(f"_file_metadata: {file_metadata_size / 1024:.1f} KB")
```

## Troubleshooting

### Missing Metadata Files

```python
# Check metadata files exist
if not dataset.has_metadata_file:
    print("No _metadata file found")
    # Update will create both files
    dataset.update(verbose=True)

if not dataset.has_file_metadata_file:
    print("No _file_metadata file found")
    dataset.update(update_metadata=False)
```

### Schema Inconsistencies

```python
# Schema repair is automatic during update
try:
    # This will trigger schema repair if needed
    dataset.update(repair=True, verbose=True)
except Exception as e:
    print(f"Schema repair issue: {e}")
```

### Performance Monitoring

```python
# Monitor metadata operations
import time

start = time.time()
dataset.update()
update_time = time.time() - start

print(f"Metadata update took {update_time:.2f} seconds")
print(f"Dataset has {dataset.num_files} files")
```

## Best Practices

1. **Let PyDala2 manage metadata automatically** - It creates and updates metadata files during write operations
2. **Use `dataset.update()` after manual file changes** - Keeps metadata synchronized
3. **Leverage the `scan()` method** for file-based filtering before reading data
4. **Use appropriate partitioning** - Reduces metadata size and improves query performance
5. **Monitor metadata file sizes** - Large datasets benefit from compaction
6. **Use SQL queries on metadata** - DuckDB integration provides powerful analysis capabilities

## Example: Complete Metadata Workflow

```python
from pydala import ParquetDataset
import pandas as pd
import numpy as np
import time

# 1. Create dataset - metadata management is automatic
dataset = ParquetDataset("data/sales")

# 2. Write data - metadata files are created automatically
data = pd.DataFrame({
    'id': range(1000000),
    'date': pd.date_range('2023-01-01', periods=1000000, freq='H'),
    'amount': np.random.randn(1000000) * 100 + 50,
    'category': np.random.choice(['A', 'B', 'C'], 1000000),
    'status': np.random.choice(['active', 'completed', 'pending'], 1000000)
})

# Write with partitioning
dataset.write_to_dataset(
    data=data,
    partition_cols=['category', 'status']
)

# 3. Verify metadata exists and check dataset info
print(f"Dataset info:")
print(f"  Rows: {dataset.num_rows:,}")
print(f"  Files: {dataset.num_files}")
print(f"  Size: {dataset.size_bytes / 1024 / 1024:.1f} MB")
print(f"  Has metadata: {dataset.has_metadata_file}")

# 4. Use metadata-based file scanning
print("\nScanning for files with specific criteria:")
start = time.time()
matching_files = dataset.scan(
    filters={
        'date': {'min': '2023-06-01', 'max': '2023-06-30'},
        'amount': {'min': 100},
        'status': ['completed']
    },
    verbose=True
)
scan_time = time.time() - start

print(f"Found {len(matching_files)} files with matching data in {scan_time:.3f}s")

# 5. Query using metadata optimization
print("\nExecuting optimized query:")
start = time.time()
result = dataset.filter(
    "date >= '2023-06-01' AND status = 'completed' AND amount > 100"
).table.to_polars()
query_time = time.time() - start

print(f"Query completed in {query_time:.2f} seconds")
print(f"Returned {len(result)} rows")

# 6. Query metadata statistics using SQL
print("\nMetadata statistics:")
stats = dataset.query_metadata("""
    SELECT
        status,
        SUM(num_rows) as total_rows,
        AVG(amount) as avg_amount,
        MIN(amount) as min_amount,
        MAX(amount) as max_amount
    FROM metadata_table
    GROUP BY status
    ORDER BY total_rows DESC
""").df()
print(stats)

# 7. After compaction, update metadata
print("\nAfter compaction:")
# dataset.optimize.compact_partitions()  # This would be your actual compaction
dataset.update(verbose=True)  # Update metadata after operations
print(f"Updated dataset has {dataset.num_files} files")
```
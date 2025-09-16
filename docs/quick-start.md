# Quick Start

Get up and running with PyDala2 in minutes with this quick start guide.

## Installation

```bash
pip install pydala2
```

## Basic Operations

### 1. Create and Write Data

```python
import polars as pl
from pydala import ParquetDataset

# Sample data
data = pl.DataFrame({
    'id': range(1, 1001),
    'name': [f'User_{i}' for i in range(1, 1001)],
    'value': [i * 2 for i in range(1, 1001)],
    'category': ['A', 'B', 'C', 'D'] * 250
})

# Create dataset (directory auto-created)
dataset = ParquetDataset("quickstart/data")

# Write data with optimization
dataset.write_to_dataset(
    data,
    partition_by=["category"],
    max_rows_per_file=250000,
    compression="zstd"
)
```

### 2. Read and Filter Data

```python
from pydala import ParquetDataset

dataset = ParquetDataset("quickstart/data")

# Read all data as Polars DataFrame
all_data = dataset.table.pl.collect()

# Filter data - automatic backend selection
filtered = dataset.filter("value > 100 AND category IN ('A', 'C')")
print(filtered.collect())

# Use SQL directly via DuckDB connection
sql_result = dataset.ddb_con.sql("""
    SELECT category, AVG(value) as avg_value
    FROM dataset
    GROUP BY category
""").pl()
```

### 3. Work with Different Data Formats

```python
# PyArrow Table
arrow_table = dataset.table.arrow()

# DuckDB operations
duckdb_result = dataset.ddb_con.sql("""
    SELECT COUNT(*) as count
    FROM dataset
    WHERE value > 500
""").pl()

# Convert to Pandas if needed
pandas_df = filtered.collect().to_pandas()
```

### 4. Catalog Management

```python
from pydala import Catalog

# Create catalog from YAML
catalog = Catalog("catalog.yaml")

# Register datasets
catalog.load_parquet("main_data", "quickstart/data")

# Get dataset from catalog
dataset = catalog.get_table("main_data")

# Dataset is automatically registered in DuckDB
result = dataset.ddb_con.sql("SELECT category, COUNT(*) FROM dataset GROUP BY category")
```

## Advanced Features

### Delta Updates

```python
# Add new data efficiently
new_data = pl.DataFrame({
    'id': range(1001, 1501),
    'name': [f'User_{i}' for i in range(1001, 1501)],
    'value': [i * 2 for i in range(1001, 1501)],
    'category': ['A', 'B', 'C', 'D'] * 125
})

# Delta mode merges with existing data
dataset.write_to_dataset(
    new_data,
    mode="delta",
    partition_by=["category"]
)
```

### Schema Evolution

```python
# Add new column
data_with_new_col = data.with_columns(
    pl.col("value").alias("double_value") * 2
)

# Allow schema changes
dataset.write_to_dataset(
    data_with_new_col,
    mode="append",
    alter_schema=True
)
```

### Optimization Operations

```python
# Compact small files
dataset.compact_partitions(
    max_rows_per_file=500000,
    sort_by="id"
)

# Optimize data types
dataset.optimize_dtypes()

# Update metadata for better performance
dataset.update_metadata()
```

### Time-based Operations

```python
# Time series data
dates = pl.date_range(start=2023-01-01, end=2023-12-31, interval="1d", eager=True)
time_data = pl.DataFrame({
    'timestamp': dates[:1000],
    'value': range(1000),
    'metric': ['cpu', 'memory', 'disk'] * 333 + ['cpu']
})

# Write with time-based partitioning
time_dataset = ParquetDataset("metrics/data")
time_dataset.write_to_dataset(
    time_data,
    partition_by=["metric"],
    timestamp_column="timestamp"
)

# Time-based compaction
time_dataset.compact_by_timeperiod(
    interval="1 month",
    timestamp_column="timestamp"
)
```

## Complete Example

Here's a complete workflow example:

```python
import polars as pl
from datetime import datetime
from pydala import ParquetDataset, Catalog

# 1. Generate sample data
dates = pl.date_range('2023-01-01', periods=100, interval='1d', eager=True)
data = pl.DataFrame({
    'date': dates,
    'product_id': range(100),
    'sales': [100 + i * 10 for i in range(100)],
    'region': ['North', 'South', 'East', 'West'] * 25
})

# 2. Create and write dataset
dataset = ParquetDataset("sales_data")
dataset.write_to_dataset(
    data,
    partition_by=["region"],
    max_rows_per_file=50,
    compression="zstd"
)

# 3. Query with filtering
result = dataset.filter("""
    date >= '2023-02-01'
    AND region IN ('North', 'South')
""")

# 4. Aggregation via DuckDB
agg_result = dataset.ddb_con.sql("""
    SELECT
        region,
        DATE_TRUNC('month', date) as month,
        SUM(sales) as total_sales,
        COUNT(*) as order_count
    FROM dataset
    WHERE date >= '2023-02-01'
    GROUP BY region, month
    ORDER BY total_sales DESC
""").pl()

print(agg_result)

# 5. Optimize the dataset
dataset.compact_partitions(
    max_rows_per_file=100,
    sort_by="date DESC"
)
```

## Working with Cloud Storage

```python
# S3 dataset
s3_dataset = ParquetDataset(
    "s3://my-bucket/sales-data",
    bucket="my-bucket",
    key="your-access-key",
    secret="your-secret-key",
    cached=True,  # Enable local caching
    cache_storage="/tmp/s3-cache"
)

# Write to S3
s3_dataset.write_to_dataset(data, partition_by=["region"])
```

## Next Steps

- 📚 [User Guide](user-guide/basic-usage.md) - Learn advanced operations
- ⚡ [Performance Guide](user-guide/performance.md) - Optimize your workflows
- 🔧 [API Reference](api/core.md) - Detailed API documentation
- 🏗️ [Deployment Guide](advanced/deployment.md) - Production best practices

## Common Patterns

### Partitioning Strategies

```python
# Date-based partitioning
dataset.write_to_dataset(
    data,
    partition_by=["year", "month", "day"]
)

# Mixed partitioning
dataset.write_to_dataset(
    data,
    partition_by=["region", "category"]
)

# Hive-style partitioning
dataset = ParquetDataset(
    "data/hive",
    partitioning="hive"  # Automatically detects from paths
)
```

### Large Dataset Processing

```python
# Process in batches using DuckDB
batch_size = 100000
total_rows = dataset.ddb_con.sql("SELECT COUNT(*) FROM dataset").fetchone()[0]

for offset in range(0, total_rows, batch_size):
    batch = dataset.ddb_con.sql(f"""
        SELECT * FROM dataset
        LIMIT {batch_size} OFFSET {offset}
    """).pl()
    process_batch(batch)
```
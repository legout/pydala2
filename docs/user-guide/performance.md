# Performance Optimization

This guide covers techniques and best practices for optimizing PyDala2 performance based on the actual features available in the library.

## Understanding Performance Factors

PyDala2's performance depends on several factors:

1. **Data Layout**: How your data is partitioned and organized
2. **File Format**: Parquet compression and encoding settings
3. **Caching**: Built-in filesystem caching for remote storage
4. **Query Patterns**: How you filter and access data
5. **Automatic Backend Selection**: Letting PyDala2 choose the optimal engine

## Partitioning Strategies

### Effective Partitioning with write_to_dataset

```python
from pydala import ParquetDataset
import polars as pl

dataset = ParquetDataset("data/sales")

# Good partitioning - low cardinality, frequently filtered
dataset.write_to_dataset(
    data,
    partition_by=["year", "month", "region"],
    max_rows_per_file=1_000_000
)

# Avoid high-cardinality partitioning
# Bad: partition_by=["user_id", "timestamp"]
```

### Time-based Partitioning

```python
# Time-based partitioning (common pattern)
dataset.write_to_dataset(
    time_series_data,
    partition_by=["year", "month", "day"],
    timestamp_column="created_at",
    max_rows_per_file=500_000
)

# Hierarchical partitioning
dataset.write_to_dataset(
    data,
    partition_by=["region", "category"],
    sort_by="date DESC"
)
```

### Analyzing Partitions

```python
# Check partition information
print(f"Partition columns: {dataset.partition_names}")
print(f"Partition values: {dataset.partition_values}")

# Access partitions data
if dataset.partitions is not None:
    print(f"Number of partitions: {len(dataset.partitions)}")
```

## Compaction Operations

### Compacting Small Files

```python
# Compact partitions with multiple small files
dataset.compact_partitions(
    max_rows_per_file=1_000_000,
    sort_by="date DESC",
    compression="zstd",
    row_group_size=100_000,
    unique=True  # Remove duplicates
)

# Compact by time period (useful for time series)
dataset.compact_by_timeperiod(
    interval="1 month",  # Can be string or timedelta
    timestamp_column="created_at",
    max_rows_per_file=500_000,
    sort_by="created_at"
)

# Compact based on row count
dataset.compact_by_rows(
    max_rows_per_file=2_000_000,
    sort_by="id",
    unique=False
)
```

### Repartitioning Data

```python
# Change partitioning scheme
dataset.repartition(
    partitioning_columns=["year", "quarter", "region"],
    max_rows_per_file=1_000_000,
    sort_by=["region", "date"],
    compression="zstd",
    unique=True
)
```

## Data Type Optimization

### Automatic Type Optimization

```python
# Optimize data types to reduce storage size
dataset.optimize_dtypes(
    exclude=["id"],  # Don't optimize ID columns
    strict=True,    # Use strict type inference
    include=None,   # Or specify columns to include
    ts_unit="ms",   # Convert timestamps to milliseconds
    tz="UTC"        # Set timezone for timestamp columns
)

# This automatically:
# - Converts int64 to int32/int16 where possible
# - Converts float64 to float32 where precision allows
# - Converts strings to categorical for low cardinality
# - Optimizes timestamp precision
```

### Manual Type Optimization

```python
# Get current schema
print(f"Current schema: {dataset.schema}")

# Write with optimized types
data = data.with_columns([
    pl.col("price").cast(pl.Float32),
    pl.col("quantity").cast(pl.Int32),
    pl.col("category").cast(pl.Categorical),
    pl.col("timestamp").dt.convert_time_zone("UTC")
])

dataset.write_to_dataset(
    data,
    partition_by=["category"],
    max_rows_per_file=1_000_000,
    compression="zstd"
)
```

## Write Performance Optimization

### Optimal Write Settings

```python
# Write with performance settings
dataset.write_to_dataset(
    data,
    partition_by=["region", "category"],
    max_rows_per_file=1_000_000,    # Target 1M rows per file
    row_group_size=250_000,        # 250K rows per row group
    compression="zstd",            # Best compression ratio
    sort_by="date DESC",           # Pre-sort for query performance
    unique=True                    # Remove duplicates
)
```

### Compression Comparison

```python
import time
import os

# Test different compression settings
compressions = ["snappy", "gzip", "zstd", "brotli"]

for comp in compressions:
    test_dataset = ParquetDataset(f"data/test_{comp}")

    start = time.time()
    test_dataset.write_to_dataset(
        data,
        compression=comp,
        max_rows_per_file=100_000
    )
    write_time = time.time() - start

    # Read performance
    start = time.time()
    df = test_dataset.table.pl.collect()
    read_time = time.time() - start

    # Calculate size
    size = sum(
        os.path.getsize(os.path.join(test_dataset.path, f))
        for f in test_dataset.files
    )

    print(f"{comp:10} | Size: {size/1024/1024:6.1f} MB | "
          f"Write: {write_time:4.2f}s | Read: {read_time:4.2f}s")

    # Clean up
    test_dataset.vacuum()
```

## Caching for Remote Storage

### Enabling Caching

```python
# Enable caching for remote filesystems
s3_dataset = ParquetDataset(
    "s3://my-bucket/data",
    bucket="my-bucket",
    key="your-access-key",
    secret="your-secret-key",
    cached=True,  # Enable caching
    cache_storage="/tmp/pydala_cache"  # Local cache directory
)

# Cache is automatically used for reads
data = s3_dataset.table.pl.collect()  # First read - from S3
data = s3_dataset.table.pl.collect()  # Subsequent reads - from cache
```

### Cache Management

```python
# Clear cache (useful when data changes)
dataset.clear_cache()

# Cache is automatically managed:
# - Files are cached on first read
# - Cache respects file modifications
# - Cache size depends on available disk space
```

## Query Optimization

### Automatic Backend Selection

PyDala2 automatically chooses the best backend for your queries:

```python
# Simple filters use PyArrow (fast scanning)
simple_result = dataset.filter("region = 'US'")

# Complex filters automatically use DuckDB
complex_result = dataset.filter("""
    region IN ('US', 'EU')
    AND amount > 1000
    AND customer_id IN (SELECT id FROM premium_customers)
""")

# Both return PydalaTable objects
df = simple_result.collect()  # Polars DataFrame
```

### Filter Pushdown

```python
# Filters are automatically pushed down when possible
# Only reads relevant partitions and rows
result = dataset.filter("""
    year = 2023
    AND month IN (1, 2, 3)
    AND amount > 100
""")

# For partitioned datasets, this automatically prunes partitions
```

### Column Pruning

```python
# Only select needed columns
result = dataset.ddb_con.sql("""
    SELECT id, name, email, amount
    FROM dataset
    WHERE date >= '2023-01-01'
""").pl()  # Convert to Polars
```

### Using DuckDB for Complex Queries

```python
# Complex aggregations
agg_result = dataset.ddb_con.sql("""
    SELECT
        region,
        category,
        COUNT(*) as order_count,
        SUM(amount) as total_revenue,
        AVG(amount) as avg_order,
        MIN(amount) as min_order,
        MAX(amount) as max_order
    FROM dataset
    WHERE date >= '2023-01-01'
    GROUP BY region, category
    HAVING COUNT(*) > 100
    ORDER BY total_revenue DESC
""").pl()

# Window functions
window_result = dataset.ddb_con.sql("""
    SELECT
        order_id,
        customer_id,
        order_date,
        amount,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id
            ORDER BY order_date
        ) as order_number,
        LAG(amount, 1) OVER (
            PARTITION BY customer_id
            ORDER BY order_date
        ) as prev_amount
    FROM dataset
""").pl()
```

## Performance Monitoring

### Using EXPLAIN

```python
# Get query execution plan
plan = dataset.ddb_con.sql("""
    EXPLAIN SELECT category, AVG(amount)
    FROM dataset
    WHERE date >= '2023-01-01'
    GROUP BY category
""").fetchall()

print("Execution plan:")
for row in plan:
    print(row[0])
```

### Dataset Statistics

```python
# Basic statistics
print(f"Total rows: {dataset.count_rows()}")
print(f"Number of files: {len(dataset.files)}")
print(f"Schema: {dataset.schema}")

# For partitioned datasets
if dataset.partitions is not None:
    print(f"Partitions: {dataset.partitions}")
```

### Memory Management

```python
# Process large datasets in chunks using DuckDB
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

# Clear memory
del batch
```

## Common Performance Patterns

### Time Series Optimization

```python
# Optimal time series layout
time_dataset = ParquetDataset("data/metrics")
time_dataset.write_to_dataset(
    metrics_data,
    partition_by=["metric_name", "year", "month", "day"],
    timestamp_column="timestamp",
    max_rows_per_file=86400,  # One day of minute data
    sort_by="timestamp"
)

# Fast time range queries with automatic partition pruning
result = time_dataset.filter("""
    metric_name = 'cpu_usage'
    AND timestamp BETWEEN '2023-01-01' AND '2023-01-31'
""")

# Compact old time series data
time_dataset.compact_by_timeperiod(
    interval="1 month",
    timestamp_column="timestamp",
    max_rows_per_file=5_000_000
)
```

### Large Dataset Processing

```python
# Write in batches for very large datasets
def write_large_dataset(dataset, large_df, batch_size=1_000_000):
    """Write large DataFrame in batches"""
    total_rows = len(large_df)

    for i in range(0, total_rows, batch_size):
        batch = large_df.slice(i, batch_size)

        dataset.write_to_dataset(
            batch,
            mode="append",
            partition_by=["batch_id", "date"],
            update_metadata=(i + batch_size >= total_rows)
        )

        print(f"Written batch {i//batch_size + 1}")

# Usage
write_large_dataset(dataset, very_large_dataframe)
```

### Schema Evolution Performance

```python
# Add new columns efficiently
new_data = existing_data.with_columns([
    (pl.col("amount") * 0.1).alias("tax"),
    (pl.col("amount") * 1.1).alias("total_with_tax")
])

# Use alter_schema for schema evolution
dataset.write_to_dataset(
    new_data,
    mode="append",
    alter_schema=True,
    partition_by=["category"]
)
```

## Performance Checklist

- [ ] Partition on low-to-medium cardinality columns (100-1000 values)
- [ ] Use appropriate compression (zstd recommended for best ratio)
- [ ] Set optimal file sizes (1M rows per file typical)
- [ ] Configure row group size (250K rows per group typical)
- [ ] Enable caching for remote filesystems
- [ ] Use `compact_partitions()` for many small files
- [ ] Run `optimize_dtypes()` to reduce storage
- [ ] Pre-sort data on frequently queried columns
- [ ] Let PyDala2 automatically select backends
- [ ] Use partition pruning in filters
- [ ] Monitor file sizes and compact regularly
- [ ] Use time-based compaction for time series data
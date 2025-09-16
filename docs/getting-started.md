# Getting Started

This guide will help you install PyDala2 and set up your environment for working with Parquet datasets.

## Installation

### Prerequisites

PyDala2 requires Python 3.8 or higher. Ensure you have a recent version of Python installed:

```bash
python --version
```

### Install from PyPI

The easiest way to install PyDala2 is using pip:

```bash
pip install pydala2
```

### Install with Optional Dependencies

PyDala2 has several optional dependencies for enhanced functionality:

```bash
# Install with all optional dependencies
pip install pydala2[all]

# Install with specific dependencies
pip install pydala2[s3]        # For S3 storage support
pip install pydala2[azure]     # For Azure Blob Storage
pip install pydala2[gcs]       # For Google Cloud Storage
```

### Install from Source

For development or to get the latest features:

```bash
git clone https://github.com/yourusername/pydala2.git
cd pydala2
pip install -e .
```

## Basic Setup

### Your First Dataset

Let's create a simple dataset and perform basic operations:

```python
import polars as pl
from pydala import ParquetDataset

# Create some sample data
data = pl.DataFrame({
    'id': range(1, 6),
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'city': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney'],
    'salary': [50000, 60000, 70000, 55000, 65000]
})

# Create a dataset (directory auto-created)
dataset = ParquetDataset("data/employees")

# Write the data with optimization
dataset.write_to_dataset(
    data,
    max_rows_per_file=1000,
    compression="zstd"
)

# Read it back
result = dataset.table.pl.collect()
print(result)
```

### Understanding the PyDala2 Architecture

PyDala2 uses a dual-engine architecture:

1. **PyArrow Dataset**: For efficient columnar operations and file scanning
2. **DuckDB**: For SQL queries and complex aggregations

The library automatically chooses the best engine for your operations:

```python
# Simple scan - uses PyArrow
data = dataset.table.pl.collect()

# Complex filter with LIKE - automatically uses DuckDB
filtered = dataset.filter("name LIKE 'A%' OR city IN ('London', 'Paris')")

# SQL operations - uses DuckDB directly
sql_result = dataset.ddb_con.sql("""
    SELECT city, AVG(salary) as avg_salary
    FROM dataset
    GROUP BY city
    ORDER BY avg_salary DESC
""").pl()
```

## Configuration

### Environment Variables

PyDala2 can be configured using environment variables:

```bash
# S3 credentials (if using S3)
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Cache directory
export PYDALA_CACHE_DIR=/tmp/pydala_cache

# Logging level
export PYDALA_LOG_LEVEL=INFO
```

### Dataset Configuration

Configuration is handled at the dataset level:

```python
# Dataset with caching enabled
dataset = ParquetDataset(
    "data/large_dataset",
    cached=True,
    cache_storage="/tmp/pydala_cache"
)

# S3 dataset
s3_dataset = ParquetDataset(
    "s3://my-bucket/data",
    bucket="my-bucket",
    key="your-access-key",
    secret="your-secret-key"
)

# Partitioned dataset
partitioned_dataset = ParquetDataset(
    "data/partitioned",
    partitioning="hive"  # or ["year", "month"]
)
```

## Catalog Setup

The catalog system helps you manage multiple datasets through a YAML configuration:

```python
from pydala import Catalog

# Create catalog YAML
catalog_config = """
tables:
  production:
    sales:
      path: /data/sales
      format: parquet
      options:
        partitioning: hive
        cached: true
    customers:
      path: /data/customers
      format: parquet
  staging:
    temp_data:
      path: /staging/temp
      format: parquet
"""

# Write catalog file
with open("catalog.yaml", "w") as f:
    f.write(catalog_config)

# Load catalog
catalog = Catalog("catalog.yaml", namespace="production")

# Access datasets
sales = catalog.get_table("sales")
customers = catalog.get_table("customers")

# Both datasets are registered in DuckDB
result = sales.ddb_con.sql("""
    SELECT s.*, c.city
    FROM sales s
    JOIN customers c ON s.customer_id = c.id
    WHERE s.date >= '2023-01-01'
""")
```

## Working with Different Data Formats

PyDala2 accepts multiple data formats for writing:

```python
# Polars DataFrame (recommended)
dataset.write_to_dataset(pl_df)

# PyArrow Table
import pyarrow as pa
dataset.write_to_dataset(pa_table)

# Pandas DataFrame
dataset.write_to_dataset(pd_df)

# DuckDB relation
dataset.write_to_dataset(duckdb_relation)
```

## Metadata Management

PyDala2 automatically manages metadata:

```python
# After writing, metadata files are created:
# - _metadata: Combined Parquet metadata for all files
# - _file_metadata: Per-file statistics (JSON, brotli compressed)

# View dataset information
print(f"Dataset name: {dataset.name}")
print(f"Partition columns: {dataset.partition_names}")
print(f"Row count: {dataset.count_rows()}")

# Update metadata (after manual file changes)
dataset.update_metadata()
```

## Next Steps

Now that you have PyDala2 installed and configured:

1. **Learn basic operations** in the [User Guide](user-guide/basic-usage.md)
2. **Explore advanced features** in the [Performance Guide](user-guide/performance.md)
3. **Check the API Reference](api/core.md) for detailed documentation
4. **See real-world examples** in the [Integration Patterns](advanced/integration.md)

## Troubleshooting

### Common Issues

**ImportError: No module named 'pydala'**
- Use `from pydala import ParquetDataset` (not pydala2)
- Ensure PyDala2 is installed correctly

**Permission denied when writing datasets**
- Check file permissions for the target directory
- PyDala2 auto-creates directories if needed

**DuckDB connection issues**
- DuckDB connection is automatically created
- Use `dataset.ddb_con` to access it directly

### Getting Help

- Check the [Troubleshooting Guide](advanced/troubleshooting.md)
- Browse existing issues on [GitHub](https://github.com/yourusername/pydala2/issues)
- Create a new issue with detailed error information

```python
# Enable debug logging for troubleshooting
import logging
logging.basicConfig(level=logging.DEBUG)

# Your code here
```
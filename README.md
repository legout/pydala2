# PyDala2 

<p align="center">
  <img src="logo.jpeg" width="400" alt="PyDala2">
</p>

[![PyPI version](https://badge.fury.io/py/pydala2.svg)](https://badge.fury.io/py/pydala2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



## Overview 📖
Pydala is a high-performance Python library for managing Parquet datasets with powerful metadata capabilities. Built on Apache Arrow, it provides an efficient, user-friendly interface for handling large-scale data operations.

## ✨ Key Features
 - 📦 Smart Dataset Management: Efficient Parquet handling with metadata optimization
 - 🔄 Robust Caching: Built-in support for faster data access
 - 🔌 Seamless Integration: Works with Polars, PyArrow, and DuckDB
 - 🔍 Advanced Querying: SQL-like filtering with predicate pushdown
 - 🛠️ Schema Management: Automatic validation and tracking

## 🚀 Quick Start
### Installation
```bash
pip install pydala2
```

### 📊 Creating a Dataset
```python
from pydala.dataset import ParquetDataset

dataset = ParquetDataset(
    path="path/to/dataset",
    partitioning="hive",         # Hive-style partitioning
    timestamp_column="timestamp", # For time-based operations
    cached=True                  # Enable performance caching
)
```

### 💾 Writing Data
```python
import polars as pl

# Create sample time-series data
df = pl.DataFrame({
    "timestamp": pl.date_range(0, 1000, "1d"),
    "value": range(1000)
})

# Write with smart partitioning and compression
dataset.write_to_dataset(
    data=df,                    # Can be a polars or pandas DataFrame or an Arrow Table, Dataset, or RecordBatch or a duckdb result 
    mode="overwrite",           # Options: "overwrite", "append", "delta"
    row_group_size=250_000,     # Optimize chunk size
    compression="zstd",         # High-performance compression
    partition_by=["year", "month"], # Auto-partition by time
    unique=True                 # Ensure data uniqueness
)
```

### 📥 Reading & Converting Data
```python
dataset.load(update_metadata=True)

# Flexible data format conversion
pt = dataset.t                  # PyDala Table
df_polars = pt.to_polars()      # Convert to Polars
df_pandas = pt.to_pandas()      # Convert to Pandas
df_arrow = pt.to_arrow()        # Convert to Arrow
rel_ddb = pt.to_ddb()           # Convert DuckDB relation

# and many more... 
```

### 🔍 Smart Querying
```python
# Efficient filtered reads with predicate pushdown
pt_filtered = dataset.filter("timestamp > '2023-01-01'")

# Chaining operations
df_filtered = (
    dataset
    .filter("column_name > 100")
    .pl.with_columns(
        pl.col("column_name").str.slice(0, 5).alias("new_column_name")
        )
    .to_pandas()
    )

# Fast metadata-only scans
pt_scanned = dataset.scan("column_name > 100")

# Access matching files
matching_files = ds.scan_files
```

### 🔄 Metadata Management
```python
# Incremental metadata update
dataset.load(update_metadata=True)   # Update for new files

# Full metadata refresh
dataset.load(reload_metadata=True)   # Reload all metadata

# Repair schema/metadata
dataset.repair_schema()
```

### ⚡ Performance Optimization Tools
```python
# Optimize storage types
dataset.opt_dtypes()              # Automatic type optimization

# Smart file management
dataset.compact_by_rows(max_rows=100_000)  # Combine small files
dataset.repartition(partitioning_columns=["date"])  # Optimize partitions
dataset.compact_by_timeperiod(interval="1d")  # Time-based optimization
dataset.compact_partitions()  # Partition structure optimization
```

## ⚠️ Important Notes
Type optimization involves full dataset rewrite
Choose compaction strategy based on your access patterns
Regular metadata updates ensure optimal query performance

## 📚 Documentation
For advanced usage and complete API documentation, visit our docs.

## 🤝 Contributing
Contributions welcome! See our contribution guidelines.

## 📝 License
[MIT License](LICENSE)

# PyDala2

<p align="center">
  <img src="logo.jpeg" width="400" alt="PyDala2">
</p>

[![PyPI version](https://badge.fury.io/py/pydala2.svg)](https://badge.fury.io/py/pydala2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://pydala2.readthedocs.io)

## Overview 📖

PyDala2 is a high-performance Python library for managing Parquet datasets with advanced metadata capabilities. Built on Apache Arrow, it provides efficient management of Parquet datasets with features including:

- Smart dataset management with metadata optimization
- Multi-format support (Parquet, CSV, JSON)
- Multi-backend integration (Polars, PyArrow, DuckDB, Pandas)
- Advanced querying with predicate pushdown
- Schema management with automatic validation
- Performance optimization with caching and partitioning
- Catalog system for centralized dataset management

## ✨ Key Features

- **🚀 High Performance**: Built on Apache Arrow with optimized memory usage and processing speed
- **📊 Smart Dataset Management**: Efficient Parquet handling with metadata optimization and caching
- **🔄 Multi-backend Support**: Seamlessly switch between Polars, PyArrow, DuckDB, and Pandas
- **🔍 Advanced Querying**: SQL-like filtering with predicate pushdown for maximum efficiency
- **📋 Schema Management**: Automatic validation, evolution, and tracking of data schemas
- **⚡ Performance Optimization**: Built-in caching, compression, and intelligent partitioning
- **🛡️ Type Safety**: Comprehensive validation and error handling throughout the library
- **🏗️ Catalog System**: Centralized dataset management across namespaces

## 🚀 Quick Start

### Installation

```bash
pip install pydala2
```

For remote storage, install `pydala2[s3]` for S3-compatible storage or
`pydala2[sftp]` for SSH/SFTP.

### Basic Usage

```python
from pathlib import Path

import pandas as pd
from pydala import ParquetDataset

dataset = ParquetDataset(str(Path("data/my_dataset").resolve()), partitioning="hive")

# Write data
data = pd.DataFrame({
    'id': range(100),
    'category': ['A', 'B', 'C'] * 33 + ['A'],
    'value': [i * 2 for i in range(100)]
})
dataset.write_to_dataset(data, partition_by=["category"])
dataset.load()

result = dataset.filter("category IN ('A', 'B') AND value > 50")
df_polars = result.to_polars(lazy=False)
df_pandas = result.to_pandas()
duckdb_rel = result.to_duckdb()
```

### Using Different Backends

```python
import polars as pl

# Polars LazyFrame
lazy_df = dataset.table.pl  # or dataset.t.pl
result = (
    lazy_df
    .filter(pl.col("value") > 100)
    .group_by("category")
    .agg(pl.mean("value"))
    .collect()
)

# DuckDB (for SQL queries)
result = dataset.ddb_con.sql("""
    SELECT category, AVG(value) as avg_value
    FROM my_dataset
    GROUP BY category
""").to_arrow_table()

# PyArrow Table (for columnar operations)
table = dataset.table.arrow  # or dataset.t.arrow

# Pandas DataFrame (for compatibility)
df_pandas = dataset.table.df  # or dataset.t.df

# Direct export methods
df_polars = dataset.table.to_polars(lazy=False)
table = dataset.table.to_arrow()
df_pandas = dataset.table.to_pandas()
```

### Catalog Management

Save this as `catalog.yaml` alongside the `data/` directory from the example above:

```yaml
filesystem:
  local:
    protocol: file
    bucket: .
tables:
  default:
    my_dataset:
      filesystem: local
      path: data/my_dataset
      format: parquet
```

```python
from pydala import Catalog

catalog = Catalog("catalog.yaml", namespace="default")
result = catalog.sql("SELECT COUNT(*) FROM my_dataset").fetchone()
```

## 📚 Documentation

Comprehensive documentation is available at [pydala2.readthedocs.io](https://pydala2.readthedocs.io):

### Getting Started

- [Installation Guide](https://pydala2.readthedocs.io/getting-started)
- [Quick Start Tutorial](https://pydala2.readthedocs.io/quick-start)

### User Guide

- [Basic Usage](https://pydala2.readthedocs.io/user-guide/basic-usage)
- [Data Operations](https://pydala2.readthedocs.io/user-guide/data-operations)
- [Performance Optimization](https://pydala2.readthedocs.io/user-guide/performance)
- [Catalog Management](https://pydala2.readthedocs.io/user-guide/catalog-management)
- [Schema Management](https://pydala2.readthedocs.io/user-guide/schema-management)

### API Reference

- [Core Classes](https://pydala2.readthedocs.io/api/core)
- [Dataset Classes](https://pydala2.readthedocs.io/api/datasets)
- [Table Operations](https://pydala2.readthedocs.io/api/table)
- [Metadata Management](https://pydala2.readthedocs.io/api/metadata)
- [Catalog System](https://pydala2.readthedocs.io/api/catalog)
- [Filesystem](https://pydala2.readthedocs.io/api/filesystem)
- [Utilities](https://pydala2.readthedocs.io/api/utilities)

### Advanced Topics

- [Performance Tuning](https://pydala2.readthedocs.io/advanced/performance-tuning)
- [Integration Patterns](https://pydala2.readthedocs.io/advanced/integration)
- [Deployment Guide](https://pydala2.readthedocs.io/advanced/deployment)
- [Troubleshooting](https://pydala2.readthedocs.io/advanced/troubleshooting)

## 🤝 Contributing

Report bugs and discuss changes in [GitHub Issues](https://github.com/legout/pydala2/issues).

## 📝 License

[MIT License](LICENSE)

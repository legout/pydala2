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
# Install PyDala2
pip install pydala2

# Install with all optional dependencies
pip install pydala2[all]

# Install with specific backends
pip install pydala2[polars,duckdb]
```

### Basic Usage

```python
from pydala import ParquetDataset
import pandas as pd

# Create a dataset
dataset = ParquetDataset("data/my_dataset")

# Write data
data = pd.DataFrame({
    'id': range(100),
    'category': ['A', 'B', 'C'] * 33 + ['A'],
    'value': [i * 2 for i in range(100)]
})
dataset.write_to_dataset(
    data=data,
    partition_cols=['category']
)

# Read with filtering - automatic backend selection
result = dataset.filter("category IN ('A', 'B') AND value > 50")

# Export to different formats
df_polars = result.table.to_polars()  # or use shortcut: result.t.pl
df_pandas = result.table.df           # or result.t.df
duckdb_rel = result.table.ddb         # or result.t.ddb
```

### Using Different Backends

```python
# PyDala2 provides automatic backend selection
# Just access data in your preferred format:

# Polars LazyFrame (recommended for performance)
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
    FROM dataset
    GROUP BY category
""").to_arrow()

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

```python
from pydala import Catalog

# Create catalog from YAML configuration
catalog = Catalog("catalog.yaml")

# YAML configuration example:
# tables:
#   sales_2023:
#     path: "/data/sales/2023"
#     filesystem: "local"
#   customers:
#     path: "/data/customers"
#     filesystem: "local"

# Query across datasets using automatic table loading
result = catalog.query("""
    SELECT
        s.*,
        c.customer_name,
        c.segment
    FROM sales_2023 s
    JOIN customers c ON s.customer_id = c.id
    WHERE s.date >= '2023-01-01'
""")

# Or access datasets directly
sales_dataset = catalog.get_dataset("sales_2023")
filtered_sales = sales_dataset.filter("amount > 1000")
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

Contributions are welcome! Please see our [Contributing Guide](https://pydala2.readthedocs.io/contributing) for details.

## 📝 License

[MIT License](LICENSE)


# PyDala2

[![PyPI version](https://badge.fury.io/py/pydala2.svg)](https://badge.fury.io/py/pydala2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://pydala2.readthedocs.io)

**PyDala2** is a high-performance Python library for managing Parquet datasets with advanced metadata capabilities, built on Apache Arrow and DuckDB.

## 🎯 What is PyDala2?

PyDala2 provides a sophisticated dataset management system that combines:
- **Automatic optimization** of Parquet file layouts
- **Intelligent metadata management** with schema evolution
- **Multi-engine query processing** (PyArrow + DuckDB)
- **Production-ready features** like keyed merge operations and time-based operations

## ✨ Key Features

### 🚀 Core Capabilities
- **Smart Parquet Management**: Automatic file optimization, compaction, and layout management
- **Dual Engine Architecture**: Seamlessly switches between PyArrow (fast scans) and DuckDB (complex queries)
- **Advanced Metadata**: Automatic schema unification, statistics tracking, and efficient updates
- **Hive Partitioning**: Native support for partitioned datasets with automatic inference

### 📊 Data Operations
- **Automatic Filter Pushdown**: Intelligently routes queries to the optimal engine
- **Keyed Merge Operations**: Explicit insert, update, and upsert strategies
- **Schema Evolution**: Handle schema changes without breaking existing data
- **Time-based Operations**: Built-in support for time series and temporal data

### 🏗️ Production Features
- **Catalog System**: Centralized dataset management across namespaces
- **Cloud Storage**: Native S3 integration with caching and credential management
- **Optimization Operations**: Automatic data type optimization, file compaction, and sorting
- **Robust Error Handling**: Automatic schema repair and graceful degradation

## 🎯 Use Cases

### Data Engineering
```python
# Keyed ETL merge
result = dataset.merge(
    new_sales_data,
    strategy="upsert",
    key_columns=["sale_id"],
    partition_by=["date", "region"],
    sort_by="timestamp DESC",
)
```

### Analytics
```python
# Complex queries with automatic optimization
result = dataset.filter("""
    date > '2023-01-01'
    AND region IN ('US', 'EU')
    AND amount > 1000
""")
```

### Data Lake Management

Use a [YAML catalog](user-guide/catalog-management.md) to organize datasets and query them with DuckDB.

## 🚀 Quick Example

```python
from pathlib import Path

import polars as pl
from pydala import ParquetDataset

dataset = ParquetDataset(str(Path("data/sales").resolve()), partitioning="hive")

data = pl.DataFrame({"region": ["US", "EU"], "amount": [750, 250]})
dataset.write_to_dataset(data, partition_by=["region"])
dataset.load()

result = dataset.filter("amount > 500 AND region = 'US'")
print(result.to_polars(lazy=False))
```

## 🏗️ Architecture

```mermaid
graph TD
    A[PyDala2 Application] --> B[ParquetDataset]
    B --> C[PydalaTable]
    C --> D[PyArrow Dataset]
    C --> E[DuckDB Relation]
    B --> F[Metadata Management]
    F --> G[_metadata File]
    F --> H[_file_metadata File]
    B --> I[Optimization]
    I --> J[Compaction]
    I --> K[Schema Evolution]
    I --> L[Keyed Merge Operations]
    B --> M[Catalog Integration]
    M --> N[YAML Configuration]
    M --> O[Namespace Management]
```

## 📚 Documentation

Our comprehensive documentation covers:

### Getting Started
- [Installation Guide](getting-started.md) - Setup and configuration
- [Quick Start](quick-start.md) - 5-minute tutorial

### User Guide
- [Basic Usage](user-guide/basic-usage.md) - Core operations and patterns
- [Merge and Delta Migration](user-guide/merge.md) - Insert, update, upsert, and delta migration
- [Data Operations](user-guide/data-operations.md) - Filtering, aggregation, and transformations
- [Performance Optimization](user-guide/performance.md) - Tuning for production workloads
- [Catalog Management](user-guide/catalog-management.md) - Organizing datasets
- [Schema Management](user-guide/schema-management.md) - Schema evolution and validation

### API Reference
- [Core Classes](api/core.md) - datasets and maintenance
- [Dataset Classes](api/datasets.md) - ParquetDataset and variants
- [Table Operations](api/table.md) - PydalaTable interface
- [Catalog System](api/catalog.md) - Catalog and namespace management
- [Filesystem](api/filesystem.md) - Storage backends and cloud integration
- [Utilities](api/utilities.md) - Helper functions and tools

### Advanced Topics
- [Deployment Guide](advanced/deployment.md) - Production deployment
- [Performance Tuning](advanced/performance-tuning.md) - Advanced optimization
- [Troubleshooting](advanced/troubleshooting.md) - Common issues and solutions
- [Integration Patterns](advanced/integration.md) - Using PyDala2 with other tools

## 📝 License

[MIT License](https://github.com/legout/pydala2/blob/main/LICENSE)
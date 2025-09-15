# PyDala2 Examples

This directory contains comprehensive examples demonstrating the capabilities of PyDala2, a powerful data lake management library.

## Getting Started

Each example is self-contained and includes:
- A Python script with detailed examples
- A Jupyter notebook for interactive exploration
- A Marimo notebook for reactive computing
- Comprehensive README with explanations

## Prerequisites

- Python 3.11+
- PyDala2 installed: `pip install pydala2`
- Optional dependencies for specific examples:
  - Jupyter: `pip install jupyter`
  - Marimo: `pip install marimo`
  - Polars: `pip install polars`
  - DuckDB: `pip install duckdb`

## Examples Overview

### 1. [Basic Dataset Operations](./basic-dataset-operations/)
**Level**: Beginner
**Focus**: Fundamental dataset operations and basic functionality

Learn how to:
- Create datasets from various sources
- Perform basic data operations
- Work with metadata
- Use the catalog system

```bash
cd basic-dataset-operations
python basic_operations.py
# or
jupyter notebook basic_operations.ipynb
# or
marimo edit basic_operations.marimo.py
```

### 2. [Data Loading and Conversion](./data-loading-conversion/)
**Level**: Beginner
**Focus**: Loading data from multiple sources and format conversion

Learn how to:
- Load data from CSV, JSON, and Parquet
- Convert between different formats
- Work with partitioned datasets
- Integrate with Pandas, Polars, and DuckDB

### 3. [Metadata Management](./metadata-management/)
**Level**: Intermediate
**Focus**: Advanced metadata handling and schema management

Learn how to:
- Manage dataset metadata
- Handle schema evolution
- Collect and synchronize metadata
- Optimize performance through metadata

### 4. [Advanced Querying](./advanced-querying/)
**Level**: Intermediate
**Focus**: Complex querying patterns and SQL integration

Learn how to:
- Write complex filters and predicates
- Use window functions and aggregations
- Integrate with DuckDB for SQL queries
- Perform multi-dataset operations

### 5. [Performance Optimization](./performance-optimization/)
**Level**: Advanced
**Focus**: Optimizing PyDala2 for large datasets

Learn how to:
- Manage memory efficiently
- Optimize query performance
- Use parallel processing
- Handle large-scale datasets

### 6. [Cloud Storage (S3)](./cloud-storage-s3/)
**Level**: Intermediate
**Focus**: Working with cloud storage

Learn how to:
- Connect to S3 buckets
- Optimize cloud storage operations
- Handle remote access patterns
- Implement error handling strategies

### 7. [Time Series Data](./time-series-data/)
**Level**: Advanced
**Focus**: Time series analysis and operations

Learn how to:
- Handle temporal data efficiently
- Perform time-based aggregations
- Analyze seasonal patterns
- Optimize time series queries

## Learning Path

For new users, we recommend following this sequence:

1. **Start with Basic Dataset Operations** - Learn the fundamentals
2. **Explore Data Loading and Conversion** - Understand data ingestion
3. **Master Metadata Management** - Learn to manage metadata effectively
4. **Dive into Advanced Querying** - Unlock complex analysis capabilities
5. **Optimize Performance** - Scale your operations
6. **Explore Cloud Storage** - Work with remote data
7. **Handle Time Series Data** - Specialized time series operations

## Running Examples

### Python Scripts
Each example can be run as a standalone Python script:

```bash
cd example-directory
python script_name.py
```

### Jupyter Notebooks
For interactive exploration:

```bash
cd example-directory
jupyter notebook notebook_name.ipynb
```

### Marimo Notebooks
For reactive computing:

```bash
cd example-directory
marimo edit notebook_name.marimo.py
```

## Example Data

All examples use synthetic data generated on-the-fly, so no external data files are required. The examples create temporary directories for data storage that are cleaned up automatically.

## Contributing

We welcome contributions to improve these examples! If you have suggestions for new examples or improvements to existing ones, please:

1. Check the [contribution guidelines](../../CONTRIBUTING.md)
2. Open an issue to discuss your proposal
3. Submit a pull request with your changes

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PyDala2 is properly installed
2. **Permission Issues**: Make sure you have write permissions for temporary directories
3. **Memory Issues**: Some examples process large datasets - ensure sufficient RAM
4. **Missing Dependencies**: Install optional dependencies as needed

### Getting Help

- Check the [main documentation](../../README.md)
- Review the [API reference](../../docs/api.md)
- Open an issue on [GitHub](../../issues)

## Best Practices

These examples demonstrate:

- **Proper error handling** with try/catch blocks
- **Resource management** using context managers
- **Type hints** for better code clarity
- **Documentation** with docstrings and comments
- **Performance considerations** for large datasets

## Next Steps

After completing these examples, you'll be well-equipped to:

- Build data pipelines with PyDala2
- Manage large-scale data lakes efficiently
- Perform complex data analysis
- Optimize data operations for performance
- Integrate PyDala2 with your existing data stack

Happy coding with PyDala2! 🚀
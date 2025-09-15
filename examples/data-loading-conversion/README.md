# Data Loading and Conversion

This example demonstrates how to load data from various sources and convert between different formats using PyDala2.

## What You'll Learn

- Loading data from multiple sources (CSV, JSON, Parquet)
- Converting between different data formats
- Working with partitioned datasets
- Handling large datasets efficiently
- Integration with popular data libraries (Pandas, Polars, DuckDB)

## Files

- `data_loading.py` - Python script demonstrating data loading and conversion
- `data_loading.ipynb` - Jupyter notebook version
- `data_loading.marimo.py` - Marimo notebook version

## Running the Examples

### Python Script
```bash
cd data-loading-conversion
python data_loading.py
```

### Jupyter Notebook
```bash
cd data-loading-conversion
jupyter notebook data_loading.ipynb
```

### Marimo Notebook
```bash
cd data-loading-conversion
marimo edit data_loading.marimo.py
```

## Prerequisites

- PyDala2 installed (`pip install pydala2`)
- Optional: Polars for advanced examples (`pip install polars`)

## Key Concepts Demonstrated

1. **Multi-format Support**: Loading from CSV, JSON, Parquet
2. **Data Conversion**: Seamless conversion between formats
3. **Partitioning**: Creating and working with partitioned datasets
4. **Integration**: Working with Pandas, Polars, and DuckDB
5. **Performance**: Efficient handling of large datasets

## Sample Data

The examples use synthetic data that is generated on-the-fly, so no external data files are required.

## Next Steps

After mastering data loading, explore:
- Metadata Management examples
- Advanced Querying techniques
- Performance Optimization strategies
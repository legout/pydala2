# API Reference Overview

This section provides comprehensive API documentation for PyDala2. The API is organized into several main modules:

## Core Components

### [Core Classes](core.md)
- `BaseDataset` - Base class for all dataset operations
- `Optimize` - Dataset optimization and compaction operations
- `Writer` - Data writing and transformation utilities

### [Dataset Classes](datasets.md)
- `ParquetDataset` - Parquet file operations with metadata support
- `PyarrowDataset` - PyArrow dataset integration
- `CSVDataset` - CSV file operations
- `JSONDataset` - JSON file operations

### [Table Operations](table.md)
- `PydalaTable` - Unified table interface with multiple export formats
- Data export methods for different backends
- Scanner and batch reader operations

### [Catalog System](catalog.md)
- `Catalog` - Centralized dataset management
- Namespace operations
- Dataset discovery and registration

### [Filesystem](filesystem.md)
- `FileSystem` - Advanced filesystem operations with caching
- Abstract filesystem implementations
- Storage backends

### [Metadata Management](metadata.md)
- `ParquetDatasetMetadata` - Parquet metadata collection and management
- `PydalaDatasetMetadata` - Extended metadata with statistics
- Schema unification and repair operations

### [Utilities](utilities.md)
- Helper functions and utilities
- SQL operations
- Data type conversions
- Security functions

## Quick API Reference

### Creating a Dataset
```python
from pydala import ParquetDataset

# Create a new dataset
dataset = ParquetDataset("data/my_dataset")
```

### Writing Data
```python
# Write data to dataset
dataset.write_to_dataset(
    dataframe,
    partition_cols=['category'],
    basename_template="data-{i}.parquet"
)
```

### Reading Data
```python
# Export to different formats
df_polars = dataset.table.to_polars(lazy=False)  # Eager Polars DataFrame
df_pandas = dataset.table.df  # Pandas DataFrame (shortcut)
table_arrow = dataset.table.arrow  # PyArrow Table (shortcut)

# Use shortcut notation
df_polars = dataset.t.to_polars()  # Same as dataset.table.to_polars()
```

### Filtering Data
```python
# Filter with automatic backend selection
filtered = dataset.filter("date > '2023-01-01'")

# Export filtered data
result = filtered.to_pandas()
```

### Using the Catalog
```python
from pydala import Catalog

# Create catalog from YAML configuration
catalog = Catalog("catalog.yaml")

# Query across datasets
result = catalog.ddb_con.sql("SELECT * FROM sales_data WHERE value > 100")
```

## Data Export Methods

PyDala2 provides multiple ways to export data:

### To Polars
```python
# Eager DataFrame
df = dataset.table.to_polars(lazy=False)

# LazyFrame (default)
lf = dataset.table.to_polars(lazy=True)
lf = dataset.table.pl  # Shortcut property

# Execute LazyFrame
df = lf.collect()
```

### To Pandas
```python
df = dataset.table.to_pandas()
df = dataset.table.df  # Shortcut property
```

### To PyArrow
```python
table = dataset.table.to_arrow()
table = dataset.table.arrow  # Shortcut property

# As dataset
ds = dataset.table.to_arrow_dataset()
```

### To DuckDB
```python
rel = dataset.table.to_duckdb()
rel = dataset.table.ddb  # Shortcut property

# Execute query
result = rel.query("SELECT category, AVG(value) FROM rel GROUP BY category")
```

## Common Patterns

### Dataset Operations
```python
# Check if dataset exists
if dataset.exists():
    print(f"Dataset has {dataset.count_rows()} rows")

# Get dataset information
print(f"Schema: {dataset.schema}")
print(f"Columns: {dataset.columns}")
print(f"Files: {len(dataset.files)}")
```

### Optimization Operations
```python
# Compact partitions
dataset.optimize.compact_partitions()

# Compact by time period
dataset.optimize.compact_by_timeperiod(
    timestamp_column="date",
    timeperiod="month"
)

# Optimize data types
dataset.optimize.optimize_dtypes()
```

### Scanner Operations
```python
# Create scanner for efficient reading
scanner = dataset.table.scanner()

# Read specific columns
scanner = dataset.table.to_arrow_scanner(columns=['id', 'name'])

# Read in batches
batch_reader = dataset.table.to_batch_reader()
for batch in batch_reader:
    process_batch(batch)
```

## Error Handling

PyDala2 provides comprehensive error handling:

```python
from pydala import PydalaException

try:
    result = dataset.filter("invalid_column > 100").to_polars()
except PydalaException as e:
    print(f"Error: {e}")
```

## Performance Considerations

- Use partitioning for large datasets
- Leverage the `_metadata` file for efficient dataset loading
- Use lazy evaluation with Polars for complex operations
- Apply filters before exporting data to reduce memory usage
- Use appropriate export format for your use case

For detailed information about each component, see the specific API reference pages.
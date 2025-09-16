# Troubleshooting

This guide helps you diagnose and resolve common issues when using PyDala2.

## Common Issues

### Installation Problems

#### ImportError: No module named 'pydala'

**Symptoms:**
```python
>>> import pydala
ModuleNotFoundError: No module named 'pydala'
```

**Solutions:**

1. **Check installation:**
```bash
pip list | grep pydala
```

2. **Install or reinstall:**
```bash
# Install from PyPI
pip install pydala2

# Or if developing from source
pip install -e .

# With all dependencies
pip install pydala2[all]
```

3. **Check Python environment:**
```bash
which python
python --version
```

#### Permission denied errors

**Symptoms:**
```python
PermissionError: [Errno 13] Permission denied: '/data/dataset'
```

**Solutions:**

1. **Check directory permissions:**
```bash
ls -la /data/
```

2. **Change ownership or permissions:**
```bash
# Change ownership
sudo chown -R user:group /data/dataset

# Or change permissions
chmod 755 /data/dataset
```

3. **Use user-writable location:**
```python
dataset = ParquetDataset("~/data/dataset")
```

### Memory Issues

#### Out of memory errors

**Symptoms:**
```python
MemoryError: Unable to allocate X GiB for an array
```

**Solutions:**

1. **Process data in chunks:**
```python
# Use batch reader
reader = dataset.to_batch_reader(batch_size=10000)
for batch in reader:
    process_batch(batch)
```

2. **Set memory limits:**
```python
from pydala import set_config

set_config({
    'max_memory': 4 * 1024 * 1024 * 1024,  # 4GB
    'n_workers': 2
})
```

3. **Use lazy evaluation:**
```python
# Use Polars lazy mode
lazy_df = dataset.read(backend="polars", lazy=True)
result = (
    lazy_df
    .filter(pl.col("amount") > 100)
    .group_by("category")
    .agg(pl.sum("amount"))
    .collect()
)
```

4. **Clear caches:**
```python
dataset.clear_cache()
```

#### Memory leaks

**Symptoms:**
Memory usage grows continuously over time.

**Solutions:**

1. **Monitor memory usage:**
```python
import psutil
import gc

def check_memory():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"RSS: {mem_info.rss / 1024 / 1024:.1f} MB")
    return mem_info.rss
```

2. **Explicit cleanup:**
```python
# Delete large objects
del large_dataframe
gc.collect()
```

3. **Use context managers:**
```python
with dataset.read(backend="polars") as df:
    # df is automatically cleaned up
    process_data(df)
```

### Performance Issues

#### Slow reads/writes

**Symptoms:**
Operations taking longer than expected.

**Diagnostics:**

1. **Check filesystem performance:**
```python
import time

start = time.time()
data = dataset.read()
read_time = time.time() - start
print(f"Read time: {read_time:.2f} seconds")
```

2. **Profile the operation:**
```python
import cProfile

def profile_operation():
    dataset.read(filters="date > '2023-01-01'")

profiler = cProfile.Profile()
profiler.enable()
profile_operation()
profiler.disable()
profiler.print_stats(sort='cumulative')
```

**Solutions:**

1. **Enable caching:**
```python
dataset.enable_cache(max_size=1024*1024*1024)  # 1GB
```

2. **Optimize file layout:**
```python
# Compact small files
dataset.compact_small_files(min_file_size=10*1024*1024)

# Repartition for better pruning
dataset.repartition(partition_cols=['year', 'month', 'day'])
```

3. **Use appropriate compression:**
```python
dataset.write(
    data,
    compression='zstd',
    compression_level=3,
    row_group_size=1000000
)
```

### Schema Issues

#### Schema mismatch errors

**Symptoms:**
```python
pyarrow.lib.ArrowInvalid: Schema at index 0 was different
```

**Solutions:**

1. **Check schema evolution:**
```python
# Compare schemas
old_schema = old_dataset.schema
new_schema = new_dataset.schema

for field in old_schema:
    if field.name not in new_schema.names:
        print(f"Missing column: {field.name}")
    elif new_schema.field(field.name).type != field.type:
        print(f"Type mismatch for {field.name}")
```

2. **Use schema validation:**
```python
from pydala import validate_schema

result = validate_schema(data, expected_schema)
if not result.valid:
    print("Validation errors:")
    for error in result.errors:
        print(f"  - {error}")
```

3. **Handle schema evolution:**
```python
def safe_write(dataset, data):
    """Write data with schema handling."""
    if dataset.exists():
        # Merge schemas
        existing_schema = dataset.schema
        new_schema = pa.unify_schemas([existing_schema, data.schema])

        # Add missing columns with nulls
        for field in new_schema:
            if field.name not in data.column_names:
                data = data.append_column(
                    field.name,
                    pa.nulls(len(data)).cast(field.type)
                )

    dataset.write(data)
```

### Cloud Storage Issues

#### S3 connection errors

**Symptoms:**
```python
NoCredentialsError: Unable to locate credentials
```

**Solutions:**

1. **Check credentials:**
```bash
# AWS CLI
aws configure list

# Environment variables
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY
```

2. **Configure credentials properly:**
```python
# Using environment variables
import os
os.environ['AWS_ACCESS_KEY_ID'] = 'your_key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret'

# Or using filesystem parameters
s3_fs = FileSystem(
    protocol="s3",
    key="your_key",
    secret="your_secret"
)
```

#### Slow cloud operations

**Solutions:**

1. **Enable caching:**
```python
# Local cache for cloud data
fs = FileSystem(
    protocol="s3",
    cached=True,
    cache_storage="/tmp/s3_cache",
    cache_options={
        'expiry_time': 3600  # 1 hour
    }
)
```

2. **Use multipart uploads:**
```python
dataset.write(
    data,
    filesystem_kwargs={
        'max_upload_parts': 10000,
        'multipart_threshold': 64 * 1024 * 1024  # 64MB
    }
)
```

3. **Optimize region selection:**
```python
# Use same region as data
fs = FileSystem(
    protocol="s3",
    client_kwargs={
        "region_name": "us-east-1"
    }
)
```

### Catalog Issues

#### Catalog not found errors

**Symptoms:**
```python
FileNotFoundError: catalog.yaml not found
```

**Solutions:**

1. **Check catalog path:**
```python
# Use absolute path
catalog = Catalog("/absolute/path/to/catalog.yaml")

# Or check current directory
import os
print(f"Current directory: {os.getcwd()}")
```

2. **Create catalog if missing:**
```python
def ensure_catalog(path):
    """Create catalog if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, 'w') as f:
            yaml.safe_dump({
                'tables': {},
                'filesystems': {}
            }, f)
    return Catalog(path)
```

#### Dataset not registered

**Symptoms:**
```python
ValueError: Dataset 'my_dataset' not found in catalog
```

**Solutions:**

1. **List available datasets:**
```python
catalog = Catalog("catalog.yaml")
print("Available datasets:", catalog.all_tables())
```

2. **Register dataset:**
```python
# Register existing dataset
catalog.load_parquet(
    "my_dataset",
    "/path/to/dataset",
    partitioning=["year", "month"]
)
```

### Backend-Specific Issues

#### Polars issues

**Symptoms:**
```python
PanicException: ...
```

**Solutions:**

1. **Update Polars:**
```bash
pip install --upgrade polars
```

2. **Check data types:**
```python
# Some operations don't support certain types
print(df.dtypes)

# Convert if necessary
df = df.with_columns([
    pl.col('string_col').cast(pl.Utf8),
    pl.col('numeric_col').cast(pl.Float64)
])
```

3. **Use eager mode for debugging:**
```python
# Switch from lazy to eager
df = dataset.read(backend="polars", lazy=False)
```

#### DuckDB issues

**Symptoms:**
```python
duckdb.OperationalError: Parser Error
```

**Solutions:**

1. **Validate SQL syntax:**
```python
# Test SQL in DuckDB directly
import duckdb
con = duckdb.connect()
con.execute("SELECT * FROM test LIMIT 1")
```

2. **Escape identifiers:**
```python
from pydala import escape_sql_identifier

safe_column = escape_sql_identifier(column_name)
sql = f"SELECT {safe_column} FROM table"
```

3. **Check data types:**
```python
# DuckDB has stricter type rules
# Convert to compatible types
import pyarrow as pa
table = table.cast(pa.schema([
    ('date_col', pa.timestamp('ns')),
    ('amount_col', pa.float64())
]))
```

### Debug Mode

Enable debug logging for detailed information:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific modules
logging.getLogger('pydala').setLevel(logging.DEBUG)
logging.getLogger('fsspec').setLevel(logging.DEBUG)

# Your code here
```

### Common Error Patterns

#### Pattern 1: Invalid filter expressions

```python
# Wrong
dataset.read("column > 'value' AND column2 < 100")

# Right
dataset.read(filters="column > 'value' AND column2 < 100")

# Or use Arrow expressions
import pyarrow.dataset as pds
filter_expr = (
    (pds.field('column') > 'value') &
    (pds.field('column2') < 100)
)
dataset.read(filter=filter_expr)
```

#### Pattern 2: Mixed data types

```python
# Check for mixed types before writing
def check_data_types(df):
    """Check for inconsistent data types."""
    for col in df.columns:
        unique_types = df[col].apply(type).nunique()
        if unique_types > 1:
            print(f"Warning: Column {col} has mixed types")
            print(df[col].apply(type).value_counts())
```

#### Pattern 3: Large file operations

```python
# Process large files incrementally
def process_large_dataset(dataset, chunk_size=100000):
    """Process large dataset in chunks."""
    total_rows = 0
    for i, chunk in enumerate(dataset.stream_read(chunk_size=chunk_size)):
        print(f"Processing chunk {i+1}")
        process_chunk(chunk)
        total_rows += len(chunk)
        del chunk  # Explicit cleanup
    return total_rows
```

## Getting Help

### Before Asking

1. **Check the logs:**
```bash
# Enable verbose logging
export PYDALA2_LOG_LEVEL=DEBUG

# Or in code
import logging
logging.getLogger('pydala').setLevel(logging.DEBUG)
```

2. **Create minimal reproducible example:**
```python
# Isolate the issue
import pandas as pd
from pydala import ParquetDataset

# Create minimal test case
test_data = pd.DataFrame({
    'id': range(100),
    'value': [i * 2 for i in range(100)]
})

# Reproduce issue
dataset = ParquetDataset("test_dataset")
dataset.write(test_data)
result = dataset.read(filters="value > 100")  # What fails?
```

3. **Gather system information:**
```python
import sys
import platform
import pydala

print("Python version:", sys.version)
print("Platform:", platform.platform())
print("PyDala2 version:", pydala.__version__)
print("PyArrow version:", pa.__version__)
print("Polars version:", pl.__version__)
```

### Where to Get Help

1. **Documentation**: Check the latest documentation
2. **GitHub Issues**: Search existing issues or create a new one
3. **Stack Overflow**: Use the `pydala2` tag
4. **Discord/Slack**: Join community channels

### Reporting Issues

When reporting issues, include:

1. **Minimal reproducible example**
2. **Expected vs actual behavior**
3. **Full error traceback**
4. **System information**
5. **Data sample (if possible)**

Example issue template:
```markdown
## Issue Description
Brief description of the issue

## Minimal Example
```python
# Code that reproduces the issue
```

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Error Traceback
```
Full error message
```

## System Information
- PyDala2 version: X.Y.Z
- Python version: X.Y.Z
- OS: Linux/Windows/MacOS
```
# Utilities

This document describes utility functions and helper classes available in PyDala2.

## Configuration Utilities

### set_config
```python
def set_config(config: dict) -> None
```
Set global configuration values.

**Parameters:**
- `config` (dict): Configuration dictionary

**Example:**
```python
from pydala import set_config

set_config({
    'default_backend': 'polars',
    'cache_enabled': True,
    'log_level': 'INFO'
})
```

### get_config
```python
def get_config(key: str, default=None)
```
Get a configuration value.

**Parameters:**
- `key` (str): Configuration key
- `default`: Default value if key not found

**Returns:**
- Configuration value or default

**Example:**
```python
from pydala import get_config

backend = get_config('default_backend', 'polars')
```

### config_context
```python
@contextmanager
def config_context(config: dict)
```
Context manager for temporary configuration changes.

**Parameters:**
- `config` (dict): Configuration to apply within context

**Example:**
```python
from pydala import config_context

with config_context({'default_backend': 'duckdb'}):
    # Operations here use DuckDB
    result = dataset.read()
# Back to previous configuration
```

## Schema Utilities

### infer_schema
```python
def infer_schema(data) -> pa.Schema
```
Infer Arrow schema from data.

**Parameters:**
- `data`: Data to infer schema from (DataFrame, list of dicts, etc.)

**Returns:**
- `pa.Schema`: Inferred schema

**Example:**
```python
from pydala import infer_schema

# From pandas DataFrame
schema = infer_schema(df)

# From list of dicts
data = [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]
schema = infer_schema(data)
```

### convert_schema
```python
def convert_schema(schema: pa.Schema, conversions: dict) -> pa.Schema
```
Convert schema data types.

**Parameters:**
- `schema` (pa.Schema): Original schema
- `conversions` (dict): Column to type mapping

**Returns:**
- `pa.Schema`: Converted schema

**Example:**
```python
from pydala import convert_schema
import pyarrow as pa

conversions = {
    'id': pa.int64(),
    'price': pa.decimal128(10, 2),
    'timestamp': pa.timestamp('ns')
}

new_schema = convert_schema(old_schema, conversions)
```

### validate_schema
```python
def validate_schema(data: pa.Table, schema: pa.Schema) -> ValidationResult
```
Validate data against schema.

**Parameters:**
- `data` (pa.Table): Data to validate
- `schema` (pa.Schema): Expected schema

**Returns:**
- `ValidationResult`: Validation result object

**Example:**
```python
from pydala import validate_schema

result = validate_schema(table, expected_schema)
if not result.valid:
    for error in result.errors:
        print(f"Schema error: {error}")
```

## Data Type Utilities

### optimize_dtypes
```python
def optimize_dtypes(table: pa.Table, backend: str = "polars") -> pa.Table
```
Optimize data types to reduce memory usage.

**Parameters:**
- `table` (pa.Table): Input table
- `backend` (str): Backend to use for optimization

**Returns:**
- `pa.Table`: Table with optimized types

**Example:**
```python
from pydala import optimize_dtypes

optimized = optimize_dtypes(table)
print(f"Memory reduced: {table.nbytes - optimized.nbytes} bytes")
```

### convert_large_types
```python
def convert_large_types(table: pa.Table) -> pa.Table
```
Convert large Arrow types to standard types.

**Parameters:**
- `table` (pa.Table): Input table

**Returns:**
- `pa.Table`: Table with standard types

**Example:**
```python
from pydala import convert_large_types

# Convert large_string to string, large_list to list
standard_table = convert_large_types(large_table)
```

## SQL Utilities

### sql2pyarrow_filter
```python
def sql2pyarrow_filter(sql_filter: str) -> pds.Expression
```
Convert SQL WHERE clause to PyArrow filter expression.

**Parameters:**
- `sql_filter` (str): SQL filter expression

**Returns:**
- `pds.Expression`: PyArrow filter expression

**Example:**
```python
from pydala import sql2pyarrow_filter

filter_expr = sql2pyarrow_filter("date > '2023-01-01' AND amount > 100")
result = dataset.read(filter=filter_expr)
```

### escape_sql_identifier
```python
def escape_sql_identifier(identifier: str) -> str
```
Escape SQL identifiers to prevent injection.

**Parameters:**
- `identifier` (str): SQL identifier

**Returns:**
- `str`: Escaped identifier

**Example:**
```python
from pydala import escape_sql_identifier

safe_name = escape_sql_identifier(column_name)
sql = f"SELECT {safe_name} FROM table"
```

### escape_sql_literal
```python
def escape_sql_literal(value) -> str
```
Escape SQL literal values.

**Parameters:**
- `value`: Value to escape

**Returns:**
- `str`: Escaped SQL literal

**Example:**
```python
from pydala import escape_sql_literal

safe_value = escape_sql_literal(user_input)
sql = f"SELECT * FROM table WHERE name = {safe_value}"
```

## Time Utilities

### get_timestamp_column
```python
def get_timestamp_column(df: pd.DataFrame | pl.DataFrame) -> list[str]
```
Identify timestamp columns in a DataFrame.

**Parameters:**
- `df`: DataFrame to analyze

**Returns:**
- `list[str]`: Names of timestamp columns

**Example:**
```python
from pydala import get_timestamp_column

timestamp_cols = get_timestamp_column(df)
print(f"Found timestamp columns: {timestamp_cols}")
```

### parse_timestamp_string
```python
def parse_timestamp_string(ts_str: str, fmt: str = None) -> pd.Timestamp
```
Parse timestamp string with automatic format detection.

**Parameters:**
- `ts_str` (str): Timestamp string
- `fmt` (str, optional): Expected format

**Returns:**
- `pd.Timestamp`: Parsed timestamp

**Example:**
```python
from pydala import parse_timestamp_string

# Auto-detect format
ts = parse_timestamp_string("2023-12-01 14:30:00")

# Specify format
ts = parse_timestamp_string("01/12/2023", fmt="%d/%m/%Y")
```

## Compression Utilities

### get_compression_extension
```python
def get_compression_extension(compression: str) -> str
```
Get file extension for compression type.

**Parameters:**
- `compression` (str): Compression algorithm

**Returns:**
- `str`: File extension

**Example:**
```python
from pydala import get_compression_extension

ext = get_compression_extension('gzip')  # Returns '.gz'
ext = get_compression_extension('zstd')  # Returns '.zst'
```

### compress_data
```python
def compress_data(data: bytes, compression: str, level: int = None) -> bytes
```
Compress data with specified algorithm.

**Parameters:**
- `data` (bytes): Data to compress
- `compression` (str): Compression algorithm
- `level` (int, optional): Compression level

**Returns:**
- `bytes`: Compressed data

**Example:**
```python
from pydala import compress_data

compressed = compress_data(raw_data, 'zstd', level=3)
```

### decompress_data
```python
def decompress_data(data: bytes, compression: str) -> bytes
```
Decompress data.

**Parameters:**
- `data` (bytes): Compressed data
- `compression` (str): Compression algorithm

**Returns:**
- `bytes`: Decompressed data

**Example:**
```python
from pydala import decompress_data

decompressed = decompress_data(compressed_data, 'zstd')
```

## Security Utilities

### validate_path
```python
def validate_path(path: str) -> bool
```
Validate filesystem path for security.

**Parameters:**
- `path` (str): Path to validate

**Returns:**
- `bool`: True if path is safe

**Example:**
```python
from pydala import validate_path

if validate_path(user_path):
    # Path is safe to use
    fs.open(user_path)
```

### sanitize_filter_expression
```python
def sanitize_filter_expression(expr: str) -> str
```
Sanitize filter expressions to prevent injection.

**Parameters:**
- `expr` (str): Filter expression

**Returns:**
- `str`: Sanitized expression

**Example:**
```python
from pydala import sanitize_filter_expression

safe_expr = sanitize_filter_expression(user_filter)
result = dataset.read(filters=safe_expr)
```

### validate_partition_name
```python
def validate_partition_name(name: str) -> bool
```
Validate partition column names.

**Parameters:**
- `name` (str): Partition name

**Returns:**
- `bool`: True if valid

**Example:**
```python
from pydala import validate_partition_name

if validate_partition_name(partition_col):
    # Safe to use as partition column
    dataset.write(data, partition_cols=[partition_col])
```

## Performance Utilities

### Timer
```python
class Timer
```
Context manager for timing operations.

**Example:**
```python
from pydala import Timer

with Timer() as t:
    # Perform operation
    dataset.read()

print(f"Operation took {t.elapsed:.2f} seconds")
```

### Memory Usage
```python
def get_memory_usage() -> dict
```
Get current memory usage statistics.

**Returns:**
- `dict`: Memory usage information

**Example:**
```python
from pydala import get_memory_usage

mem = get_memory_usage()
print(f"Memory used: {mem['used'] / 1024**3:.1f} GB")
print(f"Memory available: {mem['available'] / 1024**3:.1f} GB")
```

### ProgressBar
```python
class ProgressBar
```
Progress bar for long-running operations.

**Example:**
```python
from pydala import ProgressBar

with ProgressBar(total=len(files)) as pbar:
    for file in files:
        process_file(file)
        pbar.update(1)
```

## Logging Utilities

### setup_logger
```python
def setup_logger(
    name: str,
    level: str = "INFO",
    format: str = None,
    file: str = None
) -> logging.Logger
```
Set up a logger with custom configuration.

**Parameters:**
- `name` (str): Logger name
- `level` (str): Log level
- `format` (str, optional): Log format
- `file` (str, optional): Log file path

**Returns:**
- `logging.Logger`: Configured logger

**Example:**
```python
from pydala import setup_logger

logger = setup_logger(
    "my_app",
    level="DEBUG",
    file="app.log"
)
logger.info("Application started")
```

## Exception Classes

### PydalaException
```python
class PydalaException(Exception)
```
Base exception class for PyDala2.

### ValidationError
```python
class ValidationError(PydalaException)
```
Raised when data validation fails.

### FileSystemError
```python
class FileSystemError(PydalaException)
```
Raised for filesystem-related errors.

### CatalogError
```python
class CatalogError(PydalaException)
```
Raised for catalog-related errors.

### SchemaError
```python
class SchemaError(PydalaException)
```
Raised for schema-related errors.

**Example:**
```python
from pydala import PydalaException, ValidationError

try:
    dataset.write(data)
except ValidationError as e:
    print(f"Validation failed: {e}")
except PydalaException as e:
    print(f"PyDala2 error: {e}")
```

## Helper Functions

### safe_join
```python
def safe_join(base_path: str, *paths) -> str
```
Safely join filesystem paths.

**Parameters:**
- `base_path` (str): Base path
- `*paths`: Additional path components

**Returns:**
- `str`: Joined path

**Example:**
```python
from pydala import safe_join

path = safe_join("/data", "sales", "2023", "file.parquet")
```

### generate_unique_name
```python
def generate_unique_name(prefix: str = "pydala_") -> str
```
Generate a unique name for temporary objects.

**Parameters:**
- `prefix` (str): Name prefix

**Returns:**
- `str`: Unique name

**Example:**
```python
from pydala import generate_unique_name

temp_name = generate_unique_name("temp_")
print(temp_name)  # e.g., "temp_abc123"
```

### format_bytes
```python
def format_bytes(bytes: int) -> str
```
Format bytes as human-readable string.

**Parameters:**
- `bytes` (int): Number of bytes

**Returns:**
- `str`: Formatted string

**Example:**
```python
from pydala import format_bytes

print(format_bytes(1024))        # "1.0 KB"
print(format_bytes(1048576))      # "1.0 MB"
print(format_bytes(1073741824))   # "1.0 GB"
```

### format_duration
```python
def format_duration(seconds: float) -> str
```
Format duration as human-readable string.

**Parameters:**
- `seconds` (float): Duration in seconds

**Returns:**
- `str`: Formatted string

**Example:**
```python
from pydala import format_duration

print(format_duration(45))        # "45.0s"
print(format_duration(3661))      # "1h 1m 1s"
print(format_duration(90061))     # "1d 1h 1m 1s"
```
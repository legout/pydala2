# Catalog System

This document describes the `Catalog` class which provides centralized management of datasets across your data lake.

## Catalog

```python
class Catalog
```

The `Catalog` class provides a centralized way to manage multiple datasets across different namespaces and filesystems. It maintains a YAML configuration file that tracks table locations, formats, and other metadata.

### Constructor

```python
Catalog(
    path: str,
    namespace: str | None = None,
    ddb_con: duckdb.DuckDBPyConnection | None = None,
    filesystem: AbstractFileSystem | None = None,
    bucket: str | None = None,
    **fs_kwargs
) -> None
```

**Parameters:**
- `path` (str): Path to the catalog configuration file (YAML)
- `namespace` (str, optional): Namespace to scope table operations to
- `ddb_con` (duckdb.DuckDBPyConnection, optional): Existing DuckDB connection
- `filesystem` (AbstractFileSystem, optional): Default filesystem for catalog operations
- `bucket` (str, optional): Bucket name for cloud storage
- `**fs_kwargs`: Additional filesystem configuration

**Example:**
```python
from pydala import Catalog

# Create catalog
catalog = Catalog("catalog.yaml", namespace="sales")

# The catalog.yaml file structure:
# tables:
#   sales:
#     daily:
#       path: /data/sales/daily
#       format: parquet
#       options:
#         partitioning: [year, month]
#     monthly:
#       path: /data/sales/monthly
#       format: parquet
```

### Properties

#### namespace
```python
@property
def namespace(self) -> str | None
```
Get the current namespace.

**Returns:**
- `str | None`: Current namespace or None if not set

### Methods

#### load_catalog
```python
def load_catalog(self, namespace: str | None = None) -> Munch
```
Load catalog configuration from YAML file.

**Parameters:**
- `namespace` (str, optional): If specified, loads only tables from this namespace

**Returns:**
- `Munch`: The loaded catalog configuration

**Raises:**
- `FileNotFoundError`: If the catalog file doesn't exist
- `yaml.YAMLError`: If the catalog file is invalid YAML

**Example:**
```python
# Load full catalog
catalog.load_catalog()

# Load specific namespace
catalog.load_catalog(namespace="sales")
```

#### list_namespaces
```python
def list_namespaces(self) -> list[str]
```
List all namespaces in the catalog.

**Returns:**
- `list[str]`: List of namespace names

**Example:**
```python
namespaces = catalog.list_namespaces()
print(namespaces)  # ['sales', 'customers', 'products']
```

#### all_tables
```python
def all_tables(self) -> list[str]
```
List all tables in the current namespace.

**Returns:**
- `list[str]`: List of table names

**Example:**
```python
tables = catalog.all_tables()
print(tables)  # ['daily', 'monthly', 'quarterly']
```

#### get
```python
def get(self, table_name: str) -> Munch
```
Get table configuration.

**Parameters:**
- `table_name` (str): Name of the table

**Returns:**
- `Munch`: Table configuration

**Example:**
```python
config = catalog.get("daily")
print(config.path)  # /data/sales/daily
print(config.format)  # parquet
```

#### show
```python
def show(self, table_name: str) -> None
```
Display table configuration.

**Parameters:**
- `table_name` (str): Name of the table

**Example:**
```python
catalog.show("daily")
# Output:
# path: /data/sales/daily
# format: parquet
# partitioning: [year, month]
# ...
```

#### load_parquet
```python
def load_parquet(
    self,
    table_name: str,
    path: str,
    name: str | None = None,
    filesystem: AbstractFileSystem | None = None,
    bucket: str | None = None,
    partitioning: str | list[str] | None = None,
    cached: bool = False,
    timestamp_column: str | None = None,
    **table_options
) -> ParquetDataset
```
Load a Parquet dataset and register it in the catalog.

**Parameters:**
- `table_name` (str): Name to register the table as
- `path` (str): Path to the Parquet dataset
- `name` (str, optional): Dataset name
- `filesystem` (AbstractFileSystem, optional): Filesystem for data access
- `bucket` (str, optional): Bucket name for cloud storage
- `partitioning` (str | list[str]): Partitioning scheme
- `cached` (bool): Whether to use caching
- `timestamp_column` (str, optional): Timestamp column name
- `**table_options`: Additional table options

**Returns:**
- `ParquetDataset`: The loaded Parquet dataset

**Example:**
```python
# Load and register Parquet dataset
dataset = catalog.load_parquet(
    "sales_2023",
    "/data/sales/2023",
    partitioning=["year", "month"],
    cached=True
)
```

#### load_csv
```python
def load_csv(
    self,
    table_name: str,
    path: str,
    name: str | None = None,
    filesystem: AbstractFileSystem | None = None,
    bucket: str | None = None,
    partitioning: str | list[str] | None = None,
    cached: bool = False,
    timestamp_column: str | None = None,
    **table_options
) -> CSVDataset
```
Load a CSV dataset and register it in the catalog.

**Parameters:**
- `table_name` (str): Name to register the table as
- `path` (str): Path to the CSV file/directory
- `name` (str, optional): Dataset name
- `filesystem` (AbstractFileSystem, optional): Filesystem for data access
- `bucket` (str, optional): Bucket name for cloud storage
- `partitioning` (str | list[str]): Partitioning scheme
- `cached` (bool): Whether to use caching
- `timestamp_column` (str, optional): Timestamp column name
- `**table_options`: Additional table options

**Returns:**
- `CSVDataset`: The loaded CSV dataset

**Example:**
```python
# Load CSV with custom options
dataset = catalog.load_csv(
    "customers",
    "/data/customers.csv",
    parse_options={'delimiter': ',', 'header': True},
    convert_options={'column_types': {'id': 'int64'}}
)
```

#### load_json
```python
def load_json(
    self,
    table_name: str,
    path: str,
    name: str | None = None,
    filesystem: AbstractFileSystem | None = None,
    bucket: str | None = None,
    partitioning: str | list[str] | None = None,
    cached: bool = False,
    timestamp_column: str | None = None,
    **table_options
) -> JSONDataset
```
Load a JSON dataset and register it in the catalog.

**Parameters:**
- `table_name` (str): Name to register the table as
- `path` (str): Path to the JSON file/directory
- `name` (str, optional): Dataset name
- `filesystem` (AbstractFileSystem, optional): Filesystem for data access
- `bucket` (str, optional): Bucket name for cloud storage
- `partitioning` (str | list[str]): Partitioning scheme
- `cached` (bool): Whether to use caching
- `timestamp_column` (str, optional): Timestamp column name
- `**table_options`: Additional table options

**Returns:**
- `JSONDataset`: The loaded JSON dataset

**Example:**
```python
# Load JSON dataset
dataset = catalog.load_json(
    "events",
    "/data/events",
    read_options={'block_size': 4096}
)
```

#### load
```python
def load(
    self,
    table_name: str,
    path: str,
    format: str,
    name: str | None = None,
    filesystem: AbstractFileSystem | None = None,
    bucket: str | None = None,
    partitioning: str | list[str] | None = None,
    cached: bool = False,
    timestamp_column: str | None = None,
    **table_options
) -> BaseDataset
```
Load a dataset with specified format and register it in the catalog.

**Parameters:**
- `table_name` (str): Name to register the table as
- `path` (str): Path to the dataset
- `format` (str): File format ('parquet', 'csv', 'json')
- `name` (str, optional): Dataset name
- `filesystem` (AbstractFileSystem, optional): Filesystem for data access
- `bucket` (str, optional): Bucket name for cloud storage
- `partitioning` (str | list[str]): Partitioning scheme
- `cached` (bool): Whether to use caching
- `timestamp_column` (str, optional): Timestamp column name
- `**table_options`: Additional table options

**Returns:**
- `BaseDataset`: The loaded dataset

**Example:**
```python
# Load with explicit format
dataset = catalog.load(
    "logs",
    "/data/logs",
    format="parquet",
    partitioning=["date", "level"]
)
```

#### files
```python
def files(self, table_name: str) -> list[str]
```
List files for a table.

**Parameters:**
- `table_name` (str): Name of the table

**Returns:**
- `list[str]`: List of file paths

**Example:**
```python
files = catalog.files("daily")
print(files)  # ['/data/sales/daily/...', ...]
```

#### show_filesystem
```python
def show_filesystem(self, table_name: str) -> None
```
Display filesystem configuration for a table.

**Parameters:**
- `table_name` (str): Name of the table

**Example:**
```python
catalog.show_filesystem("daily")
# Output:
# protocol: file
# ...
```

#### all_filesystems
```python
def all_filesystems(self) -> list[str]
```
List all filesystems in the catalog.

**Returns:**
- `list[str]`: List of filesystem names

**Example:**
```python
filesystems = catalog.all_filesystems()
print(filesystems)  # ['local', 's3', 'gcs']
```

### Query Operations

The catalog supports querying across registered datasets:

```python
# Simple query
result = catalog.query("SELECT * FROM sales_2023 WHERE amount > 1000")

# Join across datasets
result = catalog.query("""
    SELECT
        s.*,
        c.customer_name,
        c.customer_segment
    FROM sales_2023 s
    JOIN customers c ON s.customer_id = c.id
    WHERE s.date >= '2023-01-01'
""")

# Aggregation
result = catalog.query("""
    SELECT
        category,
        COUNT(*) as orders,
        SUM(amount) as revenue
    FROM sales_2023
    GROUP BY category
""")
```

### Configuration Management

#### Update Table Configuration

```python
# Add metadata to table
catalog._set_table_params(
    "sales_2023",
    description="Sales data for 2023",
    owner="sales_team",
    refresh_frequency="daily"
)
```

#### Namespace Operations

```python
# Switch namespace
catalog.load_catalog(namespace="finance")

# Get current namespace
current_ns = catalog.namespace
```

## Catalog Configuration File

The catalog uses a YAML configuration file with the following structure:

```yaml
# catalog.yaml
tables:
  sales:
    daily:
      path: /data/sales/daily
      format: parquet
      options:
        partitioning: [year, month, day]
        cached: true
      metadata:
        description: Daily sales transactions
        owner: sales_team
        refresh_frequency: daily

    monthly:
      path: /data/sales/monthly
      format: parquet
      options:
        partitioning: [year, month]

  customers:
    active:
      path: /data/customers/active
      format: parquet
      options:
        cached: true

    all:
      path: /data/customers/all
      format: csv
      options:
        delimiter: ','
        header: true

filesystems:
  s3_prod:
    protocol: s3
    key: your_access_key
    secret: your_secret_key
    client_kwargs:
      region_name: us-east-1

  gcs_archive:
    protocol: gcs
    token: /path/to/service-account.json
```

## Best Practices

### Organization

```python
# Use hierarchical namespaces
catalog.load_catalog(namespace="sales/daily")
catalog.load_catalog(namespace="sales/monthly")
catalog.load_catalog(namespace="finance/revenue")

# Use descriptive table names
catalog.load_parquet("transactions_2023", "/data/transactions/2023")
catalog.load_parquet("transactions_2024", "/data/transactions/2024")
```

### Metadata Management

```python
# Document your datasets
catalog._set_table_params(
    "sales_2023",
    description="Complete sales transactions for fiscal year 2023",
    business_unit="sales",
    data_owner="sales_analytics_team",
    contact="sales-data@company.com",
    sla="Available by 9:00 AM daily",
    retention_policy="7_years",
    classification="internal",
    tags=["sales", "transactions", "fy2023"]
)
```

### Performance

```python
# Enable caching for frequently accessed data
catalog.load_parquet(
    "hot_data",
    "/data/hot",
    cached=True,
    cache_options={'max_size': '2GB'}
)

# Use appropriate partitioning
catalog.load_parquet(
    "time_series",
    "/data/time_series",
    partitioning=["date", "category"]
)
```

### Security

```python
# Store credentials securely
# Use environment variables or secret management
catalog._set_table_params(
    "sensitive_data",
    access_level="restricted",
    encryption="aes256",
    audit_required=True
)
```

## Error Handling

```python
from pydala import PydalaException

try:
    dataset = catalog.load_parquet("nonexistent", "/invalid/path")
except PydalaException as e:
    print(f"Failed to load dataset: {e}")

try:
    result = catalog.query("INVALID SQL QUERY")
except Exception as e:
    print(f"Query failed: {e}")
```
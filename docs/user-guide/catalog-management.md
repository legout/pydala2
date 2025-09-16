# Catalog Management

The catalog system provides YAML-based configuration for managing datasets across different namespaces and filesystems.

## Introduction to Catalogs

A catalog acts as a centralized registry for your datasets, storing their locations, formats, and configurations in a YAML file. This makes it easy to:

- Organize datasets by namespace or project
- Manage different filesystems (local, S3, etc.)
- Share dataset configurations across teams
- Execute SQL queries across multiple datasets

## Catalog Configuration

### Basic Catalog Structure

```yaml
# catalog.yaml
filesystem:
  local:
    protocol: file
    bucket: ./data
  s3_storage:
    protocol: s3
    bucket: my-bucket
    key: your-access-key
    secret: your-secret-key

tables:
  sales:
    orders:
      path: sales/orders
      format: parquet
      filesystem: local
      partitioning: [year, month]
      write_args:
        compression: zstd
        max_rows_per_file: 1000000

    customers:
      path: sales/customers
      format: parquet
      filesystem: local
      write_args:
        compression: snappy

  finance:
    revenue:
      path: finance/revenue
      format: parquet
      filesystem: s3_storage
      write_args:
        compression: zstd
        partition_by: [region, year]
```

### Creating a Catalog

```python
from pydala import Catalog

# Create catalog from YAML file
catalog = Catalog("catalog.yaml")

# Create catalog with specific namespace
sales_catalog = Catalog("catalog.yaml", namespace="sales")

# Use existing DuckDB connection
import duckdb
con = duckdb.connect()
catalog = Catalog("catalog.yaml", ddb_con=con)
```

## Dataset Operations

### Loading Datasets

```python
# Load Parquet dataset with metadata
orders = catalog.load("sales.orders", with_metadata=True)

# Load as DataFrame instead of dataset
customers_df = catalog.load("sales.customers", as_dataset=False)

# Auto-detect format and load
revenue = catalog.load("finance.revenue")

# Force reload from disk
orders = catalog.load("sales.orders", reload=True)
```

### Creating Tables

```python
import polars as pl

# Create table from DataFrame
data = pl.DataFrame({
    'id': range(100),
    'name': [f'item_{i}' for i in range(100)],
    'value': [i * 10 for i in range(100)]
})

catalog.create_table(
    data=data,
    table_name="inventory.items",
    format="parquet",
    filesystem="local",
    partitioning=["category"],
    write_args={
        "compression": "zstd",
        "max_rows_per_file": 50000
    }
)

# Create table from existing dataset
catalog.create_table(
    data=existing_dataset,
    table_name="archive.old_data",
    path="archive/2023/data",
    overwrite=True
)

# Create placeholder table (no data written yet)
catalog.create_table(
    table_name="future.placeholder",
    path="future/data",
    format="parquet",
    filesystem="s3_storage"
)
```

### Writing Data

```python
# Write to catalog table
new_orders = pl.read_csv("new_orders.csv")
catalog.write_table(
    data=new_orders,
    table_name="sales.orders",
    mode="append",
    partition_by=["year", "month"]
)

# Update catalog with new write arguments
catalog.write_table(
    data=quarterly_data,
    table_name="finance.revenue",
    compression="brotli",
    update_catalog=True
)
```

### Updating Table Configuration

```python
# Update table metadata
catalog.update(
    table_name="sales.orders",
    description="Customer order data",
    owner="sales_team",
    refresh_frequency="daily",
    write_args={
        "compression": "zstd",
        "max_rows_per_file": 2000000
    }
)

# Change filesystem
catalog.update(
    table_name="finance.revenue",
    filesystem="s3_storage",
    path="finance/processed/revenue"
)
```

### Deleting Tables

```python
# Remove from catalog only
catalog.delete_table("temp.data")

# Remove and delete all data files
catalog.delete_table("old.archive", vacuum=True)

# Delete entire namespace
catalog.delete_table("legacy", vacuum=True)
```

## Namespace Management

### Working with Namespaces

```python
# List available namespaces
print(catalog.list_namespaces)
# Output: ['sales', 'finance', 'inventory']

# Create catalog scoped to namespace
sales_catalog = Catalog("catalog.yaml", namespace="sales")

# Access tables in namespace
orders = sales_catalog.load("orders")  # Loads sales.orders
customers = sales_catalog.load("customers")  # Loads sales.customers

# Create new namespace
catalog.create_namespace("analytics")
catalog.create_table(
    data=metrics_data,
    table_name="analytics.metrics",
    path="analytics/metrics"
)
```

### Cross-Namespace Operations

```python
# Copy table between namespaces
def copy_table(source_catalog, dest_catalog, source_table, dest_table):
    data = source_catalog.load(source_table, as_dataset=False)
    dest_catalog.create_table(
        data=data,
        table_name=dest_table,
        write_args={"mode": "overwrite"}
    )

# Usage
copy_table(
    source_catalog=catalog,
    dest_catalog=catalog,
    source_table="sales.orders",
    dest_table="analytics.orders_archive"
)
```

## SQL Operations

### Querying Multiple Tables

```python
# SQL with automatic table loading
result = catalog.sql("""
    SELECT
        o.order_id,
        o.order_date,
        c.customer_name,
        o.total_amount
    FROM sales.orders o
    JOIN sales.customers c ON o.customer_id = c.id
    WHERE o.order_date >= '2023-01-01'
    ORDER BY o.total_amount DESC
    LIMIT 1000
""")

# Get as Polars DataFrame
df = result.pl()

# Tables are automatically loaded and registered
print(catalog.registered_tables)
```

### Complex Analytics Queries

```python
# Time-based analytics
analytics = catalog.sql("""
    WITH monthly_sales AS (
        SELECT
            DATE_TRUNC('month', order_date) as month,
            region,
            COUNT(*) as order_count,
            SUM(total_amount) as revenue
        FROM sales.orders
        WHERE order_date >= '2023-01-01'
        GROUP BY month, region
    )
    SELECT
        month,
        region,
        revenue,
        LAG(revenue, 1) OVER (PARTITION BY region ORDER BY month) as prev_revenue,
        (revenue - LAG(revenue, 1) OVER (PARTITION BY region ORDER BY month)) /
            LAG(revenue, 1) OVER (PARTITION BY region ORDER BY month) * 100 as growth_rate
    FROM monthly_sales
    ORDER BY region, month
""").pl()
```

## Filesystem Management

### Configuring Multiple Filesystems

```python
# Access configured filesystems
for fs_name in catalog.all_filesystems:
    print(f"Filesystem: {fs_name}")
    catalog.show_filesystem(fs_name)

# Files are accessed through configured filesystems
orders_files = catalog.files("sales.orders")
print(f"Orders files: {orders_files[:5]}...")  # Show first 5 files
```

### Cloud Storage Integration

```yaml
# catalog.yaml
filesystem:
  production:
    protocol: s3
    bucket: my-company-data
    key: ${S3_ACCESS_KEY}
    secret: ${S3_SECRET_KEY}
    region: us-west-2

  archive:
    protocol: s3
    bucket: my-company-archive
    key: ${S3_ACCESS_KEY}
    secret: ${S3_SECRET_KEY}
    region: us-east-1

tables:
  production:
    current:
      path: sales/current
      format: parquet
      filesystem: production
      write_args:
        compression: zstd
        partition_by: [date, region]

    archived:
      path: sales/historical
      format: parquet
      filesystem: archive
      write_args:
        compression: zstd
        partition_by: [year, quarter]
```

## Catalog Inspection

### Viewing Table Information

```python
# Show table configuration
catalog.show("sales.orders")
# Output:
# path: sales/orders
# format: parquet
# filesystem: local
# partitioning: [year, month]
# write_args:
#   compression: zstd
#   max_rows_per_file: 1000000

# Get table configuration as object
config = catalog.get("finance.revenue")
print(f"Table path: {config.path}")
print(f"Compression: {config.write_args.compression}")

# List all tables
print("All tables:", catalog.all_tables)

# Get table schema
schema = catalog.schema("sales.orders")
print(f"Schema: {schema}")
```

### Working with DuckDB

```python
# See registered tables
print("Registered tables:")
catalog.registered_tables.show()

# Manual table registration
dataset = catalog.load("sales.orders")
catalog.ddb_con.register("orders_temp", dataset.table.pl.collect())

# Use registered tables in queries
result = catalog.ddb_con.sql("""
    SELECT region, AVG(total_amount) as avg_order
    FROM orders_temp
    GROUP BY region
""").pl()
```

## Advanced Patterns

### Data Pipeline with Catalog

```python
def run_pipeline(catalog, input_table, output_table, transform_func):
    """Run a data transformation pipeline"""
    # Load input data
    input_data = catalog.load(input_table, as_dataset=False)

    # Apply transformation
    output_data = transform_func(input_data)

    # Create output table if needed
    if output_table not in catalog.all_tables:
        catalog.create_table(
            table_name=output_table,
            format="parquet",
            filesystem="local"
        )

    # Write results
    catalog.write_table(
        data=output_data,
        table_name=output_table,
        mode="overwrite"
    )

# Usage
def aggregate_daily_orders(df):
    return df.groupby(["date", "region"]).agg([
        pl.count("order_id").alias("order_count"),
        pl.sum("total_amount").alias("revenue")
    ])

run_pipeline(
    catalog=catalog,
    input_table="sales.orders",
    output_table="analytics.daily_summary",
    transform_func=aggregate_daily_orders
)
```

### Catalog Validation

```python
def validate_catalog(catalog):
    """Validate catalog configuration"""
    issues = []

    # Check all table paths exist
    for table_name in catalog.all_tables:
        try:
            files = catalog.files(table_name)
            if not files:
                issues.append(f"No files found for {table_name}")
        except Exception as e:
            issues.append(f"Error accessing {table_name}: {e}")

    # Check filesystem configurations
    for fs_name in catalog.all_filesystems:
        try:
            fs = catalog.params.filesystem[fs_name]
            if fs.protocol not in ["file", "s3"]:
                issues.append(f"Unknown protocol for {fs_name}: {fs.protocol}")
        except Exception as e:
            issues.append(f"Error in filesystem {fs_name}: {e}")

    return issues

# Usage
issues = validate_catalog(catalog)
if issues:
    print("Catalog validation issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Catalog is valid")
```

## Best Practices

### Organization

```yaml
# Use hierarchical namespaces
tables:
  raw:
    source1:
      path: raw/source1
      format: json
    source2:
      path: raw/source2
      format: csv

  processed:
    cleaned:
      path: processed/cleaned
      format: parquet
      write_args:
        compression: zstd
        partition_by: [date]

    aggregated:
      path: processed/aggregated
      format: parquet
      write_args:
        compression: zstd

  analytics:
    dashboards:
      path: analytics/dashboards
      format: parquet
```

### Performance

```python
# Cache frequently accessed datasets
catalog.load("sales.orders", with_metadata=True)  # Load with metadata

# Use SQL for complex operations across tables
result = catalog.sql("""
    SELECT * FROM sales.orders
    WHERE customer_id IN (
        SELECT id FROM sales.customers
        WHERE segment = 'Premium'
    )
""")

# Bulk operations
for table_name in catalog.all_tables:
    if table_name.startswith("temp."):
        catalog.delete_table(table_name, vacuum=True)
```

### Security

```yaml
# Use environment variables for credentials
filesystem:
  secure_storage:
    protocol: s3
    bucket: sensitive-data
    key: ${AWS_ACCESS_KEY_ID}
    secret: ${AWS_SECRET_ACCESS_KEY}

# Control access at catalog level
tables:
  restricted:
    financial:
      path: secure/financial
      format: parquet
      filesystem: secure_storage
      access_control: ["finance_team", "executives"]
```

The catalog system provides a powerful way to organize and manage your datasets while leveraging PyDala2's full capabilities for data operations.
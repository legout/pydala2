# Catalog Management

A catalog maps table names to dataset locations. Keep the configuration in
YAML and use DuckDB to query the datasets it describes.

## Catalog Configuration

Start with the [Quick Start catalog](../quick-start.md#query-through-a-catalog).
For a local filesystem, `bucket` is the root relative to the YAML file and
each table's `path` is relative to that root. Set `format` to match the data
and `filesystem` to the name of a configured filesystem.

### Working with Namespaces

Declare namespaces under `tables` in the YAML file. For example, add this
entry under `tables` alongside the existing namespaces, then open a new
catalog scoped to `analytics`:

```yaml
analytics:
  metrics:
    path: analytics/metrics
    format: parquet
    filesystem: local
```

```python
from pydala import Catalog

analytics = Catalog("catalog.yaml", namespace="analytics")
assert "metrics" in analytics.all_tables
```

Use an unqualified name such as `metrics` with a scoped catalog. The entry
describes the data location; it does not create or populate the dataset.

## Dataset Operations

### Load and query

Run this example with the three-row `sales` dataset and YAML from the Quick
Start, before inserting the fourth row:

```python
from pydala import Catalog

catalog = Catalog("catalog.yaml", namespace="default")
sales = catalog.load("sales")
df = sales.table.to_polars(lazy=False)
totals = catalog.sql("""
    SELECT region, SUM(amount) AS total
    FROM sales
    GROUP BY region
    ORDER BY region
""").fetchall()
```

By default `load()` returns a dataset. Export its table directly to obtain a
DataFrame. `sql()` loads referenced catalog entries not already registered on
the connection; for a join, configure both table names in the selected namespace.

Reopening the catalog creates a fresh DuckDB registration but does not
rebuild existing Parquet sidecars. After a managed append, verify that the
new rows are visible. After external edits, or if committed files are missing
from query results, follow [Metadata Management](metadata.md#refresh-after-external-changes)
before querying. `load_catalog()` only rereads the YAML.

### Merging Catalog Datasets

For a catalog with a `sales.orders` entry, open it as
`catalog = Catalog("catalog.yaml", namespace="sales")`. Use the dataset's
merge interface when matching rows must be updated instead of appended:

```python
orders = catalog.load("orders", with_metadata=True)
result = orders.merge(
    incoming_orders,
    strategy="upsert",
    key_columns=["order_id"],
    partition_by=["year", "month"],
)
```

Here `incoming_orders` contains the key and both partition columns. See
[Merge and Delta Migration](merge.md) for validation and strategy details.

## Catalog Inspection

Use `catalog.all_tables` and `catalog.all_filesystems` to see what is
configured, `catalog.get("sales")` to retrieve an entry, and
`catalog.files("sales")` to list the files. `catalog.registered_tables`
shows what DuckDB already knows about, which can differ from the YAML.

## Cloud Storage Integration

Install `pydala2[s3]` for S3. Configure a filesystem entry with
`protocol: s3` and `bucket: your-bucket-name`, then reference that entry
from each table's `filesystem` field. See
[Deployment](../advanced/deployment.md#s3-configuration) for storage setup.

### Security

A catalog describes where data lives; it does not enforce access control.
Configure permissions in the storage backend or the application. PyDala2
does not expand environment variable placeholders in catalog YAML. For S3,
keep credentials out of the catalog and use the AWS credential chain.

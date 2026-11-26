# Catalog API

`Catalog(path, namespace=None, ddb_con=None, filesystem=None, bucket=None, **fs_kwargs)` reads a YAML file describing dataset locations and filesystem definitions. The [Quick Start](../quick-start.md#query-through-a-catalog) has a complete local configuration.

## Load configuration

### load_catalog

`catalog.load_catalog(namespace=None)` rereads the YAML file and replaces `catalog.params`. With a namespace argument, the refreshed `params.tables` contains that namespace's tables. The method returns `None`, not the configuration. It does not rebuild filesystem instances or replace already loaded tables.

Given the Quick Start's `catalog.yaml`:

```python
from pydala import Catalog

catalog = Catalog("catalog.yaml", namespace="default")
assert catalog.load_catalog(namespace="default") is None
assert catalog.params.tables["sales"].format == "parquet"
```

`catalog.get(table_name)` returns the table's configuration. `catalog.namespace`, `catalog.list_namespaces`, `catalog.all_tables`, and `catalog.all_filesystems` are properties.

## Load and query datasets

- `catalog.load(table_name)` loads a catalog entry according to its configured format. The format-specific loaders are `load_parquet()`, `load_csv()`, and `load_json()`; each takes the configured table name, not a new path.
- `catalog.sql(query)` loads referenced catalog tables that aren't already registered and returns a DuckDB relation.
- `catalog.files(table_name)` lists files using the entry's configured filesystem.
- `catalog.registered_tables` returns DuckDB's `SHOW TABLES` relation.

See [Catalog Management](../user-guide/catalog-management.md) for loading and writing data. Configure access control and encryption in the storage backend or application; catalog YAML does not enforce them.

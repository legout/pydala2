# Quick Start

[Install PyDala2](getting-started.md), then run these examples in order in an empty working directory.

## Write and reopen a Parquet dataset

```python
from pathlib import Path

import polars as pl
from pydala import ParquetDataset

path = Path("sales").resolve()
dataset = ParquetDataset(str(path), partitioning="hive")
dataset.write_to_dataset(
    pl.DataFrame({"id": [1, 2, 3], "region": ["EU", "US", "EU"], "amount": [10, 20, 30]}),
    partition_by=["region"],
)

dataset = ParquetDataset(str(path), partitioning="hive")
assert dataset.count_rows() == 3
```

Use an absolute path for local writes. This example reopens the dataset
using the sidecars from its first write. If files change outside PyDala2,
see [Metadata Management](user-guide/metadata.md#refresh-after-external-changes).

## Filter and export

```python
filtered = dataset.filter("amount >= 20")
arrow_table = filtered.arrow
polars_frame = filtered.to_polars(lazy=False)
pandas_frame = filtered.to_pandas()

totals = dataset.ddb_con.sql(
    "SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY region"
).fetchall()
assert totals == [("EU", 40), ("US", 20)]
```

`filter()` returns a `PydalaTable`. Export it directly, or use `to_polars(lazy=True)` and collect the resulting Polars LazyFrame.

## Query through a catalog

Save this as `catalog.yaml` next to the `sales/` directory:

```yaml
filesystem:
  local:
    protocol: file
    bucket: .
tables:
  default:
    sales:
      path: sales
      format: parquet
      filesystem: local
```

```python
from pydala import Catalog

catalog = Catalog("catalog.yaml", namespace="default")
assert catalog.sql("SELECT COUNT(*) FROM sales").fetchone() == (3,)
```

`Catalog.sql()` loads referenced tables from the YAML configuration when needed.

## Insert new keys

```python
result = dataset.merge(
    pl.DataFrame({"id": [4], "region": ["EU"], "amount": [40]}),
    strategy="insert",
    key_columns=["id"],
    partition_by=["region"],
)
assert result.inserted == 1
```

- [Merge and Delta Migration](user-guide/merge.md): insert, update, upsert and legacy Delta behavior.
- [Deployment](advanced/deployment.md): remote filesystems and application configuration.
- [Troubleshooting](advanced/troubleshooting.md): installation, memory and catalog issues.

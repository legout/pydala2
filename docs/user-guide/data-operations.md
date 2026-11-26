# Data Operations

Start with the three-row `sales` dataset from the [Quick Start](../quick-start.md#write-and-reopen-a-parquet-dataset), before inserting the fourth row.

## Filter and project

```python
filtered = dataset.filter("region = 'EU' AND amount >= 20")
rows = filtered.to_arrow_table(columns=["id", "amount"])
assert rows.to_pylist() == [{"id": 3, "amount": 30}]
```

`filter()` returns a `PydalaTable`. Pass `use="pyarrow"` to require the Arrow
engine or `use="duckdb"` for predicates that need DuckDB SQL syntax. See
[Table Operations](../api/table.md) to choose an export format or read batches.

## Aggregate and join

The dataset's DuckDB connection registers the table by its name (`sales` in
the Quick Start). Use that name in SQL rather than the Python variable name.

```python
totals = dataset.ddb_con.sql("""
    SELECT region, SUM(amount) AS total
    FROM sales
    GROUP BY region
    ORDER BY region
""").fetchall()
assert totals == [("EU", 40), ("US", 20)]
```

Use a [catalog](catalog-management.md#load-and-query) for joins between
configured datasets. It loads referenced tables on the same connection.

## Transform and write a new dataset

```python
from pathlib import Path

import polars as pl
from pydala import ParquetDataset

summary = (
    dataset.table.to_polars(lazy=True)
    .group_by("region")
    .agg(pl.col("amount").sum().alias("total"))
    .collect()
)
output = ParquetDataset(str(Path("sales_summary").resolve()))
output.write_to_dataset(summary)
```

Write to a separate path when the output has a different schema. For changes
to an existing dataset, see [Schema Management](schema-management.md).

## Upsert Operations

Use the public merge interface instead of manually splitting new and existing
records. For example, with a dataset keyed by `customer_id`:

```python
result = dataset.merge(
    updated_customers,
    strategy="upsert",
    key_columns=["customer_id"],
)

print(f"Inserted {result.inserted} records")
print(f"Updated {result.updated} records")
```

Use `strategy="insert"` to leave existing keys unchanged or `strategy="update"`
to ignore source keys not in the target. See [Merge and Delta Migration](merge.md)
for key validation, duplicate handling, partition rules, and backend selection.

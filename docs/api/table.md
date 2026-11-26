# Table Operations

`dataset.table` (or `dataset.t`) returns a `PydalaTable` for a loaded dataset. `dataset.filter(...)` already returns a `PydalaTable`; export that result directly.

## Convert a filtered result

Using the `sales` dataset from [Quick Start](../quick-start.md), after the first write and reopen:

```python
filtered = dataset.filter("amount >= 20", use="pyarrow")
arrow_table = filtered.to_arrow_table(columns=["id", "amount"], sort_by="id")
polars_frame = filtered.to_polars(lazy=False)
pandas_frame = filtered.df
relation = filtered.ddb

totals = relation.aggregate("region, sum(amount) AS total", "region").order("region").fetchall()
assert totals == [("EU", 30), ("US", 20)]
```

| Output | Method | Shortcut property |
| --- | --- | --- |
| Arrow table | `to_arrow_table()` or `to_arrow()` | `arrow` |
| Polars | `to_polars(lazy=True)` on an Arrow-backed table returns a LazyFrame; use `lazy=False` for a DataFrame | `pl` |
| pandas | `to_pandas()` | `df` |
| DuckDB relation | `to_duckdb()` | `ddb` |

Pass SQL to the dataset's connection, which registers the loaded dataset by name. For operations on a filtered relation, use DuckDB relation methods as above.

```python
assert dataset.ddb_con.sql("SELECT COUNT(*) FROM sales").fetchone() == (3,)
```

## Read batches

Select the columns you need and iterate over a record batch reader instead of materializing the entire result:

```python
for batch in filtered.to_batch_reader(columns=["id", "amount"], batch_size=10_000):
    print(batch.num_rows)
```

For an Arrow-backed table, `scanner()` returns a PyArrow scanner; `arrow_dataset` exposes the underlying Arrow dataset.

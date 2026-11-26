# Basic Usage

The [Quick Start](../quick-start.md) shows a complete local workflow. Use an absolute path for local writes, or pass a [rooted filesystem](../api/filesystem.md) and a path relative to that root.

## Partitioned Writes

Supply the partition columns in the input. This example writes two months to separate Hive directories:

```python
from datetime import date
from pathlib import Path

import polars as pl
from pydala import ParquetDataset

path = str(Path("monthly_sales").resolve())
dataset = ParquetDataset(path, partitioning="hive")
data = pl.DataFrame({
    "id": [1, 2],
    "date": [date(2023, 1, 1), date(2023, 2, 1)],
    "amount": [10, 20],
}).with_columns(
    pl.col("date").dt.year().alias("year"),
    pl.col("date").dt.month().alias("month"),
)
dataset.write_to_dataset(data, partition_by=["year", "month"])
dataset = ParquetDataset(path, partitioning="hive")
```

Use `max_rows_per_file` to control row counts per file and `compression` to select the Parquet codec. Choose partition columns that you often filter on, but avoid creating a separate directory for every row.

## Read and filter

```python
february = dataset.filter("year = 2023 AND month = 2")
assert february.to_arrow_table(columns=["id", "amount"]).to_pylist() == [
    {"id": 2, "amount": 20},
]
```

A loaded dataset exposes its table through `dataset.table` or `dataset.t`. A filtered result is already a `PydalaTable`; see [Table Operations](../api/table.md) for conversions and batch reads.

## Keyed Merges

Use `write_to_dataset()` to append unconditionally. Use `merge()` when you want to match source and target rows by key:

```python
from datetime import date

new_data = pl.DataFrame({
    "id": [3],
    "date": [date(2023, 2, 2)],
    "amount": [30],
    "year": [2023],
    "month": [2],
})
result = dataset.merge(
    new_data,
    strategy="insert",
    key_columns=["id"],
    partition_by=["year", "month"],
)
```

Use `strategy="update"` to change matching rows and `strategy="upsert"` to also insert new keys. See [Merge and Delta Migration](merge.md) for details.

## Inspect and maintain the dataset

- Inspect `dataset.files`, `dataset.schema`, and `dataset.count_rows()` after loading.
- If files changed outside PyDala2, see [Metadata Management](metadata.md) for rebuilding sidecars.
- Review the [maintenance dry-run caveats](../advanced/performance-tuning.md#change-the-file-layout)
  before planning a rewrite. A dry run skips publication, but loading a dataset
  with missing sidecars during preparation can still repair Parquet files.

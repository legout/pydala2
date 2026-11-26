# Dataset Classes

| Class | Use |
| --- | --- |
| `ParquetDataset` | Managed Parquet reads, writes, keyed merges, and metadata sidecars |
| `PyarrowDataset` | Arrow-backed reads; `format` defaults to `"parquet"` |
| `CSVDataset` | Read CSV through PyArrow |
| `JSONDataset` | Read directories of JSON files |

These classes are importable from `pydala`. Pass a path, or pass a
[filesystem](filesystem.md) and a path relative to its root. For Hive-style
directories, pass `partitioning="hive"`.

## ParquetDataset

`ParquetDataset.write_to_dataset(data, partition_by=..., basename=...,
max_rows_per_file=..., row_group_size=..., compression=...)` appends files
by default. Use [keyed merge](../user-guide/merge.md) for inserts that match
on keys, updates, or upserts. See the [Quick Start](../quick-start.md) for a
complete write and reopen example.

Use `load(reload_metadata=True)` only when Parquet files remain; it can
rewrite those files. See [Metadata Management](../user-guide/metadata.md#refresh-after-external-changes)
for the empty-directory case.

## Common Dataset Operations

A loaded dataset's `table` property and its `t` alias expose a
`PydalaTable`. For Arrow-backed reads, use `dataset.table.arrow_dataset`
to inspect the underlying Arrow dataset.

### Filtering Data

Using the sales dataset from the Quick Start:

```python
filtered = dataset.filter("amount >= 20")
polars_frame = filtered.to_polars(lazy=False)
pandas_frame = filtered.df
```

`filter()` selects matching rows. On a `ParquetDataset`, `scan()` selects
candidate files using metadata and can include nonmatching rows. See
[Table Operations](table.md) for exports and batch reads.

### Performance Optimization

Call [maintenance methods](core.md#parquet-maintenance) directly on a
`ParquetDataset`. For read and write tuning, see
[Performance Optimization](../user-guide/performance.md).

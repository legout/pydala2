# Performance Optimization

Start with the slow query or write. Measure elapsed time, files read or produced,
and peak memory before changing the layout.

## Partitioning Strategies

Use Hive partitions for columns you filter on often. Supply the partition
columns in the input, as shown in [Partitioned Writes](basic-usage.md#partitioned-writes).
Avoid high-cardinality keys that turn one write into many tiny files.

For new writes, set `max_rows_per_file`, `row_group_size`, and `compression`
on `write_to_dataset()`. Test these settings on representative data instead
of assuming a fixed number of rows fits every schema.

## Compaction Operations

### Compacting Small Files

Use `dry_run=True` to see the compaction plan without publishing it. Even
without sorting, the call loads the dataset first. If sidecars are missing,
that load can repair physical Parquet files.

```python
plan = dataset.compact_partitions(
    max_rows_per_file=1_000_000,
    compression="zstd",
    unique=False,
    dry_run=True,
)
```

Omit `dry_run=True` to apply the operation. Ordinary compaction considers
small partitions with multiple files. Specify `unique=False` to preserve
duplicate rows. Deduplication (`unique=True`) does not guarantee business
order and cannot be combined with `sort_by`.

Ordered compaction with `sort_by` and `unique=False` can rewrite every
physical partition, including single-file or oversized partitions, so that
order holds within each partition and across its output files. With `sort_by`,
the call refreshes metadata before planning, which can repair physical files
even during a dry run.

Use `compact_partitions()` on date-based Hive partitions for time-series data.
`compact_by_timeperiod()` without `sort_by` currently expects a
`<dataset name>_metadata` DuckDB relation that normal dataset loading does not
create. Supplying `sort_by` instead delegates to ordered partition compaction
and ignores the time interval.

### Coordinated Rewrite Guarantees

See [Coordinated maintenance guarantees](../api/core.md#coordinated-maintenance-guarantees)
for local and object-store publication, recovery, and metadata refresh.
A dry run shows which guarantee applies to the target filesystem.

### Repartitioning Data

Use [repartition](../api/core.md#repartition) to change the Hive layout. It
preserves duplicates by default; `sort_by` is rejected. Use ordered
compaction separately if physical sort order matters.

## Data Type Optimization

Use [optimize_dtypes](../api/core.md#optimize_dtypes) to preview a narrower
schema. The default strict policy requires value-preserving casts. Review the
proposed schema before publishing it.

## Query Optimization

- Filter and project before exporting. For large outputs, consume
  [batches](../api/table.md) instead of collecting the full result.
- In DuckDB, use `EXPLAIN` with the registered dataset name to inspect the
  query plan. Use `dataset.count_rows()` and `dataset.files` to check the
  size and layout of the input.
- Measure before enabling filesystem caching for repeated remote reads. See
  [Deployment](../advanced/deployment.md) for configuration and connection
  memory limits, and [Performance Tuning](../advanced/performance-tuning.md)
  for a tuning workflow.

# Core Classes

## BaseDataset

`BaseDataset` implements storage and table access for the
[dataset classes](datasets.md). For most workflows, use the class for your
type of input.

| Member | Meaning |
| --- | --- |
| `path`, `name` | Dataset path and name. The name defaults to the last component of the path. |
| `table`, `t` | The loaded `PydalaTable`, or `None` before loading. |
| `ddb_con` | The DuckDB connection; the dataset registers its table by name. |
| `schema`, `columns` | The loaded schema and column names. |
| `files`, `has_files` | The cached input file list and whether it is nonempty. Call `load_files()` to rescan. |
| `partition_names`, `partitions` | Hive partition column names and an Arrow table of partition values, when available. |
| `count_rows()` | Row count, or zero if not loaded. |
| `filter(expr, use="auto")` | A `PydalaTable` with matching rows. |
| `clear_cache()` | Invalidate the dataset's filesystem caches. |

`ParquetDataset.vacuum()` deletes the Parquet files in `dataset.files` and the
metadata sidecars, not just obsolete files. Stop other writers and call
`dataset.load_files()` before vacuuming to empty the dataset; a cached listing
can omit files added by another writer.

## Parquet maintenance

Call maintenance methods directly on `ParquetDataset`. The `Optimize`
subclass remains available through `pydala.dataset` for compatibility.

| Method | Purpose |
| --- | --- |
| `compact_partitions(max_rows_per_file=...)` | Compact eligible Hive partitions. |
| `compact_by_rows(max_rows_per_file=...)` | Compact files to a row limit; delegates to partition compaction for partitioned data. |

See [Performance Optimization](../user-guide/performance.md#compacting-small-files)
for deduplication, ordering, and time-period compaction limitations.

### Coordinated maintenance guarantees

Compaction, repartitioning, and `optimize_dtypes()` use fsspeckit's
plan → stage → validate → publish lifecycle for their rewrites.
Use `dry_run=True` to inspect the upstream plan, including its `guarantee_level`. Native local/POSIX filesystems report `atomic_local` for
cooperating fsspeckit access; other fsspec filesystems report
`best_effort_object_store`, which has no portable distributed lock, atomic
reader visibility, or automatic rollback. Failed rewrites do not refresh
PyDala2 metadata as successful and include upstream recovery details.

These guarantees do not cover individual file replacements during
[schema repair](../user-guide/schema-management.md#repair-existing-files) or
[metadata refresh](../user-guide/metadata.md#refresh-after-external-changes).
Partition compaction loads the dataset before planning, even with
`dry_run=True`. It may repair files if sidecars are missing; ordered
compaction also refreshes existing metadata.

### repartition

`repartition(partitioning_columns, dry_run=True)` plans a new Hive layout.
Omit `dry_run` to publish it and receive the maintenance result as a plain
dictionary. It retains exact duplicate rows by default. The deprecated
`unique=True` selects global deduplication. `sort_by` is not supported; use
ordered compaction separately. Supply `max_rows_per_file` and `compression`
to adjust the output files.

### optimize_dtypes

`optimize_dtypes(dry_run=True)` proposes a target schema with Polars and
returns the target schema and rewrite plan. If the dataset is not loaded,
it loads it first; when sidecars are missing, that load can repair files.
Use `include` or `exclude` to limit which columns are considered. The default `strict=True` uses
value-preserving casts across the full physical dataset; `strict=False`
uses fsspeckit's validated loose policy. Hive partition columns remain path
metadata rather than physical file columns.

On success, the applied operation returns a plain maintenance result.
Failed validation does not publish the dtype rewrite. The error exposes
recovery details.

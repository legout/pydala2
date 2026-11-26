# Performance Tuning

Measure the same query before and after changing its layout or execution engine. Track rows returned, elapsed time and peak memory; a faster query that returns fewer rows is not a valid comparison.

## Reduce what you read

- Filter on columns relevant to the query, then select only the required columns in the export or scanner. Start with [Quick Start](../quick-start.md) for filtering and exports.
- Use Hive partitioning for frequently filtered columns. Avoid high-cardinality partition keys that produce many small files.
- For large results, consume `dataset.table.to_batch_reader(columns=[...])` rather than materializing the entire table. For joins and aggregations across datasets, use DuckDB through a [catalog](../user-guide/catalog-management.md).

## Change the file layout

Use `dry_run=True` to review the [coordinated rewrite plan](../api/core.md#coordinated-maintenance-guarantees)
for compaction, repartitioning, or dtype optimization. A dry run skips the
coordinated publish step, not preparatory loading. Partition compaction
loads the dataset before planning, even without sorting. Dtype optimization
also loads an unloaded dataset. If sidecars are missing, these loads can
repair files. Ordered partition compaction can also do so when it refreshes
existing metadata. For a separate
[per-file schema repair](../user-guide/schema-management.md#repair-existing-files),
preview the affected files with `repair_schema(dry_run=True)` on an existing
instance.

## Tune the connection and storage

Set the DuckDB memory limit on the connection your dataset uses. Enable filesystem caching for repeated remote reads only after measuring them. The [deployment guide](deployment.md) shows where to configure both; the [performance guide](../user-guide/performance.md) covers partitioning and maintenance options.

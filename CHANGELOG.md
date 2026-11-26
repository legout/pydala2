# Changelog

## Unreleased

## 0.29.1 - 2026-11-26

### Fixed

- `mode="overwrite"` no longer deletes its own replacement files or leaves the metadata sidecar naming deleted files, including the first write into an empty dataset. An overwrite that empties a dataset keeps the Hive layout for the next write.
- Row-group statistics are no longer attached to the wrong file when metadata is collected in parallel, so `scan()` stops skipping files that match.
- Appends that change the schema (`alter_schema=True`) are visible to readers immediately.
- Appending with a reused `basename` no longer replaces earlier files.
- A failure in a later batch of a list write raises `PartialWriteError` listing the files already written.
- Partition values equal to the dataset name no longer break metadata updates.
- Scoped `Catalog` edits (`create_table`, `update`, `delete_table`) keep other namespaces. `delete_table(vacuum=True)` validates before deleting data, and `Catalog.load(reload=True)` and `write_table` without `write_args` work.
- `to_batch_reader` and DuckDB-backed `to_arrow_table` honor `filter`; the filtered batch reader streams.
- Polars SQL filters keep large integers exact.
- `compact_by_timeperiod` works without `sort_by` and ignores empty time windows.
- A `threads` setting on a supplied DuckDB connection is no longer overridden.
- Explicit `cache_storage` no longer allocates unused temporary directories.
- A cached timestamp column missing from a new schema is re-inferred.

### Changed

- `basename=` now adds a unique token to file names so repeated appends cannot collide.
- Legacy pickle `_file_metadata` sidecars are no longer loaded; they are rebuilt from the Parquet files.
- The deprecated `mode="delta"` with `delta_subset=None` on a Hive-partitioned dataset raises `ValueError` instead of silently skipping rows. Pass `delta_subset` or use `merge`.
- The publish workflow runs the tests first and only runs on `main`.

## 0.29.0 - 2026-11-26

### Changed

- Made remote storage connectors optional. Install `pydala2[s3]` for S3-compatible storage or `pydala2[sftp]` for SSH/SFTP. The old `legacy` extra is no longer provided.
- Raised minimum versions to DuckDB 1.4.0, pandas 2.3.3, Polars 1.35.2, PyArrow 22.0.0, and SQLGlot 27.16.3.
- Reworked the usage guides to show current dataset and catalog APIs, and explain how to recover after a metadata refresh fails.

### Fixed

- Preserved filesystem roots and absolute local paths, and avoided adding an already specified remote root twice.
- Required Brotli 1.2.1 or newer, which includes the fix for CVE-2026-111497.

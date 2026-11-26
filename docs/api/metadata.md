# Metadata API

Use `ParquetDataset` for managed reads and writes. The metadata classes expose
sidecars and schema repair without a dataset table.

## ParquetDatasetMetadata

`ParquetDatasetMetadata(path, filesystem=..., cached=False)` manages the sidecars for a Parquet directory.

| Member | Result |
| --- | --- |
| `files` | Cached physical Parquet file paths relative to the dataset. |
| `load_files()` | Rescan the physical file listing without rebuilding sidecars. |
| `metadata` | Consolidated `pyarrow.parquet.FileMetaData`, or `None` if no metadata is available. |
| `file_metadata` | Mapping of relative file paths to `FileMetaData` entries. |
| `files_in_metadata`, `files_in_file_metadata` | File paths recorded in the respective sidecars. |
| `update(reload=True)` | Rebuild sidecars, possibly repairing Parquet files. If no data files remain, remove stale sidecars without reopening a table. |
| `repair_schema(schema=..., dry_run=True)` | Return `files` and `schema` without rewriting files or sidecars. |

## PydalaDatasetMetadata

`PydalaDatasetMetadata` adds a DuckDB `metadata_table` relation with one row per Parquet row group. Inspect the relation's columns before composing a query. `update_metadata_table()` rebuilds it from the managed metadata.

On a `ParquetDataset`, use `scan(filter_expr)` to select candidate files and `filter(filter_expr)` to match individual rows. See [Metadata Management](../user-guide/metadata.md) for refresh and query examples.

# Metadata Management

PyDala2 maintains two sidecar files for managed Parquet datasets:

- `_metadata` is consolidated Parquet metadata used to open the dataset.
- `_file_metadata` stores each file's metadata as Brotli-compressed JSON. Each Parquet metadata payload is base64-encoded inside the JSON.

The physical files are authoritative, but `dataset.files` caches its listing.
After an external change, call `dataset.load_files()` before reading
`dataset.files` to see what's on disk. This refreshes the listing without
rewriting data files, rebuilding sidecars, or reopening the table.
`files_in_metadata` and `files_in_file_metadata` show the lists recorded in
the two sidecars, which can differ from what's on disk.

## Refresh after external changes

Rescan with `dataset.load_files()` first. If `dataset.files` is nonempty,
`dataset.load(reload_metadata=True)` rebuilds the sidecars and reopens the
table. It **can also rewrite physical Parquet files** to reconcile schemas
or format versions. The first load without sidecars can also trigger repair.
See [Schema Management](schema-management.md#repair-existing-files) to
preview a schema repair separately.

If the last Parquet file was removed, do not call `load(reload_metadata=True)`:
it can delete stale sidecars and then raise `AttributeError` instead of
opening an empty table. To clear stale sidecars, call
`dataset.update(reload=True)` after confirming there are no Parquet files.
It does not reset the loaded table; discard that instance and open a new one
if you need to represent the empty directory. The new instance has no table.

## Inspect row groups

For a loaded `ParquetDataset`, access the DuckDB relation with `dataset.metadata_table`:

```python
groups = dataset.metadata_table.project("file_path, num_rows").fetchall()
```

`dataset.scan("amount > 100")` selects candidate files based on metadata. Candidate files may contain rows that do not match. Use `dataset.filter("amount > 100")` to select exact rows instead.

# Schema Management

A loaded dataset exposes its Arrow schema as `dataset.schema`. Inspect its fields before appending data, and distinguish Hive partition columns from columns stored inside the Parquet files.

## Validate incoming data

For an Arrow table, compare its schema with the expected input schema using `data.schema.equals(expected_schema)`. Normalize or cast the data before writing. A compatible schema is not a substitute for validating keys, nulls, and business rules.

## Repair existing files

On an existing dataset instance, `dataset.repair_schema(dry_run=True)` returns
`{"files": [...], "schema": target_schema}` without changing data files or
sidecars. `files` contains dataset-relative paths. Pass an Arrow `schema` to
choose the target, or omit it to derive one from the physical files. Opening a
managed dataset without sidecars can itself trigger repair.

Back up the dataset and review the plan before running without `dry_run`.
Repair replaces files one at a time, not in a coordinated dataset-wide
publish. A failed cast leaves that file intact, but other files may already
have changed. Confirm the on-disk state before retrying.

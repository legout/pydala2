# Troubleshooting

## Cannot import PyDala2

The distribution is `pydala2`, but the import is `pydala`. Check that your Python interpreter (3.12 or newer) is the one where you installed the package:

```bash
python --version
python -m pip show pydala2
```

For a complete development environment, run `uv sync --group dev` from the
source checkout. See [Getting Started](../getting-started.md) for installation
and storage extras.

## Local writes fail or rows do not appear

Check that the directory is writable and use an absolute path for local writes.
If rows are missing or a write reports a metadata error, inspect the physical
files before retrying the append: rows may already be on disk. Reopening can
reuse stale sidecars and miss committed files. Follow the listing and recovery
steps in [Metadata Management](../user-guide/metadata.md#refresh-after-external-changes).

## A query uses too much memory

Filter and project before materializing a result. For an existing dataset with `id` and `amount` columns, read the filtered rows in batches:

```python
filtered = dataset.filter("amount >= 20", use="pyarrow")
for batch in filtered.to_batch_reader(columns=["id", "amount"], batch_size=10_000):
    print(batch.num_rows)
```

For a DuckDB query, set the limit on the dataset's connection:

```python
dataset.ddb_con.execute("SET memory_limit='4GB'")
```

This does not limit PyArrow or Polars allocations. If you export a Polars LazyFrame, call `collect()` after selecting the columns and rows you need.

## The incoming schema differs

Compare the Arrow schema of the incoming table with the dataset's schema before appending. A partition column can be present in the dataset schema even when it is not stored in each file. See [Schema Management](../user-guide/schema-management.md) before casting or rewriting existing files.

## S3 cannot open a dataset

Install `pydala2[s3]` and check the credentials, bucket permissions and region used by the application. With a filesystem rooted at a bucket or prefix, pass a path relative to that root. See [Deployment](deployment.md#s3-configuration).

## Catalog tables do not resolve

Check the location of the YAML file, the namespace selected at construction, and the table's `path`, `format` and `filesystem` entries. Use `catalog.all_tables` to inspect the selected namespace and `catalog.sql(...)` to query with automatic loading. Compare with the [Quick Start catalog example](../quick-start.md#query-through-a-catalog).

## Report an issue

Include a minimal reproducible example, the full traceback, your Python and PyDala2 versions, and whether the dataset is local or remote. Remove credentials and private paths before posting to [GitHub Issues](https://github.com/legout/pydala2/issues).

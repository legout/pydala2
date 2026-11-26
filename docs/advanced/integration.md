# Integration Patterns

Use PyDala2 from the Python job that owns the dataset. The library handles Parquet data and metadata; the surrounding application handles scheduling, credentials, authorization, and retries.

- **Batch jobs:** Read filtered input with `dataset.filter(...)`. Write the output with `write_to_dataset()`, or use [keyed merge](../user-guide/merge.md) for an insert or upsert. Do not blindly retry an append after a partial-write error.
- **Analytics and notebooks:** Convert the selected rows to Arrow, Polars or pandas. Keep large results as batches or use a DuckDB relation rather than eagerly converting the whole dataset.
- **SQL across datasets:** Describe the inputs in [a YAML catalog](../user-guide/catalog-management.md) and query them with `Catalog.sql()`.
- **Remote storage:** Configure the filesystem in the application. See [Deployment](deployment.md) for the S3 extra and filesystem setup.

The [Quick Start](../quick-start.md) shows the local write, filter, catalog and merge operations these integrations build on.

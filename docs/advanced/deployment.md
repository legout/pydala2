# Deployment Guide

PyDala2 is a Python library, not a service. Install it in the environment or container that runs your application. Each application process manages its own DuckDB connection unless you supply one.

## Installation

Use Python 3.12 or newer:

```bash
pip install pydala2
```

### Docker Deployment

This Dockerfile assumes your application has an `app.py` and a
`requirements.txt` that includes `pydala2` (or the storage extra you use).

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
CMD ["python", "app.py"]
```

Size memory, temporary disk space and cache storage for your workload; the
library does not set deployment-wide resource limits.

## Storage Configuration

### Local Storage

Pass a writable dataset directory:

```python
from pydala import ParquetDataset

dataset = ParquetDataset("/data/production")
```

### S3 Configuration

Install the S3 connector with `pip install 'pydala2[s3]'`. Provide AWS credentials through your normal S3 credential configuration. Construct a filesystem rooted at the bucket, then use a path relative to that root:

```python
from pydala import FileSystem, ParquetDataset

fs = FileSystem(
    protocol="s3",
    bucket="production-data",
    cached=True,
    cache_storage="/s3_cache",
)
dataset = ParquetDataset("sales", filesystem=fs)
```

For other storage protocols, install the corresponding fsspec backend. See [Filesystem](../api/filesystem.md) for configuration.

## Performance and Operations

- Pass a shared DuckDB connection as `ddb_con` to a catalog and its datasets when they need to share registrations. Set limits on that connection, for example `ddb_con.execute("SET memory_limit='8GB'")`.
- For large results, filter and select the needed columns before exporting. Use `dataset.table.to_batch_reader()` to consume batches instead of materializing the result.
- Configure cache storage on the filesystem or dataset, not globally. Use `dataset.clear_cache()` to invalidate its filesystem caches.
- Manage access, encryption and monitoring in the application and storage backend. For a consistent backup, stop writes or use a storage snapshot; keep the Parquet files, metadata sidecars and catalog YAML together. Test the restore before relying on it.

See [Performance Tuning](performance-tuning.md) for query and layout options.

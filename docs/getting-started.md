# Getting Started

This guide will help you install PyDala2 and set up your environment for working with Parquet datasets.

## Installation

### Prerequisites

PyDala2 requires Python 3.12 or higher. Ensure you have a recent version of Python installed:

```bash
python --version
```

### Install from PyPI

The base install includes the engines for local datasets:

```bash
pip install pydala2
```

Install the extra for your remote storage protocol:

```bash
pip install 'pydala2[s3]'    # S3-compatible storage
pip install 'pydala2[sftp]'  # SSH/SFTP
```

For a development environment with tests, documentation, and notebooks:

```bash
uv sync --group dev
```

### Install from Source

For development or to get the latest features:

```bash
git clone https://github.com/legout/pydala2.git
cd pydala2
pip install -e .
```

## Next Steps

- [Quick Start](quick-start.md): write and reopen a partitioned dataset, filter its rows, and query with a YAML catalog.
- [Deployment](advanced/deployment.md): configure filesystems and DuckDB in your application.
- [Troubleshooting](advanced/troubleshooting.md): diagnose installation and runtime problems.

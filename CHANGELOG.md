# Changelog

## Unreleased

## 0.29.0 - 2026-11-26

### Changed

- Made remote storage connectors optional. Install `pydala2[s3]` for S3-compatible storage or `pydala2[sftp]` for SSH/SFTP. The old `legacy` extra is no longer provided.
- Raised minimum versions to DuckDB 1.4.0, pandas 2.3.3, Polars 1.35.2, PyArrow 22.0.0, and SQLGlot 27.16.3.
- Reworked the usage guides to show current dataset and catalog APIs, and explain how to recover after a metadata refresh fails.

### Fixed

- Preserved filesystem roots and absolute local paths, and avoided adding an already specified remote root twice.
- Required Brotli 1.2.1 or newer, which includes the fix for CVE-2026-111497.

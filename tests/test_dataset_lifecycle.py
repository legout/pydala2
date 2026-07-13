"""Characterization tests for the public ParquetDataset lifecycle.

These tests observe and lock in existing public behavior of managed Parquet
datasets, including construction, physical file discovery, metadata sidecars,
empty-dataset handling, malformed sidecar failure behavior, DuckDB connection
reuse, and the legacy Optimize compatibility subclass.

Baseline failures documented from the existing test suite at the time of
writing (commit ccb1a7b9e251731aca9e76778c3fa87323b7aea4):
- tests/test_misc.py::TestMisc::test_delattr_rec
- tests/test_misc.py::TestMisc::test_get_partitions_from_path
- tests/test_misc.py::TestMisc::test_getattr_rec
- tests/test_misc.py::TestMisc::test_setattr_rec
- tests/test_security.py::TestSqlSecurity::test_escape_sql_literal
- tests/test_security.py::TestPathSecurity::test_sanitize_filename
- tests/test_security.py::TestCredentialHandling::test_get_credentials_redaction
"""

from __future__ import annotations

import posixpath

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pydala import ParquetDataset
from pydala.dataset import Optimize
from pydala.filesystem import FileSystem
from tests.conftest import (
    SIMPLE_SCHEMA,
    assert_metadata_invariants,
    make_simple_table,
)


# --------------------------------------------------------------------------- #
# Construction & query readiness
# --------------------------------------------------------------------------- #


def test_managed_construction_is_query_ready(local_path: str) -> None:
    """A managed ParquetDataset is query-ready when constructed over existing data."""
    fs = FileSystem(bucket=str(local_path), cached=False)

    # Populate the dataset path first.
    ds = ParquetDataset(path="test_dataset", filesystem=fs)
    ds.write_to_dataset(make_simple_table(n_rows=10), mode="append")
    ds.update()
    assert ds.files

    # Constructing over the populated path yields a query-ready object.
    ds2 = ParquetDataset(path="test_dataset", filesystem=fs)
    assert ds2.table is not None, "dataset should expose a public table handle"
    assert ds2.t is not None, "dataset.t should alias the public table handle"

    # Reading back via the public table API works.
    read_back = ds2.t.to_arrow()
    assert read_back.num_rows == 10
    assert list(read_back.schema.names) == list(SIMPLE_SCHEMA.names)

    # The dataset is registered with its DuckDB connection and can be queried.
    result = ds2.ddb_con.sql("SELECT COUNT(*) AS n FROM test_dataset").fetchone()
    assert result[0] == 10


# --------------------------------------------------------------------------- #
# Physical file discovery
# --------------------------------------------------------------------------- #


def test_files_reflects_physical_parquet_files(managed_dataset: ParquetDataset) -> None:
    """dataset.files lists physical Parquet files that actually exist."""
    ds = managed_dataset

    files = ds.files
    assert files, "dataset.files should be non-empty for a populated dataset"

    for file_path in files:
        full_path = posixpath.join(ds.path, file_path)
        assert ds.fs.exists(full_path), f"file does not exist on disk: {file_path}"
        assert file_path.endswith(".parquet")

    # Every Parquet file under the dataset directory is represented.
    physical = set(ds.fs.glob(posixpath.join(ds.path, "**/*.parquet")))
    relative = {p.replace(ds.path, "").lstrip("/") for p in physical}
    assert set(files) == relative, "dataset.files must match physical parquet files"


# --------------------------------------------------------------------------- #
# Metadata sidecars as explicit views
# --------------------------------------------------------------------------- #


def test_metadata_sidecars_are_explicit_views(managed_dataset: ParquetDataset) -> None:
    """Aggregate and per-file metadata are available as explicit sidecars."""
    ds = managed_dataset

    assert ds.has_metadata_file, "aggregate metadata sidecar should exist"
    assert ds.has_file_metadata_file, "per-file metadata sidecar should exist"
    assert ds.has_metadata, "aggregate metadata should be loadable"
    assert ds.has_file_metadata, "per-file metadata should be loadable"

    # Aggregate sidecar is a PyArrow FileMetaData view.
    assert isinstance(ds.metadata, pq.FileMetaData)
    assert ds.metadata.num_row_groups > 0

    # Per-file sidecar is a mapping of relative paths to FileMetaData.
    assert isinstance(ds.file_metadata, dict)
    assert set(ds.file_metadata.keys()) == set(ds.files)
    for meta in ds.file_metadata.values():
        assert isinstance(meta, pq.FileMetaData)

    # Explicit public lists derived from the sidecars match physical discovery.
    assert set(ds.files_in_metadata) == set(ds.files)
    assert set(ds.files_in_file_metadata) == set(ds.files)


# --------------------------------------------------------------------------- #
# Empty dataset behavior
# --------------------------------------------------------------------------- #


def test_valid_empty_dataset(local_path: str) -> None:
    """An empty dataset can be constructed and populated incrementally."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="empty_dataset", filesystem=fs)

    assert ds.files == [], "empty dataset should report no files"
    assert not ds.has_metadata_file
    assert not ds.has_file_metadata_file

    ds.write_to_dataset(make_simple_table(n_rows=5), mode="append")
    ds.update()

    assert len(ds.files) == 1
    assert ds.has_metadata_file
    assert ds.has_file_metadata_file
    assert ds.has_metadata
    assert ds.has_file_metadata
    assert_metadata_invariants(ds)

    ds.write_to_dataset(make_simple_table(n_rows=5, seed=10), mode="append")
    ds.update()
    ds.load()

    assert len(ds.files) == 2
    assert ds.t.to_arrow().num_rows == 10
    assert_metadata_invariants(ds)


# --------------------------------------------------------------------------- #
# Malformed sidecar handling
# --------------------------------------------------------------------------- #


def test_malformed_aggregate_sidecar_raises(local_path: str) -> None:
    """A malformed aggregate sidecar surfaces a contextual error, not silence."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="bad_meta", filesystem=fs)
    ds.write_to_dataset(make_simple_table(n_rows=5), mode="append")
    ds.update()

    assert ds.has_metadata_file, "precondition: aggregate sidecar exists"

    # Corrupt the aggregate sidecar through the public filesystem interface.
    bad_path = posixpath.join(ds.path, "_metadata")
    with fs.open(bad_path, "wb") as f:
        f.write(b"this is not valid parquet metadata")

    # Constructing a new dataset over the corrupted sidecar does not crash, but
    # accessing the metadata must raise a contextual error rather than silently
    # returning partial or stale data.
    ds2 = ParquetDataset(path="bad_meta", filesystem=fs)
    with pytest.raises(pa.ArrowInvalid):
        _ = ds2.metadata


# --------------------------------------------------------------------------- #
# DuckDB connection reuse
# --------------------------------------------------------------------------- #


def test_caller_supplied_duckdb_connection_is_reused(local_path: str) -> None:
    """A caller-supplied DuckDB connection is reused by the dataset."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    con = duckdb.connect()

    ds = ParquetDataset(path="shared_db", filesystem=fs, ddb_con=con)
    assert ds.ddb_con is con, "dataset must reuse the supplied DuckDB connection"

    ds.write_to_dataset(make_simple_table(n_rows=10), mode="append")
    ds.update()
    ds.load()

    # Querying through the original connection sees the registered dataset.
    result = con.sql("SELECT COUNT(*) AS n FROM shared_db").fetchone()
    assert result[0] == 10

    # The public table uses the same connection.
    assert ds.t.ddb_con is con
    result2 = ds.t.sql("SELECT COUNT(*) AS n FROM shared_db").fetchone()
    assert result2[0] == 10


# --------------------------------------------------------------------------- #
# Optimize compatibility subclass
# --------------------------------------------------------------------------- #


def test_optimize_is_constructible_and_compatible(local_path: str) -> None:
    """Optimize is importable, a ParquetDataset subclass, and shares maintenance."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = Optimize(path="opt_compat", filesystem=fs)

    assert isinstance(ds, ParquetDataset)
    assert ds.files == []

    ds.write_to_dataset(make_simple_table(n_rows=10), mode="append")
    ds.update()
    ds.load()

    assert_metadata_invariants(ds)
    assert ds.t.to_arrow().num_rows == 10

    # The inherited maintenance surface is actually usable.
    ds.vacuum()
    assert ds.files == []
    assert not ds.has_metadata_file
    assert not ds.has_file_metadata_file

    ds.write_to_dataset(make_simple_table(n_rows=5), mode="append")
    ds.update()
    ds.load()
    assert ds.t.to_arrow().num_rows == 5
    assert_metadata_invariants(ds)


# --------------------------------------------------------------------------- #
# update() / reload() reconciles metadata
# --------------------------------------------------------------------------- #


def test_update_reconciles_metadata_with_physical_files(local_path: str) -> None:
    """update() reconciles sidecars with physical files written outside the API."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="reconcile", filesystem=fs)

    # Write two physical files without updating metadata sidecars.
    ds.write_to_dataset(
        [make_simple_table(n_rows=10), make_simple_table(n_rows=10, seed=10)],
        mode="append",
        update_metadata=False,
    )
    assert ds.files, "physical files should exist"
    assert len(ds.files) == 2
    assert not ds.has_metadata_file, "metadata should not be updated yet"

    # Reconcile sidecars with physical files.
    ds.update()
    ds.load()

    assert ds.has_metadata_file
    assert ds.has_file_metadata_file
    assert_metadata_invariants(ds)
    assert ds.t.to_arrow().num_rows == 20

    # Remove one physical file outside the API and reload.
    # update(reload=True) must rebuild sidecars from the remaining physical files.
    removed_file = ds.files[0]
    ds.fs.rm(posixpath.join(ds.path, removed_file))

    ds.update(reload=True)
    ds.load()

    assert_metadata_invariants(ds)
    assert len(ds.files) == 1
    assert ds.t.to_arrow().num_rows == 10

    # update(reload=True) also rebuilds when the physical file set is unchanged.
    ds.update(reload=True)
    ds.load()
    assert_metadata_invariants(ds)
    assert len(ds.files) == 1


# --------------------------------------------------------------------------- #
# vacuum()
# --------------------------------------------------------------------------- #


def test_vacuum_leaves_public_file_discovery_empty(local_path: str) -> None:
    """vacuum() removes data files and sidecars; public discovery reports empty."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="vacuumed", filesystem=fs)

    ds.write_to_dataset(make_simple_table(n_rows=10), mode="append")
    ds.write_to_dataset(make_simple_table(n_rows=10, seed=10), mode="append")
    ds.update()

    assert ds.has_files
    assert_metadata_invariants(ds)

    ds.vacuum()

    # Public file discovery is empty after vacuum and cache invalidation.
    assert ds.files == [], "dataset.files should be empty after vacuum"
    assert not ds.has_files
    assert not ds.fs.glob(posixpath.join(ds.path, "**/*.parquet"))

    # Metadata sidecars are removed as well.
    assert not ds.has_metadata_file
    assert not ds.has_file_metadata_file

    # The dataset remains writable after vacuum.
    ds.write_to_dataset(make_simple_table(n_rows=5), mode="append")
    ds.update()
    ds.load()

    assert len(ds.files) == 1
    assert ds.t.to_arrow().num_rows == 5

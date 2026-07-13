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


def assert_core_metadata_invariants(dataset: ParquetDataset) -> None:
    """Assert sidecar invariants without assuming a fixed SIMPLE_SCHEMA."""
    assert dataset.has_metadata_file, "dataset.has_metadata_file is False"
    assert dataset.has_metadata, "dataset.has_metadata is False"
    assert dataset.metadata.num_row_groups > 0, "metadata has zero row groups"
    assert dataset.has_file_metadata_file, "dataset.has_file_metadata_file is False"
    assert dataset.has_file_metadata, "dataset.has_file_metadata is False"
    files = set(dataset.files)
    assert files, "dataset contains no data files"
    assert set(dataset.files_in_metadata) == files, (
        "aggregate metadata files do not match dataset files"
    )
    assert set(dataset.files_in_file_metadata) == files, (
        "file metadata entries do not match dataset files"
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


def test_update_removes_sidecars_when_all_files_deleted(local_path: str) -> None:
    """update() clears stale sidecars when all physical files are removed."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="all_deleted", filesystem=fs)
    ds.write_to_dataset(make_simple_table(n_rows=10), mode="append")
    ds.update()
    assert_metadata_invariants(ds)

    # Remove every physical file outside the API.
    for f in list(ds.files):
        ds.fs.rm(posixpath.join(ds.path, f))

    # A fresh dataset sees no physical files but still has stale sidecars.
    ds2 = ParquetDataset(path="all_deleted", filesystem=fs)
    assert ds2.files == []
    assert ds2.has_metadata_file, "precondition: aggregate sidecar is stale"
    assert ds2.has_file_metadata_file, "precondition: per-file sidecar is stale"
    assert ds2.files_in_metadata, (
        "precondition: aggregate sidecar references deleted files"
    )

    # update() reconciles by removing the stale sidecars.
    ds2.update()
    assert ds2.files == []
    assert not ds2.has_metadata_file
    assert not ds2.has_file_metadata_file
    assert ds2.files_in_metadata == []
    assert ds2.files_in_file_metadata == []

    # The dataset remains writable after sidecar removal.
    ds2.write_to_dataset(make_simple_table(n_rows=5), mode="append")
    ds2.update()
    ds2.load()
    assert ds2.t.to_arrow().num_rows == 5
    assert_metadata_invariants(ds2)


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


# --------------------------------------------------------------------------- #
# Physical-first file discovery (ADR 0001)
# --------------------------------------------------------------------------- #


def test_files_discovers_new_physical_files_when_metadata_stale(
    local_path: str,
) -> None:
    """files reflects newly added physical Parquet files even when sidecars lag."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="stale_add", filesystem=fs)
    ds.write_to_dataset(make_simple_table(n_rows=10), mode="append")
    ds.update()
    assert len(ds.files) == 1

    # Write a second file through the public API without updating sidecars.
    ds.write_to_dataset(
        make_simple_table(n_rows=5, seed=10), mode="append", update_metadata=False
    )

    # A fresh dataset must discover both physical files.
    ds2 = ParquetDataset(path="stale_add", filesystem=fs)
    assert len(ds2.files) == 2, "files must reflect newly written physical files"
    assert len(ds2.files_in_metadata) == 1, "aggregate sidecar is stale"

    # Reconcile: both views agree after update.
    ds2.update()
    assert set(ds2.files_in_metadata) == set(ds2.files)
    assert set(ds2.files_in_file_metadata) == set(ds2.files)


def test_files_drops_deleted_physical_files_when_metadata_stale(
    local_path: str,
) -> None:
    """files drops physically deleted files even when sidecars still list them."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="stale_rm", filesystem=fs)
    ds.write_to_dataset(make_simple_table(n_rows=10), mode="append")
    ds.write_to_dataset(make_simple_table(n_rows=5, seed=10), mode="append")
    ds.update()
    assert len(ds.files) == 2

    # Remove one physical file through the public filesystem interface.
    removed = ds.files[0]
    ds.fs.rm(posixpath.join(ds.path, removed))

    # A fresh dataset must not list the deleted file.
    ds2 = ParquetDataset(path="stale_rm", filesystem=fs)
    assert removed not in ds2.files
    assert len(ds2.files) == 1
    assert len(ds2.files_in_metadata) == 2, "aggregate sidecar still references removed"
    assert removed in ds2.files_in_metadata

    # Reconcile: sidecars drop the stale reference.
    ds2.update(reload=True)
    assert removed not in ds2.files_in_metadata
    assert set(ds2.files_in_metadata) == set(ds2.files)


def test_has_files_reflects_physical_state(local_path: str) -> None:
    """has_files is True for a populated dataset and False for an empty one."""
    fs = FileSystem(bucket=str(local_path), cached=False)

    empty = ParquetDataset(path="empty_has", filesystem=fs)
    assert not empty.has_files

    populated = ParquetDataset(path="populated_has", filesystem=fs)
    populated.write_to_dataset(make_simple_table(n_rows=5), mode="append")
    populated.update()
    assert populated.has_files


# --------------------------------------------------------------------------- #
# No duplicate loading
# --------------------------------------------------------------------------- #


def test_no_duplicate_loading_during_construction(local_path: str) -> None:
    """ParquetDataset.load is called exactly once during construction."""
    from unittest.mock import patch

    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="load_count", filesystem=fs)
    ds.write_to_dataset(make_simple_table(n_rows=10), mode="append")
    ds.update()

    original_load = ParquetDataset.load
    call_count = [0]

    def counting_load(self_inner, *args, **kwargs):
        call_count[0] += 1
        return original_load(self_inner, *args, **kwargs)

    with patch.object(ParquetDataset, "load", counting_load):
        ds2 = ParquetDataset(path="load_count", filesystem=fs)

    assert call_count[0] == 1, f"load() called {call_count[0]} times, expected 1"
    assert ds2.t is not None
    assert ds2.t.to_arrow().num_rows == 10


# --------------------------------------------------------------------------- #
# Schema repair reconciliation
# --------------------------------------------------------------------------- #


def test_update_repairs_divergent_file_schemas(local_path: str) -> None:
    """update() unifies and repairs files with divergent schemas."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="schema_diverge", filesystem=fs)

    # Write two files with different integer widths directly.
    schema_32 = pa.schema([pa.field("id", pa.int32()), pa.field("name", pa.string())])
    schema_64 = pa.schema([pa.field("id", pa.int64()), pa.field("name", pa.string())])

    with fs.open(posixpath.join(ds.path, "a.parquet"), "wb") as f:
        pq.write_table(
            pa.table({"id": [1, 2], "name": ["a", "b"]}, schema=schema_32), f
        )
    with fs.open(posixpath.join(ds.path, "b.parquet"), "wb") as f:
        pq.write_table(
            pa.table({"id": [3, 4], "name": ["c", "d"]}, schema=schema_64), f
        )

    # update() triggers schema repair, promoting int32 to int64.
    ds.update()
    ds.load()

    # Every file now shares the promoted schema.
    for f in ds.files:
        full_path = posixpath.join(local_path, "schema_diverge", f)
        meta = pq.read_metadata(full_path)
        arrow_schema = meta.schema.to_arrow_schema()
        assert arrow_schema.field("id").type == pa.int64()

    # Both sidecars reconcile with the physical files.
    assert_core_metadata_invariants(ds)


# --------------------------------------------------------------------------- #
# Metadata table connection reuse
# --------------------------------------------------------------------------- #


def test_metadata_table_uses_caller_supplied_connection(local_path: str) -> None:
    """The metadata table view is built on the caller-supplied DuckDB connection."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    con = duckdb.connect()

    ds = ParquetDataset(path="shared_meta_db", filesystem=fs, ddb_con=con)
    ds.write_to_dataset(make_simple_table(n_rows=10), mode="append")
    ds.update()
    ds.load(update_metadata=True)

    # The metadata_table view is accessible through the original connection.
    assert ds.metadata_table is not None
    rows = con.sql("SELECT COUNT(*) AS n FROM metadata_table").fetchone()
    assert rows[0] > 0

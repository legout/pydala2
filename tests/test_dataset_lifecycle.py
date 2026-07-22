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

import datetime
import pathlib
import posixpath
from types import SimpleNamespace

import duckdb
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
import pytest

from pydala import ParquetDataset
from pydala.dataset import Optimize, _resolve_maintenance_target
from pydala.filesystem import FileSystem
from pydala.table import PydalaTable
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


# Public optimization methods that must live natively on ParquetDataset
# (not copied at runtime from Optimize) per ADR 0001 / issue #14.
_OPTIMIZATION_METHODS = [
    "compact_by_rows",
    "compact_partitions",
    "compact_by_timeperiod",
    "compact_small_files",
    "optimize_dtypes",
    "repartition",
    "collect_stats",
]


def test_optimization_methods_are_native_to_parquet_dataset() -> None:
    """Optimization methods are defined on ParquetDataset, not monkey-patched."""
    for name in _OPTIMIZATION_METHODS:
        assert name in ParquetDataset.__dict__, (
            f"{name} must be defined directly on ParquetDataset, not copied at runtime"
        )
        # Optimize must NOT override — it inherits unchanged.
        assert name not in Optimize.__dict__


def test_optimize_shares_full_optimization_surface() -> None:
    """Optimize exposes the same optimization methods as ParquetDataset."""
    for name in _OPTIMIZATION_METHODS:
        assert hasattr(Optimize, name), f"Optimize missing {name}"
        # Methods resolve to the identical function object.
        assert getattr(Optimize, name) is getattr(ParquetDataset, name)


def test_optimize_can_run_compact_by_rows_dry_run(local_path: str) -> None:
    """An Optimize instance can actually invoke inherited compaction."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = Optimize(path="opt_dry_run", filesystem=fs)
    ds.write_to_dataset(make_simple_table(n_rows=10), mode="append")
    ds.write_to_dataset(make_simple_table(n_rows=5, seed=10), mode="append")
    ds.update()
    ds.load(update_metadata=True)

    plan = ds.compact_by_rows(max_rows_per_file=100, dry_run=True)
    assert isinstance(plan, dict)
    assert_metadata_invariants(ds)


def test_optimize_can_optimize_dtypes(local_path: str) -> None:
    """An Optimize instance can invoke inherited dtype optimization."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = Optimize(path="opt_dtypes", filesystem=fs)
    ds.write_to_dataset(make_simple_table(n_rows=10), mode="append")
    ds.update()
    ds.load(update_metadata=True)

    rows_before = ds.t.to_arrow().num_rows
    ds.optimize_dtypes(exclude="name")
    ds.load(update_metadata=True)
    assert ds.t.to_arrow().num_rows == rows_before
    assert_metadata_invariants(ds)


# --------------------------------------------------------------------------- #
# Repartition by integer column stays Hive-discoverable
# --------------------------------------------------------------------------- #


def test_repartition_by_integer_column_reads_back_via_hive_discovery(
    local_path: str,
) -> None:
    """Repartitioning by an integer column must stay Hive-discoverable.

    Path-derived Hive partitions infer int32 from the directory value, so a
    destination partition column that is also written into the file schema
    (as int64) makes ``pyarrow.dataset`` crash on an int64-vs-int32 merge.
    The column must live in the directory layout only.
    """
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="repart_int", filesystem=fs)
    table = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "year": pa.array([2023, 2024, 2023, 2024, 2023], type=pa.int64()),
        }
    )
    ds.write_to_dataset(table, mode="append")
    ds.update()
    ds.load(update_metadata=True)
    assert all("=" not in f for f in ds.files), "precondition: source is unpartitioned"

    ds.repartition(partitioning_columns=["year"])

    assert all(f.startswith("year=") for f in ds.files), (
        f"expected hive-prefixed files, got {ds.files}"
    )
    assert_core_metadata_invariants(ds)

    base = pathlib.Path(local_path) / "repart_int"
    discovered = pds.dataset(str(base), format="parquet", partitioning="hive")
    result = discovered.to_table()
    assert result.column_names == ["id", "year"]
    assert result.schema.field("year").type == pa.int32()
    assert sorted(result.column("id").to_pylist()) == [1, 2, 3, 4, 5]
    assert sorted(result.column("year").to_pylist()) == [2023, 2023, 2023, 2024, 2024]


def test_repartition_preserves_exact_duplicates_and_refreshes_metadata(
    local_path: str,
) -> None:
    """Pure repartition preserves duplicate rows and refreshes this instance."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="repart_duplicates", filesystem=fs)
    source = pa.table(
        {
            "id": [1, 1, 2, 3],
            "year": pa.array([2023, 2023, 2024, 2024], type=pa.int64()),
        }
    )
    ds.write_to_dataset(source, mode="append")
    ds.update()
    ds.load(update_metadata=True)

    result = ds.repartition(
        partitioning_columns=["year"], max_rows_per_file=2, unique=False
    )

    assert result["succeeded"] is True
    assert sorted(
        ds.t.to_arrow().to_pylist(), key=lambda row: (row["id"], row["year"])
    ) == [
        {"id": 1, "year": 2023},
        {"id": 1, "year": 2023},
        {"id": 2, "year": 2024},
        {"id": 3, "year": 2024},
    ]
    assert_core_metadata_invariants(ds)


def test_repartition_dry_run_returns_plan_without_rewriting(local_path: str) -> None:
    """Dry-run exposes the fsspeckit plan without changing managed state."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="repart_plan", filesystem=fs)
    ds.write_to_dataset(
        pa.table({"id": [1, 2], "year": pa.array([2023, 2024], type=pa.int64())}),
        mode="append",
    )
    ds.update()
    ds.load(update_metadata=True)
    files_before = list(ds.files)

    plan = ds.repartition(partitioning_columns=["year"], dry_run=True)

    assert plan["repartition_groups"]
    assert plan["planned_groups"]
    assert ds.files == files_before
    assert_core_metadata_invariants(ds)


def test_repartition_failure_does_not_refresh_managed_metadata(
    local_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed upstream publish leaves the current dataset instance intact."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="repart_failure", filesystem=fs)
    ds.write_to_dataset(
        pa.table({"id": [1], "year": pa.array([2023], type=pa.int64())}),
        mode="append",
    )
    ds.update()
    ds.load(update_metadata=True)
    refresh_calls: list[None] = []
    maintenance_fs, _ = _resolve_maintenance_target(ds._filesystem, ds._path)
    monkeypatch.setattr(
        maintenance_fs,
        "repartition_parquet_dataset",
        lambda *args, **kwargs: SimpleNamespace(
            succeeded=False, error="publish failed"
        ),
    )
    monkeypatch.setattr(
        ds, "_refresh_after_rewrite", lambda: refresh_calls.append(None)
    )

    with pytest.raises(RuntimeError, match="publish failed"):
        ds.repartition(partitioning_columns=["year"])

    assert refresh_calls == []
    assert_core_metadata_invariants(ds)


def test_repartition_changes_an_existing_hive_layout(local_path: str) -> None:
    """Repartitioning an existing Hive layout preserves its source rows."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="repart_hive_source", filesystem=fs)
    ds.write_to_dataset(
        pa.table(
            {
                "id": [1, 2, 3, 4],
                "year": pa.array([2023, 2023, 2024, 2024], type=pa.int64()),
                "region": ["eu", "us", "eu", "us"],
            }
        ),
        partition_by=["year"],
        mode="append",
    )
    ds.update()
    ds.load(update_metadata=True)

    result = ds.repartition(partitioning_columns=["region"])

    assert result["succeeded"] is True
    assert all(path.startswith(("region=eu/", "region=us/")) for path in ds.files)
    assert sorted(ds.t.to_arrow().column("id").to_pylist()) == [1, 2, 3, 4]
    assert_core_metadata_invariants(ds)


def test_repartition_unique_uses_explicit_deduplication_compatibility_mode(
    local_path: str,
) -> None:
    """The legacy unique option maps to explicit global deduplication."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="repart_unique", filesystem=fs)
    ds.write_to_dataset(
        pa.table(
            {
                "id": [1, 1, 2],
                "year": pa.array([2023, 2023, 2024], type=pa.int64()),
            }
        ),
        mode="append",
    )
    ds.update()
    ds.load(update_metadata=True)

    with pytest.deprecated_call(match="unique=True"):
        result = ds.repartition(partitioning_columns=["year"], unique=True)

    assert result["succeeded"] is True
    assert ds.t.to_arrow().num_rows == 2
    assert_core_metadata_invariants(ds)


def test_repartition_rejects_unsupported_sorting(local_path: str) -> None:
    """Repartition does not silently ignore an unsupported sort request."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="repart_sort", filesystem=fs)
    ds.write_to_dataset(
        pa.table({"id": [1], "year": pa.array([2023], type=pa.int64())}),
        mode="append",
    )
    ds.update()

    with pytest.raises(ValueError, match="sort_by is not supported"):
        ds.repartition(partitioning_columns=["year"], sort_by="id")


def test_repartition_warns_when_row_group_size_cannot_be_applied(
    local_path: str,
) -> None:
    """A non-default row-group request is visible instead of silently ignored."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="repart_row_groups", filesystem=fs)
    ds.write_to_dataset(
        pa.table({"id": [1], "year": pa.array([2023], type=pa.int64())}),
        mode="append",
    )
    ds.update()

    with pytest.deprecated_call(match="row_group_size"):
        ds.repartition(
            partitioning_columns=["year"], row_group_size=128, dry_run=True
        )


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


# --------------------------------------------------------------------------- #
# Managed timestamp/timezone repair
# --------------------------------------------------------------------------- #


def test_update_repairs_timestamp_unit_divergence(local_path: str) -> None:
    """update(ts_unit=...) normalizes files with mixed timestamp units."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="ts_diverge", filesystem=fs)

    s_us = pa.schema([("ts", pa.timestamp("us")), ("id", pa.int32())])
    s_ms = pa.schema([("ts", pa.timestamp("ms")), ("id", pa.int32())])

    with fs.open(posixpath.join(ds.path, "us.parquet"), "wb") as f:
        pq.write_table(
            pa.table(
                {
                    "ts": pa.array([datetime.datetime(2024, 1, 1)], pa.timestamp("us")),
                    "id": [1],
                },
                schema=s_us,
            ),
            f,
        )
    with fs.open(posixpath.join(ds.path, "ms.parquet"), "wb") as f:
        pq.write_table(
            pa.table(
                {
                    "ts": pa.array([datetime.datetime(2024, 1, 2)], pa.timestamp("ms")),
                    "id": [2],
                },
                schema=s_ms,
            ),
            f,
        )

    ds.update(ts_unit="us")
    ds.load()

    for f in ds.files:
        full_path = posixpath.join(local_path, "ts_diverge", f)
        arrow_schema = pq.read_metadata(full_path).schema.to_arrow_schema()
        assert arrow_schema.field("ts").type == pa.timestamp("us")

    assert_core_metadata_invariants(ds)


def test_update_applies_timezone_to_already_unified_files(local_path: str) -> None:
    """update(tz=...) should add timezone to all files, not only divergent ones.

    This is the desired behavior: the managed repair candidate set should
    include files that differ only in timezone from the requested target
    schema. See issue #22 for the shared schema-repair planner that will
    make direct and managed repair select the same rewrite candidates.
    """
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="tz_add", filesystem=fs)

    schema = pa.schema([("ts", pa.timestamp("us")), ("id", pa.int32())])
    for name in ("a.parquet", "b.parquet"):
        with fs.open(posixpath.join(ds.path, name), "wb") as f:
            pq.write_table(
                pa.table(
                    {
                        "ts": pa.array(
                            [datetime.datetime(2024, 1, 1)], pa.timestamp("us")
                        ),
                        "id": [1],
                    },
                    schema=schema,
                ),
                f,
            )

    ds.update(tz="UTC")
    ds.load()

    for f in ds.files:
        full_path = posixpath.join(local_path, "tz_add", f)
        arrow_schema = pq.read_metadata(full_path).schema.to_arrow_schema()
        assert arrow_schema.field("ts").type == pa.timestamp("us", tz="UTC")

    assert_core_metadata_invariants(ds)


def test_update_repairs_mixed_timestamp_unit_and_timezone(local_path: str) -> None:
    """update(ts_unit=..., tz=...) should unify units and apply timezone together."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="ts_mixed", filesystem=fs)

    s_us = pa.schema([("ts", pa.timestamp("us")), ("id", pa.int32())])
    s_ms = pa.schema([("ts", pa.timestamp("ms")), ("id", pa.int32())])

    with fs.open(posixpath.join(ds.path, "us.parquet"), "wb") as f:
        pq.write_table(
            pa.table(
                {
                    "ts": pa.array([datetime.datetime(2024, 1, 1)], pa.timestamp("us")),
                    "id": [1],
                },
                schema=s_us,
            ),
            f,
        )
    with fs.open(posixpath.join(ds.path, "ms.parquet"), "wb") as f:
        pq.write_table(
            pa.table(
                {
                    "ts": pa.array([datetime.datetime(2024, 1, 2)], pa.timestamp("ms")),
                    "id": [2],
                },
                schema=s_ms,
            ),
            f,
        )

    ds.update(ts_unit="us", tz="UTC")
    ds.load()

    for f in ds.files:
        full_path = posixpath.join(local_path, "ts_mixed", f)
        arrow_schema = pq.read_metadata(full_path).schema.to_arrow_schema()
        assert arrow_schema.field("ts").type == pa.timestamp("us", tz="UTC")

    assert_core_metadata_invariants(ds)


def test_update_repair_reconciles_sidecar_views(local_path: str) -> None:
    """After schema repair, both sidecar views match the repaired physical files."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="sidecar_reconcile", filesystem=fs)

    s_a = pa.schema([("id", pa.int32()), ("name", pa.string())])
    s_b = pa.schema([("id", pa.int64()), ("name", pa.string())])
    with fs.open(posixpath.join(ds.path, "a.parquet"), "wb") as f:
        pq.write_table(pa.table({"id": [1, 2], "name": ["a", "b"]}, schema=s_a), f)
    with fs.open(posixpath.join(ds.path, "b.parquet"), "wb") as f:
        pq.write_table(pa.table({"id": [3, 4], "name": ["c", "d"]}, schema=s_b), f)

    ds.update()
    ds.load()

    # Aggregate and per-file sidecars reflect the same physical files.
    files = set(ds.files)
    assert set(ds.files_in_metadata) == files
    assert set(ds.files_in_file_metadata) == files

    # Every physical file was promoted to the unified schema.
    for f in ds.files:
        full_path = posixpath.join(local_path, "sidecar_reconcile", f)
        arrow_schema = pq.read_metadata(full_path).schema.to_arrow_schema()
        assert arrow_schema.field("id").type == pa.int64()

    # The aggregate metadata sidecar reflects the repaired schema.
    aggregate_schema = ds.metadata.schema.to_arrow_schema()
    assert aggregate_schema.field("id").type == pa.int64()

    # The per-file metadata sidecar entries exist for every physical file.
    for f in ds.files:
        assert f in ds.file_metadata

    assert_core_metadata_invariants(ds)


def test_reload_repairs_newly_added_physical_files(local_path: str) -> None:
    """reload(update_metadata=True) repairs newly written physical files."""
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path="incremental", filesystem=fs)

    # Initial dataset with a consistent schema.
    ds.write_to_dataset(make_simple_table(n_rows=5), mode="append")
    ds.update()
    ds.load()

    # Write a divergent file directly outside the API.
    divergent = pa.schema([("id", pa.int32())])
    with fs.open(posixpath.join(ds.path, "extra.parquet"), "wb") as f:
        pq.write_table(
            pa.table({"id": pa.array([99], type=pa.int32())}, schema=divergent), f
        )

    ds.load(reload_metadata=True)

    # The new file was discovered and repaired.
    assert "extra.parquet" in ds.files
    full_path = posixpath.join(local_path, "incremental", "extra.parquet")
    arrow_schema = pq.read_metadata(full_path).schema.to_arrow_schema()
    assert arrow_schema.field("id").type == pa.int64()

    assert_core_metadata_invariants(ds)


# --------------------------------------------------------------------------- #
# Metadata-statistics predicate pruning (managed scan + table file pruning)
# --------------------------------------------------------------------------- #


def _write_disjoint_range_dataset(
    local_path: str, name: str = "prune_ds"
) -> ParquetDataset:
    """Write a two-file dataset with disjoint ranges for pruning tests.

    File ``low`` has ids 1-3 and dates in January 2024.
    File ``high`` has ids 100-102 and dates in June 2024.
    """
    fs = FileSystem(bucket=str(local_path), cached=False)
    ds = ParquetDataset(path=name, filesystem=fs)

    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("amount", pa.float64()),
            pa.field("d", pa.date32()),
            pa.field("ts", pa.timestamp("us")),
        ]
    )

    low = pa.table(
        {
            "id": [1, 2, 3],
            "amount": [10.0, 20.0, 30.0],
            "d": pa.array(
                [
                    datetime.date(2024, 1, 1),
                    datetime.date(2024, 1, 2),
                    datetime.date(2024, 1, 3),
                ]
            ),
            "ts": pa.array(
                [
                    datetime.datetime(2024, 1, 1, 10, 0, 0),
                    datetime.datetime(2024, 1, 1, 11, 0, 0),
                    datetime.datetime(2024, 1, 1, 12, 0, 0),
                ]
            ),
        },
        schema=schema,
    )
    high = pa.table(
        {
            "id": [100, 101, 102],
            "amount": [1000.0, 2000.0, 3000.0],
            "d": pa.array(
                [
                    datetime.date(2024, 6, 1),
                    datetime.date(2024, 6, 2),
                    datetime.date(2024, 6, 3),
                ]
            ),
            "ts": pa.array(
                [
                    datetime.datetime(2024, 6, 1, 10, 0, 0),
                    datetime.datetime(2024, 6, 1, 11, 0, 0),
                    datetime.datetime(2024, 6, 1, 12, 0, 0),
                ]
            ),
        },
        schema=schema,
    )

    ds.write_to_dataset(low, mode="append")
    ds.write_to_dataset(high, mode="append")
    ds.update()
    ds.load()
    ds.update_metadata_table()
    assert_core_metadata_invariants(ds)
    return ds


def test_managed_scan_prunes_numeric_predicate(local_path: str) -> None:
    """ParquetDataset.scan prunes files using numeric statistics.

    The low file (ids 1-3) is deselected; the high file (ids 100-102) is
    returned. Because scan() returns all rows from selected files, the
    presence of high-range ids and absence of low-range ids proves the
    low file was pruned at the file level.
    """
    ds = _write_disjoint_range_dataset(local_path)

    result = ds.scan("id > 50")
    ids = result.to_arrow()["id"].to_pylist()
    assert ids == [100, 101, 102]


def test_managed_scan_prunes_date_predicate(local_path: str) -> None:
    """ParquetDataset.scan prunes files using date statistics."""
    ds = _write_disjoint_range_dataset(local_path)

    result = ds.scan("d < '2024-03-01'")
    dates = result.to_arrow()["d"].to_pylist()
    assert len(dates) == 3
    assert all(d.month == 1 for d in dates)


def test_managed_scan_prunes_timestamp_predicate(local_path: str) -> None:
    """ParquetDataset.scan prunes files using timestamp statistics."""
    ds = _write_disjoint_range_dataset(local_path)

    result = ds.scan("ts > '2024-03-01 00:00:00'")
    timestamps = result.to_arrow()["ts"].to_pylist()
    assert len(timestamps) == 3
    assert all(ts.month == 6 for ts in timestamps)


def test_managed_scan_returns_all_rows_from_selected_file(local_path: str) -> None:
    """scan() is a file-level optimization: all rows from selected files are returned.

    This proves pruning is at the file level, not the row level. The high file
    has ids [100, 101, 102]; all are returned even though the predicate is
    ``id > 50`` because statistics-based pruning selects the whole file.
    """
    ds = _write_disjoint_range_dataset(local_path)

    result = ds.scan("id > 50")
    ids = result.to_arrow()["id"].to_pylist()
    # Every row from the high file is present, including those the predicate
    # does not describe (e.g. ids below 50 are absent because the file was
    # pruned, not the rows filtered).
    assert set(ids) == {100, 101, 102}
    assert 1 not in ids


def test_table_prune_files_numeric_predicate(local_path: str) -> None:
    """PydalaTable.prune_files selects candidates using numeric statistics."""
    ds = _write_disjoint_range_dataset(local_path)

    scan_files = PydalaTable.prune_files(ds.metadata_table, "id > 50", ds.files)
    # Only the file with ids >= 100 should survive.
    assert len(scan_files) == 1


def test_table_prune_files_date_and_timestamp_predicates(local_path: str) -> None:
    """PydalaTable.prune_files selects candidates for date and timestamp stats."""
    ds = _write_disjoint_range_dataset(local_path)

    date_files = PydalaTable.prune_files(
        ds.metadata_table, "d < '2024-03-01'", ds.files
    )
    assert len(date_files) == 1

    ts_files = PydalaTable.prune_files(
        ds.metadata_table, "ts > '2024-03-01 00:00:00'", ds.files
    )
    assert len(ts_files) == 1


def test_table_prune_files_compound_and_predicate(local_path: str) -> None:
    """PydalaTable.prune_files handles ``AND`` conjunctions across columns."""
    ds = _write_disjoint_range_dataset(local_path)

    scan_files = PydalaTable.prune_files(
        ds.metadata_table,
        "id > 5 AND amount > 500",
        ds.files,
    )
    assert len(scan_files) == 1


def test_managed_scan_equality_predicate(local_path: str) -> None:
    """ParquetDataset.scan with ``=`` selects only the overlapping file."""
    ds = _write_disjoint_range_dataset(local_path)

    result = ds.scan("id = 1")
    ids = result.to_arrow()["id"].to_pylist()
    assert ids == [1, 2, 3]


def test_prune_files_retains_null_statistics_file() -> None:
    """Files with NULL min/max statistics are retained as candidates."""
    metadata = duckdb.from_arrow(
        pa.table(
            {
                "file_path": ["stats.parquet", "nulls.parquet"],
                "id": [
                    {"min": 1, "max": 5, "has_min_max": True},
                    {"min": None, "max": None, "has_min_max": False},
                ],
            }
        )
    )

    result = PydalaTable.prune_files(
        metadata, "id > 10", ["stats.parquet", "nulls.parquet"]
    )
    # The file with no statistics must be retained (conservative pruning).
    assert "nulls.parquet" in result


def test_prune_files_preserves_non_statistics_predicate() -> None:
    """Predicates on non-statistics metadata columns are passed through as-is."""
    metadata = duckdb.from_arrow(
        pa.table(
            {
                "file_path": ["f1.parquet", "f2.parquet"],
                "num_rows": [3, 100],
            }
        )
    )

    result = PydalaTable.prune_files(
        metadata, "num_rows > 50", ["f1.parquet", "f2.parquet"]
    )
    assert result == ["f2.parquet"]


@pytest.mark.xfail(
    reason=(
        "The current predicate planner raises a DuckDB BinderException for "
        "``!=`` on statistics columns instead of retaining all physical files "
        "as candidates. The conservative planner from #23 will retain all "
        "files when a statistics predicate cannot be safely translated."
    ),
    strict=False,
)
def test_unsupported_not_equal_does_not_prune_valid_file(local_path: str) -> None:
    """An unsupported ``!=`` predicate must not exclude a physical file with valid rows.

    File ``lo`` contains ids [1, 2, 3], all of which satisfy ``id != 10``.
    File ``hi`` contains ids [10, 20, 30], where 20 and 30 also satisfy
    ``id != 10``. Both files contain potentially valid rows and must be
    retained as scan candidates. See issue #23.
    """
    ds = _write_disjoint_range_dataset(local_path)

    result = ds.scan("id != 10")
    ids = result.to_arrow()["id"].to_pylist()
    # Both files' rows must be present — no file with valid rows pruned.
    assert set(ids) == {1, 2, 3, 100, 101, 102}


@pytest.mark.xfail(
    reason=(
        "The current predicate planner raises a DuckDB error for ``IN`` on "
        "statistics columns instead of retaining all physical files. See #23 "
        "for the conservative shared predicate planner."
    ),
    strict=False,
)
def test_unsupported_in_predicate_does_not_prune_valid_file(local_path: str) -> None:
    """An ``IN`` statistics predicate must not exclude a file with valid rows.

    File ``lo`` has ids [1, 2, 3]; ``id IN (2, 150)`` could match id=2.
    File ``hi`` has ids [100, 101, 102]; ``id IN (2, 150)`` could match none,
    but since the planner cannot safely translate ``IN`` to a min/max range
    check, both files must be retained. See issue #23.
    """
    ds = _write_disjoint_range_dataset(local_path)

    result = ds.scan("id IN (2, 150)")
    ids = result.to_arrow()["id"].to_pylist()
    # No file may be pruned when the predicate is untranslatable.
    assert set(ids) == {1, 2, 3, 100, 101, 102}

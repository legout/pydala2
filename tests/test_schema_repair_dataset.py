"""Dataset-level schema repair contract tests (issue #36).

These tests cover the dataset-facing repair workflow:

* ``ParquetDataset.repair_schema(dry_run=True)`` plans from a fresh raw
  physical snapshot without mutating files or sidecars.
* ``ParquetDataset.repair_schema()`` rewrites physical files to the canonical
  target and refreshes file metadata, aggregate metadata, Arrow dataset/table
  state, and caches.
* Failed casts leave the original physical file intact and raise.
* ``file_schemas`` is the canonicalized view while ``raw_file_schemas`` exposes
  the raw physical schema actually stored on disk.
* Mixed ``string`` / ``large_string`` files in one Hive partition can be
  repaired and then planned for compaction.
"""

from __future__ import annotations

import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pydala import ParquetDataset
from pydala.filesystem import FileSystem


def _assert_metadata_consistent(ds: ParquetDataset) -> None:
    """Schema-agnostic variant of conftest's metadata-invariant check."""
    assert ds.has_metadata_file, "dataset.has_metadata_file is False"
    assert ds.has_metadata, "dataset.has_metadata is False"
    assert ds.metadata is not None and ds.metadata.num_row_groups > 0, (
        "metadata has zero row groups"
    )
    assert ds.has_file_metadata_file, "dataset.has_file_metadata_file is False"
    assert ds.has_file_metadata, "dataset.has_file_metadata is False"

    files = set(ds.files)
    metadata_files = set(ds.files_in_metadata)
    file_metadata_files = set(ds.files_in_file_metadata)
    assert files, "dataset contains no data files"
    assert metadata_files == files, (
        f"aggregate metadata files {metadata_files} != dataset files {files}"
    )
    assert file_metadata_files == files, (
        f"file metadata entries {file_metadata_files} != dataset files {files}"
    )


def _write_large_string_file(filesystem, path: str, value: str = "large") -> None:
    """Write a Parquet file with a ``large_string`` column directly to disk."""
    table = pa.table({"x": pa.array([value], type=pa.large_string())})
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    filesystem.pipe(path, buf.getvalue())


def test_file_schemas_is_canonicalized_while_raw_view_exposes_physical_types(
    local_path: str,
) -> None:
    """file_schemas folds large types; raw_file_schemas does not (issue #36)."""
    dataset_path = "ds"
    fs = FileSystem(bucket=local_path, cached=False)
    base = pathlib.Path(local_path) / dataset_path
    base.mkdir(parents=True, exist_ok=True)

    # Construct with a canonical string file (init converges it harmlessly).
    pq.write_table(pa.table({"x": pa.array(["a"], type=pa.string())}), str(base / "normal.parquet"))
    ds = ParquetDataset(path=dataset_path, filesystem=fs)

    # Inject a large_string file *after* construction so the load-time repair
    # does not normalize it, then collect sidecar metadata (no repair).
    _write_large_string_file(fs, f"{dataset_path}/large.parquet")
    ds.clear_cache()
    ds.load_files()
    ds._file_metadata = None
    ds.update_file_metadata()

    normal_rel = next(f for f in ds.files if f.endswith("normal.parquet"))
    large_rel = next(f for f in ds.files if f.endswith("large.parquet"))

    canonical = ds.file_schemas
    raw = ds.raw_file_schemas

    assert canonical[normal_rel].field("x").type == pa.string()
    assert canonical[large_rel].field("x").type == pa.string()
    # The raw view keeps the physical difference visible.
    assert raw[normal_rel].field("x").type == pa.string()
    assert raw[large_rel].field("x").type == pa.large_string()


def test_dataset_repair_schema_dry_run_selects_large_string_file(local_path: str) -> None:
    """Dry-run identifies the raw large_string file and mutates nothing."""
    dataset_path = "ds"
    fs = FileSystem(bucket=local_path, cached=False)
    base = pathlib.Path(local_path) / dataset_path
    base.mkdir(parents=True, exist_ok=True)

    pq.write_table(pa.table({"x": pa.array(["a"], type=pa.string())}), str(base / "normal.parquet"))
    ds = ParquetDataset(path=dataset_path, filesystem=fs)

    _write_large_string_file(fs, f"{dataset_path}/large.parquet", value="hello")

    large_schema_before = pq.read_schema(str(base / "large.parquet"))
    file_meta_before = ds.has_file_metadata_file

    plan = ds.repair_schema(dry_run=True)

    assert isinstance(plan, dict)
    assert set(plan.keys()) == {"files", "schema"}
    # The method rediscovers physical files, so the injected file (absent from
    # the cached ds.files) is still detected.
    assert plan["files"] == ["large.parquet"]
    assert plan["schema"] == pa.schema([("x", pa.string())])
    # Dry-run must not mutate the physical file or the sidecars.
    assert pq.read_schema(str(base / "large.parquet")) == large_schema_before
    assert ds.has_file_metadata_file == file_meta_before


def test_dataset_repair_schema_rewrites_and_refreshes_all_state(local_path: str) -> None:
    """Real repair rewrites files, preserves values and refreshes all state."""
    dataset_path = "ds"
    fs = FileSystem(bucket=local_path, cached=False)
    base = pathlib.Path(local_path) / dataset_path
    base.mkdir(parents=True, exist_ok=True)

    pq.write_table(pa.table({"x": pa.array(["a"], type=pa.string())}), str(base / "normal.parquet"))
    ds = ParquetDataset(path=dataset_path, filesystem=fs)
    rows_before = ds.table.to_arrow().num_rows

    _write_large_string_file(fs, f"{dataset_path}/large.parquet", value="keepme")

    result = ds.repair_schema()

    assert result is None
    # Physical files are now canonical string.
    assert pq.read_schema(str(base / "normal.parquet")).field("x").type == pa.string()
    assert pq.read_schema(str(base / "large.parquet")).field("x").type == pa.string()
    # Values are preserved across the in-place rewrite.
    assert pq.read_table(str(base / "large.parquet")).column("x").to_pylist() == ["keepme"]
    # The injected row is now visible in the refreshed Arrow table.
    assert ds.table.to_arrow().num_rows == rows_before + 1
    assert ds.table.to_arrow().schema.field("x").type == pa.string()
    # All raw physical schemas are now canonical.
    assert all(
        schema.field("x").type == pa.string() for schema in ds.raw_file_schemas.values()
    )
    _assert_metadata_consistent(ds)


def test_dataset_repair_schema_failed_cast_leaves_file_intact(local_path: str) -> None:
    """A failed cast does not replace the source file and raises."""
    dataset_path = "ds"
    fs = FileSystem(bucket=local_path, cached=False)
    base = pathlib.Path(local_path) / dataset_path
    base.mkdir(parents=True, exist_ok=True)

    pq.write_table(
        pa.table({"id": pa.array([2**40], type=pa.int64())}), str(base / "overflow.parquet")
    )
    ds = ParquetDataset(path=dataset_path, filesystem=fs)

    original_schema = pq.read_schema(str(base / "overflow.parquet"))
    original_values = pq.read_table(str(base / "overflow.parquet")).to_pydict()

    with pytest.raises((pa.ArrowInvalid, pa.ArrowTypeError)):
        ds.repair_schema(schema=pa.schema([("id", pa.int32())]))

    # Source file untouched.
    assert pq.read_schema(str(base / "overflow.parquet")) == original_schema
    assert pq.read_table(str(base / "overflow.parquet")).to_pydict() == original_values


def test_dataset_repair_schema_then_compact_partition_plan(local_path: str) -> None:
    """Mixed string/large_string in one Hive partition repairs then compacts (issue #36)."""
    dataset_path = "partitioned"
    fs = FileSystem(bucket=local_path, cached=False)

    ds = ParquetDataset(path=dataset_path, partitioning="hive", filesystem=fs)
    # Write canonical string files in the same partition.
    ds.write_to_dataset(
        pa.table({"id": [1, 2], "x": ["a", "b"], "region": ["north", "north"]}),
        mode="append",
        partition_by=["region"],
    )

    partition_dir = pathlib.Path(local_path) / dataset_path / "region=north"
    assert partition_dir.exists()
    assert sorted(partition_dir.glob("*.parquet"))

    # Inject a large_string file into the same physical partition.
    injected = partition_dir / "injected.parquet"
    pq.write_table(
        pa.table({"id": [3], "x": pa.array(["c"], type=pa.large_string())}),
        str(injected),
    )

    # Before repair, the dry-run plan rediscovers and detects the injected
    # large_string file even though it is absent from the cached ds.files.
    plan = ds.repair_schema(dry_run=True)
    assert plan is not None
    assert any(f.endswith("injected.parquet") for f in plan["files"])

    # Repair converges every physical file to canonical string.
    ds.repair_schema()

    assert all(
        schema.field("x").type == pa.string() for schema in ds.raw_file_schemas.values()
    )
    _assert_metadata_consistent(ds)

    # Compaction planning succeeds on the now-uniform partition.
    compact_plan = ds.compact_partitions(
        max_rows_per_file=100, unique=False, dry_run=True
    )
    assert isinstance(compact_plan, list)
    assert len(compact_plan) >= 1

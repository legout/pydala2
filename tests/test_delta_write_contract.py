"""Contracts for deprecated ``mode="delta"`` insert-if-absent writes.

The public write API delegates delta operations to
``ParquetDataset.merge(strategy="insert")``. Explicit ``delta_subset`` values
become merge keys; when omitted, identity is inferred from common non-null
source and target columns. The compatibility path is independent of dataset
load state, rejects null source keys, and resolves duplicate source keys using
the last row.
"""

from __future__ import annotations

import datetime
import pathlib

import pyarrow as pa
import pyarrow.dataset as pds
import pytest

from pydala import ParquetDataset
from pydala.filesystem import FileSystem


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _table(**columns: list) -> pa.Table:
    """Build a PyArrow Table from keyword columns."""
    return pa.table(columns)


def _dataset(local_path: str) -> ParquetDataset:
    """Construct a fresh ParquetDataset rooted at ``local_path/ds``."""
    return ParquetDataset(
        path="ds",
        filesystem=FileSystem(bucket=local_path, cached=False),
    )


def _rows(local_path: str) -> list[dict]:
    """Read every physical parquet file under the dataset directory.

    Reads directly via ``pyarrow.dataset`` so the assertion reflects the
    ground-truth bytes on disk, independent of pydala's metadata sidecars.
    The *write* still goes through the public ``write_to_dataset`` API.
    """
    base = pathlib.Path(local_path) / "ds"
    table = pds.dataset(str(base), format="parquet", partitioning="hive").to_table()
    return sorted(
        table.to_pylist(),
        key=lambda r: tuple(
            (v is None, "" if v is None else str(v)) for v in r.values()
        ),
    )


def _loaded_dataset(local_path: str) -> ParquetDataset:
    """Return a ParquetDataset materialized over existing files."""
    ds = _dataset(local_path)
    assert ds.is_loaded, "fixture invariant: existing dataset is loaded"
    return ds


# --------------------------------------------------------------------------- #
# delta_subset=[key]: insert-if-absent (the canonical / verified example)
# --------------------------------------------------------------------------- #


def test_delta_subset_key_preserves_target_and_appends_only_new_keys(
    local_path: str,
) -> None:
    """``delta_subset=[key]`` keeps matching target rows and appends new keys.

    Target ``{1: old}``, source ``{1: new, 2: two}`` produces
    ``{1: old, 2: two}``.
    The existing key-1 row is untouched; only the absent key 2 is inserted.
    """
    ds = _dataset(local_path)
    ds.write_to_dataset(_table(id=[1], val=["old"]), mode="append")

    loaded = _loaded_dataset(local_path)
    loaded.write_to_dataset(
        _table(id=[1, 2], val=["new", "two"]),
        mode="delta",
        delta_subset=["id"],
    )

    assert _rows(local_path) == [
        {"id": 1, "val": "old"},
        {"id": 2, "val": "two"},
    ]


def test_string_delta_subset_is_normalized_as_explicit_key(local_path: str) -> None:
    """A bare-string ``delta_subset`` behaves like a one-column key list."""
    ds = _dataset(local_path)
    ds.write_to_dataset(_table(id=[1], val=["old"]), mode="append")

    loaded = _loaded_dataset(local_path)
    with pytest.warns(DeprecationWarning):
        loaded.write_to_dataset(
            _table(id=[1, 2], val=["new", "two"]),
            mode="delta",
            delta_subset="id",
        )

    assert _rows(local_path) == [
        {"id": 1, "val": "old"},
        {"id": 2, "val": "two"},
    ]


# --------------------------------------------------------------------------- #
# delta_subset=None: exact-row identity on common non-null columns
# --------------------------------------------------------------------------- #


def test_delta_subset_none_suppresses_exact_duplicate_rows(local_path: str) -> None:
    """With ``delta_subset=None`` an exact row already present is suppressed.

    Identity is over the common non-null columns: the source row ``(1, "a")``
    exists verbatim in the target, so it is dropped; the genuinely new row
    ``(3, "c")`` is appended.
    """
    ds = _dataset(local_path)
    ds.write_to_dataset(_table(id=[1, 2], val=["a", "b"]), mode="append")

    loaded = _loaded_dataset(local_path)
    loaded.write_to_dataset(_table(id=[1, 3], val=["a", "c"]), mode="delta")

    assert _rows(local_path) == [
        {"id": 1, "val": "a"},
        {"id": 2, "val": "b"},
        {"id": 3, "val": "c"},
    ]


def test_delta_subset_none_appends_row_with_same_key_but_different_values(
    local_path: str,
) -> None:
    """``delta_subset=None`` uses whole-row identity, not key identity.

    Source ``(1, "new")`` shares key 1 with target ``(1, "old")`` but is not an
    exact row duplicate, so it is appended rather than suppressing the target.
    This distinguishes row-identity (``delta_subset=None``) from key-identity
    (``delta_subset=["id"]``).
    """
    ds = _dataset(local_path)
    ds.write_to_dataset(_table(id=[1], val=["old"]), mode="append")

    loaded = _loaded_dataset(local_path)
    loaded.write_to_dataset(_table(id=[1, 2], val=["new", "two"]), mode="delta")

    assert _rows(local_path) == [
        {"id": 1, "val": "new"},
        {"id": 1, "val": "old"},
        {"id": 2, "val": "two"},
    ]


def test_delta_subset_none_identity_uses_common_columns_only(
    local_path: str,
) -> None:
    """``delta_subset=None`` keys on the *common* non-null columns only.

    The source carries an extra ``note`` column absent from the target. Identity
    is computed over the intersection of source and target columns, so
    ``(1, "a", "x")`` matches the target row ``(1, "a")`` and is suppressed;
    the genuinely new ``(2, "b", "y")`` is appended with the non-common
    ``note`` dropped to match the target schema.
    """
    ds = _dataset(local_path)
    ds.write_to_dataset(_table(id=[1], val=["a"]), mode="append")

    loaded = _loaded_dataset(local_path)
    loaded.write_to_dataset(
        _table(id=[1, 2], val=["a", "b"], note=["x", "y"]),
        mode="delta",
    )

    assert _rows(local_path) == [
        {"id": 1, "val": "a"},
        {"id": 2, "val": "b"},
    ]


# --------------------------------------------------------------------------- #
# Loaded and unloaded dataset instances
# --------------------------------------------------------------------------- #


def test_delta_runs_insert_if_absent_when_dataset_is_loaded(local_path: str) -> None:
    """When the dataset is loaded, delta performs insert-if-absent."""
    ds = _dataset(local_path)
    ds.write_to_dataset(_table(id=[1], val=["old"]), mode="append")

    loaded = _loaded_dataset(local_path)
    assert loaded.is_loaded is True
    loaded.write_to_dataset(
        _table(id=[1, 2], val=["new", "two"]),
        mode="delta",
        delta_subset=["id"],
    )

    assert _rows(local_path) == [
        {"id": 1, "val": "old"},
        {"id": 2, "val": "two"},
    ]


def test_delta_is_insert_if_absent_when_dataset_not_loaded(local_path: str) -> None:
    """Delta delegates to merge independently of the dataset's load state."""
    ds = _dataset(local_path)
    ds.write_to_dataset(_table(id=[1], val=["old"]), mode="append")
    assert ds.is_loaded is False, "writing does not load the dataset"

    with pytest.warns(DeprecationWarning):
        ds.write_to_dataset(
            _table(id=[1, 2], val=["new", "two"]),
            mode="delta",
            delta_subset=["id"],
        )

    assert _rows(local_path) == [
        {"id": 1, "val": "old"},
        {"id": 2, "val": "two"},
    ]


# --------------------------------------------------------------------------- #
# Duplicate source keys
# --------------------------------------------------------------------------- #


def test_duplicate_new_source_keys_use_the_last_source_row(local_path: str) -> None:
    """The delegated merge resolves duplicate source keys last-row-wins."""
    ds = _dataset(local_path)
    ds.write_to_dataset(_table(id=[1], val=["old"]), mode="append")

    loaded = _loaded_dataset(local_path)
    with pytest.warns(DeprecationWarning):
        loaded.write_to_dataset(
            _table(id=[2, 2], val=["x", "y"]),
            mode="delta",
            delta_subset=["id"],
        )

    assert _rows(local_path) == [
        {"id": 1, "val": "old"},
        {"id": 2, "val": "y"},
    ]


def test_duplicate_existing_source_keys_are_all_dropped(local_path: str) -> None:
    """Every source row matching an existing key is suppressed by the anti-join.

    Both source rows share the already-present key 1, so the anti-join removes
    both; the target row is preserved untouched.
    """
    ds = _dataset(local_path)
    ds.write_to_dataset(_table(id=[1], val=["old"]), mode="append")

    loaded = _loaded_dataset(local_path)
    loaded.write_to_dataset(
        _table(id=[1, 1], val=["x", "y"]),
        mode="delta",
        delta_subset=["id"],
    )

    assert _rows(local_path) == [{"id": 1, "val": "old"}]


# --------------------------------------------------------------------------- #
# Null key values
# --------------------------------------------------------------------------- #


def test_null_source_key_is_rejected_when_target_has_no_null(
    local_path: str,
) -> None:
    """Delta rejects null source keys before persistence."""
    ds = _dataset(local_path)
    ds.write_to_dataset(_table(id=[1], val=["old"]), mode="append")

    loaded = _loaded_dataset(local_path)
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ValueError, match="null values in key"),
    ):
        loaded.write_to_dataset(
            _table(id=[None, 2], val=["n", "two"]),
            mode="delta",
            delta_subset=["id"],
        )

    assert _rows(local_path) == [{"id": 1, "val": "old"}]


def test_null_source_key_is_rejected_when_target_contains_null(
    local_path: str,
) -> None:
    """Null source keys are rejected rather than matched to null target keys."""
    ds = _dataset(local_path)
    ds.write_to_dataset(_table(id=[1, None], val=["old", "tn"]), mode="append")

    loaded = _loaded_dataset(local_path)
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ValueError, match="null values in key"),
    ):
        loaded.write_to_dataset(
            _table(id=[None, 2], val=["n", "two"]),
            mode="delta",
            delta_subset=["id"],
        )

    assert _rows(local_path) == [
        {"id": 1, "val": "old"},
        {"id": None, "val": "tn"},
    ]


# --------------------------------------------------------------------------- #
# List inputs
# --------------------------------------------------------------------------- #


def test_accepts_list_of_tables_as_data(local_path: str) -> None:
    """A ``list`` of frames is processed element-by-element through the delta path."""
    ds = _dataset(local_path)
    ds.write_to_dataset(_table(id=[1], val=["old"]), mode="append")

    loaded = _loaded_dataset(local_path)
    loaded.write_to_dataset(
        [
            _table(id=[1, 2], val=["new", "two"]),
            _table(id=[3], val=["three"]),
        ],
        mode="delta",
        delta_subset=["id"],
    )

    assert _rows(local_path) == [
        {"id": 1, "val": "old"},
        {"id": 2, "val": "two"},
        {"id": 3, "val": "three"},
    ]


# --------------------------------------------------------------------------- #
# Partitioned datasets
# --------------------------------------------------------------------------- #


def test_delta_on_hive_partitioned_dataset(local_path: str) -> None:
    """Insert-if-absent works on a hive-partitioned dataset keyed by id."""
    ds = _dataset(local_path)
    ds.write_to_dataset(
        _table(id=[1, 2], region=["a", "b"], val=["old1", "old2"]),
        mode="append",
        partition_by=["region"],
    )

    loaded = _loaded_dataset(local_path)
    loaded.write_to_dataset(
        _table(id=[1, 3], region=["a", "a"], val=["new1", "three"]),
        mode="delta",
        delta_subset=["id"],
        partition_by=["region"],
    )

    assert _rows(local_path) == [
        {"id": 1, "region": "a", "val": "old1"},
        {"id": 2, "region": "b", "val": "old2"},
        {"id": 3, "region": "a", "val": "three"},
    ]


# --------------------------------------------------------------------------- #
# Empty / missing targets
# --------------------------------------------------------------------------- #


def test_delta_into_empty_dataset_appends_all_rows(local_path: str) -> None:
    """With no existing target rows there is nothing to anti-join against.

    A dataset that has never been written to is not loaded, so the delta path
    is skipped and every source row is appended -- which is also the correct
    insert-if-absent result for an empty target.
    """
    ds = _dataset(local_path)
    assert ds.is_loaded is False

    ds.write_to_dataset(
        _table(id=[1, 2], val=["a", "b"]),
        mode="delta",
        delta_subset=["id"],
    )

    assert _rows(local_path) == [
        {"id": 1, "val": "a"},
        {"id": 2, "val": "b"},
    ]


def test_delta_with_empty_source_is_a_noop(local_path: str) -> None:
    """An empty source frame writes nothing and leaves the target untouched.

    ``Writer.execute`` returns early when the incoming frame has zero rows, so
    a delta with an empty source neither appends a file nor mutates the target.
    """
    ds = _dataset(local_path)
    ds.write_to_dataset(_table(id=[1], val=["old"]), mode="append")

    loaded = _loaded_dataset(local_path)
    loaded.write_to_dataset(
        _table(id=pa.array([], pa.int64()), val=pa.array([], pa.string())),
        mode="delta",
        delta_subset=["id"],
    )

    assert _rows(local_path) == [{"id": 1, "val": "old"}]


def test_delta_matching_all_target_rows_appends_nothing(local_path: str) -> None:
    """A delta whose every row already exists writes no new file.

    When the anti-join removes every source row, the surviving frame is empty
    and ``Writer.execute`` writes nothing, so the target is left byte-for-byte
    unchanged (no second parquet file). This is the "existing-but-no-new-rows"
    target case complementing the never-written empty target above.
    """
    ds = _dataset(local_path)
    ds.write_to_dataset(_table(id=[1], val=["a"]), mode="append")

    ds_dir = pathlib.Path(local_path, "ds")
    files_before = {p.name for p in ds_dir.glob("*.parquet")}

    loaded = _loaded_dataset(local_path)
    loaded.write_to_dataset(
        _table(id=[1], val=["a"]),
        mode="delta",
        delta_subset=["id"],
    )

    assert _rows(local_path) == [{"id": 1, "val": "a"}]
    assert {p.name for p in ds_dir.glob("*.parquet")} == files_before


# --------------------------------------------------------------------------- #
# Schema casting
# --------------------------------------------------------------------------- #


def test_delta_casts_source_schema_to_target_schema(local_path: str) -> None:
    """Source columns are cast to the target schema during a delta write.

    Source ``id`` arrives as int32 but the target is int64; the writer casts it
    so the delta anti-join still recognises the existing key 1 and suppresses
    it, leaving the target row and appending only key 2.
    """
    ds = _dataset(local_path)
    ds.write_to_dataset(
        _table(id=pa.array([1], pa.int64()), val=["old"]),
        mode="append",
    )

    loaded = _loaded_dataset(local_path)
    loaded.write_to_dataset(
        _table(id=pa.array([1, 2], pa.int32()), val=["new", "two"]),
        mode="delta",
        delta_subset=["id"],
    )

    assert _rows(local_path) == [
        {"id": 1, "val": "old"},
        {"id": 2, "val": "two"},
    ]


# --------------------------------------------------------------------------- #
# Derived date-part partition columns
# --------------------------------------------------------------------------- #


def test_delta_with_derived_datepart_partition_column(local_path: str) -> None:
    """A timestamp-derived partition column (``year``) is derived on both writes.

    ``partition_by=["year"]`` with ``timestamp_column="ts"`` derives a ``year``
    partition column from the timestamp. The delta still keys on ``id``: the
    existing id 1 is kept, the new id 2 is inserted under its derived partition.
    """
    ds = _dataset(local_path)
    ts = pa.array([datetime.datetime(2024, 1, 15)], pa.timestamp("us"))
    ds.write_to_dataset(
        _table(id=[1], ts=ts, val=["old"]),
        mode="append",
        partition_by=["year"],
        timestamp_column="ts",
    )

    loaded = _loaded_dataset(local_path)
    ts2 = pa.array(
        [datetime.datetime(2024, 1, 15), datetime.datetime(2024, 6, 1)],
        pa.timestamp("us"),
    )
    loaded.write_to_dataset(
        _table(id=[1, 2], ts=ts2, val=["new", "two"]),
        mode="delta",
        delta_subset=["id"],
        partition_by=["year"],
        timestamp_column="ts",
    )

    rows = _rows(local_path)
    assert {r["id"]: r["val"] for r in rows} == {1: "old", 2: "two"}
    assert {r["year"] for r in rows} == {2024}
    assert {r["ts"] for r in rows} == {
        datetime.datetime(2024, 1, 15, 0, 0),
        datetime.datetime(2024, 6, 1, 0, 0),
    }

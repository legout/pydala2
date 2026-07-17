"""Public-API contracts for ``ParquetDataset.merge`` (issue #30).

``ParquetDataset.merge`` is the pydala-facing merge interface that normalizes
every supported source family through ``Writer.prepare``, treats list inputs
as one logical source batch, delegates persistence strictly through
``FsspeckitParquetAdapter.merge``, refreshes pydala file/metadata/table/cache
state via ``_refresh_after_rewrite``, and returns fsspeckit's typed
``MergeResult``.

These tests freeze the public contract at the dataset seam, independent of
the transport adapter (#29) and source-normalization (#25) layers below.

Baseline captured at commit ``9012dfdce12130c1db557e84e9c8257d5a4acd24``.
"""

from __future__ import annotations

import pathlib

import duckdb
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
import pytest

from fsspeckit.core.incremental import MergeResult

from pydala import ParquetDataset
from pydala.adapters.fsspeckit import FsspeckitParquetAdapter
from pydala.filesystem import FileSystem
from pydala.io import PartialMergeError


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _dataset(local_path: str, subpath: str = "ds") -> ParquetDataset:
    """Construct a fresh ParquetDataset rooted at ``local_path/subpath``."""
    return ParquetDataset(
        path=subpath,
        filesystem=FileSystem(bucket=local_path, cached=False),
    )


def _rows(local_path: str, subpath: str = "ds") -> list[dict]:
    """Read every physical parquet file under the dataset directory."""
    base = pathlib.Path(local_path) / subpath
    if not base.exists():
        return []
    return (
        pds.dataset(str(base), format="parquet", partitioning="hive")
        .to_table()
        .to_pylist()
    )


def _table(**columns: list) -> pa.Table:
    """Build a PyArrow Table from keyword columns."""
    return pa.table(columns)


# --------------------------------------------------------------------------- #
# Argument validation
# --------------------------------------------------------------------------- #


def test_merge_requires_explicit_key_columns(local_path: str) -> None:
    """``key_columns`` is mandatory and must be non-empty."""
    ds = _dataset(local_path)
    with pytest.raises(ValueError):
        ds.merge(_table(id=[1]), strategy="upsert", key_columns=None)
    with pytest.raises(ValueError):
        ds.merge(_table(id=[1]), strategy="upsert", key_columns=[])


def test_merge_rejects_invalid_strategy(local_path: str) -> None:
    ds = _dataset(local_path)
    with pytest.raises(ValueError):
        ds.merge(_table(id=[1]), strategy="delete", key_columns=["id"])


def test_merge_rejects_invalid_backend(local_path: str) -> None:
    ds = _dataset(local_path)
    with pytest.raises(ValueError):
        ds.merge(
            _table(id=[1]),
            strategy="insert",
            key_columns=["id"],
            backend="polars",
        )


def test_merge_rejects_null_key_values(local_path: str) -> None:
    """Null source key values must be rejected before persistence."""
    ds = _dataset(local_path)
    with pytest.raises(ValueError):
        ds.merge(
            _table(id=[1, None], v=["a", "b"]),
            strategy="upsert",
            key_columns=["id"],
        )
    # Nothing was written.
    assert _rows(local_path) == []


def test_merge_rejects_unsupported_source_type(local_path: str) -> None:
    """Sources outside the supported families are rejected during preparation."""
    ds = _dataset(local_path)
    with pytest.raises(Exception):
        ds.merge(
            "not a frame",  # type: ignore[arg-type]
            strategy="upsert",
            key_columns=["id"],
        )


# --------------------------------------------------------------------------- #
# Typed result and counts
# --------------------------------------------------------------------------- #


def test_merge_returns_typed_merge_result_with_counts(local_path: str) -> None:
    ds = _dataset(local_path)
    result = ds.merge(
        _table(id=[1, 2], v=["a", "b"]),
        strategy="upsert",
        key_columns=["id"],
    )
    assert isinstance(result, MergeResult)
    assert result.strategy == "upsert"
    assert result.target_count_before == 0
    assert result.target_count_after == 2
    assert result.inserted == 2
    assert result.updated == 0


# --------------------------------------------------------------------------- #
# Empty / missing targets
# --------------------------------------------------------------------------- #


def test_merge_into_missing_target_inserts_all_rows(local_path: str) -> None:
    """Merging into a target that does not yet exist inserts every source row."""
    ds = _dataset(local_path)
    result = ds.merge(
        _table(id=[1, 2, 3], v=["a", "b", "c"]),
        strategy="insert",
        key_columns=["id"],
    )
    assert result.inserted == 3
    assert result.target_count_after == 3
    rows = sorted(_rows(local_path), key=lambda r: r["id"])
    assert [r["v"] for r in rows] == ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# No-op insert / no-op update
# --------------------------------------------------------------------------- #


def test_merge_insert_noop_when_all_keys_exist(local_path: str) -> None:
    ds = _dataset(local_path)
    ds.merge(_table(id=[1, 2], v=["a", "b"]), strategy="upsert", key_columns=["id"])

    result = ds.merge(
        _table(id=[1, 2], v=["IGNORED", "IGNORED"]),
        strategy="insert",
        key_columns=["id"],
    )
    assert result.inserted == 0
    assert result.updated == 0
    assert result.target_count_after == 2
    rows = sorted(_rows(local_path), key=lambda r: r["id"])
    assert [r["v"] for r in rows] == ["a", "b"]


def test_merge_update_noop_when_no_keys_match(local_path: str) -> None:
    ds = _dataset(local_path)
    ds.merge(_table(id=[1, 2], v=["a", "b"]), strategy="upsert", key_columns=["id"])

    result = ds.merge(
        _table(id=[3, 4], v=["c", "d"]),
        strategy="update",
        key_columns=["id"],
    )
    assert result.updated == 0
    assert result.inserted == 0
    assert result.target_count_after == 2
    rows = sorted(_rows(local_path), key=lambda r: r["id"])
    assert [r["v"] for r in rows] == ["a", "b"]


# --------------------------------------------------------------------------- #
# Upsert + selective file rewrite
# --------------------------------------------------------------------------- #


def test_merge_upsert_updates_and_inserts(local_path: str) -> None:
    ds = _dataset(local_path)
    ds.merge(_table(id=[1, 2], v=["a", "b"]), strategy="upsert", key_columns=["id"])

    result = ds.merge(
        _table(id=[2, 3], v=["B", "c"]),
        strategy="upsert",
        key_columns=["id"],
    )
    assert result.updated == 1
    assert result.inserted == 1
    assert result.target_count_after == 3
    rows = {r["id"]: r["v"] for r in _rows(local_path)}
    assert rows == {1: "a", 2: "B", 3: "c"}


def test_merge_update_rewrites_matched_files_only(local_path: str) -> None:
    """An update rewrites only the files holding matched keys; others are preserved."""
    ds = _dataset(local_path)
    # Seed with two separate writes so matched and unmatched keys live in
    # distinct physical files.
    ds.merge(_table(id=[1], v=["a"]), strategy="upsert", key_columns=["id"])
    ds.merge(_table(id=[2], v=["b"]), strategy="upsert", key_columns=["id"])

    result = ds.merge(
        _table(id=[1], v=["A"]),
        strategy="update",
        key_columns=["id"],
    )
    assert result.updated == 1
    assert result.inserted == 0
    assert len(result.rewritten_files) >= 1
    assert result.preserved_files  # unmatched file(s) preserved
    rows = {r["id"]: r["v"] for r in _rows(local_path)}
    assert rows == {1: "A", 2: "b"}


# --------------------------------------------------------------------------- #
# Composite keys
# --------------------------------------------------------------------------- #


def test_merge_composite_keys(local_path: str) -> None:
    ds = _dataset(local_path)
    ds.merge(
        _table(a=[1, 1], b=["x", "y"], v=["ax", "by"]),
        strategy="upsert",
        key_columns=["a", "b"],
    )
    result = ds.merge(
        _table(a=[1, 2], b=["x", "z"], v=["AX", "cz"]),
        strategy="upsert",
        key_columns=["a", "b"],
    )
    assert result.updated == 1
    assert result.inserted == 1
    rows = sorted(_rows(local_path), key=lambda r: (r["a"], r["b"]))
    assert [(r["a"], r["b"], r["v"]) for r in rows] == [
        (1, "x", "AX"),
        (1, "y", "by"),
        (2, "z", "cz"),
    ]


# --------------------------------------------------------------------------- #
# Immutable partition columns (hive layout from partition_by)
# --------------------------------------------------------------------------- #


def test_merge_partition_by_creates_hive_layout(local_path: str) -> None:
    """``partition_by`` is translated to hive partition directories."""
    ds = _dataset(local_path)
    ds.merge(
        _table(id=[1, 2], region=["eu", "us"], v=["one", "two"]),
        strategy="upsert",
        key_columns=["id"],
        partition_by=["region"],
    )
    listing = sorted(ds.fs.find("ds"))
    assert any("region=eu" in p for p in listing)
    assert any("region=us" in p for p in listing)


def test_merge_partitioned_upsert_routes_by_partition(local_path: str) -> None:
    ds = _dataset(local_path)
    ds.merge(
        _table(id=[1, 2], region=["eu", "us"], v=["one", "two"]),
        strategy="upsert",
        key_columns=["id"],
        partition_by=["region"],
    )
    result = ds.merge(
        _table(id=[1, 3], region=["eu", "eu"], v=["ONE", "three"]),
        strategy="upsert",
        key_columns=["id"],
        partition_by=["region"],
    )
    assert result.updated == 1
    assert result.inserted == 1
    # The same instance becomes partition-aware and reads region back.
    assert ds.partition_names == ["region"]
    same_rows = sorted(ds.table.to_arrow().to_pylist(), key=lambda r: r["id"])
    assert [(r["id"], r["region"], r["v"]) for r in same_rows] == [
        (1, "eu", "ONE"),
        (2, "us", "two"),
        (3, "eu", "three"),
    ]
    rows = sorted(_rows(local_path), key=lambda r: r["id"])
    assert [(r["id"], r["v"]) for r in rows] == [
        (1, "ONE"),
        (2, "two"),
        (3, "three"),
    ]


def test_merge_partitioned_subsequent_merge_without_partition_by(
    local_path: str,
) -> None:
    """A dataset partitioned by an earlier merge stays partition-aware."""
    ds = _dataset(local_path)
    ds.merge(
        _table(id=[1], region=["eu"], v=["one"]),
        strategy="upsert",
        key_columns=["id"],
        partition_by=["region"],
    )
    # A second merge that does not repeat partition_by must reuse the
    # dataset's hive partitioning (not silently write unpartitioned files).
    result = ds.merge(
        _table(id=[2], region=["us"], v=["two"]),
        strategy="insert",
        key_columns=["id"],
    )
    assert result.inserted == 1
    assert ds.partition_names == ["region"]
    listing = sorted(ds.fs.find("ds"))
    assert any("region=us" in p for p in listing)


def test_merge_rejects_update_that_changes_partition_value(
    local_path: str,
) -> None:
    """Partition columns are immutable: a key cannot move partitions."""
    ds = _dataset(local_path)
    ds.merge(
        _table(id=[1], region=["eu"], v=["one"]),
        strategy="upsert",
        key_columns=["id"],
        partition_by=["region"],
    )
    # Attempting to relocate id=1 from eu to us must be rejected.
    with pytest.raises(ValueError):
        ds.merge(
            _table(id=[1], region=["us"], v=["moved"]),
            strategy="update",
            key_columns=["id"],
            partition_by=["region"],
        )
    # The target is left untouched.
    rows = _rows(local_path)
    assert rows == [{"id": 1, "region": "eu", "v": "one"}]


# --------------------------------------------------------------------------- #
# Duplicate source keys last-row-wins
# --------------------------------------------------------------------------- #


def test_merge_duplicate_source_keys_last_row_wins(local_path: str) -> None:
    """Duplicate source keys resolve last-row-wins as required by fsspeckit."""
    ds = _dataset(local_path)
    ds.merge(_table(id=[1], v=["orig"]), strategy="upsert", key_columns=["id"])

    result = ds.merge(
        _table(id=[2, 2, 2], v=["first", "second", "third"]),
        strategy="upsert",
        key_columns=["id"],
    )
    # fsspeckit dedups the source batch; only one new key is inserted.
    assert result.inserted == 1
    rows = {r["id"]: r["v"] for r in _rows(local_path)}
    assert rows == {1: "orig", 2: "third"}


# --------------------------------------------------------------------------- #
# Schema casting
# --------------------------------------------------------------------------- #


def test_merge_casts_source_schema_to_target_schema(local_path: str) -> None:
    """Incoming source columns are cast to the existing target schema."""
    ds = _dataset(local_path)
    # Establish a target schema where v is string.
    ds.merge(_table(id=[1], v=["a"]), strategy="upsert", key_columns=["id"])

    # Source arrives with v as int32; the prepared table must be cast to string.
    result = ds.merge(
        pa.table({"id": pa.array([2], pa.int64()), "v": pa.array([42], pa.int32())}),
        strategy="insert",
        key_columns=["id"],
    )
    assert result.inserted == 1

    rows = sorted(_rows(local_path), key=lambda r: r["id"])
    assert [r["v"] for r in rows] == ["a", "42"]
    # The on-disk schema must be the unified (target) schema.
    table = pds.dataset(
        str(pathlib.Path(local_path) / "ds"), format="parquet"
    ).to_table()
    assert table.schema.field("v").type == pa.string()


# --------------------------------------------------------------------------- #
# List input as one logical source batch
# --------------------------------------------------------------------------- #


def test_merge_list_input_is_one_logical_batch(local_path: str) -> None:
    """A list of frames is merged as a single source batch, not sequentially."""
    ds = _dataset(local_path)
    result = ds.merge(
        [_table(id=[1], v=["a"]), _table(id=[2], v=["b"])],
        strategy="insert",
        key_columns=["id"],
    )
    assert result.source_count == 2
    assert result.inserted == 2
    assert result.target_count_after == 2
    rows = sorted(_rows(local_path), key=lambda r: r["id"])
    assert [r["v"] for r in rows] == ["a", "b"]


def test_merge_list_batch_dedups_duplicate_keys_last_wins(local_path: str) -> None:
    """Duplicate keys across list elements resolve last-row-wins in one batch."""
    ds = _dataset(local_path)
    result = ds.merge(
        [_table(id=[1], v=["first"]), _table(id=[1], v=["second"])],
        strategy="upsert",
        key_columns=["id"],
    )
    assert result.inserted == 1
    rows = {r["id"]: r["v"] for r in _rows(local_path)}
    assert rows == {1: "second"}


# --------------------------------------------------------------------------- #
# Source families: every supported input merges correctly
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: pa.table({"id": [1], "v": ["a"]}), id="pyarrow-table"),
        pytest.param(
            lambda: pa.RecordBatch.from_arrays(
                [pa.array([1]), pa.array(["a"])], names=["id", "v"]
            ),
            id="pyarrow-recordbatch",
        ),
        pytest.param(
            lambda: pl.DataFrame({"id": [1], "v": ["a"]}), id="polars-dataframe"
        ),
        pytest.param(
            lambda: pl.DataFrame({"id": [1], "v": ["a"]}).lazy(), id="polars-lazyframe"
        ),
        pytest.param(lambda: pd.DataFrame({"id": [1], "v": ["a"]}), id="pandas"),
        pytest.param(
            lambda: duckdb.sql("SELECT 1 AS id, 'a' AS v"), id="duckdb-relation"
        ),
    ],
)
def test_merge_supports_every_source_family(make, local_path: str) -> None:
    ds = _dataset(local_path)
    result = ds.merge(make(), strategy="upsert", key_columns=["id"])
    assert isinstance(result, MergeResult)
    assert result.target_count_after == 1
    rows = _rows(local_path)
    assert rows == [{"id": 1, "v": "a"}]


# --------------------------------------------------------------------------- #
# Same-instance immediate readability (independent of is_loaded)
# --------------------------------------------------------------------------- #


def test_merge_same_instance_readable_after_merge_without_prior_load(
    local_path: str,
) -> None:
    """A never-loaded dataset reads the merged result from the same instance."""
    ds = _dataset(local_path)
    assert ds.is_loaded is False

    ds.merge(_table(id=[1, 2], v=["a", "b"]), strategy="upsert", key_columns=["id"])

    # The same instance must see the new rows immediately.
    assert ds.is_loaded is True
    assert ds.num_rows == 2
    rows = sorted(ds.table.to_arrow().to_pylist(), key=lambda r: r["id"])
    assert [r["v"] for r in rows] == ["a", "b"]


def test_merge_same_instance_readable_when_already_loaded(
    local_path: str,
) -> None:
    """An already-loaded dataset reflects the merged result without manual reload."""
    ds = _dataset(local_path)
    ds.merge(_table(id=[1], v=["a"]), strategy="upsert", key_columns=["id"])
    assert ds.is_loaded is True
    assert ds.num_rows == 1

    ds.merge(_table(id=[1, 2], v=["A", "b"]), strategy="upsert", key_columns=["id"])
    assert ds.num_rows == 2
    rows = {r["id"]: r["v"] for r in ds.table.to_arrow().to_pylist()}
    assert rows == {1: "A", 2: "b"}


# --------------------------------------------------------------------------- #
# Backend / error propagation
# --------------------------------------------------------------------------- #


def test_merge_propagates_backend_error_for_memory_filesystem(tmp_path) -> None:
    """DuckDB backend cannot serve a memory filesystem; the error surfaces."""
    import uuid

    ds = ParquetDataset(
        path="ev",
        filesystem=FileSystem(
            bucket=f"merge-{uuid.uuid4().hex}", protocol="memory", cached=False
        ),
    )
    with pytest.raises(ValueError) as ctx:
        ds.merge(
            _table(id=[1], v=["a"]),
            strategy="upsert",
            key_columns=["id"],
            backend="duckdb",
        )
    assert "duckdb" in str(ctx.value).lower()


def test_merge_duckdb_backend_succeeds_on_local_filesystem(
    local_path: str,
) -> None:
    """The DuckDB backend merges successfully on a local filesystem."""
    ds = _dataset(local_path)
    ds.merge(_table(id=[1], v=["a"]), strategy="upsert", key_columns=["id"])

    result = ds.merge(
        _table(id=[2], v=["b"]),
        strategy="insert",
        key_columns=["id"],
        backend="duckdb",
    )
    assert isinstance(result, MergeResult)
    assert result.inserted == 1
    assert result.target_count_after == 2


def test_merge_persistence_error_propagates_without_refresh(
    local_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persistence failure propagates verbatim without refresh or result retention."""
    ds = _dataset(local_path)
    refresh_calls = {"n": 0}
    monkeypatch.setattr(
        ds, "_refresh_after_rewrite", lambda: refresh_calls.__setitem__("n", 1)
    )

    def _boom(self, *args, **kwargs):
        raise RuntimeError("simulated fsspeckit failure")

    monkeypatch.setattr(FsspeckitParquetAdapter, "merge", _boom)
    with pytest.raises(RuntimeError, match="simulated fsspeckit failure"):
        ds.merge(_table(id=[1], v=["a"]), strategy="upsert", key_columns=["id"])

    # No refresh, no retained result, no physical files.
    assert refresh_calls["n"] == 0
    assert ds.last_merge_result is None
    assert _rows(local_path) == []


def test_merge_options_forwarded_to_fsspeckit(local_path: str) -> None:
    """Extra ``**merge_options`` reach fsspeckit's merge path."""
    ds = _dataset(local_path)
    # ``max_rows_per_file`` and ``row_group_size`` are fsspeckit sizing
    # options forwarded verbatim through ``**merge_options``.
    result = ds.merge(
        _table(id=[1], v=["a"]),
        strategy="upsert",
        key_columns=["id"],
        max_rows_per_file=1,
        row_group_size=1,
    )
    assert isinstance(result, MergeResult)
    assert result.target_count_after == 1


def test_merge_alter_schema_keeps_extra_source_columns(
    local_path: str,
) -> None:
    """``alter_schema=True`` preserves source columns absent from the target."""
    ds = _dataset(local_path)
    ds.merge(_table(id=[1], v=["a"]), strategy="upsert", key_columns=["id"])

    result = ds.merge(
        pa.table({"id": [2], "v": ["b"], "extra": [99]}),
        strategy="insert",
        key_columns=["id"],
        alter_schema=True,
    )
    assert result.inserted == 1
    rows = {r["id"]: r for r in _rows(local_path)}
    assert rows[2].get("extra") == 99


def test_merge_compression_forwarded(local_path: str) -> None:
    """``compression`` is forwarded to the fsspeckit write path."""
    ds = _dataset(local_path)
    result = ds.merge(
        _table(id=[1], v=["a"]),
        strategy="upsert",
        key_columns=["id"],
        compression="zstd",
    )
    assert isinstance(result, MergeResult)
    # Confirm the written file is zstd-compressed.
    files = list(pathlib.Path(local_path).rglob("*.parquet"))
    assert files

    # zstd codec is recorded on the column compression metadata.
    assert pq.read_metadata(str(files[0])).row_group(0).column(0).compression in {
        "zstd",
        "ZSTD",
        "Zstd",
    }


# --------------------------------------------------------------------------- #
# Refresh-failure recovery: MergeResult preserved on the instance
# --------------------------------------------------------------------------- #


def test_merge_refresh_failure_preserves_result_and_raises(
    local_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When refresh fails after a successful merge, the result is retained."""
    ds = _dataset(local_path)

    def _boom() -> None:
        raise RuntimeError("simulated refresh failure")

    monkeypatch.setattr(ds, "_refresh_after_rewrite", _boom)

    with pytest.raises(PartialMergeError) as ctx:
        ds.merge(_table(id=[1], v=["a"]), strategy="upsert", key_columns=["id"])

    # The successful result is exposed on the error and on the instance.
    assert isinstance(ctx.value.merge_result, MergeResult)
    assert ctx.value.merge_result.target_count_after == 1
    assert isinstance(ds.last_merge_result, MergeResult)
    assert ds.last_merge_result.target_count_after == 1

    # The physical merge actually succeeded even though refresh failed.
    rows = _rows(local_path)
    assert rows == [{"id": 1, "v": "a"}]


def test_merge_refresh_failure_then_manual_recovery(
    local_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a refresh failure, a manual refresh exposes the merged data."""
    ds = _dataset(local_path)
    calls = {"n": 0}

    def _flaky() -> None:
        calls["n"] += 1
        raise RuntimeError("simulated refresh failure")

    monkeypatch.setattr(ds, "_refresh_after_rewrite", _flaky)

    with pytest.raises(PartialMergeError):
        ds.merge(_table(id=[1], v=["a"]), strategy="upsert", key_columns=["id"])

    # Stop flaking; recover by refreshing from the successful on-disk state.
    monkeypatch.undo()
    ds._refresh_after_rewrite()
    assert ds.is_loaded is True
    assert ds.num_rows == 1


# --------------------------------------------------------------------------- #
# Append / overwrite regression: merge does not perturb existing modes
# --------------------------------------------------------------------------- #


def test_append_and_overwrite_still_work_after_merge(local_path: str) -> None:
    """``write_to_dataset`` append/overwrite behavior is unchanged by merge."""
    ds = _dataset(local_path)
    ds.merge(_table(id=[1], v=["a"]), strategy="upsert", key_columns=["id"])

    # Append still works.
    ds.write_to_dataset(_table(id=[1], v=["b"]), mode="append")
    # File scan order is not guaranteed across the merged and appended files.
    rows = sorted(_rows(local_path), key=lambda r: (r["id"], r["v"]))
    assert rows == [{"id": 1, "v": "a"}, {"id": 1, "v": "b"}]

    # Overwrite replaces everything.
    ds.write_to_dataset(_table(id=[9], v=["only"]), mode="overwrite")
    assert _rows(local_path) == [{"id": 9, "v": "only"}]

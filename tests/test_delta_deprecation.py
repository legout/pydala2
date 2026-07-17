"""Contracts for deprecated ``mode="delta"`` writes."""

from __future__ import annotations

import pathlib

import pyarrow as pa
import pyarrow.dataset as pds
import pytest
from fsspeckit.core.incremental import MergeResult

from pydala import ParquetDataset
from pydala.filesystem import FileSystem


def _dataset(tmp_path: pathlib.Path) -> ParquetDataset:
    return ParquetDataset(
        path="events",
        filesystem=FileSystem(bucket=str(tmp_path), cached=False),
    )


def _rows(tmp_path: pathlib.Path) -> list[dict[str, object]]:
    path = tmp_path / "events"
    if not path.exists() or not list(path.rglob("*.parquet")):
        return []
    return sorted(
        pds.dataset(path, format="parquet", partitioning="hive").to_table().to_pylist(),
        key=lambda row: row["id"],
    )


def test_unknown_write_mode_is_rejected_before_writing(tmp_path: pathlib.Path) -> None:
    dataset = _dataset(tmp_path)

    with pytest.raises(ValueError, match="Unsupported write mode"):
        dataset.write_to_dataset(pa.table({"id": [1]}), mode="typo")

    assert _rows(tmp_path) == []


def test_keyed_delta_warns_delegates_to_insert_and_returns_merge_result(
    tmp_path: pathlib.Path,
) -> None:
    dataset = _dataset(tmp_path)
    dataset.write_to_dataset(
        pa.table({"id": [1], "value": ["old"]}),
        mode="append",
    )

    with pytest.warns(DeprecationWarning) as warnings:
        result = dataset.write_to_dataset(
            pa.table({"id": [1, 2], "value": ["new", "two"]}),
            mode="delta",
            delta_subset=["id"],
        )

    message = str(warnings[0].message)
    assert "mode='delta' is deprecated" in message
    assert "merge(strategy='insert')" in message
    assert "null keys are rejected" in message
    assert "duplicate source keys use the last row" in message
    assert "independent of load state" in message
    assert "MergeResult" in message
    assert isinstance(result, MergeResult)
    assert result.target_count_before == 1
    assert result.inserted == 1
    assert result.updated == 0
    assert result.target_count_after == 2
    assert _rows(tmp_path) == [
        {"id": 1, "value": "old"},
        {"id": 2, "value": "two"},
    ]


def test_delta_without_subset_warns_explicit_keys_will_be_required(
    tmp_path: pathlib.Path,
) -> None:
    dataset = _dataset(tmp_path)
    dataset.write_to_dataset(
        pa.table({"id": [1], "value": ["old"]}),
        mode="append",
    )

    with pytest.warns(DeprecationWarning) as warnings:
        result = dataset.write_to_dataset(
            pa.table({"id": [1, 2], "value": ["old", "two"]}),
            mode="delta",
            delta_subset=None,
        )

    message = str(warnings[0].message)
    assert "delta_subset=None" in message
    assert "common non-null source/target columns" in message
    assert "explicit keys will be required" in message
    assert isinstance(result, MergeResult)
    assert result.inserted == 1
    assert _rows(tmp_path) == [
        {"id": 1, "value": "old"},
        {"id": 2, "value": "two"},
    ]


def test_delta_without_subset_excludes_nullable_target_columns(
    tmp_path: pathlib.Path,
) -> None:
    dataset = _dataset(tmp_path)
    dataset.write_to_dataset(
        pa.table({"id": [1, 2], "value": ["old", None]}),
        mode="append",
    )

    with pytest.warns(DeprecationWarning):
        dataset.write_to_dataset(
            pa.table({"id": [1, 3], "value": ["new", "three"]}),
            mode="delta",
            delta_subset=None,
        )

    assert _rows(tmp_path) == [
        {"id": 1, "value": "old"},
        {"id": 2, "value": None},
        {"id": 3, "value": "three"},
    ]


def test_delta_without_subset_discovers_existing_hive_partition_keys(
    tmp_path: pathlib.Path,
) -> None:
    dataset = _dataset(tmp_path)
    dataset.write_to_dataset(
        pa.table({"id": [1], "value": ["same"], "region": ["A"]}),
        mode="append",
        partition_by=["region"],
    )

    with pytest.warns(DeprecationWarning):
        dataset.write_to_dataset(
            pa.table({"id": [1], "value": ["same"], "region": ["B"]}),
            mode="delta",
            delta_subset=None,
        )

    assert sorted(_rows(tmp_path), key=lambda row: row["region"]) == [
        {"id": 1, "value": "same", "region": "A"},
        {"id": 1, "value": "same", "region": "B"},
    ]


def test_keyed_delta_empty_list_returns_typed_noop_result(
    tmp_path: pathlib.Path,
) -> None:
    dataset = _dataset(tmp_path)
    dataset.write_to_dataset(
        pa.table({"id": [1], "value": ["old"]}),
        mode="append",
    )

    with pytest.warns(DeprecationWarning):
        result = dataset.write_to_dataset(
            [],
            mode="delta",
            delta_subset=["id"],
        )

    assert isinstance(result, MergeResult)
    assert dataset.is_loaded is True
    assert result.source_count == 0
    assert result.target_count_before == 1
    assert result.target_count_after == 1
    assert result.inserted == 0
    assert result.updated == 0
    assert _rows(tmp_path) == [{"id": 1, "value": "old"}]

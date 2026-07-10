"""Focused contracts for fsspeckit-backed managed Parquet maintenance."""

from __future__ import annotations

from tests.conftest import assert_metadata_invariants


def test_compaction_dry_run_returns_plan_without_mutating_managed_metadata(
    managed_dataset,
) -> None:
    files_before = set(managed_dataset.files)
    metadata_before = set(managed_dataset.files_in_metadata)

    plan = managed_dataset.compact_by_rows(
        max_rows_per_file=100,
        compression="zstd",
        dry_run=True,
    )

    assert isinstance(plan, dict)
    assert set(managed_dataset.files) == files_before
    assert set(managed_dataset.files_in_metadata) == metadata_before
    assert_metadata_invariants(managed_dataset)


def test_compaction_refreshes_managed_metadata_and_preserves_rows(managed_dataset) -> None:
    managed_dataset.load()
    rows_before = managed_dataset.table.to_arrow().num_rows

    managed_dataset.compact_by_rows(
        max_rows_per_file=100,
        compression="zstd",
        unique=True,
    )

    assert managed_dataset.table.to_arrow().num_rows == rows_before
    assert_metadata_invariants(managed_dataset)


def test_stats_are_collected_through_maintenance_adapter(managed_dataset) -> None:
    stats = managed_dataset.collect_stats()

    assert stats["total_rows"] == 15
    assert len(stats["files"]) == 2

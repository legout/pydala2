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


def test_compaction_refreshes_managed_metadata_and_preserves_rows(
    managed_dataset,
) -> None:
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


def test_unique_true_removes_exact_duplicates(managed_dataset) -> None:
    managed_dataset.load()
    rows_before = managed_dataset.table.to_arrow().num_rows

    # Write the same batch again to create exact duplicates.
    from tests.conftest import make_simple_table

    managed_dataset.write_to_dataset(make_simple_table(n_rows=5, seed=0), mode="append")
    managed_dataset.update()
    managed_dataset.load()

    managed_dataset.compact_by_rows(
        max_rows_per_file=100,
        compression="zstd",
        unique=True,
    )

    rows_after = managed_dataset.table.to_arrow().num_rows
    assert rows_after == rows_before, f"expected {rows_before}, got {rows_after}"
    assert_metadata_invariants(managed_dataset)


def test_optimize_dtypes_preserves_rows_and_metadata(managed_dataset) -> None:
    """dtype optimization rewrites files in place and preserves row count."""
    managed_dataset.load(update_metadata=True)
    rows_before = managed_dataset.table.to_arrow().num_rows

    managed_dataset.optimize_dtypes(exclude="name")
    managed_dataset.load(update_metadata=True)

    assert managed_dataset.table.to_arrow().num_rows == rows_before
    assert_metadata_invariants(managed_dataset)


def test_scan_files_returns_filtered_subset(managed_dataset) -> None:
    """scan() must populate _metadata_table_scanned so scan_files is scoped."""
    managed_dataset.load(update_metadata=True)
    first_file = managed_dataset.files[0]

    managed_dataset.scan(f"file_path='{first_file}'")
    assert managed_dataset.scan_files == [first_file]
    managed_dataset.reset_scan()

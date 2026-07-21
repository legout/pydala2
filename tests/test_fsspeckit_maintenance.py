"""Focused contracts for fsspeckit-backed managed Parquet maintenance."""

from __future__ import annotations


import pyarrow as pa

from pydala import ParquetDataset
from pydala.filesystem import FileSystem

import pyarrow.parquet as pq
import pytest

from pydala.dataset import _normalize_compaction_sort_by
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


@pytest.mark.parametrize(
    ("sort_by", "expected"),
    [
        ("id desc, name asc", [("id", True), ("name", False)]),
        (["id", "name"], [("id", False), ("name", False)]),
        ([("id", "descending"), ["name", "asc"]], [("id", True), ("name", False)]),
        (("id", "desc"), [("id", True)]),
    ],
)
def test_compaction_sort_by_normalizes_public_forms(sort_by, expected) -> None:
    keys = _normalize_compaction_sort_by(sort_by, ["id", "name"])

    assert [(key.column, key.descending) for key in keys] == expected
    assert all(key.nulls_first is False for key in keys)


@pytest.mark.parametrize("sort_by", ["missing", "id sideways", [("id", "up")]])
def test_compaction_sort_by_rejects_invalid_columns_and_directions(sort_by) -> None:
    with pytest.raises(ValueError):
        _normalize_compaction_sort_by(sort_by, ["id", "name"])


def test_ordered_compaction_sorts_within_and_across_output_files(
    managed_dataset,
) -> None:
    managed_dataset.compact_by_rows(
        max_rows_per_file=4,
        sort_by=[("id", "descending")],
        compression="zstd",
    )

    files = sorted(managed_dataset.files)
    file_values = [
        pq.ParquetFile(managed_dataset.fs.open(f"{managed_dataset.path}/{path}"))
        .read()
        .column("id")
        .to_pylist()
        for path in files
    ]
    flattened = [value for values in file_values for value in values]

    assert all(values == sorted(values, reverse=True) for values in file_values)
    assert all(
        left[-1] >= right[0] for left, right in zip(file_values, file_values[1:])
    )
    assert (
        managed_dataset.schema
        == pq.ParquetFile(
            managed_dataset.fs.open(f"{managed_dataset.path}/{files[0]}")
        ).schema_arrow
    )
    assert flattened == sorted(flattened, reverse=True)
    assert len(flattened) == 15
    assert all(len(values) <= 4 for values in file_values)
    assert_metadata_invariants(managed_dataset)


def test_ordered_compaction_dry_run_reports_ordered_groups(managed_dataset) -> None:
    plan = managed_dataset.compact_by_rows(
        max_rows_per_file=4,
        sort_by="id",
        compression="zstd",
        dry_run=True,
    )

    assert "ordered_groups" in plan
    assert plan["planned_groups"] == [
        [file["path"] for file in group["files"]] for group in plan["ordered_groups"]
    ]


def test_ordinary_compaction_dry_run_remains_unordered(managed_dataset) -> None:
    plan = managed_dataset.compact_by_rows(
        max_rows_per_file=4,
        compression="zstd",
        dry_run=True,
    )

    assert "ordered_groups" not in plan


def test_ordered_compaction_rejects_deduplication(managed_dataset) -> None:
    with pytest.raises(ValueError, match="sort_by cannot be combined with unique"):
        managed_dataset.compact_by_rows(
            max_rows_per_file=4,
            sort_by="id",
            unique=True,
        )


def test_partitioned_compaction_keeps_ordered_output_in_its_partition(
    local_path: str,
) -> None:
    dataset = ParquetDataset(
        path="partitioned",
        filesystem=FileSystem(bucket=local_path, cached=False),
    )
    for ids in ([4, 1], [3, 2]):
        dataset.write_to_dataset(
            pa.table({"id": ids, "region": ["north"] * len(ids)}),
            mode="append",
            partition_by=["region"],
        )
    dataset.update()

    dataset.compact_by_rows(
        max_rows_per_file=10,
        sort_by="id",
        unique=False,
        compression="zstd",
    )

    assert dataset.files
    assert all(path.startswith("region=north/") for path in dataset.files)
    output = pq.ParquetFile(
        dataset.fs.open(f"{dataset.path}/{dataset.files[0]}")
    ).read()
    assert output.column("id").to_pylist() == [1, 2, 3, 4]

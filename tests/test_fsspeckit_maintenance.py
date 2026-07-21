"""Focused contracts for fsspeckit-backed managed Parquet maintenance."""

from __future__ import annotations
import datetime as dt


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


def test_optimize_dtypes_dry_run_exposes_schema_rewrite_plan_without_mutation(
    managed_dataset,
) -> None:
    """The Polars proposal is published only after an explicit coordinated plan."""
    files_before = set(managed_dataset.files)
    metadata_before = set(managed_dataset.files_in_metadata)

    plan = managed_dataset.optimize_dtypes(exclude="name", dry_run=True)

    assert plan["target_schema"].field("name").type == pa.string()
    assert "schema_rewrite_groups" in plan
    assert plan["planned_groups"]
    assert set(managed_dataset.files) == files_before
    assert set(managed_dataset.files_in_metadata) == metadata_before
    assert_metadata_invariants(managed_dataset)


def test_optimize_dtypes_publishes_one_target_schema_and_timestamp_options(
    local_path: str,
) -> None:
    """Narrowing and timestamp conversion are coordinated across all files."""
    dataset = ParquetDataset(
        path="typed",
        filesystem=FileSystem(bucket=local_path, cached=False),
    )
    table = pa.table(
        {
            "integer_text": pa.array(["1", "2"], type=pa.string()),
            "float_text": pa.array(["1.0", "2.0"], type=pa.string()),
            "null_only": pa.nulls(2),
            "excluded": pa.array([1, 2], type=pa.int64()),
            "ts": pa.array(
                [0, 1_000],
                type=pa.timestamp("us", tz="UTC"),
            ),
        }
    )
    dataset.write_to_dataset(table, mode="append")
    dataset.write_to_dataset(table, mode="append")
    dataset.update()
    dataset.load(update_metadata=True)

    proposal = dataset.optimize_dtypes(
        exclude="excluded",
        ts_unit="ms",
        tz="UTC",
        dry_run=True,
    )
    assert proposal["target_schema"].field("excluded").type == pa.int64()
    assert proposal["target_schema"].field("integer_text").type == pa.uint8()
    assert proposal["target_schema"].field("float_text").type == pa.float32()
    assert proposal["target_schema"].field("null_only").type == pa.null()
    assert proposal["target_schema"].field("ts").type == pa.timestamp("ms", tz="UTC")
    include_only = dataset.optimize_dtypes(
        include="small_int", strict=False, dry_run=True
    )
    assert include_only["target_schema"].field("integer_text").type == pa.string()
    assert include_only["cast_policy"] == "loose"

    result = dataset.optimize_dtypes(
        exclude="excluded",
        ts_unit="ms",
        tz="UTC",
    )

    assert result["succeeded"] is True
    physical_schemas = [
        pq.read_schema(f"{dataset.path}/{file_path}", filesystem=dataset.fs)
        for file_path in dataset.files
    ]
    expected_fields = [
        (field.name, field.type)
        for field in proposal["target_schema"]
        if field.name != "ts"
    ]
    assert all(
        [(field.name, field.type) for field in schema if field.name != "ts"]
        == expected_fields
        for schema in physical_schemas
    )
    assert all(schema.field("ts").type.tz == "UTC" for schema in physical_schemas)
    assert dataset.has_metadata
    assert set(dataset.files_in_metadata) == set(dataset.files)
    assert set(dataset.files_in_file_metadata) == set(dataset.files)


def test_optimize_dtypes_publishes_explicit_float64_to_float32_proposal(
    local_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A value-preserving proposal publishes float64 values as float32."""
    dataset = ParquetDataset(
        path="float-narrowing",
        filesystem=FileSystem(bucket=local_path, cached=False),
    )
    dataset.write_to_dataset(
        pa.table({"value": pa.array([1.5, 2.25], type=pa.float64())}),
        mode="append",
    )
    dataset.update()
    dataset.load(update_metadata=True)
    target_schema = pa.schema([pa.field("value", pa.float32())])
    monkeypatch.setattr(dataset, "_dtype_schema_proposal", lambda **_: target_schema)

    result = dataset.optimize_dtypes()

    assert result["succeeded"] is True
    assert all(
        pq.read_schema(f"{dataset.path}/{file_path}", filesystem=dataset.fs)
        == target_schema
        for file_path in dataset.files
    )
    assert all(
        pq.read_table(f"{dataset.path}/{file_path}", filesystem=dataset.fs)
        .column("value")
        .to_pylist()
        == [1.5, 2.25]
        for file_path in dataset.files
    )
    assert dataset.has_metadata
    assert set(dataset.files_in_metadata) == set(dataset.files)
    assert set(dataset.files_in_file_metadata) == set(dataset.files)
    assert (
        pq.read_schema(
            f"{dataset.path}/_metadata",
            filesystem=dataset.fs,
        )
        .field("value")
        .type
        == pa.float32()
    )


def test_optimize_dtypes_partition_columns_are_not_in_physical_target(
    local_path: str,
) -> None:
    """Hive partition values stay in paths rather than the rewrite schema."""
    dataset = ParquetDataset(
        path="partitioned-types",
        filesystem=FileSystem(bucket=local_path, cached=False),
    )
    dataset.write_to_dataset(
        pa.table({"id": pa.array([1, 2], type=pa.int64()), "region": ["north"] * 2}),
        mode="append",
        partition_by=["region"],
    )
    dataset.update()
    dataset.load(update_metadata=True)

    plan = dataset.optimize_dtypes(dry_run=True)
    assert "region" not in plan["target_schema"].names

    dataset.optimize_dtypes()

    assert all(path.startswith("region=north/") for path in dataset.files)
    assert all(
        "region"
        not in pq.read_schema(
            f"{dataset.path}/{file_path}", filesystem=dataset.fs
        ).names
        for file_path in dataset.files
    )
    assert dataset.has_metadata
    assert set(dataset.files_in_metadata) == set(dataset.files)
    assert set(dataset.files_in_file_metadata) == set(dataset.files)


def test_optimize_dtypes_unsafe_target_schema_preserves_live_files_and_metadata(
    managed_dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LOOSE validates every value and aborts an overflowing supplied target."""
    managed_dataset.write_to_dataset(
        pa.table(
            {
                "id": pa.array([999], type=pa.int64()),
                "name": ["overflow"],
                "value": pa.array([1.5], type=pa.float64()),
                "timestamp": pa.array([0], type=pa.timestamp("us")),
            }
        ),
        mode="append",
    )
    managed_dataset.update()
    managed_dataset.load(update_metadata=True)
    files_before = list(managed_dataset.files)
    schemas_before = {
        file_path: pq.read_schema(
            f"{managed_dataset.path}/{file_path}",
            filesystem=managed_dataset.fs,
        )
        for file_path in files_before
    }
    values_before = {
        file_path: pq.read_table(
            f"{managed_dataset.path}/{file_path}",
            filesystem=managed_dataset.fs,
        ).to_pydict()
        for file_path in files_before
    }
    metadata_before = set(managed_dataset.files_in_metadata)
    sidecars_before = {
        metadata_file: managed_dataset.fs.cat(f"{managed_dataset.path}/{metadata_file}")
        for metadata_file in ("_metadata", "_file_metadata")
    }

    monkeypatch.setattr(
        managed_dataset,
        "_dtype_schema_proposal",
        lambda **_: pa.schema(
            [
                pa.field("id", pa.int8()),
                pa.field("name", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("timestamp", pa.timestamp("us")),
            ]
        ),
    )

    with pytest.raises(RuntimeError, match="would overflow.*recovery details"):
        managed_dataset.optimize_dtypes(strict=False)

    assert managed_dataset.files == files_before
    assert set(managed_dataset.files_in_metadata) == metadata_before
    assert all(
        pq.read_schema(
            f"{managed_dataset.path}/{file_path}",
            filesystem=managed_dataset.fs,
        )
        == schema
        for file_path, schema in schemas_before.items()
    )
    assert all(
        pq.read_table(
            f"{managed_dataset.path}/{file_path}",
            filesystem=managed_dataset.fs,
        ).to_pydict()
        == values
        for file_path, values in values_before.items()
    )
    assert_metadata_invariants(managed_dataset)

    assert all(
        managed_dataset.fs.cat(f"{managed_dataset.path}/{metadata_file}") == contents
        for metadata_file, contents in sidecars_before.items()
    )


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


def test_compaction_sort_by_treats_list_strings_as_columns() -> None:
    keys = _normalize_compaction_sort_by(["id", "desc"], ["id", "desc"])

    assert [(key.column, key.descending) for key in keys] == [
        ("id", False),
        ("desc", False),
    ]


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
    dataset.write_to_dataset(
        pa.table({"id": [4, 1, 3, 2], "region": ["north"] * 4}),
        mode="append",
        partition_by=["region"],
    )
    dataset.update()

    dataset.compact_by_rows(
        max_rows_per_file=2,
        sort_by="id",
        unique=False,
        compression="zstd",
    )

    assert dataset.files
    assert all(path.startswith("region=north/") for path in dataset.files)
    file_values = [
        pq.ParquetFile(dataset.fs.open(f"{dataset.path}/{path}"))
        .read()
        .column("id")
        .to_pylist()
        for path in sorted(dataset.files)
    ]
    assert file_values == [[1, 2], [3, 4]]


def test_timeperiod_ordered_compaction_rewrites_full_physical_partition(
    local_path: str,
) -> None:
    dataset = ParquetDataset(
        path="timepartitioned",
        partitioning="hive",
        filesystem=FileSystem(bucket=local_path, cached=False),
    )
    dataset.write_to_dataset(
        pa.table(
            {
                "id": [4, 1],
                "ts": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
                "region": ["north"] * 2,
            }
        ),
        mode="append",
        partition_by=["region"],
    )
    dataset.write_to_dataset(
        pa.table(
            {
                "id": [3, 2],
                "ts": [dt.datetime(2024, 1, 3), dt.datetime(2024, 1, 4)],
                "region": ["north"] * 2,
            }
        ),
        mode="append",
        partition_by=["region"],
    )
    dataset.update()

    dataset.compact_by_timeperiod(
        interval="1d",
        timestamp_column="ts",
        max_rows_per_file=1,
        sort_by="id",
        unique=False,
        compression="zstd",
    )

    file_values = [
        pq.ParquetFile(dataset.fs.open(f"{dataset.path}/{path}"))
        .read()
        .column("id")
        .to_pylist()
        for path in sorted(dataset.files)
    ]
    flattened = [value for values in file_values for value in values]

    assert all(path.startswith("region=north/") for path in dataset.files)
    assert file_values == [[1], [2], [3], [4]]
    assert flattened == sorted(flattened)

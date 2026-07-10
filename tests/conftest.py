"""Shared fixtures for pydala2 characterization tests.

Provides a small managed Parquet dataset whose metadata invariants
can be checked before and after later transport/maintenance migrations.
"""

from __future__ import annotations

import pathlib
import tempfile

import pyarrow as pa

import pytest

from pydala import ParquetDataset
from pydala.filesystem import FileSystem


# --------------------------------------------------------------------------- #
# Schemas and sample data
# --------------------------------------------------------------------------- #

SIMPLE_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("name", pa.string()),
        pa.field("value", pa.float64()),
        pa.field("timestamp", pa.timestamp("us")),
    ]
)


def make_simple_table(n_rows: int = 10, seed: int = 0) -> pa.Table:
    """Return a small deterministic PyArrow table for fixture use."""
    ids = list(range(seed, seed + n_rows))
    names = [f"row_{i}" for i in ids]
    values = [float(i) * 1.5 for i in ids]
    timestamps = pa.array([f"2024-01-{(i % 28) + 1:02d}T00:00:00" for i in ids]).cast(
        pa.timestamp("us")
    )
    return pa.table(
        {
            "id": ids,
            "name": names,
            "value": values,
            "timestamp": timestamps,
        },
        schema=SIMPLE_SCHEMA,
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def local_fs() -> FileSystem:
    """Return a pydala FileSystem rooted at a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = FileSystem(bucket=tmpdir, cached=False)
        yield fs


@pytest.fixture
def local_path() -> str:
    """Return a temporary directory path that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def managed_dataset(local_path: str) -> ParquetDataset:
    """Return a small managed Parquet dataset with metadata.

    Writes two batches of rows so the dataset has multiple files,
    then updates metadata so ``_metadata`` and ``_file_metadata``
    exist on disk.
    """
    dataset_path = pathlib.Path(local_path) / "test_dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)

    ds = ParquetDataset(
        path="test_dataset",
        filesystem=FileSystem(bucket=str(local_path), cached=False),
    )

    table = make_simple_table(n_rows=10)
    ds.write_to_dataset(table, mode="append")
    ds.write_to_dataset(make_simple_table(n_rows=5, seed=10), mode="append")
    ds.update()

    yield ds


# --------------------------------------------------------------------------- #
# Metadata invariant assertions
# --------------------------------------------------------------------------- #


def assert_metadata_invariants(dataset: ParquetDataset) -> None:
    """Assert the canonical metadata invariants for a managed ParquetDataset.

    Every characterization test that exercises a managed dataset should call
    this after any operation that could change files or metadata.
    """
    # The dataset must report a metadata file present.
    assert dataset.has_metadata_file, "dataset.has_metadata_file is False"

    # The aggregate metadata must be loadable.
    assert dataset.has_metadata, "dataset.has_metadata is False"

    # The metadata row-group count must exceed zero.
    assert dataset.metadata.num_row_groups > 0, "metadata has zero row groups"

    # The metadata schema must match the number of columns in SIMPLE_SCHEMA.
    arrow_schema = dataset.metadata.schema.to_arrow_schema()
    assert len(arrow_schema.names) == len(SIMPLE_SCHEMA.names), (
        "metadata schema column count mismatch"
    )

    # Both metadata sidecars must exist and agree with the data files
    # discovered through the public dataset interface.
    assert dataset.has_file_metadata_file, "dataset.has_file_metadata_file is False"
    assert dataset.has_file_metadata, "dataset.has_file_metadata is False"

    files = set(dataset.files)
    metadata_files = set(dataset.files_in_metadata)
    file_metadata_files = set(dataset.files_in_file_metadata)
    assert files, "dataset contains no data files"
    assert metadata_files == files, "aggregate metadata files do not match dataset files"
    assert file_metadata_files == files, "file metadata entries do not match dataset files"

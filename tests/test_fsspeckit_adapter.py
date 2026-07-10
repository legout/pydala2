"""Focused contracts for the fsspeckit-backed managed Parquet read seam."""

from __future__ import annotations

import tempfile
import unittest
import uuid

import pyarrow as pa
import pyarrow.dataset as pds

from pydala import ParquetDataset
from pydala.filesystem import FileSystem
from pydala.table import PydalaTable


class TestFsspeckitManagedParquetAdapter(unittest.TestCase):
    """Public pydala filtering stays stable over fsspeckit backends."""

    @staticmethod
    def _write_dataset(filesystem: FileSystem) -> ParquetDataset:
        dataset = ParquetDataset(path="events", filesystem=filesystem)
        dataset.write_to_dataset(
            pa.table({"id": [1, 2, 3], "name": ["one", "two", "three"]}),
            mode="append",
        )
        dataset.update()
        return dataset

    def test_local_pyarrow_adapter_preserves_sql_filter_and_arrow_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._write_dataset(FileSystem(bucket=directory, cached=False))

            filtered = dataset.filter("id >= 2", use="fsspeckit")

            self.assertIsInstance(filtered, PydalaTable)
            self.assertEqual(filtered.to_arrow().column("id").to_pylist(), [2, 3])

    def test_local_duckdb_adapter_translates_expression_string_literals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._write_dataset(FileSystem(bucket=directory, cached=False))

            filtered = dataset.filter(
                pds.field("name") == "two", use="fsspeckit-duckdb"
            )

            self.assertEqual(
                filtered.to_arrow().column("name").to_pylist(), ["two"]
            )

    def test_memory_pyarrow_adapter_routes_through_fsspeckit(self) -> None:
        filesystem = FileSystem(
            bucket=f"adapter-{uuid.uuid4().hex}", protocol="memory", cached=False
        )
        dataset = self._write_dataset(filesystem)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            filtered = dataset.filter("id >= 2", use="fsspeckit")

        self.assertIsInstance(filtered, PydalaTable)
        self.assertEqual(filtered.to_arrow().column("id").to_pylist(), [2, 3])

    def test_memory_duckdb_adapter_routes_through_fsspeckit(self) -> None:
        filesystem = FileSystem(
            bucket=f"adapter-{uuid.uuid4().hex}", protocol="memory", cached=False
        )
        dataset = self._write_dataset(filesystem)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            filtered = dataset.filter(
                pds.field("name") == "two", use="fsspeckit-duckdb"
            )

        self.assertIsInstance(filtered, PydalaTable)
        self.assertEqual(filtered.to_arrow().column("name").to_pylist(), ["two"])

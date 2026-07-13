"""Tests for P0 bugs identified in the deep code review.

Each test targets a specific crash or silent-failure bug:
  1. write_table(filesystem=None) crashes with AttributeError
  2. _get_unified_schema crashes with NameError when has_file_metadata is False
  3. vacuum() is an empty stub that deletes nothing
  4. interrupt_duckdb() is an empty stub that never interrupts
  5. Catalog.create_table(data=dataset) crashes on _partitioning_columns
  6. registered_tables crashes because .arrow() returns RecordBatchReader
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from pydala import ParquetDataset
from pydala.filesystem import FileSystem


def _make_simple_table(n_rows: int = 5) -> pa.Table:
    return pa.table(
        {
            "id": list(range(n_rows)),
            "name": [f"row_{i}" for i in range(n_rows)],
            "value": [float(i) * 1.5 for i in range(n_rows)],
        }
    )


class TestWriteTableFilesystemNone(unittest.TestCase):
    """write_table must not crash when filesystem is None."""

    def test_write_table_creates_local_filesystem_when_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(pathlib.Path(tmpdir) / "out.parquet")
            table = _make_simple_table()

            from pydala.io import write_table

            written_path, metadata = write_table(table, path, filesystem=None)

            self.assertTrue(pathlib.Path(written_path).exists())
            self.assertIsInstance(metadata, pq.FileMetaData)


class TestGetUnifiedSchemaNoFileMetadata(unittest.TestCase):
    """_get_unified_schema must not crash when has_file_metadata is False."""

    def test_returns_schema_without_file_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystem(bucket=tmpdir, cached=False)
            ds = ParquetDataset(path="ds", filesystem=fs)
            ds.write_to_dataset(_make_simple_table(), mode="append")
            ds.update()

            # Simulate missing _file_metadata sidecar.
            if ds.has_file_metadata_file:
                ds._filesystem.rm(ds._file_metadata_file)
                ds._file_metadata = None
                ds.clear_cache()

            schema, equal = ds._get_unified_schema()
            self.assertIsInstance(schema, pa.Schema)
            self.assertIn("id", schema.names)


class TestVacuumDeletesFiles(unittest.TestCase):
    """vacuum() must delete all data files, not be a silent no-op."""

    def test_vacuum_removes_data_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystem(bucket=tmpdir, cached=False)
            ds = ParquetDataset(path="ds", filesystem=fs)
            ds.write_to_dataset(_make_simple_table(), mode="append")
            ds.write_to_dataset(_make_simple_table(), mode="append")
            ds.update()

            self.assertTrue(len(ds.files) > 0)

            ds.vacuum()

            self.assertEqual(len(ds.files), 0)


class _FakeConnection:
    """Minimal stand-in for DuckDBPyConnection."""

    def __init__(self):
        self.interrupt_called = False

    def interrupt(self):
        self.interrupt_called = True


class TestInterruptDuckdb(unittest.TestCase):
    """interrupt_duckdb() must actually call the connection's interrupt()."""

    def test_interrupt_duckdb_invokes_connection_interrupt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystem(bucket=tmpdir, cached=False)
            ds = ParquetDataset(path="ds", filesystem=fs)

            fake = _FakeConnection()
            ds.ddb_con = fake

            ds.interrupt_duckdb()

            self.assertTrue(
                fake.interrupt_called,
                "interrupt_duckdb must call ddb_con.interrupt()",
            )


class TestRegisteredTables(unittest.TestCase):
    """registered_tables must return a list, not crash on .arrow()."""

    def test_registered_tables_returns_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystem(bucket=tmpdir, cached=False)
            ds = ParquetDataset(path="ds", filesystem=fs)
            ds.write_to_dataset(_make_simple_table(), mode="append")

            tables = ds.registered_tables
            self.assertIsInstance(tables, list)

    def test_reset_duckdb_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystem(bucket=tmpdir, cached=False)
            ds = ParquetDataset(path="ds", filesystem=fs)
            ds.write_to_dataset(_make_simple_table(), mode="append")

            ds.reset_duckdb()
            result = ds.ddb_con.sql("SELECT 42 as v").fetchone()
            self.assertEqual(result[0], 42)


class TestCatalogCreateTableWithDataset(unittest.TestCase):
    """Catalog.create_table(data=dataset) must not crash on _partitioning_columns."""

    def test_create_table_from_dataset_object(self):
        from pydala import Catalog

        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystem(bucket=tmpdir, cached=False)

            src = ParquetDataset(path="src", filesystem=fs)
            src.write_to_dataset(_make_simple_table(), mode="append")

            # Create a minimal catalog.yml so load_catalog doesn't fail.
            catalog_path = pathlib.Path(tmpdir) / "catalog.yml"
            with open(catalog_path, "w") as f:
                yaml.dump({"tables": {"ns": {}}}, f)

            cat = Catalog(path="catalog.yml", namespace="ns", bucket=tmpdir)

            cat.create_table(
                data=src,
                table_name="my_table",
                write_catalog=True,
            )


if __name__ == "__main__":
    unittest.main()

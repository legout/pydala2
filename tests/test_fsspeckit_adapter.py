"""Focused contracts for the fsspeckit-backed managed Parquet read/merge seam."""

from __future__ import annotations

import tempfile
import unittest
import uuid

import pyarrow as pa
import pyarrow.dataset as pds

from pydala import ParquetDataset
from pydala.filesystem import FileSystem
from pydala.table import PydalaTable
from fsspeckit.core.incremental import MergeResult

from pydala.adapters.fsspeckit import FsspeckitParquetAdapter


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

            self.assertEqual(filtered.to_arrow().column("name").to_pylist(), ["two"])

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


class TestFsspeckitParquetAdapterMerge(unittest.TestCase):
    """Typed merge contracts for ``FsspeckitParquetAdapter.merge``.

    The adapter owns pydala-relative path conversion, filesystem reuse,
    backend branching, and ``partition_by`` -> ``partition_columns``
    translation.  It must return fsspeckit's typed ``MergeResult`` unchanged
    and never expose fsspeckit IO handler construction to callers.
    """

    @staticmethod
    def _bootstrap_one_row(adapter: FsspeckitParquetAdapter, path: str) -> None:
        """Seed the target dataset with a single existing key."""
        adapter.merge(
            pa.table({"id": [1], "value": ["old"]}),
            path,
            strategy="upsert",
            key_columns=["id"],
        )

    # ----------------------------------------------------------------- #
    # Insert contract: verified result counts
    # ----------------------------------------------------------------- #

    def test_local_pyarrow_insert_returns_typed_result_with_verified_counts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = FsspeckitParquetAdapter(
                filesystem=FileSystem(bucket=directory, cached=False)
            )
            self._bootstrap_one_row(adapter, "events")

            result = adapter.merge(
                pa.table({"id": [1, 2], "value": ["new", "two"]}),
                "events",
                strategy="insert",
                key_columns=["id"],
            )

            self.assertIsInstance(result, MergeResult)
            self.assertEqual(result.strategy, "insert")
            self.assertEqual(result.target_count_before, 1)
            self.assertEqual(result.inserted, 1)
            self.assertEqual(result.updated, 0)
            self.assertEqual(result.target_count_after, 2)
            # Insert never rewrites existing target files.
            self.assertEqual(result.rewritten_files, [])
            # The existing key is preserved untouched.
            rows = sorted(
                adapter.read_parquet("events").to_pylist(), key=lambda r: r["id"]
            )
            self.assertEqual([r["value"] for r in rows], ["old", "two"])

    def test_memory_pyarrow_insert_returns_typed_result_with_verified_counts(
        self,
    ) -> None:
        adapter = FsspeckitParquetAdapter(
            filesystem=FileSystem(
                bucket=f"merge-{uuid.uuid4().hex}", protocol="memory", cached=False
            )
        )
        self._bootstrap_one_row(adapter, "events")

        result = adapter.merge(
            pa.table({"id": [1, 2], "value": ["new", "two"]}),
            "events",
            strategy="insert",
            key_columns=["id"],
        )

        self.assertIsInstance(result, MergeResult)
        self.assertEqual(result.target_count_before, 1)
        self.assertEqual(result.inserted, 1)
        self.assertEqual(result.updated, 0)
        self.assertEqual(result.target_count_after, 2)
        rows = sorted(adapter.read_parquet("events").to_pylist(), key=lambda r: r["id"])
        self.assertEqual([r["value"] for r in rows], ["old", "two"])

    def test_pyarrow_is_the_default_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = FsspeckitParquetAdapter(
                filesystem=FileSystem(bucket=directory, cached=False)
            )
            self._bootstrap_one_row(adapter, "events")

            default = adapter.merge(
                pa.table({"id": [2], "value": ["two"]}),
                "events",
                strategy="insert",
                key_columns=["id"],
            )
            explicit = adapter.merge(
                pa.table({"id": [3], "value": ["three"]}),
                "events",
                strategy="insert",
                key_columns=["id"],
                backend="pyarrow",
            )

            self.assertIsInstance(default, MergeResult)
            self.assertIsInstance(explicit, MergeResult)
            self.assertEqual(default.target_count_after, 2)
            self.assertEqual(explicit.target_count_after, 3)

    def test_pyarrow_works_with_generic_fsspec_filesystem(self) -> None:
        # A bare fsspec LocalFileSystem (not wrapped by pydala's
        # DirFileSystem) must still flow through the PyArrow backend, which
        # supports arbitrary fsspec filesystems.
        import os

        from fsspec import filesystem

        with tempfile.TemporaryDirectory() as directory:
            adapter = FsspeckitParquetAdapter(filesystem=filesystem("file"))
            path = os.path.join(directory, "events")

            result = adapter.merge(
                pa.table({"id": [1], "value": ["one"]}),
                path,
                strategy="upsert",
                key_columns=["id"],
            )

            self.assertIsInstance(result, MergeResult)
            self.assertEqual(result.target_count_after, 1)
            self.assertEqual(adapter.read_parquet(path).column("id").to_pylist(), [1])

    # ----------------------------------------------------------------- #
    # DuckDB explicit backend
    # ----------------------------------------------------------------- #

    def test_local_duckdb_insert_explicit_backend_with_verified_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = FsspeckitParquetAdapter(
                filesystem=FileSystem(bucket=directory, cached=False)
            )
            self._bootstrap_one_row(adapter, "events")

            result = adapter.merge(
                pa.table({"id": [1, 2], "value": ["new", "two"]}),
                "events",
                strategy="insert",
                key_columns=["id"],
                backend="duckdb",
            )

            self.assertIsInstance(result, MergeResult)
            self.assertEqual(result.target_count_before, 1)
            self.assertEqual(result.inserted, 1)
            self.assertEqual(result.updated, 0)
            self.assertEqual(result.target_count_after, 2)
            rows = sorted(
                adapter.read_parquet("events").to_pylist(), key=lambda r: r["id"]
            )
            self.assertEqual([r["value"] for r in rows], ["old", "two"])

    def test_duckdb_backend_rejects_memory_filesystem(self) -> None:
        # DuckDB cannot write to fsspec MemoryFileSystem.  The adapter must
        # encode this source-verified limitation explicitly rather than
        # silently diverging to the PyArrow backend.
        adapter = FsspeckitParquetAdapter(
            filesystem=FileSystem(
                bucket=f"merge-{uuid.uuid4().hex}", protocol="memory", cached=False
            )
        )

        with self.assertRaises(ValueError) as ctx:
            adapter.merge(
                pa.table({"id": [1], "value": ["one"]}),
                "events",
                strategy="upsert",
                key_columns=["id"],
                backend="duckdb",
            )

        self.assertIn("duckdb", str(ctx.exception).lower())

    # ----------------------------------------------------------------- #
    # Composite keys
    # ----------------------------------------------------------------- #

    def test_composite_keys_insert_update_and_upsert(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = FsspeckitParquetAdapter(
                filesystem=FileSystem(bucket=directory, cached=False)
            )
            # Bootstrap two composite keys.
            adapter.merge(
                pa.table({"a": [1, 1], "b": ["x", "y"], "value": ["ax", "by"]}),
                "events",
                strategy="upsert",
                key_columns=["a", "b"],
            )

            # Insert: existing (1,x) preserved, new (2,z) appended.
            insert_result = adapter.merge(
                pa.table({"a": [1, 2], "b": ["x", "z"], "value": ["IGNORED", "cz"]}),
                "events",
                strategy="insert",
                key_columns=["a", "b"],
            )
            self.assertEqual(insert_result.inserted, 1)
            self.assertEqual(insert_result.updated, 0)
            self.assertEqual(insert_result.target_count_after, 3)

            # Update: rewrite the matched (1,y) row.
            update_result = adapter.merge(
                pa.table({"a": [1], "b": ["y"], "value": ["BY"]}),
                "events",
                strategy="update",
                key_columns=["a", "b"],
            )
            self.assertEqual(update_result.updated, 1)
            self.assertEqual(update_result.inserted, 0)
            self.assertEqual(update_result.target_count_after, 3)
            self.assertEqual(len(update_result.rewritten_files), 1)

            # Upsert: update (1,x) and insert (3,w).
            upsert_result = adapter.merge(
                pa.table({"a": [1, 3], "b": ["x", "w"], "value": ["AX", "cw"]}),
                "events",
                strategy="upsert",
                key_columns=["a", "b"],
            )
            self.assertEqual(upsert_result.updated, 1)
            self.assertEqual(upsert_result.inserted, 1)
            self.assertEqual(upsert_result.target_count_after, 4)

            rows = sorted(
                adapter.read_parquet("events").to_pylist(),
                key=lambda r: (r["a"], r["b"]),
            )
            self.assertEqual(
                [(r["a"], r["b"], r["value"]) for r in rows],
                [(1, "x", "AX"), (1, "y", "BY"), (2, "z", "cz"), (3, "w", "cw")],
            )

    # ----------------------------------------------------------------- #
    # Partitioned insert/update/upsert (partition_by -> partition_columns)
    # ----------------------------------------------------------------- #

    def test_partition_by_translates_to_hive_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = FsspeckitParquetAdapter(
                filesystem=FileSystem(bucket=directory, cached=False)
            )

            adapter.merge(
                pa.table(
                    {"id": [1, 2], "region": ["eu", "us"], "value": ["one", "two"]}
                ),
                "events",
                strategy="upsert",
                key_columns=["id"],
                partition_by=["region"],
            )

            # Hive partition directories must exist under the dataset root.
            listing = adapter.filesystem.find("events")
            self.assertTrue(any("region=eu" in p for p in listing))
            self.assertTrue(any("region=us" in p for p in listing))

    def test_partitioned_insert_update_and_upsert(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = FsspeckitParquetAdapter(
                filesystem=FileSystem(bucket=directory, cached=False)
            )
            # Bootstrap a partitioned dataset; key column differs from the
            # partition column (fsspeckit requires the key to be physically
            # present in each parquet file).
            adapter.merge(
                pa.table(
                    {"id": [1, 2], "region": ["eu", "us"], "value": ["one", "two"]}
                ),
                "events",
                strategy="upsert",
                key_columns=["id"],
                partition_by=["region"],
            )

            # Insert: existing id=1 preserved, new id=3 appended into eu.
            insert_result = adapter.merge(
                pa.table(
                    {
                        "id": [1, 3],
                        "region": ["eu", "eu"],
                        "value": ["IGNORED", "three"],
                    }
                ),
                "events",
                strategy="insert",
                key_columns=["id"],
                partition_by=["region"],
            )
            self.assertEqual(insert_result.inserted, 1)
            self.assertEqual(insert_result.updated, 0)
            self.assertEqual(insert_result.target_count_after, 3)

            # Update: rewrite the matched id=2 row in the us partition.
            update_result = adapter.merge(
                pa.table({"id": [2], "region": ["us"], "value": ["TWO"]}),
                "events",
                strategy="update",
                key_columns=["id"],
                partition_by=["region"],
            )
            self.assertEqual(update_result.updated, 1)
            self.assertEqual(update_result.inserted, 0)
            self.assertEqual(update_result.target_count_after, 3)
            self.assertEqual(len(update_result.rewritten_files), 1)

            # Upsert: update id=1 (eu) and insert id=4 (us).
            upsert_result = adapter.merge(
                pa.table(
                    {"id": [1, 4], "region": ["eu", "us"], "value": ["ONE", "four"]}
                ),
                "events",
                strategy="upsert",
                key_columns=["id"],
                partition_by=["region"],
            )
            self.assertEqual(upsert_result.updated, 1)
            self.assertEqual(upsert_result.inserted, 1)
            self.assertEqual(upsert_result.target_count_after, 4)

            rows = sorted(
                adapter.read_parquet("events").to_pylist(), key=lambda r: r["id"]
            )
            self.assertEqual(
                [(r["id"], r["value"]) for r in rows],
                [(1, "ONE"), (2, "TWO"), (3, "three"), (4, "four")],
            )

    # ----------------------------------------------------------------- #
    # Argument validation
    # ----------------------------------------------------------------- #

    def test_invalid_strategy_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = FsspeckitParquetAdapter(
                filesystem=FileSystem(bucket=directory, cached=False)
            )
            with self.assertRaises(ValueError):
                adapter.merge(
                    pa.table({"id": [1]}),
                    "events",
                    strategy="delete",
                    key_columns=["id"],
                )

    def test_invalid_backend_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = FsspeckitParquetAdapter(
                filesystem=FileSystem(bucket=directory, cached=False)
            )
            with self.assertRaises(ValueError):
                adapter.merge(
                    pa.table({"id": [1]}),
                    "events",
                    strategy="insert",
                    key_columns=["id"],
                    backend="polars",
                )

    def test_merge_options_are_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = FsspeckitParquetAdapter(
                filesystem=FileSystem(bucket=directory, cached=False)
            )
            # ``compression`` and ``row_group_size`` are fsspeckit merge/write
            # tuning options forwarded through ``**merge_options``.
            result = adapter.merge(
                pa.table({"id": [1], "value": ["one"]}),
                "events",
                strategy="upsert",
                key_columns=["id"],
                compression="zstd",
                row_group_size=128,
            )
            self.assertIsInstance(result, MergeResult)
            self.assertEqual(result.target_count_after, 1)

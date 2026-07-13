"""Characterization contract tests for the pydala filesystem seam (issue #4).

These tests pin down the *current, observable* behaviour of the
``pydala.filesystem.FileSystem`` factory and the ``clear_cache`` helper.

``FileSystem`` is a thin wrapper around ``fsspeckit.core.filesystem.filesystem``
(see ``pydala/filesystem.py``). By default it returns a
:class:`~fsspec.implementations.dirfs.DirFileSystem` rooted at ``bucket`` (for a
local root) or at the memory bucket. With ``cached=True`` it instead returns a
``MonitoredSimpleCacheFileSystem`` layered over that same ``DirFileSystem``.

The tests are intentionally behavioural rather than structural: they describe
what the seam does *today* -- including the parts of the catalog's
``as_dataset=False`` (direct IO) path that are currently broken -- so that a
future migration off the old monkey-patched methods can detect regressions and
fixes alike.

Conventions follow ``tests/test_table.py``: ``unittest.TestCase`` classes with
``setUp``/``tearDown`` owning their own temp directories.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem
from pydala.catalog import Catalog
from pydala.dataset import JSONDataset, PyarrowDataset
from pydala.filesystem import FileSystem, clear_cache

from tests.conftest import assert_metadata_invariants


# --------------------------------------------------------------------------- #
# 1. Local filesystem root
# --------------------------------------------------------------------------- #
class TestLocalFilesystemRoot(unittest.TestCase):
    """``FileSystem(bucket=<tmpdir>, cached=False)`` is a working local fs."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self.fs = FileSystem(bucket=self.tmpdir, cached=False)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_factory_returns_dirfilesystem_over_local(self) -> None:
        self.assertIsInstance(self.fs, DirFileSystem)
        self.assertIsInstance(self.fs.fs, LocalFileSystem)

    def test_write_then_read_roundtrip(self) -> None:
        with self.fs.open("hello.txt", "wt") as f:
            f.write("hello world")
        with self.fs.open("hello.txt", "rt") as f:
            self.assertEqual(f.read(), "hello world")
        self.assertTrue(self.fs.exists("hello.txt"))

    def test_file_actually_lands_on_disk(self) -> None:
        with self.fs.open("on_disk.txt", "wt") as f:
            f.write("persisted")
        on_disk = os.path.join(self.tmpdir, "on_disk.txt")
        self.assertTrue(os.path.exists(on_disk))
        with open(on_disk) as f:
            self.assertEqual(f.read(), "persisted")

    def test_read_and_write_bytes(self) -> None:
        payload = b"\x00\x01\x02\xff"
        self.fs.write_bytes("blob.bin", payload)
        self.assertEqual(self.fs.read_bytes("blob.bin"), payload)

    def test_glob_and_ls_list_contents(self) -> None:
        self.fs.write_bytes("a.txt", b"a")
        self.fs.write_bytes("b.txt", b"b")
        listing = self.fs.ls("")
        self.assertIn("a.txt", listing)
        self.assertIn("b.txt", listing)
        self.assertEqual(sorted(self.fs.glob("*.txt")), ["a.txt", "b.txt"])

    def test_nested_paths_with_makedirs(self) -> None:
        self.fs.makedirs("nested/deep", exist_ok=True)
        with self.fs.open("nested/deep/file.txt", "wt") as f:
            f.write("deep")
        self.assertTrue(self.fs.exists("nested/deep/file.txt"))
        self.assertEqual(self.fs.read_bytes("nested/deep/file.txt"), b"deep")


# --------------------------------------------------------------------------- #
# 2. Memory filesystem
# --------------------------------------------------------------------------- #
class TestMemoryFilesystem(unittest.TestCase):
    """``FileSystem(bucket=..., protocol='memory', cached=False)`` round-trips."""

    def setUp(self) -> None:
        # A unique bucket keeps each test isolated in the process-wide
        # in-memory store.
        self.bucket = f"mem-{uuid.uuid4().hex[:8]}"
        self.fs = FileSystem(bucket=self.bucket, protocol="memory", cached=False)

    def test_factory_returns_dirfilesystem_over_memory(self) -> None:
        self.assertIsInstance(self.fs, DirFileSystem)
        self.assertIsInstance(self.fs.fs, MemoryFileSystem)

    def test_write_then_read_roundtrip(self) -> None:
        with self.fs.open("note.txt", "wt") as f:
            f.write("in memory")
        with self.fs.open("note.txt", "rt") as f:
            self.assertEqual(f.read(), "in memory")
        self.assertTrue(self.fs.exists("note.txt"))

    def test_read_write_bytes_and_cat_file(self) -> None:
        payload = b"memory-bytes"
        self.fs.write_bytes("data.bin", payload)
        self.assertEqual(self.fs.read_bytes("data.bin"), payload)
        self.assertEqual(self.fs.cat_file("data.bin"), payload)

    def test_glob_lists_contents(self) -> None:
        self.fs.write_bytes("x.json", b"{}")
        self.fs.write_bytes("y.json", b"{}")
        self.assertEqual(sorted(self.fs.glob("*.json")), ["x.json", "y.json"])


# --------------------------------------------------------------------------- #
# 3. Cached vs uncached lifecycle
# --------------------------------------------------------------------------- #
class TestMemoryCachedFilesystem(unittest.TestCase):
    """Cached memory roots expose the same public read/write interface."""

    def setUp(self) -> None:
        self.bucket = f"mem-cached-{uuid.uuid4().hex[:8]}"
        self._cache = tempfile.TemporaryDirectory()
        self.fs = FileSystem(
            bucket=self.bucket,
            protocol="memory",
            cached=True,
            cache_storage=self._cache.name,
        )

    def tearDown(self) -> None:
        self._cache.cleanup()

    def test_cached_memory_roundtrip(self) -> None:
        self.fs.write_bytes("cached.bin", b"cached-memory")
        self.assertEqual(self.fs.read_bytes("cached.bin"), b"cached-memory")
        self.assertTrue(self.fs.exists("cached.bin"))
        self.assertTrue(getattr(self.fs, "is_cache_fs", False))


class TestCachedVsUncached(unittest.TestCase):
    """``cached=True`` yields a different class than ``cached=False``.

    Both must read/write through the same public interface (``.open()``).
    """

    def setUp(self) -> None:
        self._data = tempfile.TemporaryDirectory()
        self._cache = tempfile.TemporaryDirectory()
        self.datadir = self._data.name
        self.cachedir = self._cache.name
        self.uncached = FileSystem(bucket=self.datadir, cached=False)
        self.cached = FileSystem(
            bucket=self.datadir, cached=True, cache_storage=self.cachedir
        )

    def tearDown(self) -> None:
        self._data.cleanup()
        self._cache.cleanup()

    def test_uncached_is_plain_dirfilesystem(self) -> None:
        self.assertIsInstance(self.uncached, DirFileSystem)
        self.assertFalse(getattr(self.uncached, "is_cache_fs", True))

    def test_cached_is_cache_filesystem_not_dirfilesystem(self) -> None:
        self.assertNotIsInstance(self.cached, DirFileSystem)
        self.assertTrue(getattr(self.cached, "is_cache_fs", False))

    def test_cached_and_uncached_are_different_classes(self) -> None:
        self.assertIsNot(type(self.cached), type(self.uncached))

    def test_cached_overlays_a_dirfilesystem(self) -> None:
        # The cache sits on top of the same DirFileSystem-backed local fs.
        self.assertIsInstance(self.cached.fs, DirFileSystem)
        self.assertIsInstance(self.cached.fs.fs, LocalFileSystem)

    def test_uncached_writes_and_reads(self) -> None:
        with self.uncached.open("u.txt", "wt") as f:
            f.write("uncached")
        with self.uncached.open("u.txt", "rt") as f:
            self.assertEqual(f.read(), "uncached")

    def test_cached_writes_and_reads_through_same_interface(self) -> None:
        with self.cached.open("c.txt", "wt") as f:
            f.write("cached")
        with self.cached.open("c.txt", "rb") as f:
            self.assertEqual(f.read(), b"cached")


# --------------------------------------------------------------------------- #
# 4. Nested DirFileSystem cache invalidation
# --------------------------------------------------------------------------- #
class TestCacheInvalidation(unittest.TestCase):
    """``clear_cache`` works on filesystems that wrap a nested DirFileSystem.

    By default the factory returns a ``DirFileSystem`` (uncached) or a
    ``MonitoredSimpleCacheFileSystem`` over a ``DirFileSystem`` (cached).
    ``clear_cache`` must invalidate the outer layer *and* the wrapped ``.fs``
    without raising, and a file written after invalidation must be visible.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._cache = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self.cachedir = self._cache.name
        self.uncached = FileSystem(bucket=self.tmpdir, cached=False)
        self.cached = FileSystem(
            bucket=self.tmpdir, cached=True, cache_storage=self.cachedir
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()
        self._cache.cleanup()

    def test_clear_cache_returns_none_without_error(self) -> None:
        # Both the plain DirFileSystem and the nested cache fs must clear.
        self.assertIsNone(clear_cache(self.uncached))
        self.assertIsNone(clear_cache(self.cached))

    def test_new_file_visible_after_clearing_uncached(self) -> None:
        self.uncached.write_bytes("first.txt", b"1")
        _ = self.uncached.ls("")  # populate the dir listing cache
        clear_cache(self.uncached)
        self.uncached.write_bytes("second.txt", b"2")
        self.assertTrue(self.uncached.exists("second.txt"))
        self.assertIn("second.txt", self.uncached.ls(""))

    def test_external_file_becomes_visible_to_cached_fs(self) -> None:
        # A file written *around* the cache (directly to disk) must become
        # visible to the cached filesystem once the cache is invalidated.
        self.cached.write_bytes("seed.txt", b"seed")
        _ = self.cached.ls("")
        with open(os.path.join(self.tmpdir, "external.txt"), "wt") as f:
            f.write("from outside")
        clear_cache(self.cached)
        self.assertTrue(self.cached.exists("external.txt"))


# --------------------------------------------------------------------------- #
# 5. Catalog direct IO (the as_dataset=False path)
# --------------------------------------------------------------------------- #
class TestCatalogDirectIO(unittest.TestCase):
    """Characterize the catalog's ``as_dataset=False`` (direct IO) path.

    ``Catalog.load_parquet`` / ``load_csv`` / ``load_json`` with
    ``as_dataset=False`` now read directly from the underlying filesystem.
    Parquet paths use ``pyarrow.parquet.read_table`` with the catalog
    filesystem; CSV and JSON paths use the fsspeckit-registered
    ``read_csv`` / ``read_json`` methods on the underlying filesystem.
    """

    def setUp(self) -> None:
        self._root = tempfile.TemporaryDirectory()
        self.root = self._root.name

        self._table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        # single-file parquet table
        self.single_dir = os.path.join(self.root, "single")
        os.makedirs(self.single_dir)
        pq.write_table(self._table, os.path.join(self.single_dir, "single.parquet"))

        # directory parquet table
        self.dir_root = os.path.join(self.root, "dirtable")
        os.makedirs(self.dir_root)
        pq.write_table(self._table, os.path.join(self.dir_root, "part.parquet"))
        self.empty_root = os.path.join(self.root, "empty")
        os.makedirs(self.empty_root)

        self._write_catalog()

    def _write_catalog(self) -> None:
        cfg = {
            "filesystem": {
                "single": {
                    "protocol": "file",
                    "bucket": self.single_dir,
                    "cached": False,
                },
                "dir": {
                    "protocol": "file",
                    "bucket": self.dir_root,
                    "cached": False,
                },
                "empty": {
                    "protocol": "file",
                    "bucket": self.empty_root,
                    "cached": False,
                },
            },
            "tables": {
                "default": {
                    "single_table": {
                        "path": "single.parquet",
                        "format": "parquet",
                        "filesystem": "single",
                    },
                    "dir_table": {
                        "path": ".",
                        "format": "parquet",
                        "filesystem": "dir",
                    },
                    "empty_csv": {
                        "path": ".",
                        "format": "csv",
                        "filesystem": "empty",
                    },
                    "empty_json": {
                        "path": ".",
                        "format": "json",
                        "filesystem": "empty",
                    },
                }
            },
        }
        self.catalog_path = os.path.join(self.root, "catalog.yml")
        with open(self.catalog_path, "w") as f:
            yaml.dump(cfg, f)

    def _catalog(self) -> Catalog:
        return Catalog(path="catalog.yml", namespace="default", bucket=self.root)

    def tearDown(self) -> None:
        self._root.cleanup()

    # -- current, working surface ------------------------------------------- #

    def test_direct_reader_method_surface(self) -> None:
        """Document which direct-IO methods the fs currently exposes.

        ``fsspeckit`` registers the single-file readers/writers, but the
        ``*_dataset`` helpers are absent. The catalog therefore uses
        ``pq.read_table`` directly for directory parquet paths and the
        single-file readers for csv/json paths.
        """
        fs = FileSystem(bucket=self.single_dir, cached=False)
        for present in ("read_parquet", "read_csv", "read_json", "write_parquet"):
            self.assertTrue(hasattr(fs, present), f"{present} should be registered")
        for absent in (
            "read_parquet_dataset",
            "read_json_dataset",
            "read_csv_dataset",
        ):
            self.assertFalse(hasattr(fs, absent), f"{absent} should be absent")

    def test_single_file_csv_and_json_read_directly(self) -> None:
        """The single-file csv/json direct readers still work today."""
        fs = FileSystem(bucket=self.single_dir, cached=False)
        import polars as pl

        pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}).write_csv(
            os.path.join(self.single_dir, "data.csv")
        )
        pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}).write_json(
            os.path.join(self.single_dir, "data.json")
        )
        csv_df = fs.read_csv("data.csv")
        json_df = fs.read_json("data.json")
        self.assertEqual(csv_df.height, 2)
        self.assertEqual(json_df.height, 2)

    # -- known-broken behaviour (xfail) ------------------------------------- #

    def test_load_parquet_directory_direct_io(self) -> None:
        catalog = self._catalog()
        result = catalog.load_parquet("dir_table", as_dataset=False)
        self.assertEqual(
            getattr(result, "num_rows", getattr(result, "height", None)), 3
        )

    def test_load_parquet_single_file_direct_io(self) -> None:
        catalog = self._catalog()
        result = catalog.load_parquet("single_table", as_dataset=False)
        self.assertEqual(
            getattr(result, "num_rows", getattr(result, "height", None)), 3
        )

    def test_load_parquet_supports_compatible_fsspeckit_options(self) -> None:
        catalog = self._catalog()

        result = catalog.load_parquet(
            "single_table", as_dataset=False, include_file_path=True, opt_dtypes=True
        )

        self.assertIn("file_path", result.columns)
        self.assertEqual(
            result.get_column("file_path").to_list(), ["single.parquet"] * 3
        )
        with self.assertRaisesRegex(ValueError, "batch_size"):
            catalog.load_parquet("single_table", as_dataset=False, batch_size=1)
        with self.assertRaisesRegex(ValueError, "concat=True"):
            catalog.load_parquet("single_table", as_dataset=False, concat=False)

    def test_load_parquet_tracks_each_directory_source_path(self) -> None:
        pq.write_table(
            pa.table({"a": [4], "b": ["other"]}),
            os.path.join(self.dir_root, "other.parquet"),
        )

        result = self._catalog().load_parquet(
            "dir_table", as_dataset=False, include_file_path=True
        )

        self.assertEqual(result.height, 4)
        self.assertEqual(
            {os.path.basename(path) for path in result.get_column("file_path")},
            {"part.parquet", "other.parquet"},
        )

    def test_empty_direct_csv_and_json_paths_raise_clear_errors(self) -> None:
        catalog = self._catalog()

        with self.assertRaisesRegex(FileNotFoundError, "No CSV files"):
            catalog.load_csv("empty_csv", as_dataset=False)
        with self.assertRaisesRegex(FileNotFoundError, "No JSON files"):
            catalog.load_json("empty_json", as_dataset=False)


class TestManagedJSONDataset(unittest.TestCase):
    """Managed JSON datasets use fsspeckit's registered direct reader."""

    def test_loads_without_removed_dataset_reader(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            filesystem = FileSystem(bucket=root, cached=False)
            filesystem.makedirs("data")
            filesystem.pipe_file("data/part.json", b'[{"value": 1}, {"value": 2}]')
            dataset = JSONDataset("data", filesystem=filesystem)

            dataset.load()

            self.assertEqual(dataset._arrow_dataset.to_table().num_rows, 2)


class TestManagedDatasetMetadataInvariants:
    """Verify that a small managed Parquet dataset has consistent metadata."""

    def test_metadata_invariants_hold(self, managed_dataset) -> None:
        """The managed-dataset fixture must satisfy all metadata invariants.

        This is the characterization test that catches metadata corruption
        during transport and maintenance migrations.
        """
        assert_metadata_invariants(managed_dataset)

    def test_files_exist_on_disk(self, managed_dataset) -> None:
        """Both written batches must remain represented by physical data files."""
        assert len(managed_dataset.files_in_metadata) == 2
        assert set(managed_dataset.files_in_metadata) == set(managed_dataset.files)


class TestGenericDatasetEagerLoading(unittest.TestCase):
    """Generic dataset types are eager and load exactly once during construction."""

    def setUp(self) -> None:
        self._root = tempfile.TemporaryDirectory()
        self.root = self._root.name
        self.fs = FileSystem(bucket=self.root, cached=False)
        self.fs.makedirs("data")
        pq.write_table(
            pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}),
            os.path.join(self.root, "data", "part.parquet"),
        )

    def tearDown(self) -> None:
        self._root.cleanup()

    def test_pyarrow_dataset_is_query_ready(self) -> None:
        ds = PyarrowDataset("data", filesystem=self.fs)
        self.assertTrue(ds.is_loaded)
        self.assertEqual(ds.count_rows(), 3)

    def test_base_dataset_is_query_ready(self) -> None:
        from pydala.dataset import BaseDataset

        ds = BaseDataset("data", filesystem=self.fs, format="parquet")
        self.assertTrue(ds.is_loaded)
        self.assertEqual(ds.count_rows(), 3)

    def test_load_called_exactly_once(self) -> None:
        """Each concrete type invokes load exactly once during construction."""
        from pydala.dataset import BaseDataset

        call_count = [0]
        original_load = BaseDataset.load

        def counting_load(self):
            call_count[0] += 1
            return original_load(self)

        BaseDataset.load = counting_load
        try:
            PyarrowDataset("data", filesystem=self.fs)
        finally:
            BaseDataset.load = original_load
        self.assertEqual(call_count[0], 1)

    def test_no_dynamic_dispatch_during_base_init(self) -> None:
        """BaseDataset.__init__ must not call an overridden load() on subclasses."""
        from pydala.dataset import BaseDataset

        load_during_base_init = [False]
        original_init = BaseDataset.__init__

        def tracking_init(self, *args, **kwargs):
            original_load_method = type(self).load
            def spy_load(self_inner):
                load_during_base_init[0] = True
                return original_load_method(self_inner)
            type(self).load = spy_load
            try:
                original_init(self, *args, **kwargs)
            finally:
                type(self).load = original_load_method

        BaseDataset.__init__ = tracking_init
        try:
            PyarrowDataset("data", filesystem=self.fs)
        finally:
            BaseDataset.__init__ = original_init
        self.assertFalse(
            load_during_base_init[0],
            "load() must not be dispatched during BaseDataset.__init__",
        )
if __name__ == "__main__":
    unittest.main()

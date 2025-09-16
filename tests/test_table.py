import unittest
from unittest.mock import Mock
import pytest
import pyarrow.dataset as pds
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb as _duckdb
import polars as pl
from pydala.table import PydalaTable


class TestPydalaTable(unittest.TestCase):
    def setUp(self):
        # Create a mock table object
        # Create a simple PyArrow table for testing
        data = pa.table({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [10, 20, 30, 40, 50]
        })
        dataset = pds.dataset(data)
        self.table = PydalaTable(dataset)

    def test_scanner(self):
        # Test that scanner method returns a Scanner object
        scanner = self.table.scanner()
        self.assertIsInstance(scanner, pds.Scanner)

    def test_scanner_with_columns(self):
        # Test scanner with column selection
        scanner = self.table.scanner(columns=["col1", "col2"])
        self.assertIsInstance(scanner, pds.Scanner)

    def test_scanner_with_batch_size(self):
        # Test scanner with custom batch size
        scanner = self.table.scanner(batch_size=1024)
        self.assertIsInstance(scanner, pds.Scanner)

    def test_scanner_with_sort_by(self):
        # Test scanner with sorting
        scanner = self.table.scanner(sort_by="col1")
        self.assertIsInstance(scanner, pds.Scanner)

    def test_scanner_invalid_batch_size(self):
        # Test that invalid batch size raises ValueError
        with self.assertRaises(ValueError):
            self.table.scanner(batch_size=-1)

    def test_scanner_invalid_batch_readahead(self):
        # Test that invalid batch_readahead raises ValueError
        with self.assertRaises(ValueError):
            self.table.scanner(batch_readahead=-1)

    def test_to_scanner_deprecation(self):
        # Test that to_scanner shows deprecation warning
        with self.assertWarns(DeprecationWarning):
            scanner = self.table.to_scanner()
        self.assertIsInstance(scanner, pds.Scanner)

    def test_get_sort_by_string(self):
        # Test _get_sort_by with string input
        result = PydalaTable._get_sort_by("col1 desc, col2 asc")
        expected = {"sorting": [("col1", "descending"), ("col2", "ascending")]}
        self.assertEqual(result, expected)

    def test_get_sort_by_list(self):
        # Test _get_sort_by with list input
        result = PydalaTable._get_sort_by([["col1", "desc"], ["col2", "asc"]])
        expected = {"sorting": [("col1", "descending"), ("col2", "ascending")]}
        self.assertEqual(result, expected)

    def test_get_sort_by_invalid_direction(self):
        # Test _get_sort_by with invalid direction
        with self.assertRaises(ValueError):
            PydalaTable._get_sort_by([["col1", "invalid"]])

    def test_to_scanner_deprecation_warning(self):
        # Test that to_scanner shows deprecation warning
        with pytest.warns(DeprecationWarning, match="to_scanner\\(\\) is deprecated"):
            scanner = self.table.to_scanner()
        self.assertIsInstance(scanner, pds.Scanner)

    def test_scanner_with_config_defaults(self):
        # Test scanner uses config defaults when parameters are None
        scanner = self.table.scanner(
            batch_size=None,
            batch_readahead=None,
            fragment_readahead=None,
            use_threads=None,
            memory_pool=None
        )
        self.assertIsInstance(scanner, pds.Scanner)

    def test_scanner_invalid_batch_size(self):
        # Test that invalid batch size raises ValueError
        with self.assertRaises(ValueError):
            self.table.scanner(batch_size=0)

    def test_scanner_negative_batch_readahead(self):
        # Test that negative batch_readahead raises ValueError
        with self.assertRaises(ValueError):
            self.table.scanner(batch_readahead=-1)

    def test_scanner_negative_fragment_readahead(self):
        # Test that negative fragment_readahead raises ValueError
        with self.assertRaises(ValueError):
            self.table.scanner(fragment_readahead=-1)

    def test_to_arrow_dataset_property(self):
        # Test arrow_dataset property
        dataset = self.table.arrow_dataset
        self.assertIsInstance(dataset, pds.Dataset)

    def test_to_duckdb_with_sorting(self):
        # Test to_duckdb with sorting
        duckdb_rel = self.table.to_duckdb(sort_by="col1")
        self.assertIsInstance(duckdb_rel, _duckdb.DuckDBPyRelation)

    def test_to_duckdb_distinct(self):
        # Test to_duckdb with distinct
        duckdb_rel = self.table.to_duckdb(distinct=True)
        self.assertIsInstance(duckdb_rel, _duckdb.DuckDBPyRelation)

    def test_to_batches_with_parameters(self):
        # Test to_batches with various parameters
        table = self.table.to_batches(
            columns=["col1", "col2"],
            batch_size=100,
            sort_by="col1",
            distinct=True
        )
        # to_batches actually returns a Table (despite the name)
        self.assertIsInstance(table, pa.Table)
        self.assertEqual(table.num_columns, 2)

    def test_to_arrow_table_with_filter(self):
        # Test to_arrow_table with column selection
        table = self.table.to_arrow_table(columns=["col1", "col2"])
        self.assertEqual(table.num_columns, 2)

    def test_ddb_property(self):
        # Test ddb property
        ddb_rel = self.table.ddb
        self.assertIsInstance(ddb_rel, _duckdb.DuckDBPyRelation)

    def test_to_polars_lazy(self):
        # Test to_polars with lazy=True
        polars_df = self.table.to_polars(lazy=True)
        self.assertIsInstance(polars_df, pl.LazyFrame)

    def test_to_polars_eager(self):
        # Test to_polars with lazy=False
        polars_df = self.table.to_polars(lazy=False)
        self.assertIsInstance(polars_df, pl.DataFrame)

    def test_to_polars_with_sorting(self):
        # Test to_polars with sorting
        polars_df = self.table.to_polars(lazy=False, sort_by="col1")
        self.assertIsInstance(polars_df, pl.DataFrame)

    def test_to_polars_distinct(self):
        # Test to_polars with distinct
        polars_df = self.table.to_polars(lazy=False, distinct=True)
        self.assertIsInstance(polars_df, pl.DataFrame)


if __name__ == "__main__":
    unittest.main()

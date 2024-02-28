import unittest
from unittest.mock import Mock
import pyarrow.dataset as pds
import pyarrow as pa
from pydala.table import YourTableClass


class TestYourTableClass(unittest.TestCase):
    def setUp(self):
        # Create a mock table object
        self.table = YourTableClass()

    def test_scanner(self):
        # Define the expected arguments for the to_arrow_scanner method
        expected_columns = ["col1", "col2"]
        expected_filter = pds.Expression.equal("col3", 10)
        expected_batch_size = 1000
        expected_sort_by = ["col1", ("col2", "desc")]
        expected_batch_readahead = 8
        expected_fragment_readahead = 2
        expected_fragment_scan_options = Mock()
        expected_use_threads = False
        expected_memory_pool = pa.default_memory_pool()

        # Call the scanner method with the test arguments
        scanner = self.table.scanner(
            columns=expected_columns,
            filter=expected_filter,
            batch_size=expected_batch_size,
            sort_by=expected_sort_by,
            batch_readahead=expected_batch_readahead,
            fragment_readahead=expected_fragment_readahead,
            fragment_scan_options=expected_fragment_scan_options,
            use_threads=expected_use_threads,
            memory_pool=expected_memory_pool,
        )

        # Assert that the to_arrow_scanner method was called with the expected arguments
        self.table.to_arrow_scanner.assert_called_once_with(
            columns=expected_columns,
            filter=expected_filter,
            batch_size=expected_batch_size,
            batch_readahead=expected_batch_readahead,
            fragment_readahead=expected_fragment_readahead,
            fragment_scan_options=expected_fragment_scan_options,
            use_threads=expected_use_threads,
            memory_pool=expected_memory_pool,
            sort_by=expected_sort_by,
        )
        # Assert that the returned scanner object is of type pds.Scanner
        self.assertIsInstance(scanner, pds.Scanner)


if __name__ == "__main__":
    unittest.main()

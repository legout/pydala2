import unittest
from pydala.helpers.misc import (
    humanize_size,
    humanized_size_to_bytes,
    getattr_rec,
    setattr_rec,
    delattr_rec,
    get_nested_keys,
    unify_schemas_pl,
    get_partitions_from_path
)


class TestMisc(unittest.TestCase):

    def test_humanize_size(self):
        # Test basic conversions
        self.assertEqual(humanize_size(1024, "b"), 1024.0)
        self.assertEqual(humanize_size(1024, "kb"), 1.0)
        self.assertEqual(humanize_size(1024**2, "mb"), 1.0)
        self.assertEqual(humanize_size(1024**3, "gb"), 1.0)
        self.assertEqual(humanize_size(1024**4, "tb"), 1.0)
        self.assertEqual(humanize_size(1024**5, "pb"), 1.0)

        # Test default unit
        self.assertEqual(humanize_size(1024**2), 1.0)

        # Test invalid unit
        with self.assertRaises(ValueError):
            humanize_size(1024, "invalid")

        # Test negative size
        with self.assertRaises(ValueError):
            humanize_size(-1024)

    def test_humanized_size_to_bytes(self):
        # Test basic conversions
        self.assertEqual(humanized_size_to_bytes("1024 B"), 1024)
        self.assertEqual(humanized_size_to_bytes("1 KB"), 1024)
        self.assertEqual(humanized_size_to_bytes("1 MB"), 1024**2)
        self.assertEqual(humanized_size_to_bytes("1 GB"), 1024**3)
        self.assertEqual(humanized_size_to_bytes("1 TB"), 1024**4)
        self.assertEqual(humanized_size_to_bytes("1 PB"), 1024**5)

        # Test case insensitive
        self.assertEqual(humanized_size_to_bytes("1 mb"), 1024**2)

        # Test invalid input type
        with self.assertRaises(ValueError):
            humanized_size_to_bytes(1024)

        # Test empty string
        with self.assertRaises(ValueError):
            humanized_size_to_bytes("")

        # Test invalid numeric value
        with self.assertRaises(ValueError):
            humanized_size_to_bytes("invalid MB")

        # Test negative value
        with self.assertRaises(ValueError):
            humanized_size_to_bytes("-1 MB")

    def test_getattr_rec(self):
        # Test simple attribute
        class TestObj:
            def __init__(self):
                self.a = self.b
                self.b = "value"

        obj = TestObj()
        obj.a.b = "nested_value"
        self.assertEqual(getattr_rec(obj, "a.b"), "nested_value")

        # Test invalid input
        with self.assertRaises(ValueError):
            getattr_rec(obj, "")

        with self.assertRaises(ValueError):
            getattr_rec(obj, "a..b")

    def test_setattr_rec(self):
        # Test setting nested attribute
        class TestObj:
            def __init__(self):
                self.a = self.b
                self.b = ""

        obj = TestObj()
        setattr_rec(obj, "a.b", "new_value")
        self.assertEqual(obj.a.b, "new_value")

        # Test invalid input
        with self.assertRaises(ValueError):
            setattr_rec(obj, "", "value")

    def test_delattr_rec(self):
        # Test deleting nested attribute
        class TestObj:
            def __init__(self):
                self.a = self.b
                self.b = ""

        obj = TestObj()
        obj.a.b = "temp"
        delattr_rec(obj, "a.b")
        self.assertFalse(hasattr(obj.a, "b"))

        # Test invalid input
        with self.assertRaises(ValueError):
            delattr_rec(obj, "")

    def test_get_nested_keys(self):
        # Test with nested dictionary
        test_dict = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        }
        keys = get_nested_keys(test_dict)
        expected = ["a", "b", "b.c", "b.d", "b.d.e"]
        self.assertEqual(sorted(keys), sorted(expected))

    def test_get_partitions_from_path(self):
        # Test hive partitioning
        path = "data/year=2024/month=01/day=01/file.parquet"
        partitions = get_partitions_from_path(path, partitioning="hive")
        expected = [("year", "2024"), ("month", "01"), ("day", "01")]
        self.assertEqual(partitions, expected)

        # Test list partitioning
        path = "data/2024/01/01/file.parquet"
        partitions = get_partitions_from_path(path, partitioning=["year", "month", "day"])
        expected = [("year", "2024"), ("month", "01"), ("day", "01")]
        self.assertEqual(partitions, expected)

        # Test string partitioning
        path = "data/2024/file.parquet"
        partitions = get_partitions_from_path(path, partitioning="year")
        expected = [("year", "2024")]
        self.assertEqual(partitions, expected)


if __name__ == "__main__":
    unittest.main()
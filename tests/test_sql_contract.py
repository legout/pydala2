"""Characterization contract tests for the SQL helper seam.

These tests pin the current behavior of ``pydala.helpers.sql`` so that a
future port of these helpers (e.g. into ``fsspeckit``) can be validated
behavior-for-behavior. They are the acceptance criteria for issue #4 and
must not import from any future location until the migration is complete.

Scope:
    * ``sql2pyarrow_filter`` -- SQL WHERE clause -> ``pyarrow.compute`` expression
    * ``sql2polars_filter``  -- SQL WHERE clause -> ``polars`` expression
    * ``get_table_names``    -- SQL query -> list of referenced table names
"""

import unittest

import pyarrow as pa
import pyarrow.compute as pc
import polars as pl

from pydala.helpers.sql import get_table_names, sql2polars_filter, sql2pyarrow_filter

# A schema shared by the pyarrow filter tests: an int64 id, a string name and
# a float64 value. The same column layout is mirrored for the polars tests.
PA_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("name", pa.string()),
        pa.field("value", pa.float64()),
    ]
)

PL_SCHEMA = pl.Schema(
    {
        "id": pl.Int64,
        "name": pl.String,
        "value": pl.Float64,
    }
)


class TestSql2PyarrowFilter(unittest.TestCase):
    """``sql2pyarrow_filter`` translates a SQL predicate into a compute expr."""

    def setUp(self) -> None:
        self.schema = PA_SCHEMA
        self.table = pa.table(
            {
                "id": [1, 2, 3],
                "name": ["a", "test", "c"],
                "value": [1.0, 6.0, 10.0],
            }
        )

    def test_simple_comparison_returns_compute_expression(self) -> None:
        expr = sql2pyarrow_filter("value > 5.0", self.schema)
        self.assertIsInstance(expr, pc.Expression)

    def test_simple_comparison_filters_table(self) -> None:
        expr = sql2pyarrow_filter("value > 5.0", self.schema)
        filtered = self.table.filter(expr)
        self.assertEqual(filtered.column("id").to_pylist(), [2, 3])

    def test_in_clause_filters_table(self) -> None:
        expr = sql2pyarrow_filter("id IN (1, 3)", self.schema)
        filtered = self.table.filter(expr)
        self.assertEqual(filtered.column("id").to_pylist(), [1, 3])

    def test_and_clause_filters_table(self) -> None:
        expr = sql2pyarrow_filter("id > 0 AND name = 'test'", self.schema)
        filtered = self.table.filter(expr)
        self.assertEqual(filtered.column("id").to_pylist(), [2])

    def test_or_clause_filters_table(self) -> None:
        expr = sql2pyarrow_filter("value > 9.0 OR name = 'a'", self.schema)
        filtered = self.table.filter(expr)
        self.assertEqual(filtered.column("id").to_pylist(), [1, 3])


class TestSql2PolarsFilter(unittest.TestCase):
    """``sql2polars_filter`` translates a SQL predicate into a polars expr."""

    def setUp(self) -> None:
        self.schema = PL_SCHEMA
        self.df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["a", "b", "c"],
                "value": [1.0, 6.0, 10.0],
            }
        )

    def test_simple_comparison_returns_polars_expr(self) -> None:
        expr = sql2polars_filter("value > 5.0", self.schema)
        self.assertIsInstance(expr, pl.Expr)

    def test_simple_comparison_filters_dataframe(self) -> None:
        expr = sql2polars_filter("value > 5.0", self.schema)
        filtered = self.df.filter(expr)
        self.assertEqual(filtered.get_column("id").to_list(), [2, 3])

    def test_in_clause_filters_dataframe(self) -> None:
        expr = sql2polars_filter("id IN (1, 3)", self.schema)
        filtered = self.df.filter(expr)
        self.assertEqual(filtered.get_column("id").to_list(), [1, 3])

    def test_and_clause_filters_dataframe(self) -> None:
        expr = sql2polars_filter("value > 5.0 AND name = 'c'", self.schema)
        filtered = self.df.filter(expr)
        self.assertEqual(filtered.get_column("id").to_list(), [3])


class TestGetTableNames(unittest.TestCase):
    """``get_table_names`` extracts every table referenced by a SQL query."""

    def test_single_table(self) -> None:
        self.assertEqual(get_table_names("SELECT * FROM users"), ["users"])

    def test_join_returns_both_tables(self) -> None:
        names = get_table_names(
            "SELECT * FROM users u JOIN orders o ON u.id = o.user_id"
        )
        self.assertEqual(names, ["users", "orders"])

    def test_cte_includes_definition_and_reference(self) -> None:
        # A CTE contributes both its inner source table and its alias where it
        # is referenced. Membership is asserted (not order) because the
        # traversal order is an implementation detail of the SQL parser.
        names = get_table_names("WITH t AS (SELECT * FROM raw) SELECT * FROM t")
        self.assertEqual(sorted(names), ["raw", "t"])

    def test_subquery_returns_inner_table_only(self) -> None:
        # The derived-table alias ("sub") is not itself a table; only the
        # concrete source inside the subquery is reported.
        names = get_table_names("SELECT * FROM (SELECT * FROM raw) sub")
        self.assertEqual(names, ["raw"])


if __name__ == "__main__":
    unittest.main()

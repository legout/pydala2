"""Characterization tests for PydalaTable conversion behavior (issue #16).

Baseline failures documented after running the full suite at the end:
- TestMisc.test_delattr_rec: AttributeError in TestObj.__init__
- TestMisc.test_get_partitions_from_path: incorrect partition parsing
- TestMisc.test_getattr_rec: AttributeError in TestObj.__init__
- TestMisc.test_setattr_rec: AttributeError in TestObj.__init__
- TestSqlSecurity.test_escape_sql_literal: boolean literal casing
- TestPathSecurity.test_sanitize_filename: empty filename handling
- TestCredentialHandling.test_get_credentials_redaction: missing import
"""

from __future__ import annotations

import warnings

import duckdb as _duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import pytest

from pydala.table import PydalaTable
from tests.conftest import SIMPLE_SCHEMA, make_simple_table


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def arrow_table() -> PydalaTable:
    """Return a PydalaTable backed by a PyArrow dataset."""
    return PydalaTable(result=make_simple_table(n_rows=10))


@pytest.fixture
def duckdb_table() -> PydalaTable:
    """Return a PydalaTable backed by a DuckDB relation.

    The connection is closed after the test to keep resource usage isolated.
    """
    con = _duckdb.connect()
    try:
        rel = con.from_arrow(make_simple_table(n_rows=10))
        table = PydalaTable(result=rel, ddb_con=con)
        yield table
    finally:
        con.close()


@pytest.fixture(
    params=["arrow", "duckdb"],
)
def table(
    request: pytest.FixtureRequest, arrow_table: PydalaTable, duckdb_table: PydalaTable
) -> PydalaTable:
    """Parametrize over both Arrow-backed and DuckDB-backed tables."""
    return arrow_table if request.param == "arrow" else duckdb_table


@pytest.fixture
def duped_arrow_table() -> PydalaTable:
    """Return an Arrow-backed table containing duplicated rows."""
    base = make_simple_table(n_rows=5)
    return PydalaTable(result=pa.concat_tables([base, base]))


@pytest.fixture
def duped_duckdb_table() -> PydalaTable:
    """Return a DuckDB-backed table containing duplicated rows."""
    base = make_simple_table(n_rows=5)
    con = _duckdb.connect()
    try:
        rel = con.from_arrow(pa.concat_tables([base, base]))
        table = PydalaTable(result=rel, ddb_con=con)
        yield table
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _sort_table(table: pa.Table, columns: list[str]) -> pa.Table:
    """Return a deterministically sorted copy of ``table`` for comparisons."""
    return table.sort_by([(col, "ascending") for col in columns])


# --------------------------------------------------------------------------- #
# Scanner equivalence
# --------------------------------------------------------------------------- #


def test_scanner_and_to_arrow_scanner_equivalent(arrow_table: PydalaTable) -> None:
    """``scanner()`` and ``to_arrow_scanner()`` produce the same results for the
    same projection, filter, and scan options on an Arrow-backed table.
    """
    filter_expr = pds.field("id") > 3

    s1 = arrow_table.scanner(
        columns=["id", "name"],
        filter=filter_expr,
        batch_size=1024,
        batch_readahead=8,
        fragment_readahead=2,
        use_threads=False,
    )
    s2 = arrow_table.to_arrow_scanner(
        columns=["id", "name"],
        filter=filter_expr,
        batch_size=1024,
        batch_readahead=8,
        fragment_readahead=2,
        use_threads=False,
    )

    assert isinstance(s1, pds.Scanner)
    assert isinstance(s2, pds.Scanner)
    t1 = s1.to_table()
    t2 = s2.to_table()
    assert t1.schema.equals(t2.schema)
    assert _sort_table(t1, ["id"]).equals(_sort_table(t2, ["id"]))


def test_scanner_uses_config_defaults_when_none(arrow_table: PydalaTable) -> None:
    """``scanner()`` substitutes ScannerConfig defaults when parameters are None."""
    scanner = arrow_table.scanner(
        batch_size=None,
        batch_readahead=None,
        fragment_readahead=None,
        use_threads=None,
        memory_pool=None,
    )
    assert isinstance(scanner, pds.Scanner)
    # Defaults should still yield the full dataset.
    assert scanner.to_table().num_rows == 10
    assert scanner.to_table().column_names == list(SIMPLE_SCHEMA.names)


# --------------------------------------------------------------------------- #
# Projection semantics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "columns",
    [
        "id",
        ["id"],
    ],
)
def test_string_projection_matches_list_projection(
    table: PydalaTable, columns: str | list[str]
) -> None:
    """A single-column string projection behaves like a one-element list."""
    result = table.to_arrow_table(columns=columns)
    assert result.column_names == ["id"]
    assert result.schema.field("id").type == pa.int64()
    assert result.num_rows == 10


def test_arrow_conversion_methods_equivalent(duped_arrow_table: PydalaTable) -> None:
    """Projection, ordering, and distinctness are equivalent across conversion
    methods on an Arrow-backed table with duplicated rows.
    """
    columns = ["id", "value"]
    sort_by = [["id", "descending"]]
    expected_ids = [4, 3, 2, 1, 0]
    expected_schema = pa.schema(
        [pa.field("id", pa.int64()), pa.field("value", pa.float64())]
    )

    # Build expected from a single canonical conversion.
    expected = duped_arrow_table.to_arrow_table(
        columns=columns, sort_by=sort_by, distinct=True
    )
    assert expected.column_names == ["id", "value"]
    assert expected.schema.equals(expected_schema)
    assert expected.column("id").to_pylist() == expected_ids

    # DuckDB relation (lazy): DuckDB DISTINCT does not guarantee ORDER BY order.
    ddb = duped_arrow_table.to_duckdb(columns=columns, sort_by=sort_by, distinct=True)
    ddb_arrow = ddb.to_arrow_table()
    assert ddb_arrow.column_names == ["id", "value"]
    assert sorted(ddb_arrow.column("id").to_pylist()) == sorted(expected_ids)

    # Polars eager: routes through DuckDB for sort/distinct, so compare values only.
    pl = duped_arrow_table.to_polars(
        lazy=False, columns=columns, sort_by=sort_by, distinct=True
    )
    assert pl.columns == ["id", "value"]
    assert sorted(pl.to_arrow().column("id").to_pylist()) == sorted(expected_ids)

    # Pandas
    pdf = duped_arrow_table.to_pandas(columns=columns, sort_by=sort_by, distinct=True)
    assert isinstance(pdf, pd.DataFrame)
    assert list(pdf.columns) == columns
    assert sorted(pdf["id"].tolist()) == sorted(expected_ids)


@pytest.mark.xfail(
    reason="DuckDB-backed tables currently call scanner() inside to_duckdb(lazy=False), which fails because scanner() requires an Arrow dataset."
)
def test_duckdb_eager_to_duckdb_and_polars_equivalent(
    duped_duckdb_table: PydalaTable,
) -> None:
    """DuckDB-backed tables support eager conversion methods.

    This is the desired behavior; the current implementation raises because
    ``to_duckdb(lazy=False)`` routes through ``scanner()``, which is Arrow-only.
    """
    columns = ["id", "value"]
    sort_by = [["id", "descending"]]

    expected = duped_duckdb_table.to_arrow_table(
        columns=columns, sort_by=sort_by, distinct=True
    )

    ddb = duped_duckdb_table.to_duckdb(
        lazy=False, columns=columns, sort_by=sort_by, distinct=True
    )
    assert (
        ddb.to_arrow_table().column("id").to_pylist()
        == expected.column("id").to_pylist()
    )

    pl = duped_duckdb_table.to_polars(
        lazy=False, columns=columns, sort_by=sort_by, distinct=True
    )
    assert pl.to_arrow().column("id").to_pylist() == expected.column("id").to_pylist()


def test_duckdb_lazy_conversion_methods_equivalent(
    duped_duckdb_table: PydalaTable,
) -> None:
    """Lazy conversion methods work on DuckDB-backed tables and return the same
    unique values as the Arrow-backed equivalent.
    """
    columns = ["id", "value"]
    sort_by = [["id", "descending"]]
    expected_ids = sorted(range(5))

    expected = duped_duckdb_table.to_arrow_table(
        columns=columns, sort_by=sort_by, distinct=True
    )
    assert expected.column_names == ["id", "value"]
    assert sorted(expected.column("id").to_pylist()) == expected_ids

    ddb = duped_duckdb_table.to_duckdb(
        lazy=True, columns=columns, sort_by=sort_by, distinct=True
    )
    assert ddb.to_arrow_table().column_names == ["id", "value"]
    assert sorted(ddb.to_arrow_table().column("id").to_pylist()) == expected_ids

    pl = duped_duckdb_table.to_polars(
        lazy=True, columns=columns, sort_by=sort_by, distinct=True
    )
    assert pl.columns == ["id", "value"]
    assert sorted(pl.to_arrow().column("id").to_pylist()) == expected_ids

    pdf = duped_duckdb_table.to_pandas(columns=columns, sort_by=sort_by, distinct=True)
    assert isinstance(pdf, pd.DataFrame)
    assert list(pdf.columns) == columns
    assert sorted(pdf["id"].tolist()) == expected_ids


# --------------------------------------------------------------------------- #
# Filtering semantics
# --------------------------------------------------------------------------- #


def test_arrow_scanner_honors_arrow_filter(arrow_table: PydalaTable) -> None:
    """``to_arrow_scanner()`` applies the supplied PyArrow filter."""
    filter_expr = pds.field("id") > 3
    scanner = arrow_table.to_arrow_scanner(filter=filter_expr)
    table = scanner.to_table()
    assert table.column("id").to_pylist() == [4, 5, 6, 7, 8, 9]


def test_to_arrow_table_honors_filter(arrow_table: PydalaTable) -> None:
    """``to_arrow_table()`` applies the supplied filter via the scanner."""
    filter_expr = pds.field("id") >= 7
    table = arrow_table.to_arrow_table(columns=["id"], filter=filter_expr)
    assert table.column("id").to_pylist() == [7, 8, 9]


# --------------------------------------------------------------------------- #
# Batch-size and scan-option behavior
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(
    reason="Current implementation switches to DuckDB for non-default batch sizes, ignoring Arrow filters/scan options."
)
def test_custom_batch_size_preserves_arrow_filter_and_options(
    arrow_table: PydalaTable,
) -> None:
    """A custom batch size must keep the Arrow scanner path and honor filters.

    With a filter that selects six rows and a batch size of three, the reader
    should return the selected rows in two batches of three.
    """
    filter_expr = pds.field("id") > 3
    reader = arrow_table.to_batch_reader(
        columns=["id"],
        filter=filter_expr,
        batch_size=3,
        batch_readahead=8,
        fragment_readahead=2,
        use_threads=False,
    )
    batches = list(reader)
    total = pa.Table.from_batches(batches)
    assert total.column("id").to_pylist() == [4, 5, 6, 7, 8, 9]
    assert [batch.num_rows for batch in batches] == [3, 3]


def test_default_batch_size_honors_arrow_options(arrow_table: PydalaTable) -> None:
    """At the default batch size, Arrow-backed batch reads honor filters and
    scan options without falling back to DuckDB.
    """
    filter_expr = pds.field("id") > 3
    reader = arrow_table.to_batch_reader(
        columns=["id"],
        filter=filter_expr,
        batch_size=131072,
        batch_readahead=8,
        fragment_readahead=2,
        use_threads=False,
    )
    result = reader.read_all()
    assert result.column("id").to_pylist() == [4, 5, 6, 7, 8, 9]


@pytest.mark.parametrize(
    ("method", "value"),
    [
        ("scanner", 0),
        ("scanner", -1),
        ("to_scanner", 0),
        ("to_scanner", -1),
        ("to_arrow_table", 0),
        ("to_arrow_table", -1),
        ("to_arrow", 0),
        ("to_arrow", -1),
        ("to_table", 0),
        ("to_table", -1),
        ("to_duckdb", 0),
        ("to_duckdb", -1),
        ("to_pandas", 0),
        ("to_pandas", -1),
        ("to_df", 0),
        ("to_df", -1),
        ("to_arrow_scanner", 0),
        ("to_arrow_scanner", -1),
        pytest.param(
            "to_batches",
            0,
            marks=pytest.mark.xfail(
                reason="to_batches routes invalid batch sizes to DuckDB, which raises RuntimeError instead of a consistent ValueError."
            ),
        ),
        pytest.param(
            "to_batches",
            -1,
            marks=pytest.mark.xfail(
                reason="to_batches routes invalid batch sizes to DuckDB, which raises TypeError instead of a consistent ValueError."
            ),
        ),
        pytest.param(
            "to_batch_reader",
            0,
            marks=pytest.mark.xfail(
                reason="to_batch_reader routes invalid batch sizes to DuckDB, which raises RuntimeError instead of a consistent ValueError."
            ),
        ),
        pytest.param(
            "to_batch_reader",
            -1,
            marks=pytest.mark.xfail(
                reason="to_batch_reader routes invalid batch sizes to DuckDB, which raises TypeError instead of a consistent ValueError."
            ),
        ),
        pytest.param(
            "to_polars",
            0,
            marks=pytest.mark.xfail(
                reason="to_polars(lazy=False) routes invalid batch sizes to DuckDB, which raises RuntimeError instead of a consistent ValueError."
            ),
        ),
        pytest.param(
            "to_polars",
            -1,
            marks=pytest.mark.xfail(
                reason="to_polars(lazy=False) routes invalid batch sizes to DuckDB, which raises TypeError instead of a consistent ValueError."
            ),
        ),
    ],
)
def test_invalid_batch_size_raises_value_error(
    method: str, value: int, arrow_table: PydalaTable
) -> None:
    """Invalid batch sizes raise a consistent ValueError on all public methods."""
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        if method == "scanner":
            arrow_table.scanner(batch_size=value)
        elif method == "to_arrow_scanner":
            arrow_table.to_arrow_scanner(batch_size=value)
        elif method == "to_scanner":
            arrow_table.to_scanner(batch_size=value)
        elif method == "to_arrow_table":
            arrow_table.to_arrow_table(batch_size=value)
        elif method == "to_arrow":
            arrow_table.to_arrow(batch_size=value)
        elif method == "to_table":
            arrow_table.to_table(batch_size=value)
        elif method == "to_duckdb":
            arrow_table.to_duckdb(lazy=False, batch_size=value)
        elif method == "to_batches":
            arrow_table.to_batches(batch_size=value)
        elif method == "to_batch_reader":
            arrow_table.to_batch_reader(batch_size=value)
        elif method == "to_polars":
            arrow_table.to_polars(lazy=False, batch_size=value)
        elif method == "to_pandas":
            arrow_table.to_pandas(batch_size=value)
        else:
            arrow_table.to_df(batch_size=value)


@pytest.mark.parametrize(
    ("method", "option"),
    [
        ("scanner", "batch_readahead"),
        ("scanner", "fragment_readahead"),
        ("to_arrow_scanner", "batch_readahead"),
        ("to_arrow_scanner", "fragment_readahead"),
        ("to_scanner", "batch_readahead"),
        ("to_scanner", "fragment_readahead"),
        ("to_arrow_table", "batch_readahead"),
        ("to_arrow_table", "fragment_readahead"),
        ("to_arrow", "batch_readahead"),
        ("to_arrow", "fragment_readahead"),
        ("to_table", "batch_readahead"),
        ("to_table", "fragment_readahead"),
        ("to_duckdb", "batch_readahead"),
        ("to_duckdb", "fragment_readahead"),
        ("to_batch_reader", "batch_readahead"),
        ("to_batch_reader", "fragment_readahead"),
        ("to_batches", "batch_readahead"),
        ("to_batches", "fragment_readahead"),
        ("to_polars", "batch_readahead"),
        ("to_polars", "fragment_readahead"),
        ("to_pandas", "batch_readahead"),
        ("to_pandas", "fragment_readahead"),
        ("to_df", "batch_readahead"),
        ("to_df", "fragment_readahead"),
    ],
)
@pytest.mark.parametrize("value", [-1, -5])
def test_negative_readahead_raises_across_methods(
    method: str, option: str, value: int, arrow_table: PydalaTable
) -> None:
    """Negative read-ahead options raise consistently across every public method
    that exposes them.
    """
    kwargs = {option: value}
    with pytest.raises(ValueError, match=f"{option} must be a non-negative integer"):
        if method == "scanner":
            arrow_table.scanner(**kwargs)
        elif method == "to_arrow_scanner":
            arrow_table.to_arrow_scanner(**kwargs)
        elif method == "to_scanner":
            arrow_table.to_scanner(**kwargs)
        elif method == "to_arrow_table":
            arrow_table.to_arrow_table(**kwargs)
        elif method == "to_arrow":
            arrow_table.to_arrow(**kwargs)
        elif method == "to_table":
            arrow_table.to_table(**kwargs)
        elif method == "to_duckdb":
            arrow_table.to_duckdb(lazy=False, **kwargs)
        elif method == "to_batch_reader":
            arrow_table.to_batch_reader(**kwargs)
        elif method == "to_batches":
            arrow_table.to_batches(**kwargs)
        elif method == "to_polars":
            arrow_table.to_polars(lazy=False, **kwargs)
        elif method == "to_pandas":
            arrow_table.to_pandas(**kwargs)
        else:
            arrow_table.to_df(**kwargs)


# --------------------------------------------------------------------------- #
# Sorting and distinctness
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", ["arrow", "duckdb"])
def test_sorting_preserves_semantics(
    backend: str, arrow_table: PydalaTable, duckdb_table: PydalaTable
) -> None:
    """Sort requests are honored regardless of backend selection."""
    t = arrow_table if backend == "arrow" else duckdb_table

    table = t.to_arrow_table(columns=["id", "value"], sort_by=[["id", "descending"]])
    ids = table.column("id").to_pylist()
    assert ids == sorted(ids, reverse=True)

    # String sort spec also works
    table2 = t.to_arrow_table(columns=["id"], sort_by="id ascending")
    assert table2.column("id").to_pylist() == list(range(10))


@pytest.mark.parametrize("backend", ["arrow", "duckdb"])
def test_distinct_preserves_semantics(
    backend: str, duped_arrow_table: PydalaTable, duped_duckdb_table: PydalaTable
) -> None:
    """Distinct requests return unique rows on both backends."""
    t = duped_arrow_table if backend == "arrow" else duped_duckdb_table

    result = t.to_arrow_table(distinct=True)
    assert result.num_rows == 5
    assert set(result.column("id").to_pylist()) == set(range(5))


# --------------------------------------------------------------------------- #
# Batch reader / batch list behavior
# --------------------------------------------------------------------------- #


def test_to_batch_reader_returns_record_batch_reader(arrow_table: PydalaTable) -> None:
    """``to_batch_reader()`` returns a PyArrow RecordBatchReader."""
    reader = arrow_table.to_batch_reader(columns=["id", "name"])
    assert isinstance(reader, pa.RecordBatchReader)
    result = reader.read_all()
    assert result.column_names == ["id", "name"]
    assert result.num_rows == 10


def test_to_batches_returns_table(arrow_table: PydalaTable) -> None:
    """``to_batches()`` returns a materialized PyArrow Table."""
    result = arrow_table.to_batches(columns=["id"])
    assert isinstance(result, pa.Table)
    assert result.column_names == ["id"]
    assert result.num_rows == 10


@pytest.mark.xfail(
    reason="DuckDB's DISTINCT currently does not preserve the ORDER BY ordering, so sorted distinct results are not ordered."
)
def test_batch_reader_with_sort_and_distinct(arrow_table: PydalaTable) -> None:
    """``to_batch_reader`` supports sorting and distinctness requests."""
    reader = arrow_table.to_batch_reader(
        columns=["id"],
        sort_by="id descending",
        distinct=True,
    )
    assert isinstance(reader, pa.RecordBatchReader)
    result = reader.read_all()
    assert set(result.column("id").to_pylist()) == set(range(10))
    assert result.column("id").to_pylist() == sorted(range(10), reverse=True)


# --------------------------------------------------------------------------- #
# Deprecation
# --------------------------------------------------------------------------- #


def test_to_scanner_emits_deprecation_warning(arrow_table: PydalaTable) -> None:
    """``to_scanner()`` emits the existing DeprecationWarning and remains functional."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scanner = arrow_table.to_scanner()

    assert any(
        issubclass(warning.category, DeprecationWarning)
        and "to_scanner() is deprecated" in str(warning.message)
        for warning in w
    )
    assert isinstance(scanner, pds.Scanner)
    assert scanner.to_table().num_rows == 10


# --------------------------------------------------------------------------- #
# Backend-specific behavior
# --------------------------------------------------------------------------- #


def test_duckdb_backed_table_raises_for_arrow_scanner(
    duckdb_table: PydalaTable,
) -> None:
    """Arrow-only scanners reject DuckDB-backed tables."""
    with pytest.raises(ValueError, match="only available for pyarrow datasets"):
        duckdb_table.to_arrow_scanner()


def test_duckdb_backed_table_converts_to_arrow(duckdb_table: PydalaTable) -> None:
    """DuckDB-backed tables convert to Arrow tables via DuckDB."""
    result = duckdb_table.to_arrow_table()
    assert isinstance(result, pa.Table)
    assert result.num_rows == 10
    assert result.column_names == list(SIMPLE_SCHEMA.names)

    expected = make_simple_table(n_rows=10)
    assert _sort_table(result, ["id"]).equals(_sort_table(expected, ["id"]))


def test_duckdb_backed_table_converts_to_polars(duckdb_table: PydalaTable) -> None:
    """DuckDB-backed tables convert to Polars."""
    result = duckdb_table.to_polars(lazy=True)
    assert result.shape == (10, 4)
    assert sorted(result.to_arrow().column("id").to_pylist()) == list(range(10))


def test_duckdb_backed_table_converts_to_pandas(duckdb_table: PydalaTable) -> None:
    """DuckDB-backed tables convert to pandas."""
    result = duckdb_table.to_pandas()
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (10, 4)
    assert sorted(result["id"].tolist()) == list(range(10))


# --------------------------------------------------------------------------- #
# Shorthand conversion aliases
# --------------------------------------------------------------------------- #


def test_to_arrow_alias(arrow_table: PydalaTable) -> None:
    """``to_arrow()`` behaves like ``to_arrow_table()``."""
    t1 = arrow_table.to_arrow_table(columns=["id"])
    t2 = arrow_table.to_arrow(columns=["id"])
    assert t1.equals(t2)


def test_to_table_alias(arrow_table: PydalaTable) -> None:
    """``to_table()`` behaves like ``to_arrow_table()``."""
    t1 = arrow_table.to_arrow_table(columns=["id"])
    t2 = arrow_table.to_table(columns=["id"])
    assert t1.equals(t2)


def test_to_df_alias(arrow_table: PydalaTable) -> None:
    """``to_df()`` behaves like ``to_pandas()``."""
    d1 = arrow_table.to_pandas(columns=["id"])
    d2 = arrow_table.to_df(columns=["id"])
    assert d1.equals(d2)


def test_arrow_property(arrow_table: PydalaTable) -> None:
    """The ``arrow`` property behaves like ``to_arrow()``."""
    t1 = arrow_table.to_arrow()
    t2 = arrow_table.arrow
    assert isinstance(t2, pa.Table)
    assert t1.equals(t2)


def test_ddb_property(arrow_table: PydalaTable) -> None:
    """The ``ddb`` property behaves like ``to_duckdb()``."""
    r1 = arrow_table.to_duckdb()
    r2 = arrow_table.ddb
    assert isinstance(r2, _duckdb.DuckDBPyRelation)
    assert r1.to_arrow_table().equals(r2.to_arrow_table())


def test_pl_property(arrow_table: PydalaTable) -> None:
    """The ``pl`` property behaves like ``to_polars()``."""
    df1 = arrow_table.to_polars(lazy=True)
    df2 = arrow_table.pl
    assert isinstance(df2, type(df1))
    assert df1.collect().equals(df2.collect())


def test_df_property(arrow_table: PydalaTable) -> None:
    """The ``df`` property behaves like ``to_pandas()``."""
    d1 = arrow_table.to_pandas()
    d2 = arrow_table.df
    assert isinstance(d2, pd.DataFrame)
    assert d1.equals(d2)

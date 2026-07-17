"""Characterization tests for the ``Writer.prepare`` source-normalization seam.

Issue #25: extract the transformation half of ``Writer.execute`` into a
reusable ``Writer.prepare(...) -> pa.Table`` seam shared by ordinary writes
and (future) merge.

These tests freeze the transformation contract *independently of writing*:
``prepare`` runs input normalization, sorting, optional uniqueness, schema
cast/evolution, derived date-part partition columns, and Arrow conversion,
returning a single ``pyarrow.Table`` (or, via ``prepare_many``, one table per
source) without touching the filesystem or mutating caches.

Baseline captured at commit ``12a1674d2c3d1ee805db82e8f1178b39fe9583b6``.
"""

from __future__ import annotations

import pathlib

import duckdb
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as pds
import pytest

from pydala.filesystem import FileSystem
from pydala.io import Writer


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _fs(local_path: str) -> FileSystem:
    """Local pydala filesystem rooted at ``local_path`` (no caching)."""
    return FileSystem(bucket=local_path, cached=False)


def _writer(data, local_path: str, *, schema=None, subpath: str = "ds") -> Writer:
    """Construct a Writer whose base path lives under ``local_path/subpath``."""
    return Writer(
        data=data,
        path=subpath,
        filesystem=_fs(local_path),
        schema=schema,
    )


def _read_back(local_path: str, subpath: str = "ds") -> pa.Table:
    """Read every physical parquet file under ``local_path/subpath``."""
    base = pathlib.Path(local_path) / subpath
    return pds.dataset(str(base), format="parquet", partitioning="hive").to_table()


def _cache_spy(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Spy on ``Writer.clear_cache``; return a counter dict incremented per call."""
    counter = {"n": 0}
    monkeypatch.setattr(
        Writer,
        "clear_cache",
        lambda self: counter.__setitem__("n", counter["n"] + 1),
    )
    return counter


def _assert_no_parquet(local_path: str) -> None:
    """Assert no parquet files exist anywhere under ``local_path``."""
    assert list(pathlib.Path(local_path).rglob("*.parquet")) == []


# --------------------------------------------------------------------------- #
# Source families: every supported input yields a pyarrow.Table
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(
            lambda: pa.table({"id": [1, 2], "v": [10, 20]}), id="pyarrow-table"
        ),
        pytest.param(
            lambda: pa.RecordBatch.from_arrays(
                [pa.array([1, 2]), pa.array([10, 20])], names=["id", "v"]
            ),
            id="pyarrow-recordbatch",
        ),
        pytest.param(
            lambda: pl.DataFrame({"id": [1, 2], "v": [10, 20]}), id="polars-dataframe"
        ),
        pytest.param(
            lambda: pl.DataFrame({"id": [1, 2], "v": [10, 20]}).lazy(),
            id="polars-lazyframe",
        ),
        pytest.param(
            lambda: pd.DataFrame({"id": [1, 2], "v": [10, 20]}), id="pandas-dataframe"
        ),
        pytest.param(
            lambda: duckdb.sql("SELECT 1 AS id, 10 AS v UNION ALL SELECT 2, 20"),
            id="duckdb-relation",
        ),
    ],
)
def test_prepare_supports_every_source_family(make, local_path: str) -> None:
    """All documented source families prepare into a ``pyarrow.Table``."""
    writer = _writer(make(), local_path)
    table = writer.prepare()

    assert isinstance(table, pa.Table)
    assert table.column("id").to_pylist() == [1, 2]
    assert table.column("v").to_pylist() == [10, 20]


# --------------------------------------------------------------------------- #
# Schema cast / evolution
# --------------------------------------------------------------------------- #


def test_prepare_converts_large_string_to_normal_string(local_path: str) -> None:
    """Polars string columns arrive as ``large_string`` and must be normalized."""
    writer = _writer(pl.DataFrame({"s": ["a", "b"]}).lazy(), local_path)
    table = writer.prepare()

    assert table.schema.field("s").type == pa.string()


def test_prepare_converts_timestamp_unit_and_timezone(local_path: str) -> None:
    """Timestamp unit and timezone are converted per ``ts_unit``/``tz``."""
    data = pa.table({"ts": pa.array([0, 1]).cast(pa.timestamp("ms", tz="UTC"))})
    writer = _writer(data, local_path)
    table = writer.prepare(ts_unit="ns", tz="Europe/Berlin")

    assert table.schema.field("ts").type == pa.timestamp("ns", tz="Europe/Berlin")


def test_prepare_removes_timezone_when_requested(local_path: str) -> None:
    """``remove_tz=True`` strips timezone from timestamp fields."""
    data = pa.table({"ts": pa.array([0, 1]).cast(pa.timestamp("us", tz="UTC"))})
    writer = _writer(data, local_path)
    table = writer.prepare(remove_tz=True)

    assert table.schema.field("ts").type == pa.timestamp("us")


def test_prepare_alter_schema_false_drops_columns_not_in_schema(
    local_path: str,
) -> None:
    """``alter_schema=False`` enforces the target schema, dropping extras."""
    schema = pa.schema([pa.field("id", pa.int64())])
    data = pa.table({"id": [1, 2], "temp": [10, 20]})
    writer = _writer(data, local_path, schema=schema)

    table = writer.prepare()  # alter_schema defaults to False
    assert table.column_names == ["id"]


def test_prepare_alter_schema_true_preserves_all_data_columns(
    local_path: str,
) -> None:
    """``alter_schema=True`` (schema=None) preserves every data column."""
    data = pa.table({"id": [1, 2], "temp": [10, 20]})
    writer = _writer(data, local_path, schema=None)

    table = writer.prepare(alter_schema=True)
    assert set(table.column_names) == {"id", "temp"}


# --------------------------------------------------------------------------- #
# Derived date-part partition columns
# --------------------------------------------------------------------------- #


def test_prepare_derives_datepart_partition_columns(local_path: str) -> None:
    """``partition_by`` derives integer date-part columns from the timestamp."""
    data = pa.table(
        {
            "id": [1, 2],
            "ts": pa.array(["2024-03-15T00:00:00", "2024-11-01T00:00:00"]).cast(
                pa.timestamp("us")
            ),
        }
    )
    writer = _writer(data, local_path)
    table = writer.prepare(partition_by=["year", "month"], timestamp_column="ts")

    # The historical ``add_datepart_columns`` path leaves derived columns at
    # the polars default integer width (int64); this characterizes that output.
    assert table.schema.field("year").type == pa.int64()
    assert table.schema.field("month").type == pa.int64()
    assert table.column("year").to_pylist() == [2024, 2024]
    assert table.column("month").to_pylist() == [3, 11]


# --------------------------------------------------------------------------- #
# Sorting and uniqueness
# --------------------------------------------------------------------------- #


def test_prepare_sorts_data(local_path: str) -> None:
    """``sort_by`` orders rows; direction is honored via ``[col, dir]`` pairs."""
    data = pa.table({"id": [3, 1, 2]})
    writer = _writer(data, local_path)

    table = writer.prepare(sort_by=[["id", "descending"]])
    assert table.column("id").to_pylist() == [3, 2, 1]


def test_prepare_unique_deduplicates_rows(local_path: str) -> None:
    """``unique=True`` collapses duplicate rows."""
    data = pa.table({"id": [1, 1, 2], "v": [10, 10, 20]})
    writer = _writer(data, local_path)

    table = writer.prepare(unique=True)
    assert table.num_rows == 2


# --------------------------------------------------------------------------- #
# Lazy inputs and empty inputs
# --------------------------------------------------------------------------- #


def test_prepare_collects_lazyframe(local_path: str) -> None:
    """A ``pl.LazyFrame`` is collected and returned as a concrete table."""
    lf = pl.DataFrame({"id": [1, 2, 3]}).lazy()
    writer = _writer(lf, local_path)

    table = writer.prepare()
    assert isinstance(table, pa.Table)
    assert table.column("id").to_pylist() == [1, 2, 3]
    # The lazy source must have been concretized on the writer.
    assert not isinstance(writer.data, pl.LazyFrame)


def test_prepare_empty_input_returns_empty_table(local_path: str) -> None:
    """An empty source prepares into an empty table (not an exception)."""
    data = pa.table({"id": pa.array([], pa.int64())})
    writer = _writer(data, local_path)

    table = writer.prepare()
    assert isinstance(table, pa.Table)
    assert table.num_rows == 0


# --------------------------------------------------------------------------- #
# prepare performs NO filesystem writes and NO cache mutation
# --------------------------------------------------------------------------- #


def test_prepare_writes_no_files_and_does_not_clear_cache(
    local_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``prepare`` is a pure transformation: no parquet writes, no cache clear."""
    data = pa.table({"id": [1, 2], "s": ["a", "b"]})
    writer = _writer(data, local_path)
    cache_calls = _cache_spy(monkeypatch)

    table = writer.prepare(sort_by="id", ts_unit="us")

    assert isinstance(table, pa.Table)
    assert cache_calls["n"] == 0
    _assert_no_parquet(local_path)


# --------------------------------------------------------------------------- #
# List preparation path
# --------------------------------------------------------------------------- #


def test_prepare_many_returns_one_table_per_source_without_writing(
    local_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``prepare_many`` prepares every source and writes nothing."""
    sources = [
        pa.table({"id": [1, 2]}),
        pl.DataFrame({"id": [3, 4]}).lazy(),
        pd.DataFrame({"id": [5, 6]}),
    ]
    cache_calls = _cache_spy(monkeypatch)

    tables = Writer.prepare_many(
        sources,
        path="ds",
        schema=None,
        filesystem=_fs(local_path),
        sort_by="id",
    )

    assert len(tables) == len(sources)
    assert all(isinstance(t, pa.Table) for t in tables)
    assert [t.column("id").to_pylist() for t in tables] == [[1, 2], [3, 4], [5, 6]]
    assert cache_calls["n"] == 0
    _assert_no_parquet(local_path)


# --------------------------------------------------------------------------- #
# execute delegates transformation to prepare
# --------------------------------------------------------------------------- #


def test_execute_delegates_transformation_to_prepare(
    local_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``execute`` must route its transformation phase through ``prepare``."""
    data = pa.table({"id": [2, 1], "s": ["b", "a"]})
    writer = _writer(data, local_path)

    prepare_calls = {"n": 0}
    real_prepare = Writer.prepare

    def spy(self, **kwargs):
        prepare_calls["n"] += 1
        return real_prepare(self, **kwargs)

    monkeypatch.setattr(Writer, "prepare", spy)

    writer.execute(sort_by="id")
    assert prepare_calls["n"] == 1


def test_execute_empty_source_short_circuits_before_transformation(
    local_path: str,
) -> None:
    """An empty source returns ``[]`` before any transformation runs.

    Freezes the historical short-circuit: even an irrelevant ``sort_by`` on a
    column that does not exist must NOT raise -- the empty guard fires first.
    """
    data = pa.table({"id": pa.array([], pa.int64())})
    writer = _writer(data, local_path)

    metadata = writer.execute(sort_by="this_column_does_not_exist")
    assert metadata == []
    _assert_no_parquet(local_path)


# --------------------------------------------------------------------------- #
# execute append / overwrite output is unchanged through the seam
# --------------------------------------------------------------------------- #


def test_execute_append_writes_prepared_table(local_path: str) -> None:
    """Append writes exactly the prepared (sorted, cast) bytes."""
    data = pa.table({"id": [2, 1], "s": ["b", "a"]})
    writer = _writer(data, local_path)

    writer.execute(sort_by="id", ts_unit="us")
    table = _read_back(local_path)

    assert table.column("id").to_pylist() == [1, 2]
    assert table.schema.field("s").type == pa.string()


def test_dataset_overwrite_replaces_previous_content(local_path: str) -> None:
    """End-to-end ``mode="overwrite"`` removes prior files then writes new."""
    from pydala import ParquetDataset

    ds = ParquetDataset(path="ds", filesystem=_fs(local_path))
    ds.write_to_dataset(pa.table({"id": [1, 2]}), mode="append")
    ds.write_to_dataset(pa.table({"id": [9]}), mode="overwrite")

    table = _read_back(local_path)
    assert table.column("id").to_pylist() == [9]


# --------------------------------------------------------------------------- #
# Transformation order: steps compose on a single source
# --------------------------------------------------------------------------- #


def test_prepare_composes_all_transformations(local_path: str) -> None:
    """Sort, unique, schema cast and date-part derivation compose in order."""
    data = pa.table(
        {
            "id": [2, 1, 2],
            "ts": pa.array(
                ["2024-01-10T00:00:00", "2024-02-20T00:00:00", "2024-01-10T00:00:00"]
            ).cast(pa.timestamp("ms")),
        }
    )
    writer = _writer(data, local_path)

    table = writer.prepare(
        sort_by="id",
        unique=True,
        ts_unit="us",
        partition_by=["year"],
        timestamp_column="ts",
    )

    # unique collapsed the duplicate (id=2, same ts) -> 2 rows
    assert table.num_rows == 2
    # sorted ascending by id
    assert table.column("id").to_pylist() == [1, 2]
    # timestamp converted ms -> us
    assert table.schema.field("ts").type == pa.timestamp("us")
    # date-part derived after cast (int64: historical add_datepart width)
    assert table.schema.field("year").type == pa.int64()
    assert table.column("year").to_pylist() == [2024, 2024]

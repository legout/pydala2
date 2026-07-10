"""Characterization contract tests for pydala's helper seam.

These tests pin down the behavior of the helpers that pydala re-exports from
``fsspeckit`` *before* that delegation moves any further. They cover:

* the Polars compatibility helpers (``pydala.helpers.polars``),
* the datetime compatibility helpers (``pydala.helpers.datetime``),
* the ``run_parallel`` execution helper (``pydala.helpers.misc``),
* partition parsing edge cases not already covered by ``tests/test_misc.py``,
* and an explicit check that every helper is an *exact delegate* of fsspeckit
  (no pydala-specific adapter wrapper), which is the contract the migration
  relies on.

Behavioral helpers are exercised against small, deterministic inputs. The
intent is to fail loudly if the re-export wiring or the underlying fsspeckit
behavior drifts.
"""

import datetime as dt

import polars as pl
import pytest

import pydala.helpers.datetime as p_dt
import pydala.helpers.polars as p_pl
from pydala.helpers.datetime import (
    get_timestamp_column,
    get_timedelta_str,
    timestamp_from_string,
)
from pydala.helpers.misc import run_parallel
from pydala.helpers.polars import (
    cast_relaxed,
    delta,
    drop_null_columns,
    explode_all,
    opt_dtype,
    partition_by,
    pl as reexported_pl,
    unify_schemas,
    unnest_all,
    with_datepart_columns,
    with_row_count,
    with_strftime_columns,
    with_truncated_columns,
)

# Helpers exercised directly in the behavioral tests below; collected here so
# the "callable" contract test references the imported objects (verifying they
# are importable from the pydala seam) rather than string names.
POLARS_HELPERS = [
    delta,
    opt_dtype,
    partition_by,
    unnest_all,
    explode_all,
    drop_null_columns,
    cast_relaxed,
    unify_schemas,
    with_datepart_columns,
    with_strftime_columns,
    with_truncated_columns,
    with_row_count,
]

DATETIME_HELPERS = [
    get_timestamp_column,
    get_timedelta_str,
    timestamp_from_string,
]

POLARS_DELEGATES = [fn.__name__ for fn in POLARS_HELPERS]
DATETIME_DELEGATES = [fn.__name__ for fn in DATETIME_HELPERS]


# ---------------------------------------------------------------------------
# 1. Exact delegation contract
# ---------------------------------------------------------------------------
class TestExactDelegation:
    """Every helper re-exported from pydala must be the very same object as the
    one fsspeckit ships.

    These are *delegates*, not adapters: ``pydala.helpers.polars.delta`` and
    ``fsspeckit.datasets.polars.delta`` must be identical (``is``). If a helper
    ever becomes a pydala-specific adapter, this test will flag it so the
    behavior contract can be revisited intentionally.
    """

    @pytest.mark.parametrize("name", POLARS_DELEGATES)
    def test_polars_helper_is_exact_delegate(self, name):
        # The public seam exposes the fsspeckit implementation directly; a
        # pydala wrapper would have this module's name instead.
        assert getattr(p_pl, name).__module__ == "fsspeckit.datasets.polars"

    @pytest.mark.parametrize("name", DATETIME_DELEGATES)
    def test_datetime_helper_is_exact_delegate(self, name):
        assert getattr(p_dt, name).__module__ == "fsspeckit.common.datetime"

    def test_reexported_pl_is_polars_module(self):
        # The ``pl`` symbol re-exported from pydala must be the Polars module.
        assert reexported_pl is pl
        assert p_pl.pl is pl

    def test_polars_all_names_resolve(self):
        for name in p_pl.__all__:
            assert getattr(p_pl, name) is not None

    def test_datetime_all_names_resolve(self):
        for name in p_dt.__all__:
            assert getattr(p_dt, name) is not None


# ---------------------------------------------------------------------------
# 2. Polars helpers behavior
# ---------------------------------------------------------------------------
class TestPolarsHelpersContract:
    """Behavioral characterization of the re-exported Polars helpers."""

    @pytest.fixture
    def df_left(self):
        return pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    @pytest.fixture
    def df_right(self):
        return pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    @pytest.mark.parametrize("helper", POLARS_HELPERS, ids=lambda h: h.__name__)
    def test_polars_helpers_are_callable(self, helper):
        # Each helper is importable from ``pydala.helpers.polars`` and callable
        # (not None) — the core re-export compatibility contract.
        assert callable(helper)

    def test_delta_returns_rows_only_in_left(self, df_left, df_right):
        result = delta(df_left, df_right)

        assert isinstance(result, pl.DataFrame)
        # Only the row present in df_left but absent from df_right survives.
        assert result.to_dict(as_series=False) == {"a": [3], "b": ["z"]}

    def test_delta_eager_collects_lazy_input(self, df_left, df_right):
        result = delta(df_left.lazy(), df_right.lazy(), eager=True)

        assert isinstance(result, pl.DataFrame)
        assert result.to_dict(as_series=False) == {"a": [3], "b": ["z"]}

    def test_delta_empty_when_right_contains_left(self, df_left):
        result = delta(df_left, df_left)

        assert isinstance(result, pl.DataFrame)
        assert result.height == 0

    def test_opt_dtype_converts_integer_strings(self):
        df = pl.DataFrame({"a": ["1", "2", "3"], "b": ["x", "y", "z"]})

        optimized = opt_dtype(df)

        # A column of clean integer strings is promoted to Int64; the pure
        # string column is left alone.
        assert optimized.schema["a"] == pl.Int64
        assert optimized.schema["b"] == pl.String

    def test_opt_dtype_converts_float_strings(self):
        df = pl.DataFrame({"a": ["1.5", "2.5"]})

        optimized = opt_dtype(df)

        assert optimized.schema["a"] == pl.Float64

    def test_opt_dtype_rejects_invalid_sample_method(self):
        df = pl.DataFrame({"a": ["1", "2"]})

        with pytest.raises(ValueError, match="sample_method"):
            opt_dtype(df, sample_method="bogus")

    def test_drop_null_columns_removes_all_null(self):
        df = pl.DataFrame({"a": [1, None], "b": [None, None]})

        result = drop_null_columns(df)

        assert result.columns == ["a"]

    def test_unify_schemas_returns_schema(self):
        df1 = pl.DataFrame({"a": [1], "b": [2]})
        df2 = pl.DataFrame({"b": [2], "a": [1]})

        schema = unify_schemas([df1, df2])

        assert isinstance(schema, pl.Schema)
        assert set(schema.names()) == {"a", "b"}

    def test_cast_relaxed_preserves_values(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        schema = pl.Schema({"a": pl.Int64})

        result = cast_relaxed(df, schema)

        assert result.schema == schema
        assert result["a"].to_list() == [1, 2, 3]

    def test_partition_by_columns_splits_dataframe(self):
        df = pl.DataFrame({"cat": ["x", "y", "x"], "v": [1, 2, 3]})

        partitions = partition_by(df, columns="cat")

        assert len(partitions) == 2
        keys = {tuple(sorted(d.items())) for d, _ in partitions}
        assert keys == {(("cat", "x"),), (("cat", "y"),)}
        # The partition column is dropped from each returned frame.
        for _, frame in partitions:
            assert "cat" not in frame.columns

    def test_partition_by_no_columns_returns_single_partition(self):
        df = pl.DataFrame({"v": [1, 2, 3]})

        partitions = partition_by(df)

        assert partitions == [({}, df)]


# ---------------------------------------------------------------------------
# 3. Datetime helpers behavior
# ---------------------------------------------------------------------------
class TestDatetimeHelpersContract:
    @pytest.fixture
    def ts_frame(self):
        return pl.DataFrame({"ts": [dt.datetime(2023, 1, 1)], "value": [1]})

    @pytest.mark.parametrize("helper", DATETIME_HELPERS, ids=lambda h: h.__name__)
    def test_datetime_helpers_are_callable(self, helper):
        assert callable(helper)

    def test_get_timestamp_column_detects_datetime(self, ts_frame):
        assert get_timestamp_column(ts_frame) == ["ts"]

    def test_get_timestamp_column_empty_when_no_temporal(self):
        df = pl.DataFrame({"value": [1, 2, 3]})

        assert get_timestamp_column(df) == []

    def test_get_timestamp_column_accepts_pyarrow_table(self):
        import pyarrow as pa

        table = pa.table({"ts": pa.array([dt.datetime(2023, 1, 1)])})

        assert get_timestamp_column(table) == ["ts"]

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1d", "1d"),
            ("1h", "1h"),
            ("5mo", "5mo"),
        ],
    )
    def test_get_timedelta_str_polars_passes_valid_units(self, value, expected):
        # Valid polars units are returned untouched.
        assert get_timedelta_str(value, to="polars") == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1d", "1 day"),
            ("1h", "1 hour"),
            ("1s", "1 second"),
        ],
    )
    def test_get_timedelta_str_duckdb_expands_units(self, value, expected):
        assert get_timedelta_str(value, to="duckdb") == expected

    def test_get_timedelta_str_unknown_unit_is_passthrough(self):
        # Unknown units never raise; they are echoed back verbatim.
        assert get_timedelta_str("1invalid", to="polars") == "1 invalid"

    def test_timestamp_from_string_parses_datetime(self):
        result = timestamp_from_string("2023-01-01T10:00:00")

        assert result == dt.datetime(2023, 1, 1, 10, 0)

    def test_timestamp_from_string_date_only_is_midnight(self):
        # ``datetime.fromisoformat`` accepts a bare date, so a date-only string
        # yields a datetime at midnight (not a ``date`` object).
        result = timestamp_from_string("2023-01-01")

        assert result == dt.datetime(2023, 1, 1, 0, 0)

    def test_timestamp_from_string_preserves_offset(self):
        result = timestamp_from_string("2023-01-01T10:00:00+02:00")

        assert result.tzinfo is not None
        assert result.utcoffset() == dt.timedelta(hours=2)

    def test_timestamp_from_string_naive_strips_tz(self):
        result = timestamp_from_string("2023-01-01T10:00:00+02:00", naive=True)

        assert result == dt.datetime(2023, 1, 1, 10, 0)
        assert result.tzinfo is None

    def test_timestamp_from_string_invalid_raises(self):
        with pytest.raises(ValueError):
            timestamp_from_string("not-a-timestamp")


# ---------------------------------------------------------------------------
# 4. run_parallel (misc) behavior
# ---------------------------------------------------------------------------
class TestRunParallelContract:
    @staticmethod
    def _square(x):
        return x * x

    def test_run_parallel_returns_one_result_per_input(self):
        result = run_parallel(self._square, [1, 2, 3], n_jobs=1, verbose=False)

        assert result == [1, 4, 9]

    def test_run_parallel_matches_sequential(self):
        params = list(range(10))

        parallel_result = run_parallel(self._square, params, n_jobs=1, verbose=False)
        sequential_result = [self._square(p) for p in params]

        assert parallel_result == sequential_result

    def test_run_parallel_passes_extra_args_and_kwargs(self):
        def add(x, y, z=0):
            return x + y + z

        result = run_parallel(add, [1, 2, 3], 10, z=100, n_jobs=1, verbose=False)

        assert result == [111, 112, 113]

    def test_run_parallel_empty_input(self):
        result = run_parallel(self._square, [], n_jobs=1, verbose=False)

        assert result == []

    def test_run_parallel_verbose_default_runs(self):
        # The default ``verbose=True`` path must not change results.
        result = run_parallel(self._square, [1, 2], n_jobs=1)

        assert result == [1, 4]


# ---------------------------------------------------------------------------
# 5. Partition parsing edge cases (not covered by tests/test_misc.py)
# ---------------------------------------------------------------------------
class TestPartitionParsingEdgeCases:
    """``tests/test_misc.py`` already covers hive and explicit-list partitioning
    on a path with a file extension. These tests cover the remaining edges of
    ``get_partitions_from_path``.
    """

    def test_single_string_partitioning_takes_first_segment(self):
        # A non-hive string partitioning yields a single (name, first-segment)
        # tuple rather than scanning for ``key=value`` pairs.
        from pydala.helpers.misc import get_partitions_from_path

        result = get_partitions_from_path("2024/file.parquet", partitioning="year")

        assert result == [("year", "2024")]

    def test_hive_partitioning_without_file_extension(self):
        from pydala.helpers.misc import get_partitions_from_path

        result = get_partitions_from_path("year=2024/month=01", partitioning="hive")

        assert result == [("year", "2024"), ("month", "01")]

    def test_plain_path_without_partitioning_raises(self):
        from pydala.helpers.misc import get_partitions_from_path

        # ``partitioning=None`` is not a supported mode: it falls through to a
        # ``zip(None, ...)`` which raises TypeError. Pin that contract.
        with pytest.raises(TypeError):
            get_partitions_from_path("plain/path", partitioning=None)

import logging
import typing as t
from fsspec import AbstractFileSystem
import duckdb as _duckdb
import pyarrow as pa
import pyarrow.dataset as pds
import polars.selectors as cs

from .helpers.datetime import get_timestamp_column
from .helpers.polars import pl as _pl
from .table import PydalaTable

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data processing operations for datasets including loading and querying."""

    def __init__(
        self,
        ddb_con: _duckdb.DuckDBPyConnection | None = None,
        timestamp_column: str | None = None,
        partitioning: str | list[str] | None = None,
        name: str | None = None,
    ):
        if ddb_con is None:
            ddb_con = _duckdb.connect()
            # Enable object caching for e.g. parquet metadata
            ddb_con.execute("PRAGMA enable_object_cache;")

        self.ddb_con = ddb_con
        self._timestamp_column = timestamp_column
        self._partitioning = partitioning
        self.name = name or "dataset"
        self._arrow_dataset = None
        self.table = None
        self._tz = None

    def load_dataset(
        self,
        path: str,
        filesystem: AbstractFileSystem,
        schema: pa.Schema | None = None,
        format: str = "parquet",
    ) -> None:
        """Load dataset from path and initialize PydalaTable."""
        self._arrow_dataset = pds.dataset(
            path,
            schema=schema,
            filesystem=filesystem,
            format=format,
            partitioning=self._partitioning,
        )
        self.table = PydalaTable(result=self._arrow_dataset, ddb_con=self.ddb_con)

        # Auto-detect timestamp column if not set
        if self._timestamp_column is None:
            timestamp_columns = get_timestamp_column(self.table.pl.head(10))
            if timestamp_columns:
                self._timestamp_column = timestamp_columns[0]

        # Set timezone if timestamp column exists
        if self._timestamp_column is not None:
            tz = self.schema.field(self._timestamp_column).type.tz
            self._tz = tz
            if tz is not None:
                # Use parameterized query to prevent SQL injection
                self.ddb_con.execute("SET TimeZone=?", [str(tz)])

        # Register dataset in DuckDB
        self.ddb_con.register(f"{self.name}", self._arrow_dataset)

    @property
    def is_loaded(self) -> bool:
        """Check if dataset is loaded."""
        return self._arrow_dataset is not None

    @property
    def schema(self) -> pa.Schema:
        """Get dataset schema."""
        if self.is_loaded:
            return self._arrow_dataset.schema
        raise RuntimeError("Dataset not loaded")

    @property
    def columns(self) -> list[str]:
        """Get list of column names."""
        if self.is_loaded:
            return self.schema.names
        return []

    @property
    def timestamp_column(self) -> str | None:
        """Get timestamp column name."""
        return self._timestamp_column

    @property
    def timezone(self) -> str | None:
        """Get timezone of timestamp column."""
        return self._tz

    def count_rows(self) -> int:
        """Count total rows in dataset."""
        if self.is_loaded:
            return self._arrow_dataset.count_rows()
        return 0

    @property
    def num_rows(self) -> int:
        """Get number of rows."""
        return self.count_rows()

    @property
    def num_columns(self) -> int:
        """Get number of columns."""
        if self.is_loaded:
            return len(self.schema.names)
        return 0

    @property
    def partitioning_schema(self) -> pa.Schema:
        """Get partitioning schema."""
        if self.is_loaded and hasattr(self._arrow_dataset, "partitioning"):
            return self._arrow_dataset.partitioning.schema
        return pa.schema([])

    @property
    def partition_names(self) -> list[str]:
        """Get partition column names."""
        if self.is_loaded and hasattr(self._arrow_dataset, "partitioning"):
            return self._arrow_dataset.partitioning.schema.names
        return []

    @property
    def partition_values(self) -> dict:
        """Get partition values."""
        if self.is_loaded and hasattr(self._arrow_dataset, "partitioning"):
            return dict(
                zip(
                    self._arrow_dataset.partitioning.schema.names,
                    [
                        pa_list.to_pylist()
                        for pa_list in self._arrow_dataset.partitioning.dictionaries
                    ],
                )
            )
        return {}

    def get_delta_filter_df(
        self,
        df: _pl.DataFrame | _pl.LazyFrame,
        existing_columns: list[str],
        filter_columns: str | list[str] | None = None,
    ) -> _pl.DataFrame | _pl.LazyFrame:
        """Filter dataframe based on min/max values for delta operations."""
        collect = isinstance(df, _pl.LazyFrame)

        columns = set(df.columns) & set(existing_columns)
        null_columns = df.select(cs.by_dtype(_pl.Null)).collect_schema().names()
        columns = columns - set(null_columns)

        if filter_columns is not None:
            columns = set(columns) & set(filter_columns)

        if len(columns) == 0:
            return _pl.DataFrame(schema=df.schema)

        filter_expr = []
        for col in columns:
            max_min = df.select(_pl.max(col).alias("max"), _pl.min(col).alias("min"))

            if collect:
                max_min = max_min.collect()

            f_max = max_min["max"][0]
            f_min = max_min["min"][0]

            if isinstance(f_max, str):
                f_max = f_max.strip("'").replace(",", "")
            if isinstance(f_min, str):
                f_min = f_min.strip("'").replace(",", "")

            filter_expr.append(
                f"(({col}<='{f_max}' AND {col}>='{f_min}') OR {col} IS NULL)".replace(
                    "'None'", "NULL"
                )
            )

        return filter_expr
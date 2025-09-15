"""
Dataset filtering functionality for simplified dataset module.
"""

import logging
from typing import Union, Optional

import pyarrow as pa
import pyarrow.dataset as pds

from ..helpers.misc import sql2pyarrow_filter
from ..table import PydalaTable

logger = logging.getLogger(__name__)


class DatasetFilter:
    """Handles dataset filtering with multiple backends."""

    def __init__(self, dataset):
        self.dataset = dataset

    def filter(self, expression: str, backend: str = "auto") -> "BaseDataset":
        """
        Filter the dataset using the given expression.

        Args:
            expression: Filter expression (SQL-like syntax)
            backend: Filtering backend ("auto", "duckdb", "pyarrow")

        Returns:
            Filtered dataset
        """
        if not self.dataset.is_loaded:
            logger.warning("Cannot filter dataset that is not loaded")
            return self.dataset

        # Determine appropriate backend
        chosen_backend = self._select_backend(expression, backend)

        # Apply filter
        try:
            if chosen_backend == "duckdb":
                result = self._filter_duckdb(expression)
            else:
                result = self._filter_pyarrow(expression)

            # Wrap result in new dataset-like object
            return self._create_filtered_dataset(result, chosen_backend)

        except Exception as e:
            logger.error(f"Failed to filter dataset: {e}")
            raise

    def _select_backend(self, expression: str, preferred: str) -> str:
        """Select appropriate filtering backend."""
        if preferred != "auto":
            return preferred

        # Check for DuckDB-specific operations
        duckdb_patterns = ["%", "like", "similar to", "*", "regexp", "~("
        if any(pattern in expression.lower() for pattern in duckdb_patterns):
            logger.debug("Detected DuckDB-specific pattern, using DuckDB")
            return "duckdb"

        # Check for complex expressions that might require DuckDB
        if self._is_complex_expression(expression):
            logger.debug("Complex expression detected, using DuckDB")
            return "duckdb"

        return "pyarrow"

    def _is_complex_expression(self, expression: str) -> bool:
        """Check if expression is complex (multiple conditions, subqueries, etc.)."""
        # Count boolean operators
        operators = ["and", "or", "not", "(", ")", "case", "when"]
        op_count = sum(1 for op in operators if op in expression.lower())

        # Consider complex if multiple operators or parentheses
        return op_count > 2 or expression.count("(") > 1

    def _filter_duckdb(self, expression: str) -> duckdb.DuckDBPyRelation:
        """Apply filter using DuckDB."""
        logger.debug(f"Filtering with DuckDB: {expression[:100]}...")
        return self.dataset.table.ddb.filter(expression)

    def _filter_pyarrow(self, expression: str) -> pds.Dataset:
        """Apply filter using PyArrow."""
        logger.debug(f"Filtering with PyArrow: {expression[:100]}...")

        # Convert SQL-like expression to PyArrow expression
        pa_expression = sql2pyarrow_filter(expression, self.dataset.schema)
        return self.dataset._arrow_dataset.filter(pa_expression)

    def _create_filtered_dataset(self, result, backend: str):
        """Create appropriate result object."""
        # Create PydalaTable from result
        table = PydalaTable(result=result, ddb_con=self.dataset.db_manager.connection)

        # For now, return the table directly
        # In a more complete implementation, we might create a ViewDataset or similar
        logger.info(f"Successfully filtered dataset using {backend}")
        return table


class DateRangeFilter:
    """Specialized filter for date/time ranges."""

    def __init__(self, dataset, timestamp_column: str):
        self.dataset = dataset
        self.timestamp_column = timestamp_column

    def filter_range(
        self,
        start: Optional[Union[str, pa.TimestampScalar]] = None,
        end: Optional[Union[str, pa.TimestampScalar]] = None,
        inclusive: bool = True
    ):
        """Filter dataset to date/time range."""
        if not start and not end:
            return self.dataset

        # Build filter expression
        conditions = []

        if start:
            op = ">=" if inclusive else ">"
            conditions.append(f"{self.timestamp_column} {op} '{start}'")

        if end:
            op = "<=" if inclusive else "<"
            conditions.append(f"{self.timestamp_column} {op} '{end}'")

        expression = " AND ".join(conditions)

        # Apply filter
        filter_engine = DatasetFilter(self.dataset)
        return filter_engine.filter(expression)

    def filter_last(self, duration: str) -> "BaseDataset":
        """Filter for last N time units (e.g., '7d', '24h')."""
        import datetime as dt

        # Parse duration
        units = {
            's': 'seconds',
            'm': 'minutes',
            'h': 'hours',
            'd': 'days',
            'w': 'weeks',
            'M': 'months'
        }

        value = int(duration[:-1])
        unit = duration[-1]

        if unit not in units:
            raise ValueError(f"Unknown duration unit: {unit}")

        # Calculate time range
        now = dt.datetime.now()
        if unit == 'M':
            start = now - dt.timedelta(days=value * 30)  # Approximate
        else:
            start = now - dt.timedelta(**{units[unit]: value})

        return self.filter_range(start=start)
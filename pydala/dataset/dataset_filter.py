import logging
from typing import Union
import duckdb as _duckdb
import pyarrow.dataset as pds

from .helpers.misc import sql2pyarrow_filter
from .table import PydalaTable

logger = logging.getLogger(__name__)


class DatasetFilter:
    """Handles filtering operations for datasets."""

    def __init__(self, arrow_dataset: pds.Dataset, table: PydalaTable):
        self._arrow_dataset = arrow_dataset
        self.table = table

    def filter_with_duckdb(self, filter_expr: str) -> _duckdb.DuckDBPyRelation:
        """Filter using DuckDB."""
        return self.table.ddb.filter(filter_expr)

    def filter_with_arrow(
        self, filter_expr: Union[str, pds.Expression]
    ) -> pds.FileSystemDataset:
        """Filter using PyArrow."""
        if isinstance(filter_expr, str):
            filter_expr = sql2pyarrow_filter(filter_expr, self._arrow_dataset.schema)
        return self._arrow_dataset.filter(filter_expr)

    def filter(
        self,
        filter_expr: Union[str, pds.Expression],
        use: str = "auto",
    ) -> PydalaTable:
        """
        Filter the dataset using either PyArrow or DuckDB.

        Args:
            filter_expr: Filter expression as string or PyArrow expression
            use: Filter method to use ("auto", "pyarrow", or "duckdb")
                 If "auto", will try PyArrow first and fallback to DuckDB

        Returns:
            Filtered dataset as PydalaTable
        """
        # Auto-detect if DuckDB should be used
        if isinstance(filter_expr, str):
            if any([s in filter_expr.lower() for s in ["%", "like", "similar to", "*", "(", ")"]]):
                use = "duckdb"

        if use == "auto":
            try:
                result = self.filter_with_arrow(filter_expr)
            except Exception as e:
                logger.debug(f"PyArrow filtering failed: {e}")
                result = self.filter_with_duckdb(filter_expr)
        elif use == "pyarrow":
            result = self.filter_with_arrow(filter_expr)
        elif use == "duckdb":
            result = self.filter_with_duckdb(filter_expr)
        else:
            raise ValueError(f"Invalid filter method: {use}")

        return PydalaTable(result=result, ddb_con=self.table.ddb)
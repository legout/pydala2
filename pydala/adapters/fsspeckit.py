"""fsspeckit-backed adapter for managed Parquet loading with filtering.

This module provides a thin, pydala-owned adapter that sits on top of the
public fsspeckit PyArrow and DuckDB dataset I/O classes.  It exists because
fsspeckit 0.22.0's public ``PyarrowDatasetIO`` ships a broken string-filter
normalization path (it imports ``fsspeckit.common.sql_filters``, which is not
present in the released package).  The adapter therefore performs its own
filter normalization using the public ``fsspeckit.sql.filters.sql2pyarrow_filter``
helper, then hands a PyArrow-native filter expression to
``PyarrowDatasetIO.read_parquet``.

Design constraints:
* Only public fsspeckit imports are used.
* Filesystem credentials/configuration flow through fsspeckit's own connection
  helpers (``create_duckdb_connection`` / ``filesystem``).
* No fsspeckit class leaks into pydala's public domain types.
* Pydala's accepted filter contract (``str`` SQL predicate or
  ``pyarrow.compute.Expression``) is preserved.
"""

from __future__ import annotations

import posixpath
import re
import warnings

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from sqlglot import parse_one

from fsspeckit.datasets.duckdb import DuckDBDatasetIO, create_duckdb_connection
from fsspeckit.datasets.pyarrow import PyarrowDatasetIO
from fsspeckit.sql.filters import sql2pyarrow_filter


class FsspeckitParquetAdapter:
    """Read managed Parquet data through fsspeckit PyArrow/DuckDB adapters.

    Args:
        filesystem: An fsspec-compatible filesystem instance.  pydala's
            ``FileSystem`` factory already returns instances produced by
            fsspeckit's own ``filesystem`` helper, so passing one of those
            keeps credential routing inside fsspeckit.
    """

    def __init__(self, filesystem: AbstractFileSystem) -> None:
        self._filesystem = filesystem
        self._adapter_filesystem = getattr(filesystem, "fs", filesystem)
        self._adapter_root = getattr(filesystem, "path", "")
        self._pyarrow_io: PyarrowDatasetIO | None = None
        self._duckdb_io: DuckDBDatasetIO | None = None
        self._duckdb_connection = None

    def _adapter_path(self, path: str) -> str:
        """Return the backend filesystem path for a pydala-relative path."""
        return posixpath.join(self._adapter_root, path)

    @property
    def _uses_supported_fsspeckit_protocol(self) -> bool:
        protocol = self._adapter_filesystem.protocol
        protocols = protocol if isinstance(protocol, (list, tuple)) else (protocol,)
        return any(value in {"file", "local"} for value in protocols)

    @property
    def filesystem(self) -> AbstractFileSystem:
        """The pydala filesystem whose configuration fsspeckit receives."""
        return self._filesystem

    @property
    def _pyarrow(self) -> PyarrowDatasetIO:
        if self._pyarrow_io is None:
            self._pyarrow_io = PyarrowDatasetIO(filesystem=self._adapter_filesystem)
        return self._pyarrow_io

    @property
    def _duckdb(self) -> DuckDBDatasetIO:
        if self._duckdb_io is None:
            self._duckdb_connection = create_duckdb_connection(
                filesystem=self._adapter_filesystem
            )
            self._duckdb_io = DuckDBDatasetIO(connection=self._duckdb_connection)
        return self._duckdb_io

    @staticmethod
    def _to_sql_string(filter_expr: str | pc.Expression) -> str:
        """Translate a pydala filter into a DuckDB WHERE-clause string.

        pydala accepts either a SQL string or a PyArrow compute expression.
        Strings are passed through unchanged.  Expressions are converted by
        mapping their string representation to standard SQL, including the
        common predicate shapes supported by pydala's own filter path.
        """
        if isinstance(filter_expr, str):
            return filter_expr

        s = str(filter_expr)
        s = re.sub(r'"([^"]*)"', r"'\1'", s)


        # PyArrow compute helpers that are not valid SQL.
        s = re.sub(r"is_valid\(([^)]+)\)", r"(\1 IS NOT NULL)", s)
        s = re.sub(r"is_null\(([^,]+)(?:,[^)]+)?\)", r"(\1 IS NULL)", s)

        def _replace_is_in(match: re.Match[str]) -> str:
            field = match.group(1)
            values_block = match.group(2)
            array_match = re.search(r"\[[\s\S]*?\]", values_block)
            if array_match is None:
                raise ValueError(
                    f"Could not parse is_in value set from expression: {match.group(0)}"
                )
            values = array_match.group(0)[1:-1].strip()
            values = re.sub(r"\s*,\s*", ", ", values)
            return f"({field} IN ({values}))"

        s = re.sub(r"is_in\(([^,]+),\s*\{value_set=([^}]+)\}[^)]*\)", _replace_is_in, s)

        # sqlglot normalises ``==`` to ``=`` and ``and``/``or`` to ``AND``/``OR``.
        return parse_one(s).sql()

    @staticmethod
    def _to_pyarrow_expression(
        filter_expr: str | pc.Expression, schema: pa.Schema
    ) -> pc.Expression | None:
        """Translate a pydala filter into a PyArrow compute expression."""
        if filter_expr is None:
            return None
        if isinstance(filter_expr, pc.Expression):
            return filter_expr
        return sql2pyarrow_filter(filter_expr, schema)

    def read_parquet(
        self,
        path: str,
        filters: str | pc.Expression | None = None,
        columns: list[str] | None = None,
        backend: str = "pyarrow",
    ) -> pa.Table:
        """Read parquet through a fsspeckit adapter, applying a pydala filter.

        Args:
            path: Parquet file or directory understood by the filesystem.
            filters: pydala filter; either a SQL predicate string or a
                ``pyarrow.compute.Expression``.
            columns: Optional column projection.
            backend: Either ``"pyarrow"`` or ``"duckdb"``.

        Returns:
            A ``pyarrow.Table`` containing the filtered data.
        """
        if backend not in {"pyarrow", "duckdb"}:
            raise ValueError(f"Unsupported adapter backend: {backend!r}")

        if not self._uses_supported_fsspeckit_protocol:
            warnings.warn(
                "fsspeckit 0.22 does not support this filesystem protocol through "
                "its public dataset IO adapters; using pydala's Arrow fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            schema = pds.dataset(
                path, filesystem=self._filesystem, format="parquet"
            ).schema
            pa_filter = (
                self._to_pyarrow_expression(filters, schema)
                if filters is not None
                else None
            )
            return pds.dataset(
                path, filesystem=self._filesystem, format="parquet"
            ).to_table(columns=columns, filter=pa_filter)

        adapter_path = self._adapter_path(path)
        if backend == "pyarrow":
            pa_filter: pc.Expression | None = None
            if filters is not None:
                if self._adapter_filesystem.isfile(adapter_path):
                    schema = pq.read_schema(
                        adapter_path, filesystem=self._adapter_filesystem
                    )
                else:
                    schema = pds.dataset(
                        adapter_path,
                        filesystem=self._adapter_filesystem,
                        format="parquet",
                    ).schema
                pa_filter = self._to_pyarrow_expression(filters, schema)
            return self._pyarrow.read_parquet(
                adapter_path, columns=columns, filters=pa_filter
            )

        sql_filter = self._to_sql_string(filters) if filters is not None else None
        return self._duckdb.read_parquet(
            adapter_path, columns=columns, filters=sql_filter
        )

    def close(self) -> None:
        """Release the public fsspeckit DuckDB connection, if it was created."""
        if self._duckdb_connection is not None:
            self._duckdb_connection.close()
            self._duckdb_connection = None
            self._duckdb_io = None

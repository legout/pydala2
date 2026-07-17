"""fsspeckit-backed adapter for managed Parquet loading with filtering.

This module provides a thin, pydala-owned adapter that sits on top of the
public fsspeckit PyArrow and DuckDB dataset I/O classes.  It exists because
fsspeckit 0.22.0's public ``PyarrowDatasetIO`` ships a broken string-filter
normalization path (it imports ``fsspeckit.common.sql_filters``, which is not
present in the released package).  The adapter therefore performs its own
filter normalization using the public ``fsspeckit.sql.filters.sql2pyarrow_filter``
helper, then hands a PyArrow-native filter expression to
``PyarrowDatasetIO.read_parquet``.

fsspeckit 0.22's ``normalize_path`` prepends the filesystem protocol to the
path (e.g. ``memory://events``) and then validates existence.  Some fsspec
backends (notably ``MemoryFileSystem``) reject protocol-prefixed paths in
``exists()``/``isfile()``, so the adapter bypasses ``_normalize_path`` on the
IO objects for those protocols.  The underlying PyArrow and DuckDB IO still
flows through fsspeckit's public classes; only the path-validation wrapper is
skipped.

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
from typing import TYPE_CHECKING, Any, Literal
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from sqlglot import parse_one

from fsspeckit.datasets.duckdb import DuckDBDatasetIO, create_duckdb_connection
from fsspeckit.datasets.pyarrow import PyarrowDatasetIO
from fsspeckit.sql.filters import sql2pyarrow_filter

if TYPE_CHECKING:
    from fsspeckit.core.incremental import MergeResult


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

    def _make_pyarrow_io(self) -> PyarrowDatasetIO:
        """Create a PyarrowDatasetIO, bypassing path validation for unsupported protocols."""
        io = PyarrowDatasetIO(filesystem=self._adapter_filesystem)
        if not self._uses_supported_fsspeckit_protocol:
            adapter_root = self._adapter_root
            io._normalize_path = lambda path, operation="other": posixpath.join(
                adapter_root, path
            )
        return io

    def _make_duckdb_io(self) -> DuckDBDatasetIO:
        """Create a DuckDBDatasetIO, bypassing path validation for unsupported protocols."""
        conn = create_duckdb_connection(filesystem=self._adapter_filesystem)
        io = DuckDBDatasetIO(connection=conn)
        if not self._uses_supported_fsspeckit_protocol:
            adapter_root = self._adapter_root
            io._normalize_path = lambda path, operation="other": posixpath.join(
                adapter_root, path
            )
        return io

    @property
    def filesystem(self) -> AbstractFileSystem:
        """The pydala filesystem whose configuration fsspeckit receives."""
        return self._filesystem

    @property
    def _pyarrow(self) -> PyarrowDatasetIO:
        if self._pyarrow_io is None:
            self._pyarrow_io = self._make_pyarrow_io()
        return self._pyarrow_io

    @property
    def _duckdb(self) -> DuckDBDatasetIO:
        if self._duckdb_io is None:
            self._duckdb_io = self._make_duckdb_io()
            self._duckdb_connection = self._duckdb_io._connection
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
        if backend not in self._BACKENDS:
            raise ValueError(
                f"Unsupported adapter backend: {backend!r}. "
                f"Expected one of {self._BACKENDS}."
            )

        adapter_path = self._adapter_path(path)

        # DuckDB cannot read from fsspec backends it does not register natively
        # (e.g. MemoryFileSystem). Route through the fsspeckit PyArrow IO
        # instead so the read still flows through fsspeckit's public adapter.
        effective_backend = (
            "pyarrow"
            if (backend == "duckdb" and not self._uses_supported_fsspeckit_protocol)
            else backend
        )

        if effective_backend == "pyarrow":
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

    _MERGE_STRATEGIES = ("insert", "update", "upsert")
    _BACKENDS = ("pyarrow", "duckdb")

    def merge(
        self,
        data: pa.Table | list[pa.Table],
        path: str,
        strategy: Literal["insert", "update", "upsert"],
        key_columns: str | list[str],
        *,
        partition_by: str | list[str] | None = None,
        backend: Literal["pyarrow", "duckdb"] = "pyarrow",
        **merge_options: Any,
    ) -> MergeResult:
        """Merge prepared Arrow table(s) into a dataset through fsspeckit.

        Performs an incremental merge (insert / update / upsert) of already
        prepared PyArrow tables into the target dataset.  Source preparation
        (input normalization, schema casting, derived partition columns) is
        the caller's responsibility; this adapter only owns the transport
        seam.

        The adapter owns pydala-relative path conversion, filesystem and
        credential reuse, backend construction and lifetime, and the
        translation from pydala's ``partition_by`` name to fsspeckit's
        ``partition_columns`` name.  fsspeckit's typed ``MergeResult`` is
        forwarded unchanged.

        Args:
            data: Prepared PyArrow table or list of tables to merge.
            path: pydala-relative dataset path understood by the filesystem.
            strategy: Merge strategy (``"insert"``, ``"update"`` or
                ``"upsert"``).
            key_columns: Column(s) used as unique identifiers.
            partition_by: Optional pydala-style partition column(s); passed
                to fsspeckit as ``partition_columns``.
            backend: ``"pyarrow"`` (the default, supports arbitrary fsspec
                filesystems) or ``"duckdb"`` (explicit; limited to
                filesystems DuckDB can write to natively).
            **merge_options: Additional fsspeckit merge/write tuning options
                forwarded verbatim (e.g. ``compression``, ``row_group_size``,
                ``max_rows_per_file``, ``schema``).

        Returns:
            fsspeckit's typed ``MergeResult``.

        Raises:
            ValueError: If ``strategy`` or ``backend`` is unsupported, or if
                the DuckDB backend is requested for a filesystem it cannot
                serve (e.g. ``MemoryFileSystem``).  DuckDB cannot write to
                fsspec backends it does not register natively, so the
                limitation is encoded explicitly rather than silently
                diverging to the PyArrow backend.
        """
        if strategy not in self._MERGE_STRATEGIES:
            raise ValueError(
                f"Unsupported merge strategy: {strategy!r}. "
                f"Expected one of {self._MERGE_STRATEGIES}."
            )
        if backend not in self._BACKENDS:
            raise ValueError(
                f"Unsupported merge backend: {backend!r}. "
                f"Expected one of {self._BACKENDS}."
            )
        if backend == "duckdb" and not self._uses_supported_fsspeckit_protocol:
            protocol = self._adapter_filesystem.protocol
            raise ValueError(
                f"DuckDB merge backend cannot serve filesystem protocol "
                f"{protocol!r}; use backend='pyarrow' instead."
            )

        adapter_path = self._adapter_path(path)
        io = self._duckdb if backend == "duckdb" else self._pyarrow
        return io.merge(
            data,
            adapter_path,
            strategy=strategy,
            key_columns=key_columns,
            partition_columns=partition_by,
            **merge_options,
        )

    def close(self) -> None:
        """Release the public fsspeckit DuckDB connection, if it was created."""
        if self._duckdb_connection is not None:
            self._duckdb_connection.close()
            self._duckdb_connection = None
            self._duckdb_io = None

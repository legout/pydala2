# Third-party imports
import duckdb as _duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds

# Local imports
from .helpers.polars import pl as _pl


class PydalaTable:
    """A unified table interface supporting multiple backends.

    This class provides a consistent interface for working with tabular data
    across different backends (PyArrow and DuckDB). It supports conversion
    to various formats and provides efficient data access methods.

    Attributes:
        ddb_con: DuckDB connection for SQL operations.
        _type: Backend type ('pyarrow' or 'duckdb').
        _dataset: Underlying PyArrow dataset.
        _ddb: DuckDB relation object.
    """

    def __init__(
        self,
        result: pds.Dataset | _duckdb.DuckDBPyRelation,
        ddb_con: _duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        """Initialize a PydalaTable instance.

        Args:
            result: The data source - either a PyArrow dataset or DuckDB relation.
            ddb_con: Existing DuckDB connection. If None, creates a new one.
        """
        if ddb_con is None:
            self.ddb_con = _duckdb.connect()
        else:
            self.ddb_con = ddb_con

        self._type = (
            "duckdb" if isinstance(result, _duckdb.DuckDBPyRelation) else "pyarrow"
        )
        if self._type == "pyarrow":
            self._dataset = result
            self._ddb = self.ddb_con.from_arrow(result)
        else:
            arrow_result = result.arrow()
            self._dataset = pds.dataset(arrow_result, schema=arrow_result.schema)
            self._ddb = result

    @staticmethod
    def _parse_sort_by_string(sort_by: str) -> list[list[str]]:
        """Parse a string sort specification into a list of [field, direction] pairs."""
        result = []
        for spec in sort_by.split(","):
            spec = spec.strip()
            parts = spec.rsplit(" ", 1)
            if len(parts) == 1:
                result.append([parts[0], "ascending"])
            else:
                field, direction = parts
                result.append([field, direction.lower()])
        return result

    @staticmethod
    def _parse_sort_by_list(sort_by: list | tuple) -> list[list[str]]:
        """Parse a list/tuple sort specification into a list of [field, direction] pairs."""
        if isinstance(sort_by[0], str):
            return [[s, "ascending"] if isinstance(s, str) else s for s in sort_by]
        elif isinstance(sort_by[0], list):
            return [[s[0], s[1]] for s in sort_by]
        else:
            raise ValueError("Invalid sort_by format")

    @staticmethod
    def _normalize_sort_directions(sort_by: list[list[str]]) -> list[list[str]]:
        """Normalize sort directions to 'ascending' or 'descending'."""
        normalized = []
        for field, direction in sort_by:
            direction_lower = direction.lower()
            if direction_lower in ["asc", "ascending"]:
                normalized.append([field, "ascending"])
            elif direction_lower in ["desc", "descending"]:
                normalized.append([field, "descending"])
            else:
                raise ValueError(f"Invalid sort direction: {direction}")
        return normalized

    @staticmethod
    def _format_for_pyarrow(sort_by: list[list[str]]) -> dict:
        """Format sort specification for PyArrow."""
        return {"sorting": [(s[0], s[1]) for s in sort_by]}

    @staticmethod
    def _format_for_duckdb(sort_by: list[list[str]]) -> str:
        """Format sort specification for DuckDB."""
        return ",".join(
            [
                " ".join([
                    s[0],
                    s[1].replace("ascending", "ASC").replace("descending", "DESC"),
                ])
                for s in sort_by
            ]
        )

    @staticmethod
    def _format_for_polars(sort_by: list[list[str]]) -> dict:
        """Format sort specification for Polars."""
        by = [s[0] for s in sort_by]
        descending = [s[1].lower() == "descending" for s in sort_by]
        return {"by": by, "descending": descending}

    @staticmethod
    def _get_sort_by(
        sort_by: str | list[str] | list[tuple[str, str]], type_: str | None = "pyarrow"
    ) -> dict | str:
        """
        A static method to get the sorted fields and their order, formatted for different types of data storage.

        Args:
            sort_by (str | list[str] | list[tuple[str, str]]): The fields to sort by.
            type_ (str | None, optional): The type of data storage. Defaults to "pyarrow".

        Returns:
            dict | str: A dictionary or a string of sorted fields and their order, formatted based on the data
                storage type.

        Raises:
            ValueError: If sort_by is not a string, list of strings, or list of tuples.
        """
        # Parse the input into a consistent format
        if isinstance(sort_by, str):
            sort_by = PydalaTable._parse_sort_by_string(sort_by)
        elif isinstance(sort_by, (list, tuple)):
            sort_by = PydalaTable._parse_sort_by_list(sort_by)
        else:
            raise ValueError(
                "sort_by must be a string, list of strings, or list of tuples."
            )

        # Normalize sort directions
        sort_by = PydalaTable._normalize_sort_directions(sort_by)

        # Format based on the target type
        if type_ == "pyarrow":
            return PydalaTable._format_for_pyarrow(sort_by)
        elif type_ == "duckdb":
            return PydalaTable._format_for_duckdb(sort_by)
        elif type_ == "polars":
            return PydalaTable._format_for_polars(sort_by)
        else:
            raise ValueError(f"Unsupported type_: {type_}")

    def to_arrow_dataset(self) -> pds.Dataset:
        """
        Convert the object to a PyArrow dataset if the type is "pyarrow".

        Returns:
            pds.Dataset: The PyArrow dataset if the type is "pyarrow", otherwise None.
        """
        return self._dataset

    @property
    def arrow_dataset(self) -> pds.Dataset:
        """
        This is a property method that returns an arrow dataset.
        Returns:
            pds.Dataset: The arrow dataset.
        """
        return self.to_arrow_dataset()

    def to_arrow_scanner(
        self,
        columns: str | list[str] | None = None,
        filter: pds.Expression | None = None,
        batch_size: int | None = None,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pds.FragmentScanOptions | None = None,
        use_threads: bool | None = None,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pds.Scanner:
        """
        Converts the table to an Arrow scanner.

        Args:
            columns (str | list[str] | None, optional): Columns to include in the scanner.
                Defaults to None.
            filter (pds.Expression | None, optional): Filter expression to apply. Defaults to None.
            batch_size (int | None, optional): Batch size for scanning.
                If None, uses default from ScannerConfig.
            sort_by (str | list[str] | list[tuple[str, str]] | None, optional): Columns to sort by.
                Defaults to None.
            batch_readahead (int | None, optional): Number of batches to read ahead.
                If None, uses default from ScannerConfig.
            fragment_readahead (int | None, optional): Number of fragments to read ahead.
                If None, uses default from ScannerConfig.
            fragment_scan_options (pds.FragmentScanOptions | None, optional): Fragment scan options.
                Defaults to None.
            use_threads (bool | None, optional): Whether to use multiple threads.
                If None, uses default from ScannerConfig.
            memory_pool (pa.MemoryPool | None, optional): Memory pool to use. Defaults to None.

        Returns:
            pds.Scanner: The Arrow scanner.

        Raises:
            ValueError: If the table type is not "pyarrow".
        """
        from .constants import ScannerConfig

        if self._type != "pyarrow":
            raise ValueError("This method is only available for pyarrow datasets.")

        # Use config defaults if not specified
        config = ScannerConfig()
        batch_size = batch_size or config.batch_size
        batch_readahead = batch_readahead or config.batch_readahead
        fragment_readahead = fragment_readahead or config.fragment_readahead
        use_threads = use_threads if use_threads is not None else config.use_threads
        fragment_scan_options = fragment_scan_options or config.fragment_scan_options
        memory_pool = memory_pool or config.memory_pool

        # Validate parameters
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if batch_readahead < 0:
            raise ValueError("batch_readahead must be a non-negative integer")
        if fragment_readahead < 0:
            raise ValueError("fragment_readahead must be a non-negative integer")

        if isinstance(columns, str):
            columns = [columns]

        if sort_by is not None:
            sort_by = self._get_sort_by(sort_by, "pyarrow")
            return self._dataset.sort_by(**sort_by).scanner(
                columns=columns,
                filter=filter,
                batch_size=batch_size,
                batch_readahead=batch_readahead,
                fragment_readahead=fragment_readahead,
                fragment_scan_options=fragment_scan_options,
                use_threads=use_threads,
                memory_pool=memory_pool,
            )
        else:
            return self._dataset.scanner(
                columns=columns,
                filter=filter,
                batch_size=batch_size,
                batch_readahead=batch_readahead,
                fragment_readahead=fragment_readahead,
                fragment_scan_options=fragment_scan_options,
                use_threads=use_threads,
                memory_pool=memory_pool,
            )

    def to_scanner(
        self,
        columns: str | list[str] | None = None,
        filter: pds.Expression | None = None,
        batch_size: int = 131072,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: pds.FragmentScanOptions | None = None,
        use_threads: bool = True,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pds.Scanner:
        """
        Converts the table to a scanner object for efficient data scanning.

        .. deprecated::
            Use scanner() instead. This method will be removed in a future version.

        Args:
            columns (str | list[str] | None, optional): Columns to include in the scanner.
                Defaults to None.
            filter (pds.Expression | None, optional): Filter expression to apply on the data.
                Defaults to None.
            batch_size (int, optional): Batch size for scanning. Defaults to 131072.
            sort_by (str | list[str] | list[tuple[str, str]] | None, optional): Columns to sort by.
                Defaults to None.
            batch_readahead (int, optional): Number of batches to read ahead. Defaults to 16.
            fragment_readahead (int, optional): Number of fragments to read ahead. Defaults to 4.
            fragment_scan_options (pds.FragmentScanOptions | None, optional): Fragment scan options.
                Defaults to None.
            use_threads (bool, optional): Whether to use multiple threads for scanning.
                Defaults to True.
            memory_pool (pa.MemoryPool | None, optional): Memory pool for scanning. Defaults to None.

        Returns:
            pds.Scanner: The scanner object for efficient data scanning.
        """
        import warnings

        warnings.warn(
            "to_scanner() is deprecated, use scanner() instead",
            DeprecationWarning,
            stacklevel=2
        )

        return self.scanner(
            columns=columns,
            filter=filter,
            batch_size=batch_size,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            fragment_scan_options=fragment_scan_options,
            use_threads=use_threads,
            memory_pool=memory_pool,
            sort_by=sort_by,
        )

    def scanner(
        self,
        columns: str | list[str] | None = None,
        filter: pds.Expression | None = None,
        batch_size: int | None = None,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        batch_readahead: int | None = None,
        fragment_readahead: int | None = None,
        fragment_scan_options: pds.FragmentScanOptions | None = None,
        use_threads: bool | None = None,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pds.Scanner:
        """
        Returns a scanner object that can be used to scan the table.

        Args:
            columns (str | list[str] | None, optional): Columns to include in the scan. Defaults to None.
            filter (pds.Expression | None, optional): Filter expression to apply during the scan.
                Defaults to None.
            batch_size (int | None, optional): Number of rows to read per batch.
                If None, uses default from ScannerConfig.
            sort_by (str | list[str] | list[tuple[str, str]] | None, optional): Columns to sort the scan by.
                Defaults to None.
            batch_readahead (int | None, optional): Number of batches to read ahead.
                If None, uses default from ScannerConfig.
            fragment_readahead (int | None, optional): Number of fragments to read ahead.
                If None, uses default from ScannerConfig.
            fragment_scan_options (pds.FragmentScanOptions | None, optional): Fragment scan options.
                Defaults to None.
            use_threads (bool | None, optional): Whether to use multiple threads for scanning.
                If None, uses default from ScannerConfig.
            memory_pool (pa.MemoryPool | None, optional): Memory pool to use for allocations.
                Defaults to None.

        Returns:
            pds.Scanner: Scanner object for scanning the table.
        """
        from .constants import ScannerConfig

        # Use config defaults if not specified
        config = ScannerConfig()

        # Validate parameters before passing to to_arrow_scanner
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if batch_readahead is not None and batch_readahead < 0:
            raise ValueError("batch_readahead must be a non-negative integer")
        if fragment_readahead is not None and fragment_readahead < 0:
            raise ValueError("fragment_readahead must be a non-negative integer")

        return self.to_arrow_scanner(
            columns=columns,
            filter=filter,
            batch_size=batch_size or config.batch_size,
            batch_readahead=batch_readahead or config.batch_readahead,
            fragment_readahead=fragment_readahead or config.fragment_readahead,
            fragment_scan_options=fragment_scan_options or config.fragment_scan_options,
            use_threads=use_threads if use_threads is not None else config.use_threads,
            memory_pool=memory_pool or config.memory_pool,
            sort_by=sort_by,
        )

    def to_duckdb(
        self,
        lazy: bool = True,
        columns: str | list[str] | None = None,
        batch_size: int = 131072,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        **kwargs,
    ) -> _duckdb.DuckDBPyRelation:
        """Convert the table to a DuckDB relation.

        This method provides access to DuckDB's SQL engine and optimized
        query execution. It supports both lazy and eager evaluation modes.

        Args:
            lazy: If True, returns a lazy relation for further optimization.
                  If False, materializes the result immediately.
            columns: Columns to select from the table. Defaults to all columns.
            batch_size: Batch size for scanning when materializing.
            sort_by: Column(s) to sort by in the result.
            distinct: Whether to return only distinct rows.
            **kwargs: Additional arguments for underlying operations.

        Returns:
            A DuckDBPyRelation object representing the table data.
        """

        if isinstance(columns, str):
            columns = [columns]

        columns = "*" if columns is None else ",".join(columns)

        if lazy:
            if sort_by is not None:
                sort_by = self._get_sort_by(sort_by, "duckdb")
                ddb = self._ddb.select(columns).order(sort_by)
            else:
                ddb = self._ddb.select(columns)
        else:
            ddb = self.ddb_con.from_arrow(
                self.scanner(
                    columns=columns, sort_by=sort_by, batch_size=batch_size, **kwargs
                ).to_table()
            )

        return ddb.distinct() if distinct else ddb

    def to_batch_reader(
        self,
        columns: str | list[str] | None = None,
        filter: pds.Expression | None = None,
        lazy: bool = True,
        batch_size: int = 131072,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: pds.FragmentScanOptions | None = None,
        use_threads: bool = True,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pa.RecordBatchReader:
        """
        Converts the table to a batch reader that can be used to read data in batches.

        Args:
            columns (str | list[str] | None, optional): Columns to include in the reader. Defaults to None.
            filter (pds.Expression | None, optional): Filter expression to apply. Defaults to None.
            lazy (bool, optional): Whether to lazily evaluate the reader. Defaults to True.
            batch_size (int, optional): Size of each batch. Defaults to 131072.
            sort_by (str | list[str] | list[tuple[str, str]] | None, optional): Columns to sort by.
                Defaults to None.
            distinct (bool, optional): Whether to return distinct rows. Defaults to False.
            batch_readahead (int, optional): Number of batches to read ahead. Defaults to 16.
            fragment_readahead (int, optional): Number of fragments to read ahead. Defaults to 4.
            fragment_scan_options (pds.FragmentScanOptions | None, optional): Fragment scan options.
                Defaults to None.
            use_threads (bool, optional): Whether to use multiple threads. Defaults to True.
            memory_pool (pa.MemoryPool | None, optional): Memory pool to use. Defaults to None.

        Returns:
            pa.RecordBatchReader: The batch reader object.
        """

        if self._type == "pyarrow" and sort_by is None and not distinct:
            if batch_size == 131072:
                return self.to_arrow_scanner(
                    columns=columns,
                    filter=filter,
                    batch_size=batch_size,
                    sort_by=sort_by,
                    batch_readahead=batch_readahead,
                    fragment_readahead=fragment_readahead,
                    fragment_scan_options=fragment_scan_options,
                    use_threads=use_threads,
                    memory_pool=memory_pool,
                ).to_reader()

        return self.to_duckdb(
            columns=columns, lazy=lazy, sort_by=sort_by, distinct=distinct
        ).fetch_arrow_reader(batch_size=batch_size)

    def to_batches(
        self,
        columns: str | list[str] | None = None,
        filter: pds.Expression | None = None,
        batch_size: int = 131072,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: pds.FragmentScanOptions | None = None,
        use_threads: bool = True,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pa.RecordBatch:
        """
        Converts the table to a sequence of RecordBatches.

        Args:
            columns (str | list[str] | None, optional): Columns to include in the RecordBatches.
                Defaults to None.
            filter (pds.Expression | None, optional): Filter expression to apply. Defaults to None.
            batch_size (int, optional): Size of each RecordBatch. Defaults to 131072.
            sort_by (str | list[str] | list[tuple[str, str]] | None, optional): Columns to sort by.
                Defaults to None.
            distinct (bool, optional): Whether to return distinct rows. Defaults to False.
            batch_readahead (int, optional): Number of batches to read ahead. Defaults to 16.
            fragment_readahead (int, optional): Number of fragments to read ahead. Defaults to 4.
            fragment_scan_options (pds.FragmentScanOptions | None, optional): Fragment scan options.
                Defaults to None.
            use_threads (bool, optional): Whether to use multiple threads. Defaults to True.
            memory_pool (pa.MemoryPool | None, optional): Memory pool to use. Defaults to None.

        Returns:
            pa.RecordBatch: The resulting sequence of RecordBatches.
        """

        return self.to_batch_reader(
            columns=columns,
            filter=filter,
            batch_size=batch_size,
            sort_by=sort_by,
            distinct=distinct,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            fragment_scan_options=fragment_scan_options,
            use_threads=use_threads,
            memory_pool=memory_pool,
        ).read_all()

    @property
    def ddb(self) -> _duckdb.DuckDBPyRelation:
        """
        Returns a DuckDBPyRelation object representing the table.

        Returns:
            _duckdb.DuckDBPyRelation: The DuckDBPyRelation object representing the table.
        """
        return self.to_duckdb()

    def to_arrow_table(
        self,
        columns: str | list[str] | None = None,
        filter: pds.Expression | None = None,
        batch_size: int = 131072,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: pds.FragmentScanOptions | None = None,
        use_threads: bool = True,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pa.Table:
        """
        Converts the table to an Arrow Table.

        Args:
            columns (str | list[str] | None, optional): Columns to include in the Arrow Table.
                Defaults to None.
            filter (pds.Expression | None, optional): Filter expression to apply. Defaults to None.
            batch_size (int, optional): Batch size for reading data. Defaults to 131072.
            sort_by (str | list[str] | list[tuple[str, str]] | None, optional): Columns to sort by.
                Defaults to None.
            distinct (bool, optional): Whether to return distinct rows. Defaults to False.
            batch_readahead (int, optional): Number of batches to read ahead. Defaults to 16.
            fragment_readahead (int, optional): Number of fragments to read ahead. Defaults to 4.
            fragment_scan_options (pds.FragmentScanOptions | None, optional): Fragment scan options.
                Defaults to None.
            use_threads (bool, optional): Whether to use multiple threads. Defaults to True.
            memory_pool (pa.MemoryPool | None, optional): Memory pool to use. Defaults to None.

        Returns:
            pa.Table: The converted Arrow Table.
        """

        if self._type == "pyarrow":
            t = self.scanner(
                columns=columns,
                filter=filter,
                batch_size=batch_size,
                sort_by=sort_by,
                batch_readahead=batch_readahead,
                fragment_readahead=fragment_readahead,
                fragment_scan_options=fragment_scan_options,
                use_threads=use_threads,
                memory_pool=memory_pool,
            ).to_table()
            if distinct:
                return _pl.from_arrow(t).unique(maintain_order=True).to_arrow()
            return t

        else:
            return self.to_duckdb(
                columns=columns, sort_by=sort_by, distinct=distinct
            ).to_arrow_table()

    def to_arrow(
        self,
        columns: str | list[str] | None = None,
        filter: pds.Expression | None = None,
        batch_size: int = 131072,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: pds.FragmentScanOptions | None = None,
        use_threads: bool = True,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pa.Table:
        """Convert the table to a PyArrow Table.

        This method provides a convenient alias for to_arrow_table() with
        the same functionality and parameters.

        Args:
            columns: Columns to include in the Arrow table. Defaults to all columns.
            filter: Filter expression to apply during conversion.
            batch_size: Batch size for reading data from the source.
            sort_by: Column(s) to sort by before returning the table.
            distinct: Whether to return only distinct rows.
            batch_readahead: Number of batches to read ahead for performance.
            fragment_readahead: Number of fragments to read ahead.
            fragment_scan_options: Options for fragment scanning.
            use_threads: Whether to use multiple threads for reading.
            memory_pool: Memory pool for Arrow allocations.

        Returns:
            The converted PyArrow Table.
        """

        return self.to_arrow_table(
            columns=columns,
            filter=filter,
            batch_size=batch_size,
            sort_by=sort_by,
            distinct=distinct,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            fragment_scan_options=fragment_scan_options,
            use_threads=use_threads,
            memory_pool=memory_pool,
        )

    def to_table(
        self,
        columns: str | list[str] | None = None,
        filter: pds.Expression | None = None,
        batch_size: int = 131072,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: pds.FragmentScanOptions | None = None,
        use_threads: bool = True,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pa.Table:
        """
        Converts the Pydala Table to an Arrow Table.

        Args:
            columns (str | list[str] | None, optional): Columns to include in the resulting table.
                Defaults to None.
            filter (pds.Expression | None, optional): Filter expression to apply. Defaults to None.
            batch_size (int, optional): Batch size for reading data. Defaults to 131072.
            sort_by (str | list[str] | list[tuple[str, str]] | None, optional): Columns to sort by.
                Defaults to None.
            distinct (bool, optional): Whether to return distinct rows. Defaults to False.
            batch_readahead (int, optional): Number of batches to read ahead. Defaults to 16.
            fragment_readahead (int, optional): Number of fragments to read ahead. Defaults to 4.
            fragment_scan_options (pds.FragmentScanOptions | None, optional): Fragment scan options.
                Defaults to None.
            use_threads (bool, optional): Whether to use multiple threads. Defaults to True.
            memory_pool (pa.MemoryPool | None, optional): Memory pool to use. Defaults to None.

        Returns:
            pa.Table: The resulting Arrow Table.
        """

        return self.to_arrow_table(
            columns=columns,
            filter=filter,
            batch_size=batch_size,
            sort_by=sort_by,
            distinct=distinct,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            fragment_scan_options=fragment_scan_options,
            use_threads=use_threads,
            memory_pool=memory_pool,
        )

    @property
    def arrow(self) -> pa.Table:
        """
        Converts the table to an Arrow table.

        Returns:
            pa.Table: The Arrow table representation of the table.
        """
        return self.to_arrow()

    def to_polars(
        self,
        lazy: bool = True,
        columns: str | list[str] | None = None,
        sort_by: str | list[str] | None = None,
        distinct: bool = False,
        batch_size: int = 131072,
        **kwargs,
    ) -> _pl.DataFrame | _pl.LazyFrame:
        """Convert the table to a Polars DataFrame or LazyFrame.

        This method efficiently converts the table to Polars format, with
        support for both lazy and eager evaluation. When using PyArrow backend,
        it leverages Polars' native PyArrow scanner for optimal performance.

        Args:
            lazy: If True, returns a LazyFrame for lazy evaluation.
                   If False, returns a materialized DataFrame.
            columns: Columns to include in the result. Defaults to all columns.
            sort_by: Column(s) to sort by before returning the result.
            distinct: Whether to return only distinct rows.
            batch_size: Batch size for scanning from the data source.
            **kwargs: Additional arguments passed to underlying conversion methods.

        Returns:
            A Polars DataFrame or LazyFrame depending on the lazy parameter.
        """

        if isinstance(columns, str):
            columns = [columns]

        if self._type == "pyarrow":
            if lazy:
                df = _pl.scan_pyarrow_dataset(self._dataset, batch_size=batch_size)
                if columns is not None:
                    df = df.select(columns)
                if sort_by is not None:
                    sort_by = self._get_sort_by(sort_by, "polars")
                    df = df.sort(**sort_by)
                if distinct:
                    df = df.unique(maintain_order=True)
            else:
                df = _pl.from_arrow(
                    self.to_batches(
                        columns=columns,
                        batch_size=batch_size,
                        sort_by=sort_by,
                        distinct=distinct,
                        **kwargs,
                    )
                )
        else:
            df = self.to_duckdb(
                lazy=lazy,
                columns=columns,
                sort_by=sort_by,
                distinct=distinct,
                batch_size=batch_size,
                **kwargs,
            ).pl(batch_size=batch_size)
        return df

    @property
    def pl(self) -> _pl.DataFrame | _pl.LazyFrame:
        """
        Returns the table as a Polars DataFrame or LazyFrame.

        Returns:
            _pl.DataFrame or _pl.LazyFrame: The table as a Polars DataFrame or LazyFrame.
        """
        return self.to_polars()

    def to_pandas(
        self,
        columns: str | list[str] | None = None,
        sort_by: str | list[str] | None = None,
        distinct: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Convert the table to a pandas DataFrame.

        This method converts the table to pandas format via PyArrow for
        optimal performance and memory efficiency.

        Args:
            columns: Columns to include in the DataFrame. Defaults to all columns.
            sort_by: Column(s) to sort by before conversion.
            distinct: Whether to return only distinct rows.
            **kwargs: Additional arguments passed to to_arrow_table().

        Returns:
            A pandas DataFrame containing the table data.
        """
        return self.to_arrow_table(
            columns=columns, sort_by=sort_by, distinct=distinct, **kwargs
        ).to_pandas()

    def to_df(
        self,
        columns: str | list[str] | None = None,
        sort_by: str | list[str] | None = None,
        distinct: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Converts the table to a pandas DataFrame.

        Args:
            columns (str | list[str] | None, optional): Columns to include in the DataFrame. Defaults to None.
            sort_by (str | list[str] | None, optional): Columns to sort the DataFrame by. Defaults to None.
            distinct (bool, optional): Whether to return distinct rows. Defaults to False.
            **kwargs: Additional keyword arguments to be passed to the to_pandas method.

        Returns:
            pd.DataFrame: The table as a pandas DataFrame.
        """
        return self.to_pandas(
            columns=columns, sort_by=sort_by, distinct=distinct, **kwargs
        )

    @property
    def df(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame representation of the table.

        Returns:
            pd.DataFrame: A pandas DataFrame representation of the table.
        """
        return self.to_pandas()

    def sql(self, sql: str) -> _duckdb.DuckDBPyRelation:
        """Execute a SQL query using the table's DuckDB connection.

        This method provides direct SQL access to the table data through
        the DuckDB connection. The table is automatically registered and
        can be referenced in SQL queries.

        Args:
            sql: The SQL query to execute.

        Returns:
            The result of the SQL query as a DuckDB relation.

        Note:
            The table is registered with DuckDB and can be referenced
            by name in SQL queries. Use parameterized queries for security
            when dealing with user input.
        """
        return self.ddb_con.sql(sql)

    def __repr__(self) -> str:
        """Return a string representation of the table.

        Returns a preview of the table showing the first 10 rows.
        The representation is generated through DuckDB for consistent
        formatting across different backends.

        Returns:
            String representation of the table preview.
        """
        return self.to_duckdb().limit(10).__repr__()

    def __call__(self) -> pds.Dataset | _duckdb.DuckDBPyRelation:
        """Return the underlying result object.

        This allows the table to be called directly to access the
        underlying PyArrow dataset or DuckDB relation.

        Returns:
            The underlying data object (PyArrow dataset or DuckDB relation).
        """
        return self.result

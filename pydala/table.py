import duckdb as _duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds

from .helpers.polars import pl as _pl


class PydalaTable:
    def __init__(
        self,
        result: pds.Dataset | _duckdb.DuckDBPyRelation,
        ddb_con: _duckdb.DuckDBPyConnection | None = None,
    ):
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
            self._dataset = pds.dataset(result.arrow())
            self._ddb = result

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
        if isinstance(sort_by, str):
            sort_by = [s.split(" ") for s in sort_by.split(",")]
            sort_by = [[s[0], "ascending"] if len(s) == 1 else s for s in sort_by]

        elif isinstance(sort_by, list | tuple):
            if isinstance(sort_by[0], str):
                sort_by = [
                    [s, "ascending"] if isinstance(s, str) else s for s in sort_by
                ]
            if isinstance(sort_by[0], list):
                sort_by = [[s[0], s[1]] for s in sort_by]

        for i in range(len(sort_by)):
            if sort_by[i][1].lower() in ["asc", "ascending"]:
                sort_by[i][1] = "ascending"
            elif sort_by[i][1].lower() in ["desc", "descending"]:
                sort_by[i][1] = "descending"

        if type_ == "pyarrow":
            return {"sorting": [(s[0], s[1]) for s in sort_by]}

        elif type_ == "duckdb":
            sort_by = ",".join(
                [
                    " ".join(
                        [
                            s[0],
                            s[1]
                            .replace("ascending", "ASC")
                            .replace("descending", "DESC"),
                        ]
                    )
                    for s in sort_by
                ]
            )
            return sort_by

        elif type_ == "polars":
            by = [s[0] for s in sort_by]
            descending = [s[1].lower() == "descending" for s in sort_by]
            return {"by": by, "descending": descending}

        else:
            raise ValueError(
                "sort_by must be a string, list of strings, or list of tuples."
            )

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
        batch_size: int = 131072,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: pds.FragmentScanOptions | None = None,
        use_threads: bool = True,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pds.Scanner:
        """
        Converts the table to an Arrow scanner.

        Args:
            columns (str | list[str] | None, optional): Columns to include in the scanner.
                Defaults to None.
            filter (pds.Expression | None, optional): Filter expression to apply. Defaults to None.
            batch_size (int, optional): Batch size for scanning. Defaults to 131072.
            sort_by (str | list[str] | list[tuple[str, str]] | None, optional): Columns to sort by.
                Defaults to None.
            batch_readahead (int, optional): Number of batches to read ahead. Defaults to 16.
            fragment_readahead (int, optional): Number of fragments to read ahead. Defaults to 4.
            fragment_scan_options (pds.FragmentScanOptions | None, optional): Fragment scan options.
                Defaults to None.
            use_threads (bool, optional): Whether to use multiple threads. Defaults to True.
            memory_pool (pa.MemoryPool | None, optional): Memory pool to use. Defaults to None.

        Returns:
            pds.Scanner: The Arrow scanner.

        Raises:
            ValueError: If the table type is not "pyarrow".
        """

        if self._type != "pyarrow":
            raise ValueError("This method is only available for pyarrow datasets.")

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

        return self.to_arrow_scanner(
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
        batch_size: int = 131072,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: pds.FragmentScanOptions | None = None,
        use_threads: bool = True,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pds.Scanner:
        """
        Returns a scanner object that can be used to scan the table.

        Args:
            columns (str | list[str] | None, optional): Columns to include in the scan. Defaults to None.
            filter (pds.Expression | None, optional): Filter expression to apply during the scan.
                Defaults to None.
            batch_size (int, optional): Number of rows to read per batch. Defaults to 131072.
            sort_by (str | list[str] | list[tuple[str, str]] | None, optional): Columns to sort the scan by.
                Defaults to None.
            batch_readahead (int, optional): Number of batches to read ahead. Defaults to 16.
            fragment_readahead (int, optional): Number of fragments to read ahead. Defaults to 4.
            fragment_scan_options (pds.FragmentScanOptions | None, optional): Fragment scan options.
                Defaults to None.
            use_threads (bool, optional): Whether to use multiple threads for scanning. Defaults to True.
            memory_pool (pa.MemoryPool | None, optional): Memory pool to use for allocations.
                Defaults to None.

        Returns:
            pds.Scanner: Scanner object for scanning the table.
        """

        return self.to_arrow_scanner(
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

    def to_duckdb(
        self,
        lazy: bool = True,
        columns: str | list[str] | None = None,
        batch_size: int = 131072,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        **kwargs,
    ) -> _duckdb.DuckDBPyRelation:
        """
        Converts the table to a DuckDBPyRelation object.

        Args:
            lazy (bool, optional): If True, the conversion is lazy. Defaults to True.
            columns (str | list[str] | None, optional): The columns to select. Defaults to None.
            batch_size (int, optional): The batch size for scanning the table. Defaults to 131072.
            sort_by (str | list[str] | list[tuple[str, str]] | None, optional): The columns to sort by.
                Defaults to None.
            distinct (bool, optional): If True, removes duplicate rows. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            _duckdb.DuckDBPyRelation: The converted DuckDBPyRelation object.
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
        """
        Converts the table to an Arrow table.

        Args:
            columns (str | list[str] | None, optional): Columns to include in the Arrow table.
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
            pa.Table: The converted Arrow table.
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
        """
        Converts the table to a Polars DataFrame or LazyFrame.

        Args:
            lazy (bool, optional): If True, returns a LazyFrame. If False, returns a DataFrame. Defaults to True.
            columns (str | list[str] | None, optional): Columns to include in the resulting DataFrame/LazyFrame.
                Defaults to None.
            sort_by (str | list[str] | None, optional): Columns to sort the resulting DataFrame/LazyFrame by.
                Defaults to None.
            distinct (bool, optional): If True, removes duplicate rows from the resulting DataFrame/LazyFrame.
                Defaults to False.
            batch_size (int, optional): The batch size used for scanning the PyArrow dataset. Defaults to 131072.
            **kwargs: Additional keyword arguments to be passed to the underlying functions.

        Returns:
            _pl.DataFrame | _pl.LazyFrame: The resulting Polars DataFrame or LazyFrame.
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
        """
        Convert the table to a pandas DataFrame.

        Args:
            columns (str | list[str] | None): Optional. Columns to include in the DataFrame.
            sort_by (str | list[str] | None): Optional. Columns to sort the DataFrame by.
            distinct (bool): Optional. Whether to return distinct rows in the DataFrame.
            **kwargs: Additional keyword arguments to be passed to the underlying methods.

        Returns:
            pd.DataFrame: A pandas DataFrame representing the table.
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

    def sql(self, sql: str):
        """
        Executes the given SQL query on the database connection.

        Args:
            sql (str): The SQL query to execute.

        Returns:
            The result of the SQL query execution.
        """
        return self.ddb_con.sql(sql)

    def __repr__(self):
        # if self._type == "pyarrow":
        #    return self.to_polars().head(10).collect().__repr__()
        # return self.result.limit(10).__repr__()
        return self.to_duckdb().limit(10).__repr__()

    def __call__(self):
        return self.result

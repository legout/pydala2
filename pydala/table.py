import duckdb as _duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds

from .helpers.polars_ext import pl as _pl


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
            self._dataset = None
            self._ddb = result

    # @staticmethod
    def _get_sort_by(
        self, sort_by: str | list[str] | list[tuple[str, str]], type_: str | None = None
    ):
        if type_ is None:
            type_ = self._type

        if type_ == "pyarrow":
            if isinstance(sort_by, str):
                return [(sort_by, "ascending")]
            elif isinstance(sort_by, list):
                if isinstance(sort_by[0], str):
                    return [
                        (s, "ascending") if isinstance(s, str) else s for s in sort_by
                    ]
                else:
                    return sort_by

        if isinstance(sort_by, str):
            if "asc" in sort_by.lower() or "desc" in sort_by.lower():
                return sort_by
            elif "," in sort_by:
                return sort_by
            else:
                return f"{sort_by} ASC"

        elif isinstance(sort_by, list):
            if isinstance(sort_by[0], str):
                return ",".join(
                    [f"{s} ASC" if isinstance(s, str) else s for s in sort_by]
                )
            elif isinstance(sort_by[0], tuple):
                return ",".join([f"{s[0]} {s[1]}" for s in sort_by])

        else:
            raise ValueError(
                "sort_by must be a string, list of strings, or list of tuples."
            )

    # def sort_by(self, by: str | list[str] | list[tuple[str, str]] | None = None, type_:str|None=None):
    #     type_ = type_ or self._type
    #
    #     if by is not None:
    #         if type_ == "pyarrow":
    #             sort_by_pa = self._get_sort_by(by, "pyarrow")
    #             self._dataset = self._dataset.sort_by(sort_by_pa)
    #
    #         if type_ == "duckdb":
    #             sort_by_ddb = self._get_sort_by(by, "duckdb")
    #             self._ddb = self._ddb.order(sort_by_ddb)
    #     return self

    def to_arrow_dataset(self) -> pds.Dataset:
        if self._type == "pyarrow":
            return self._dataset

    @property
    def arrow_dataset(self) -> pds.Dataset:
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
        if self._type != "pyarrow":
            raise ValueError("This method is only available for pyarrow datasets.")

        if isinstance(columns, str):
            columns = [columns]

        if sort_by is not None:
            sort_by = self._get_sort_by(sort_by, "pyarrow")
            return self._dataset.sort_by(sort_by).scanner(
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
        # batch_size: int | None = None,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        **kwargs,
    ) -> _duckdb.DuckDBPyRelation:
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
                self.scanner(columns=columns, sort_by=sort_by, **kwargs).to_table()
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
        # if sort_by is not None:
        #    _ = self.sort_by(sort_by)

        if self._type == "pyarrow" and sort_by is not None and not distinct:
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

        else:
            return self.to_duckdb(
                columns=columns, lazy=lazy, sort_by=sort_by, distinct=distinct
            ).to_batch_reader(batch_size=batch_size)

    def to_batchtes(
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

    def to_arrow(self, *args, **kwargs) -> pa.Table:
        return self.to_arrow_table(*args, **kwargs)

    def to_table(self, *args, **kwargs) -> pa.Table:
        return self.to_arrow(*args, **kwargs)

    @property
    def arrow(self) -> pa.Table:
        return self.to_arrow()

    def to_polars(
        self,
        lazy: bool = True,
        columns: str | list[str] | None = None,
        sort_by: str | list[str] | None = None,
        distinct: bool = False,
        batch_size: int | None = None,
        **kwargs,
    ) -> _pl.DataFrame | _pl.LazyFrame:
        if isinstance(columns, str):
            columns = [columns]

        if self._type == "pyarrow":
            if lazy:
                df = _pl.scan_pyarrow_dataset(self._dataset)
                if columns is not None:
                    df = df.select(columns)
            else:
                df = _pl.from_arrow(self.scanner(columns=columns)
                          


    @property
    def pl(self) -> _pl.DataFrame | _pl.LazyFrame:
        return self.to_polars()

    def to_pandas(
        self,
        columns: str | list[str] | None = None,
        sort_by: str | list[str] | None = None,
        distinct: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        return self.to_duckdb(columns=columns, sort_by=sort_by, distinct=distinct).df()

    def to_df(self, *args, **kwargs) -> pd.DataFrame:
        return self.to_pandas(*args, **kwargs)

    @property
    def df(self) -> pd.DataFrame:
        return self.to_pandas()

    def __repr__(self):
        if self._type == "pyarrow":
            return self.to_polars().head(10).collect().__repr__()
        return self.result.limit(10).__repr__()

    def __call__(self):
        return self.result

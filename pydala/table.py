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

        self.result = result
        self._type = (
            "duckdb" if isinstance(result, _duckdb.DuckDBPyRelation) else "pyarrow"
        )

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

    def sort_by(self, by: str | list[str] | list[tuple[str, str]] | None = None):
        if by is not None:
            sort_by = self._get_sort_by(self, by)
            if self._type == "pyarrow":
                self.result = self.result.sort_by(sort_by)
            else:
                self.result = self.result.order(sort_by)
        return self

    def to_arrow_dataset(self) -> pds.Dataset:
        if self._type == "pyarrow":
            return self.result

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
        if sort_by is not None:
            _ = self.sort_by(sort_by)

        if self._type == "pyarrow":
            return self.result.scanner(
                columns=columns,
                filter=filter,
                batch_size=batch_size,
                batch_readahead=batch_readahead,
                fragment_readahead=fragment_readahead,
                fragment_scan_options=fragment_scan_options,
                use_threads=use_threads,
                memory_pool=memory_pool,
            )

    def to_scanner(self, *args, **kwargs) -> pds.Scanner:
        return self.to_arrow_scanner(*args, **kwargs)

    def scanner(self, *args, **kwargs) -> pds.Scanner:
        return self.to_arrow_scanner(*args, **kwargs)

    def to_batch_reader(
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
    ) -> pa.RecordBatchReader:
        if isinstance(columns, str):
            columns = [columns]

        if self._type == "pyarrow":
            ddb = self.to_duckdb(
                columns=columns,
                filter=filter,
                batch_size=batch_size,
                batch_readahead=batch_readahead,
                fragment_readahead=fragment_readahead,
                fragment_scan_options=fragment_scan_options,
                use_threads=use_threads,
                memory_pool=memory_pool,
            )
            if sort_by is not None:
                sort_by = self._get_sort_by(self, sort_by, "duckdb")
                return (
                    ddb.order(sort_by)
                    .distinct()
                    .fetch_arrow_reader(batch_size=batch_size or 131072)
                    if distinct
                    else ddb.order(sort_by).fetch_arrow_reader(
                        batch_size=batch_size or 131072
                    )
                )

            return (
                ddb.distinct().fetch_arrow_reader(batch_size=batch_size or 131072)
                if distinct
                else ddb.fetch_arrow_reader(batch_size=batch_size or 131072)
            )

        columns = "*" if columns is None else ",".join(columns)
        if sort_by is not None:
            sort_by = self._get_sort_by(self, sort_by, "duckdb")
            return (
                self.result.select(columns)
                .order(sort_by)
                .distinct()
                .fetch_arrow_reader(batch_size=batch_size or 131072)
                if distinct
                else self.result.select(columns).order(sort_by).fetch_arrow_reader()
            )
        return (
            self.result.select(columns)
            .distinct()
            .fetch_arrow_reader(batch_size=batch_size or 131072)
            if distinct
            else self.result.select(columns).fetch_arrow_reader(
                batch_size=batch_size or 131072
            )
        )

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

    def to_duckdb(
        self,
        lazy: bool = True,
        columns: str | list[str] | None = None,
        batch_size: int | None = None,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        **kwargs,
    ) -> _duckdb.DuckDBPyRelation:
        if isinstance(columns, str):
            columns = [columns]

        if self._type == "pyarrow":
            if lazy:
                ddb = self.ddb_con.from_arrow(
                    self.to_arrow_scanner(
                        columns=columns, batch_size=batch_size or 131072, **kwargs
                    )
                )
            else:
                ddb = self.ddb_con.from_arrow(
                    self.result.to_table(
                        columns=columns, batch_size=batch_size or 131072, **kwargs
                    )
                )
            if sort_by is not None:
                sort_by = self._get_sort_by(self, sort_by, "duckdb")
                return ddb.order(sort_by).distinct() if distinct else ddb.order(sort_by)

        return ddb.distinct() if distinct else ddb

        columns = "*" if columns is None else ",".join(columns)
        if sort_by is not None:
            sort_by = self._get_sort_by(self, sort_by, "duckdb")
            return (
                self.result.select(columns).order(sort_by).distinct()
                if distinct
                else self.result.select(columns).order(sort_by)
            )
        return (
            self.result.select(columns).distinct()
            if distinct
            else self.result.select(columns)
        )

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
        if isinstance(columns, str):
            columns = [columns]

        if self._type == "pyarrow":
            arrow_table = self.result.to_table(
                columns=columns,
                filter=filter,
                batch_size=batch_size,
                batch_readahead=batch_readahead,
                fragment_readahead=fragment_readahead,
                fragment_scan_options=fragment_scan_options,
                use_threads=use_threads,
                memory_pool=memory_pool,
            )
            if sort_by is not None:
                sort_by = self._get_sort_by(self, sort_by)
                return (
                    self.ddb_con.from_arrow(arrow_table.sort_by(sort_by))
                    .distinct()
                    .arrow()
                    if distinct
                    else arrow_table.sort_by(sort_by)
                )
            return (
                self.ddb_con.from_arrow(arrow_table).distinct().arrow()
                if distinct
                else arrow_table
            )

        columns = "*" if columns is None else ",".join(columns)
        if sort_by is not None:
            sort_by = self._get_sort_by(self, sort_by, "duckdb")
            return (
                self.result.select(columns).order(sort_by).distinct().arrow()
                if distinct
                else self.result.select(columns).order(sort_by).arrow()
            )

        return (
            self.result.select(columns)
            .distinct()
            .arrow(batch_size=batch_size or 131072)
            if distinct
            else self.result.select(columns).arrow()
        )

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
                df = _pl.scan_pyarrow_dataset(self.result, batch_size=batch_size)
                if columns is not None:
                    df = df.select(columns)
                if sort_by is not None:
                    df = df.sort(sort_by)
                if distinct:
                    df = df.unique()
                return df
            else:
                df = _pl.from_arrow(
                    self.result.to_table(
                        columns=columns, batch_size=batch_size or 131072, **kwargs
                    )
                )
                if sort_by is not None:
                    df = df.sort(sort_by)
                if distinct:
                    df = df.unique()
                return df

        columns = "*" if columns is None else ",".join(columns)
        if sort_by is not None:
            sort_by = self._get_sort_by(self, sort_by, "duckdb")
            if batch_size:
                return (
                    self.result.select(columns)
                    .order(sort_by)
                    .distinct()
                    .pl(batch_size=batch_size)
                    if distinct
                    else self.result.select(columns)
                    .order(sort_by)
                    .pl(batch_size=batch_size)
                )
            return (
                self.result.select(columns).order(sort_by).distinct().pl()
                if distinct
                else self.result.select(columns).order(sort_by).pl()
            )
        return (
            self.result.select(columns).pl()
            if batch_size is None
            else self.result.select(columns).pl(batch_size=batch_size)
        )

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

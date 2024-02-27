import duckdb as _duckdb
import pyarrow as pa
import pyarrow.dataset as pds
from polars import col

from .polars_ext import pl as _pl


class PyDalaTable:
    def __init__(
        self,
        table: pds.Dataset | _duckdb.DuckDBPyRelation,
        ddb_con: _duckdb.DuckDBPyConnection | None = None,
    ):
        if ddb_con is None:
            self.ddb_con = _duckdb.connect()
        else:
            self.ddb_con = ddb_con

        self._table = table
        self._type = (
            "duckdb" if isinstance(table, _duckdb.DuckDBPyRelation) else "pyarrow"
        )

    def to_arrow_dataset(self) -> pds.Dataset:
        if self._type == "pyarrow":
            return self._table

    @property
    def arrow_dataset(self) -> pds.Dataset:
        return self.to_arrow_dataset()

    def to_arrow_scanner(
        self,
        columns: str | list[str] | None = None,
        filter: pds.Expression | None = None,
        batch_size: int = 131072,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: pds.FragmentScanOptions | None = None,
        use_threads: bool = True,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pds.Scanner:
        if self._type == "pyarrow":
            return self._table.scanner(
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
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: pds.FragmentScanOptions | None = None,
        use_threads: bool = True,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pa.RecordBatchReader:
        if isinstance(columns, str):
            columns = [columns]

        if self._type == "pyarrow":
            return self.to_duckdb(
                columns=colums,
                filter=filter,
                batch_size=batch_size,
                batch_readahead=batch_readahead,
                fragment_readahead=fragment_readahead,
                fragment_scan_options=fragment_scan_options,
                use_threads=use_threads,
                memory_pool=memory_pool,
            ).fetch_arrow_reader(batch_size=batch_size or 131072)

        columns = "*" if columns is None else ",".join(columns)
        return self._table.select(columns).fetch_arrow_reader(
            batch_size=batch_size or 131072
        )

    def to_batchtes(
        self,
        columns: str | list[str] | None = None,
        filter: pds.Expression | None = None,
        batch_size: int = 131072,
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
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            fragment_scan_options=fragment_scan_options,
            use_threads=use_threads,
            memory_pool=memory_pool,
        ).read_all()

    def to_polars(
        self,
        lazy: bool = True,
        columns: str | list[str] | None = None,
        batch_size: int | None = None,
        **kwargs,
    ) -> _pl.DataFrame | _pl.LazyFrame:
        if isinstance(columns, str):
            columns = [columns]

        if self._type == "pyarrow":
            if lazy:
                return _pl.scan_pyarrow_dataset(
                    self._table, batch_size=batch_size
                ).select(columns)

            return _pl.from_arrow(
                self._table.to_table(
                    columns=columns, batch_size=batch_size or 131072, **kwargs
                )
            )

        columns = "*" if columns is None else ",".join(columns)
        return (
            self._table.select(columns).pl()
            if batch_size is None
            else self._table.select(columns).pl(batch_size=batch_size)
        )

    @property
    def pl(self) -> _pl.DataFrame | _pl.LazyFrame:
        return self.to_polars()

    def to_duckdb(
        self,
        lazy: bool = True,
        columns: str | list[str] | None = None,
        batch_size: int | None = None,
        **kwargs,
    ) -> _duckdb.DuckDBPyRelation:
        if isinstance(columns, str):
            columns = [columns]

        if self._type == "pyarrow":
            if lazy:
                return self.ddb_con.from_arrow(
                    self.to_arrow_scanner(
                        columns=columns, batch_size=batch_size or 131072, **kwargs
                    )
                )
            return self.ddb_con.from_arrow(
                self._table.to_table(
                    columns=columns, batch_size=batch_size or 131072, **kwargs
                )
            )

        columns = "*" if columns is None else ",".join(columns)
        return self._table.select(columns)

    @prperty
    def ddb(self) -> _duckdb.DuckDBPyRelation:
        return self.to_duckdb()

    def to_arrow_table(
        self,
        columns: str | list[str] | None = None,
        filter: pds.Expression | None = None,
        batch_size: int = 131072,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: pds.FragmentScanOptions | None = None,
        use_threads: bool = True,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pa.Table:
        if isinstance(columns, str):
            columns = [columns]

        if self._type == "pyarrow":
            return self._table.to_table(
                columns=columns,
                filter=filter,
                batch_size=batch_size,
                batch_readahead=batch_readahead,
                fragment_readahead=fragment_readahead,
                fragment_scan_options=fragment_scan_options,
                use_threads=use_threads,
                memory_pool=memory_pool,
            )
        columns = "*" if columns is None else ",".join(columns)
        return self._table.select(columns).arrow(batch_size=batch_size or 131072)

    def to_arrow(self, *args, **kwargs) -> pa.Table:
        return self.to_arrow_table(*args, **kwargs)

    def to_table(self, *args, **kwargs) -> pa.Table:
        return self.to_arrow(*args, **kwargs)

    @property
    def arrow(self) -> pa.Table:
        return self.to_arrow()

    def to_pandas(
        self,
        columns: str | list[str] | None = None,
        batch_size: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        if isinstance(columns, str):
            columns = [columns]

        if self._type == "pyarrow":
            return self._table.to_table(
                columns=columns, batch_size=batch_size or 131072
            ).to_pandas()

        columns = "*" if columns is None else ",".join(columns)
        return self._table.select("*").df()

    def to_df(self, *args, **kwargs) -> pd.DataFrame:
        return self.to_pandas(*args, **kwargs)

    @property
    def df(self) -> pd.DataFrame:
        return self.to_pandas()

    def __repr__(self):
        if self._type == "pyarrow":
            return self.to_polars().head(10).collect().__repr__()
        return self._table.limit(10).__repr__()

    def __call__(self):
        return self._table

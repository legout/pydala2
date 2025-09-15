import logging
import posixpath
import typing as t
from fsspec import AbstractFileSystem
import duckdb as _duckdb
import pyarrow as pa
import pyarrow.dataset as pds
import polars as pl

from .base import BaseDataset
from .metadata import PydalaDatasetMetadata
from .table import PydalaTable
from .helpers.datetime import get_timestamp_column

logger = logging.getLogger(__name__)


class ParquetDataset(PydalaDatasetMetadata, BaseDataset):
    """Dataset class for Parquet files with metadata support."""

    def __init__(
        self,
        path: str,
        name: str | None = None,
        filesystem: AbstractFileSystem | None = None,
        bucket: str | None = None,
        partitioning: str | list[str] | None = None,
        cached: bool = False,
        timestamp_column: str | None = None,
        ddb_con: _duckdb.DuckDBPyConnection | None = None,
        **fs_kwargs,
    ):
        # Initialize metadata handling
        PydalaDatasetMetadata.__init__(
            self,
            path=path,
            filesystem=filesystem,
            bucket=bucket,
            cached=cached,
            partitioning=partitioning,
            ddb_con=ddb_con,
            **fs_kwargs,
        )

        # Initialize base dataset
        BaseDataset.__init__(
            self,
            path=path,
            name=name,
            filesystem=filesystem,
            bucket=bucket,
            partitioning=partitioning,
            format="parquet",
            cached=cached,
            timestamp_column=timestamp_column,
            ddb_con=self.ddb_con,
            **fs_kwargs,
        )

        try:
            self.load()
        except Exception as e:
            logger.debug(f"Failed to load during initialization: {e}")

    def load(
        self,
        update_metadata: bool = False,
        reload_metadata: bool = False,
        schema: pa.Schema | None = None,
        ts_unit: str = "us",
        tz: str | None = None,
        format_version: str = "2.6",
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Load Parquet dataset with metadata support."""
        # Handle metadata updates
        if not self.has_file_metadata_file:
            if len(self.fs.lss(self.path)) == 0:
                return
            else:
                update_metadata = True

        if kwargs.pop("update", None):
            update_metadata = True
        if kwargs.pop("reload", None):
            reload_metadata = True

        if update_metadata or reload_metadata:
            self.update(
                reload=reload_metadata,
                schema=schema,
                ts_unit=ts_unit,
                tz=tz,
                format_version=format_version,
                verbose=verbose,
                **kwargs,
            )
            if not hasattr(self, "_schema"):
                self._schema = self.metadata.schema.to_arrow_schema()
            self.update_metadata_table()

        # Load Arrow dataset
        self._data_processor._arrow_dataset = pds.parquet_dataset(
            self._metadata_file,
            partitioning=self._data_processor._partitioning,
            filesystem=self._file_manager.filesystem,
        )

        self._data_processor.table = PydalaTable(
            result=self._data_processor._arrow_dataset,
            ddb_con=self._data_processor.ddb_con,
        )

        # Handle timestamp column
        if self._data_processor._timestamp_column is None:
            timestamp_columns = get_timestamp_column(self._data_processor.table.pl.head(10))
            if timestamp_columns:
                self._data_processor._timestamp_column = timestamp_columns[0]

        if self._data_processor._timestamp_column is not None:
            ts_type = self.schema.field(self._data_processor._timestamp_column).type
            tz = getattr(ts_type, "tz", None)
            self._data_processor._tz = tz
            if tz is not None:
                self.ddb_con.execute("SET TimeZone=?", [str(tz)])

        # Register in DuckDB
        self.ddb_con.register(f"{self.name}", self._data_processor._arrow_dataset)

        # Initialize filter
        self._filter = super()._filter.__class__(
            arrow_dataset=self._data_processor._arrow_dataset,
            table=self._data_processor.table,
        )

    def scan(self, filter_expr: str | None = None) -> PydalaTable:
        """Scan dataset using metadata."""
        PydalaDatasetMetadata.scan(self, filter_expr=filter_expr)

        if len(self.scan_files) == 0:
            return PydalaTable(result=self.table.ddb.limit(0))

        return PydalaTable(
            result=pds.dataset(
                [posixpath.join(self.path, fn) for fn in self.scan_files],
                filesystem=self._file_manager.filesystem,
                format="parquet",
                partitioning=self._data_processor._partitioning,
            ),
            ddb_con=self.ddb_con,
        )

    def __repr__(self) -> str:
        """String representation of the dataset."""
        if self.table is not None:
            return self.table.__repr__()
        else:
            return f"{self.path} is empty."

    def write_to_dataset(
        self,
        data: t.Any,
        mode: str = "append",
        basename: str | None = None,
        partition_by: str | list[str] | None = None,
        max_rows_per_file: int | None = 2_500_000,
        row_group_size: int | None = 250_000,
        compression: str = "zstd",
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        unique: bool | str | list[str] = False,
        ts_unit: str = "us",
        tz: str | None = None,
        remove_tz: bool = False,
        delta_subset: str | list[str] | None = None,
        update_metadata: bool = True,
        alter_schema: bool = False,
        timestamp_column: str | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Write data to Parquet dataset with metadata update."""
        # Write data using base class method
        metadata = BaseDataset.write_to_dataset(
            self,
            data=data,
            mode=mode,
            basename=basename,
            partition_by=partition_by,
            max_rows_per_file=max_rows_per_file,
            row_group_size=row_group_size,
            compression=compression,
            sort_by=sort_by,
            unique=unique,
            ts_unit=ts_unit,
            tz=tz,
            remove_tz=remove_tz,
            delta_subset=delta_subset,
            alter_schema=alter_schema,
            timestamp_column=timestamp_column,
            update_metadata=update_metadata,
            verbose=verbose,
            **kwargs,
        )

        # Update metadata if requested
        if update_metadata and metadata:
            for md in metadata:
                metadata_ = list(md.values())[0]
                if self.metadata is not None:
                    if metadata_.schema != self.metadata.schema:
                        continue

                f = list(md.keys())[0].replace(self.path, "").lstrip("/")
                metadata_.set_file_path(f)
                if self._file_metadata is None:
                    self._file_metadata = {f: metadata_}
                else:
                    self._file_metadata[f] = metadata_
            self._update_metadata(verbose=verbose)


class PyarrowDataset(BaseDataset):
    """Dataset class for generic PyArrow datasets."""

    def __init__(
        self,
        path: str,
        name: str | None = None,
        filesystem: AbstractFileSystem | None = None,
        bucket: str | None = None,
        partitioning: str | list[str] | None = None,
        format: str = "parquet",
        cached: bool = False,
        timestamp_column: str | None = None,
        ddb_con: _duckdb.DuckDBPyConnection | None = None,
        **fs_kwargs,
    ):
        super().__init__(
            path=path,
            name=name,
            filesystem=filesystem,
            bucket=bucket,
            partitioning=partitioning,
            format=format,
            cached=cached,
            timestamp_column=timestamp_column,
            ddb_con=ddb_con,
            **fs_kwargs,
        )


class CsvDataset(PyarrowDataset):
    """Dataset class for CSV files."""

    def __init__(
        self,
        path: str,
        name: str | None = None,
        filesystem: AbstractFileSystem | None = None,
        bucket: str | None = None,
        partitioning: str | list[str] | None = None,
        cached: bool = False,
        timestamp_column: str | None = None,
        ddb_con: _duckdb.DuckDBPyConnection | None = None,
        **fs_kwargs,
    ):
        super().__init__(
            path=path,
            name=name,
            filesystem=filesystem,
            bucket=bucket,
            partitioning=partitioning,
            format="csv",
            cached=cached,
            timestamp_column=timestamp_column,
            ddb_con=ddb_con,
            **fs_kwargs,
        )


class JsonDataset(BaseDataset):
    """Dataset class for JSON files."""

    def __init__(
        self,
        path: str,
        name: str | None = None,
        filesystem: AbstractFileSystem | None = None,
        bucket: str | None = None,
        partitioning: str | list[str] | None = None,
        cached: bool = False,
        timestamp_column: str | None = None,
        ddb_con: _duckdb.DuckDBPyConnection | None = None,
        **fs_kwargs,
    ):
        super().__init__(
            path=path,
            name=name,
            filesystem=filesystem,
            bucket=bucket,
            partitioning=partitioning,
            format="json",
            cached=cached,
            timestamp_column=timestamp_column,
            ddb_con=ddb_con,
            **fs_kwargs,
        )

    def load(self) -> None:
        """Load JSON dataset."""
        self._data_processor._arrow_dataset = pds.dataset(
            self._file_manager.filesystem.read_json_dataset(
                self._file_manager.path,
            )
            .opt_dtype(strict=False)
            .to_arrow()
        )
        self._data_processor.table = PydalaTable(
            result=self._data_processor._arrow_dataset,
            ddb_con=self._data_processor.ddb_con,
        )

        self.ddb_con.register(f"{self.name}", self._data_processor._arrow_dataset)

        # Handle timestamp column
        if self._data_processor._timestamp_column is None:
            timestamp_columns = get_timestamp_column(self._data_processor.table.pl.head(10))
            if len(timestamp_columns) > 1:
                self._data_processor._timestamp_column = timestamp_columns[0]

        # Initialize filter
        self._filter = super()._filter.__class__(
            arrow_dataset=self._data_processor._arrow_dataset,
            table=self._data_processor.table,
        )
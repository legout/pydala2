import logging
import posixpath
import typing as t
from fsspec import AbstractFileSystem
import duckdb as _duckdb
import pyarrow as pa
import pyarrow.dataset as pds

from .file_manager import FileManager
from .data_processor import DataProcessor
from .dataset_filter import DatasetFilter
from .dataset_writer import DatasetWriter
from .table import PydalaTable

logger = logging.getLogger(__name__)


class BaseDataset:
    """
    Simplified base dataset class using composition for better separation of concerns.
    """

    def __init__(
        self,
        path: str,
        name: str | None = None,
        schema: pa.Schema | None = None,
        filesystem: AbstractFileSystem | None = None,
        bucket: str | None = None,
        partitioning: str | list[str] | None = None,
        format: str | None = "parquet",
        cached: bool = False,
        timestamp_column: str | None = None,
        ddb_con: _duckdb.DuckDBPyConnection | None = None,
        **fs_kwargs,
    ):
        # Set name
        self.name = name or posixpath.basename(path)

        # Initialize components using composition
        self._file_manager = FileManager(
            path=path,
            filesystem=filesystem,
            bucket=bucket,
            cached=cached,
            format=format or "parquet",
            **fs_kwargs,
        )

        # Infer partitioning if not provided
        if partitioning is None:
            partitioning = self._infer_partitioning()
        elif partitioning == "ignore":
            partitioning = None

        self._data_processor = DataProcessor(
            ddb_con=ddb_con,
            timestamp_column=timestamp_column,
            partitioning=partitioning,
            name=self.name,
        )

        self._writer = DatasetWriter(
            path=path,
            filesystem=self._file_manager.filesystem,
            data_processor=self._data_processor,
            file_manager=self._file_manager,
        )

        # Store schema and try to load
        self._schema = schema
        self._filter = None

        try:
            self.load()
        except FileNotFoundError as e:
            logger.debug(f"Dataset path does not exist yet: {path}")
        except Exception as e:
            logger.warning(f"Failed to load dataset {path}: {e}")

    def _infer_partitioning(self) -> str | list[str] | None:
        """Infer partitioning from existing files."""
        try:
            files = self._file_manager.filesystem.lss(self._file_manager.path)
            if any(["=" in obj for obj in files]):
                return "hive"
        except FileNotFoundError:
            pass
        return None

    def load(self) -> None:
        """Load the dataset from the specified path."""
        if self._file_manager.has_files:
            self._data_processor.load_dataset(
                path=self._file_manager.path,
                filesystem=self._file_manager.filesystem,
                schema=self._schema,
                format=self._file_manager.format,
            )
            # Initialize filter after loading
            if self._data_processor.is_loaded:
                self._filter = DatasetFilter(
                    arrow_dataset=self._data_processor._arrow_dataset,
                    table=self._data_processor.table,
                )

    # Delegated properties to FileManager
    @property
    def files(self) -> list[str]:
        """Get list of files in the dataset."""
        return self._file_manager.files

    @property
    def has_files(self) -> bool:
        """Check if dataset has any files."""
        return self._file_manager.has_files

    @property
    def path(self) -> str:
        """Get the dataset path."""
        return self._file_manager.path

    @property
    def fs(self):
        """Get the filesystem."""
        return self._file_manager.filesystem

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._file_manager.clear_cache()

    def delete_files(self, files: str | list[str]) -> None:
        """Delete files from the dataset."""
        self._file_manager.delete_files(files)

    def vacuum(self) -> None:
        """Delete all files in the dataset."""
        self.delete_files(self.files)

    # Delegated properties to DataProcessor
    @property
    def schema(self) -> pa.Schema:
        """Get dataset schema."""
        return self._data_processor.schema

    @property
    def is_loaded(self) -> bool:
        """Check if dataset is loaded."""
        return self._data_processor.is_loaded

    @property
    def columns(self) -> list[str]:
        """Get column names."""
        return self._data_processor.columns

    @property
    def table(self) -> PydalaTable | None:
        """Get the table object."""
        return self._data_processor.table

    @property
    def t(self) -> PydalaTable | None:
        """Get the table object (shorthand)."""
        return self.table

    def count_rows(self) -> int:
        """Count rows in dataset."""
        return self._data_processor.count_rows()

    @property
    def num_rows(self) -> int:
        """Get number of rows."""
        return self._data_processor.num_rows

    @property
    def num_columns(self) -> int:
        """Get number of columns."""
        return self._data_processor.num_columns

    @property
    def partitioning_schema(self) -> pa.Schema:
        """Get partitioning schema."""
        return self._data_processor.partitioning_schema

    @property
    def partition_names(self) -> list[str]:
        """Get partition column names."""
        return self._data_processor.partition_names

    @property
    def partition_values(self) -> dict:
        """Get partition values."""
        return self._data_processor.partition_values

    @property
    def ddb_con(self) -> _duckdb.DuckDBPyConnection:
        """Get DuckDB connection."""
        return self._data_processor.ddb_con

    # Delegated methods to DatasetFilter
    def filter(
        self,
        filter_expr: str | pds.Expression,
        use: str = "auto",
    ) -> PydalaTable:
        """Filter the dataset."""
        if not self._filter:
            raise RuntimeError("Dataset not loaded")
        return self._filter.filter(filter_expr, use)

    # DuckDB operations
    @property
    def registered_tables(self) -> list[str]:
        """Get list of registered tables in DuckDB."""
        return self.ddb_con.sql("SHOW TABLES").arrow().column("name").to_pylist()

    def interrupt_duckdb(self) -> None:
        """Interrupt DuckDB operations."""
        self.ddb_con.interrupt()

    def reset_duckdb(self) -> None:
        """Reset DuckDB connection and re-register tables."""
        self.interrupt_duckdb()

        if self.name not in self.registered_tables:
            if hasattr(self._data_processor, "_arrow_table"):
                self.ddb_con.register(f"{self.name}", self._data_processor._arrow_table)
            elif hasattr(self._data_processor, "_arrow_dataset"):
                self.ddb_con.register(f"{self.name}", self._data_processor._arrow_dataset)

    # Delegated method to DatasetWriter
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
        alter_schema: bool = False,
        timestamp_column: str | None = None,
        update_metadata: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> list | None:
        """Write data to dataset."""
        return self._writer.write_data(
            data=data,
            mode=mode,
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
            basename=basename,
            verbose=verbose,
            **kwargs,
        )
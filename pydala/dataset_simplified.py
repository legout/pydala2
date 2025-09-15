import datetime as dt
import logging
import posixpath
import re
import tempfile
import typing as t

import duckdb as _duckdb
import pandas as pd
import polars.selectors as cs
import psutil
import pyarrow as pa
import pyarrow.dataset as pds
import tqdm
from fsspec import AbstractFileSystem

from .filesystem import FileSystem, clear_cache
from .helpers.datetime import get_timestamp_column
from .helpers.misc import sql2pyarrow_filter
from .helpers.security import (
    escape_sql_identifier,
    escape_sql_literal,
    validate_partition_name,
    validate_partition_value,
    sanitize_filter_expression,
    validate_path,
    safe_join,
)
from .helpers.polars import pl as _pl
from .io import Writer
from .metadata import ParquetDatasetMetadata, PydalaDatasetMetadata
from .schema import replace_schema
from .schema import convert_large_types_to_normal
from .table import PydalaTable

logger = logging.getLogger(__name__)


class BaseDatasetConfig:
    """Configuration for BaseDataset initialization."""

    def __init__(self, path: str, **kwargs):
        self.path = path
        self.name = kwargs.get('name')
        self.schema = kwargs.get('schema')
        self.filesystem = kwargs.get('filesystem')
        self.bucket = kwargs.get('bucket')
        self.partitioning = kwargs.get('partitioning')
        self.format = kwargs.get('format', 'parquet')
        self.cached = kwargs.get('cached', False)
        self.timestamp_column = kwargs.get('timestamp_column')
        self.ddb_con = kwargs.get('ddb_con')
        self.fs_kwargs = kwargs


class FilesystemManager:
    """Manages filesystem operations and configuration."""

    def __init__(self, config: BaseDatasetConfig):
        self.config = config
        self._setup_cache_storage()
        self._initialize_filesystem()

    def _setup_cache_storage(self):
        """Setup cache storage if caching is enabled."""
        if self.config.cached:
            self.cache_storage = self.config.fs_kwargs.pop(
                "cache_storage", tempfile.mkdtemp(prefix="pydala2_")
            )
        else:
            self.cache_storage = None

    def _initialize_filesystem(self):
        """Initialize the filesystem."""
        self.instance = FileSystem(
            bucket=self.config.bucket,
            fs=self.config.filesystem,
            cached=self.config.cached,
            cache_storage=self.cache_storage,
            **self.config.fs_kwargs,
        )

    def create_directory(self, path: str) -> None:
        """Create directory with fallback approach."""
        if self.instance.exists(path):
            return

        try:
            self.instance.mkdirs(path)
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to create directory {path}: {e}")
            self._try_alternative_creation(path)

    def _try_alternative_creation(self, path: str) -> None:
        """Try alternative approach for directory creation."""
        try:
            self.instance.touch(posixpath.join(path, "tmp.delete"))
            self.instance.rm(posixpath.join(path, "tmp.delete"))
        except Exception as e:
            logger.error(f"Alternative directory creation also failed: {e}")
            raise


class DuckDBConnectionManager:
    """Manages DuckDB connection and configuration."""

    def __init__(self, ddb_con=None):
        self.connection = ddb_con or _duckdb.connect()
        self._configure_connection()

    def _configure_connection(self):
        """Configure DuckDB connection settings."""
        self.connection.execute(
            f"""PRAGMA enable_object_cache;
            SET THREADS={psutil.cpu_count() * 2};"""
        )

    def register_dataset(self, name: str, dataset):
        """Register dataset with DuckDB."""
        self.connection.register(name, dataset)

    def set_timezone(self, timezone: str) -> None:
        """Set timezone for DuckDB connection."""
        self.connection.execute("SET TimeZone=?", [str(timezone)])


class PartitioningManager:
    """Manages dataset partitioning logic."""

    def __init__(self, filesystem, path: str, partitioning: t.Optional[str]):
        self.filesystem = filesystem
        self.path = path
        self.partitioning = partitioning

    def infer_partitioning(self) -> t.Optional[str]:
        """Infer partitioning scheme from existing files."""
        if self.partitioning is not None:
            if self.partitioning == "ignore":
                return None
            return self.partitioning

        # Try to infer partitioning
        try:
            if any(["=" in obj for obj in self.filesystem.lss(self.path)]):
                return "hive"
        except FileNotFoundError:
            pass

        return None


class DatasetLoader:
    """Handles dataset loading operations."""

    def __init__(self, dataset, name: str):
        self.dataset = dataset
        self.name = name

    def load_arrow_dataset(self, schema=None) -> None:
        """Load dataset into Arrow format."""
        if not self.dataset.has_files:
            return

        self.dataset._arrow_dataset = pds.dataset(
            self.dataset._path,
            schema=schema,
            filesystem=self.dataset._filesystem,
            format=self.dataset._format,
            partitioning=self.dataset._partitioning,
        )

        self.dataset.table = PydalaTable(
            result=self.dataset._arrow_dataset,
            ddb_con=self.dataset.ddb_con
        )

    def setup_timestamp_handling(self) -> None:
        """Setup timestamp column detection and timezone handling."""
        if self.dataset._timestamp_column is None:
            self._detect_timestamp_column()

        if self.dataset._timestamp_column is not None:
            self._configure_timezone()
        else:
            self.dataset._tz = None

    def _detect_timestamp_column(self) -> None:
        """Detect timestamp columns in the dataset."""
        self.dataset._timestamp_columns = get_timestamp_column(
            self.dataset.table.pl.head(10)
        )
        if len(self.dataset._timestamp_columns) > 0:
            self.dataset._timestamp_column = self.dataset._timestamp_columns[0]

    def _configure_timezone(self) -> None:
        """Configure timezone based on timestamp column."""
        tz = self.dataset.schema.field(self.dataset._timestamp_column).type.tz
        self.dataset._tz = tz
        if tz is not None:
            self.dataset.ddb_con.execute("SET TimeZone=?", [str(tz)])


class BaseDataset:
    """Simplified BaseDataset with separated concerns."""

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
        # Store configuration
        self._path = path
        self._schema = schema
        self._format = format
        self._timestamp_column = timestamp_column

        # Setup managers
        config = BaseDatasetConfig(path, **locals())
        self.filesystem_manager = FilesystemManager(config)
        self.ddb_manager = DuckDBConnectionManager(ddb_con)
        self.partitioning_manager = PartitioningManager(
            self.filesystem, path, partitioning
        )

        # Initialize properties
        self.table = None
        self._filesystem = self.filesystem_manager.instance
        self.ddb_con = self.ddb_manager.connection

        # Perform initialization tasks
        self._initialize_dataset(name, timestamp_column)
        self._attempt_initial_load()

    def _initialize_dataset(self, name: str | None, timestamp_column: str | None):
        """Initialize dataset properties."""
        self.name = name or posixpath.basename(self._path)
        self._timestamp_column = timestamp_column
        self._partitioning = self.partitioning_manager.infer_partitioning()

        # Create directory structure
        self.filesystem_manager.create_directory(self._path)

    def _attempt_initial_load(self):
        """Attempt to load the dataset initially."""
        try:
            self.load()
        except FileNotFoundError:
            logger.debug(f"Dataset path does not exist yet: {self._path}")
        except Exception as e:
            logger.warning(f"Failed to load dataset {self._path}: {e}")

    @property
    def filesystem(self):
        """Get the filesystem instance."""
        return self._filesystem

    @property
    def has_files(self):
        """Check if dataset has files."""
        return len(self.files) > 0

    @property
    def is_loaded(self):
        """Check if dataset is loaded."""
        return hasattr(self, "_arrow_dataset")

    def load_files(self):
        """Load file list from path."""
        self.clear_cache()
        glob_pattern = safe_join(self._path, f"**/*.{self._format}")
        self._files = [
            fn.replace(self._path, "").lstrip("/")
            for fn in sorted(self._filesystem.glob(glob_pattern))
        ]

    def load(self):
        """Load the dataset."""
        if self.has_files:
            loader = DatasetLoader(self, self.name)
            loader.load_arrow_dataset(self._schema)
            loader.setup_timestamp_handling()
            self.ddb_manager.register_dataset(self.name, self._arrow_dataset)

    def clear_cache(self):
        """Clear filesystem cache."""
        if hasattr(self._filesystem, "fs"):
            clear_cache(self._filesystem.fs)
        clear_cache(self._filesystem)
        if hasattr(self, '_base_filesystem'):
            clear_cache(self._base_filesystem)

    @property
    def files(self):
        """Get list of files in dataset."""
        if not hasattr(self, "_files"):
            self.load_files()
        return self._files

    @property
    def schema(self):
        """Get dataset schema."""
        if self.is_loaded:
            if not hasattr(self, "_schema") or self._schema is None:
                self._schema = self._arrow_dataset.schema
            return self._schema

    @property
    def columns(self):
        """Get column names."""
        if self.is_loaded:
            return self.schema.names


# Example of simplified filter method
class DatasetFilter:
    """Handles dataset filtering operations."""

    def __init__(self, dataset):
        self.dataset = dataset

    def filter(self, filter_expr: str | pds.Expression, use: str = "auto"):
        """Filter dataset with simplified logic."""
        # Simple check for DuckDB-specific operations
        if self._requires_duckdb(filter_expr):
            use = "duckdb"

        # Use appropriate filter method
        if use == "auto":
            return self._auto_filter(filter_expr)
        elif use == "pyarrow":
            return self._filter_pyarrow(filter_expr)
        elif use == "duckdb":
            return self._filter_duckdb(filter_expr)

    def _requires_duckdb(self, filter_expr) -> bool:
        """Check if filter expression requires DuckDB."""
        duckdb_patterns = ["%", "like", "similar to", "*", "(", ")"]
        return any(pattern in filter_expr for pattern in duckdb_patterns)

    def _auto_filter(self, filter_expr):
        """Automatically choose best filtering method."""
        try:
            return self._filter_pyarrow(filter_expr)
        except Exception as e:
            logger.info("PyArrow filtering failed, falling back to DuckDB")
            return self._filter_duckdb(filter_expr)

    def _filter_pyarrow(self, filter_expr: str | pds.Expression):
        """Filter using PyArrow."""
        if isinstance(filter_expr, str):
            filter_expr = sql2pyarrow_filter(filter_expr, self.dataset.schema)
        result = self.dataset._arrow_dataset.filter(filter_expr)
        return PydalaTable(result=result, ddb_con=self.dataset.ddb_con)

    def _filter_duckdb(self, filter_expr: str):
        """Filter using DuckDB."""
        result = self.dataset.table.ddb.filter(filter_expr)
        return PydalaTable(result=result, ddb_con=self.dataset.ddb_con)
"""
Simplified dataset module with clean architecture and separated concerns.
"""

import datetime as dt
import logging
import posixpath
import re
import tempfile
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict, Any

import duckdb
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as pds
from fsspec import AbstractFileSystem

from ..filesystem import FileSystem
from ..helpers.security import safe_join
from .config import DatasetConfig
from .filesystem_manager import FilesystemManager
from .db_manager import DatabaseManager
from .partitioning import PartitioningManager
from .loader import DatasetLoader
from .writer import DatasetWriter
from ..table import PydalaTable

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """Abstract base class for datasets with clean separation of concerns."""

    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        schema: Optional[pa.Schema] = None,
        filesystem: Optional[AbstractFileSystem] = None,
        bucket: Optional[str] = None,
        partitioning: Optional[Union[str, List[str]]] = None,
        format: str = "parquet",
        cached: bool = False,
        timestamp_column: Optional[str] = None,
        ddb_con: Optional[duckdb.DuckDBPyConnection] = None,
        **fs_kwargs,
    ):
        # Create configuration object
        self.config = DatasetConfig(
            path=path,
            name=name,
            schema=schema,
            filesystem=filesystem,
            bucket=bucket,
            partitioning=partitioning,
            format=format,
            cached=cached,
            timestamp_column=timestamp_column,
            ddb_con=ddb_con,
            **fs_kwargs
        )

        # Initialize managers
        self.filesystem_manager = FilesystemManager(self.config)
        self.db_manager = DatabaseManager(self.config)
        self.partitioning_manager = PartitioningManager(self.config, self.filesystem_manager)
        self.loader = DatasetLoader(self, self.config)
        self.writer = DatasetWriter(self, self.config)

        # Initialize dataset state
        self._initialize_state()

    def _initialize_state(self):
        """Initialize dataset state variables."""
        self.table = None
        self._files = None
        self._schema = self.config.schema
        self._tz = None

    @property
    def filesystem(self) -> FileSystem:
        """Get filesystem manager."""
        return self.filesystem_manager.filesystem

    @property
    def files(self) -> List[str]:
        """Get list of files in the dataset."""
        if self._files is None:
            self._files = self.filesystem_manager.load_files(
                self.config.path,
                self.config.format
            )
        return self._files

    @property
    def has_files(self) -> bool:
        """Check if dataset has files."""
        return len(self.files) > 0

    @property
    def is_loaded(self) -> bool:
        """Check if dataset is loaded."""
        return hasattr(self, "_arrow_dataset")

    @property
    def name(self) -> str:
        """Get dataset name."""
        return self.config.name

    @property
    def schema(self) -> Optional[pa.Schema]:
        """Get dataset schema."""
        if self.is_loaded and (self._schema is None):
            self._schema = self._arrow_dataset.schema
        return self._schema

    @property
    def columns(self) -> Optional[List[str]]:
        """Get column names."""
        if self.is_loaded:
            return list(self.schema.names) if self.schema else []
        return None

    def load(self) -> None:
        """Load the dataset."""
        if self.has_files:
            self.loader.load()

    def unload(self) -> None:
        """Unload the dataset from memory."""
        if hasattr(self, "_arrow_dataset"):
            delattr(self, "_arrow_dataset")
        self.table = None

    def reload(self) -> None:
        """Reload the dataset."""
        self.unload()
        self.load()

    def write(
        self,
        data: Union[pl.DataFrame, pl.LazyFrame, pa.Table, pd.DataFrame],
        mode: str = "append",
        **kwargs
    ) -> None:
        """Write data to the dataset."""
        self.writer.write(data, mode=mode, **kwargs)

    def filter(self, expression: str) -> "BaseDataset":
        """Filter the dataset."""
        from .filtering import DatasetFilter
        filter_engine = DatasetFilter(self)
        return filter_engine.filter(expression)

    def __len__(self) -> int:
        """Get number of rows."""
        if self.is_loaded:
            return self._arrow_dataset.count_rows()
        return 0

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(path='{self.config.path}', files={len(self.files)})"


class ParquetDataset(BaseDataset):
    """Parquet dataset implementation."""

    def __init__(self, **kwargs):
        kwargs["format"] = "parquet"
        super().__init__(**kwargs)

    def optimize(self, **kwargs) -> None:
        """Optimize the dataset."""
        from .optimization import DatasetOptimizer
        optimizer = DatasetOptimizer(self)
        optimizer.optimize(**kwargs)

    def repartition(self, partitioning_columns: List[str], **kwargs) -> None:
        """Repartition the dataset."""
        from .partitioning import Repartitioner
        repartitioner = Repartitioner(self)
        repartitioner.repartition(partitioning_columns, **kwargs)


class JsonDataset(BaseDataset):
    """JSON dataset implementation."""

    def __init__(self, **kwargs):
        kwargs["format"] = "json"
        super().__init__(**kwargs)


class CsvDataset(BaseDataset):
    """CSV dataset implementation."""

    def __init__(self, **kwargs):
        kwargs["format"] = "csv"
        super().__init__(**kwargs)
"""
Dataset loading functionality for simplified dataset module.
"""

import logging
from typing import Optional

import pyarrow as pa
import pyarrow.dataset as pds

from ..helpers.datetime import get_timestamp_column
from ..table import PydalaTable

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Handles loading datasets into memory."""

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

    def load(self) -> None:
        """Load the dataset into memory."""
        if not self.dataset.has_files:
            logger.debug(f"No files to load for dataset at {self.config.path}")
            return

        logger.info(f"Loading dataset from {self.config.path}")

        # Load the Arrow dataset
        self._load_arrow_dataset()

        # Create table wrapper
        self._create_table()

        # Setup timestamp handling
        self._setup_timestamp_column()

        # Register with DuckDB
        self._register_with_duckdb()

        logger.info(f"Successfully loaded dataset with {len(self.dataset.files)} files")

    def _load_arrow_dataset(self) -> None:
        """Load the PyArrow dataset."""
        try:
            self.dataset._arrow_dataset = pds.dataset(
                self.config.path,
                schema=self.config.schema,
                filesystem=self.dataset.filesystem,
                format=self.config.format,
                partitioning=self.dataset._partitioning,
            )
            logger.debug("Loaded PyArrow dataset")
        except Exception as e:
            logger.error(f"Failed to load PyArrow dataset: {e}")
            raise

    def _create_table(self) -> None:
        """Create PydalaTable wrapper."""
        try:
            self.dataset.table = PydalaTable(
                result=self.dataset._arrow_dataset,
                ddb_con=self.dataset.db_manager.connection
            )
            logger.debug("Created PydalaTable wrapper")
        except Exception as e:
            logger.error(f"Failed to create PydalaTable: {e}")
            raise

    def _setup_timestamp_column(self) -> None:
        """Setup timestamp column detection and timezone."""
        if self.config.timestamp_column is None:
            self._detect_timestamp_column()

        if self.dataset._timestamp_column is not None:
            self._configure_timezone()

    def _detect_timestamp_column(self) -> None:
        """Detect timestamp columns from schema or sample data."""
        try:
            # Get schema
            schema = self.dataset.schema
            if not schema:
                return

            # Check for timestamp columns in schema
            timestamp_fields = [
                field.name for field in schema
                if pa.types.is_timestamp(field.type)
            ]

            if timestamp_fields:
                self.dataset._timestamp_column = timestamp_fields[0]
                logger.debug(f"Detected timestamp column from schema: {self.dataset._timestamp_column}")
                return

            # Try to detect from sample data
            sample = self.dataset.table.pl.head(10)
            timestamp_columns = get_timestamp_column(sample)
            if timestamp_columns:
                self.dataset._timestamp_column = timestamp_columns[0]
                logger.debug(f"Detected timestamp column from sample data: {self.dataset._timestamp_column}")
            else:
                logger.debug("No timestamp column detected")

        except Exception as e:
            logger.warning(f"Failed to detect timestamp column: {e}")

    def _configure_timezone(self) -> None:
        """Configure timezone from timestamp column."""
        try:
            field = self.dataset.schema.field(self.dataset._timestamp_column)
            if hasattr(field.type, 'tz'):
                tz = field.type.tz
                self.dataset._tz = tz
                if tz is not None:
                    self.dataset.db_manager.set_timezone(tz)
                    logger.debug(f"Set timezone to {tz}")
            else:
                self.dataset._tz = None
        except Exception as e:
            logger.warning(f"Failed to configure timezone: {e}")

    def _register_with_duckdb(self) -> None:
        """Register the dataset with DuckDB."""
        try:
            self.dataset.db_manager.register_dataset(
                self.config.name,
                self.dataset._arrow_dataset
            )
            logger.debug(f"Registered dataset '{self.config.name}' with DuckDB")
        except Exception as e:
            logger.error(f"Failed to register dataset with DuckDB: {e}")
            raise
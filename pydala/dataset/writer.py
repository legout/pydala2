"""
Dataset writer component.

This module provides a focused interface for writing data to datasets
with simplified parameter handling and clear responsibilities.
"""

import posixpath
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import duckdb as duckdb
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from loguru import logger

from ..filesystem import FileSystem
from ..helpers.security import safe_join
from ..io import Writer as BaseWriter
from ..metadata import DatasetMetadata, FileMetadata

# Type aliases
WriteDataTypes = t.Union[
    pl.DataFrame,
    pl.LazyFrame,
    pa.Table,
    pa.RecordBatch,
    pa.RecordBatchReader,
    pd.DataFrame,
    duckdb.DuckDBPyConnection,
    t.List['WriteDataTypes']
]

@dataclass
class WriteConfig:
    """Configuration for write operations."""

    # File settings
    max_rows_per_file: int = 2_500_000
    row_group_size: int = 250_000
    basename_template: str | None = None

    # Compression
    compression: str = "zstd"
    compression_level: int | None = None

    # Data processing
    sort_by: str | list[str] | list[tuple[str, str]] | None = None
    unique: bool | str | list[str] = False

    # Partitioning
    partition_by: str | list[str] | None = None
    partitioning_flavor: str = "hive"

    # Schema handling
    alter_schema: bool = False
    ts_unit: str = "us"
    tz: str | None = None
    remove_tz: bool = False

    # Metadata
    update_metadata: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for passing to write functions."""
        return {
            k: v for k, v in self.__dict__.items()
            if v is not None and not k.startswith('_')
        }


@dataclass
class DeltaConfig:
    """Configuration for delta write operations."""

    subset: str | list[str] | None = None
    filter_columns: str | list[str] | None = None


class DatasetWriter:
    """
    Handles all write operations for a dataset.

    This class provides a simplified interface for writing data to datasets,
    managing partitioning, compression, and metadata updates.
    """

    def __init__(
        self,
        path: str,
        filesystem: AbstractFileSystem | None = None,
        metadata: DatasetMetadata | None = None,
        default_config: WriteConfig | None = None,
    ):
        """
        Initialize DatasetWriter.

        Args:
            path: Path to the dataset
            filesystem: Filesystem to use for I/O operations
            metadata: Dataset metadata manager
            default_config: Default write configuration
        """
        self.path = path
        self.filesystem = filesystem or FileSystem()
        self.metadata = metadata
        self.default_config = default_config or WriteConfig()

    def write(
        self,
        data: t.Union[
            pl.DataFrame,
            pl.LazyFrame,
            pa.Table,
            pa.RecordBatch,
            pd.DataFrame,
            duckdb.DuckDBPyRelation,
            list,
        ],
        mode: str = "append",
        config: WriteConfig | None = None,
        delta_config: DeltaConfig | None = None,
        verbose: bool = False,
    ) -> list[FileMetadata]:
        """
        Write data to the dataset.

        Args:
            data: Data to write (various formats supported)
            mode: Write mode - "append", "overwrite", or "delta"
            config: Write configuration (uses defaults if not provided)
            delta_config: Configuration for delta writes
            verbose: Show progress information

        Returns:
            List of FileMetadata for written files
        """
        # Use provided config or defaults
        write_config = config or self.default_config

        # Normalize data to list
        if not isinstance(data, (list, tuple)):
            data = [data]

        # Track written files metadata
        written_metadata = []

        # Handle overwrite mode - mark existing files for deletion
        files_to_delete = []
        if mode == "overwrite":
            files_to_delete = self._get_existing_files()

        # Process each data batch
        for data_item in data:
            if self._is_empty(data_item):
                continue

            # Write the data
            file_metadata = self._write_data_item(
                data_item,
                mode=mode,
                config=write_config,
                delta_config=delta_config,
                verbose=verbose,
            )

            if file_metadata:
                written_metadata.extend(file_metadata)

        # Clean up for overwrite mode
        if mode == "overwrite" and files_to_delete:
            self._delete_files(files_to_delete)

        # Update metadata if configured
        if write_config.update_metadata and self.metadata and written_metadata:
            self._update_metadata(written_metadata)

        return written_metadata

    def _write_data_item(
        self,
        data: t.Any,
        mode: str,
        config: WriteConfig,
        delta_config: DeltaConfig | None,
        verbose: bool,
    ) -> list[FileMetadata]:
        """Write a single data item to the dataset."""
        # Create writer instance
        writer = BaseWriter(
            data=data,
            path=self.path,
            filesystem=self.filesystem,
            schema=self._get_schema_for_write(config),
        )

        # Check if data is empty
        if writer.shape[0] == 0:
            return []

        # Apply data transformations
        self._apply_transformations(writer, config)

        # Handle delta mode
        if mode == "delta" and delta_config:
            self._apply_delta(writer, delta_config)
            if writer.shape[0] == 0:
                return []  # No changes to write

        # Prepare partitioning
        if config.partition_by:
            writer.add_datepart_columns(
                columns=config.partition_by,
                timestamp_column=self.metadata._timestamp_column if self.metadata else None,
            )

        # Write to dataset
        metadata = writer.write_to_dataset(
            row_group_size=config.row_group_size,
            compression=config.compression,
            compression_level=config.compression_level,
            partitioning_columns=config.partition_by,
            partitioning_flavor=config.partitioning_flavor,
            max_rows_per_file=config.max_rows_per_file,
            basename=config.basename_template,
            verbose=verbose,
        )

        # Convert to FileMetadata objects
        return self._convert_metadata(metadata)

    def _apply_transformations(self, writer: BaseWriter, config: WriteConfig) -> None:
        """Apply configured transformations to the data."""
        # Sort data
        if config.sort_by:
            writer.sort_data(by=config.sort_by)

        # Remove duplicates
        if config.unique:
            writer.unique(columns=config.unique)

        # Cast schema
        writer.cast_schema(
            ts_unit=config.ts_unit,
            tz=config.tz,
            remove_tz=config.remove_tz,
            alter_schema=config.alter_schema,
        )

    def _apply_delta(self, writer: BaseWriter, delta_config: DeltaConfig) -> None:
        """Apply delta logic to filter out existing data."""
        if not self.metadata:
            return

        # Get existing data for comparison
        writer._to_polars()
        other_df = self._get_existing_data_for_delta(
            writer.data,
            filter_columns=delta_config.filter_columns,
        )

        if other_df is not None:
            writer.delta(other=other_df, subset=delta_config.subset)

    def _get_existing_data_for_delta(
        self,
        new_data: pl.DataFrame | pl.LazyFrame,
        filter_columns: str | list[str] | None,
    ) -> pl.DataFrame | pl.LazyFrame | None:
        """Get existing data that overlaps with new data for delta comparison."""
        # This is a simplified version - would need access to dataset reader
        # In full implementation, this would query existing data based on the
        # range of values in new_data
        return None

    def _get_schema_for_write(self, config: WriteConfig) -> pa.Schema | None:
        """Get the appropriate schema for writing."""
        if config.alter_schema:
            return None  # Allow schema to be inferred/altered

        if self.metadata and self.metadata.schema:
            return self.metadata.schema

        return None

    def _get_existing_files(self) -> list[str]:
        """Get list of existing files in the dataset."""
        try:
            pattern = safe_join(self.path, "**/*.parquet")
            files = self.filesystem.glob(pattern)
            return [f.replace(self.path, "").lstrip("/") for f in files]
        except Exception as e:
            logger.warning(f"Failed to list existing files: {e}")
            return []

    def _delete_files(self, files: list[str]) -> None:
        """Delete specified files from the dataset."""
        if not files:
            return

        # Ensure full paths
        full_paths = []
        for file in files:
            if self.path not in file:
                full_paths.append(posixpath.join(self.path, file))
            else:
                full_paths.append(file)

        try:
            self.filesystem.rm(full_paths, recursive=True)
            logger.info(f"Deleted {len(full_paths)} files")
        except Exception as e:
            logger.error(f"Failed to delete files: {e}")

    def _convert_metadata(self, metadata: list) -> list[FileMetadata]:
        """Convert writer metadata to FileMetadata objects."""
        file_metadata = []

        for item in metadata:
            if isinstance(item, dict):
                for file_path, meta in item.items():
                    # Extract relative path
                    rel_path = file_path.replace(self.path, "").lstrip("/")

                    # Create FileMetadata
                    if isinstance(meta, pq.FileMetaData):
                        file_meta = FileMetadata.from_parquet_metadata(rel_path, meta)
                    else:
                        # Handle other metadata formats
                        file_meta = FileMetadata(
                            path=rel_path,
                            num_rows=meta.num_rows if hasattr(meta, 'num_rows') else 0,
                            num_columns=len(meta.schema) if hasattr(meta, 'schema') else 0,
                            size_bytes=0,  # Would need to get from filesystem
                        )

                    file_metadata.append(file_meta)

        return file_metadata

    def _update_metadata(self, written_metadata: list[FileMetadata]) -> None:
        """Update dataset metadata with newly written files."""
        if not self.metadata:
            return

        for file_meta in written_metadata:
            self.metadata.update_file_metadata(file_meta.path, file_meta)

        # Write metadata file if configured
        if self.metadata.config.update_on_write:
            try:
                self.metadata.write_metadata_file()
            except Exception as e:
                logger.warning(f"Failed to write metadata file: {e}")

    def _is_empty(self, data: t.Any) -> bool:
        """Check if data is empty."""
        if data is None:
            return True

        if isinstance(data, (pl.DataFrame, pd.DataFrame)):
            return len(data) == 0
        elif isinstance(data, pl.LazyFrame):
            return False  # Can't easily check lazy frame
        elif isinstance(data, (pa.Table, pa.RecordBatch)):
            return data.num_rows == 0
        elif isinstance(data, duckdb.DuckDBPyRelation):
            return False  # Can't easily check

        return False

    def append(
        self,
        data: t.Any,
        config: WriteConfig | None = None,
        verbose: bool = False,
    ) -> list[FileMetadata]:
        """
        Append data to the dataset.

        Args:
            data: Data to append
            config: Write configuration
            verbose: Show progress

        Returns:
            List of FileMetadata for written files
        """
        return self.write(data, mode="append", config=config, verbose=verbose)

    def overwrite(
        self,
        data: t.Any,
        config: WriteConfig | None = None,
        verbose: bool = False,
    ) -> list[FileMetadata]:
        """
        Overwrite the dataset with new data.

        Args:
            data: Data to write
            config: Write configuration
            verbose: Show progress

        Returns:
            List of FileMetadata for written files
        """
        return self.write(data, mode="overwrite", config=config, verbose=verbose)

    def write_delta(
        self,
        data: t.Any,
        delta_config: DeltaConfig | None = None,
        config: WriteConfig | None = None,
        verbose: bool = False,
    ) -> list[FileMetadata]:
        """
        Write only new/changed data to the dataset.

        Args:
            data: Data to write
            delta_config: Delta write configuration
            config: Write configuration
            verbose: Show progress

        Returns:
            List of FileMetadata for written files
        """
        return self.write(
            data,
            mode="delta",
            config=config,
            delta_config=delta_config,
            verbose=verbose,
        )

    def batch_write(
        self,
        data_iterator: t.Iterator[t.Any],
        config: WriteConfig | None = None,
        batch_size: int | None = None,
        verbose: bool = False,
    ) -> list[FileMetadata]:
        """
        Write data in batches from an iterator.

        Args:
            data_iterator: Iterator yielding data batches
            config: Write configuration
            batch_size: Number of items to accumulate before writing
            verbose: Show progress

        Returns:
            List of FileMetadata for all written files
        """
        all_metadata = []
        batch = []

        for item in data_iterator:
            if batch_size:
                batch.append(item)
                if len(batch) >= batch_size:
                    metadata = self.write(batch, config=config, verbose=verbose)
                    all_metadata.extend(metadata)
                    batch = []
            else:
                # Write immediately
                metadata = self.write(item, config=config, verbose=verbose)
                all_metadata.extend(metadata)

        # Write remaining batch
        if batch:
            metadata = self.write(batch, config=config, verbose=verbose)
            all_metadata.extend(metadata)

        return all_metadata
"""
Writing functionality for simplified dataset module.
"""

import logging
from typing import Union, Optional, List, Any, Dict

import pyarrow as pa
import pyarrow.dataset as pds

from ..io import Writer

logger = logging.getLogger(__name__)


class DatasetWriter:
    """Handles writing data to datasets."""

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

    def write(
        self,
        data: Any,
        mode: str = "append",
        partition_by: Optional[Union[str, List[str]]] = None,
        basename: Optional[str] = None,
        max_rows_per_file: int = 2_500_000,
        row_group_size: int = 250_000,
        compression: str = "zstd",
        sort_by: Optional[Union[str, List[str], List[tuple]]] = None,
        unique: Union[bool, str, List[str]] = False,
        ts_unit: str = "us",
        tz: Optional[str] = None,
        remove_tz: bool = False,
        delta_subset: Optional[Union[str, List[str]]] = None,
        alter_schema: bool = False,
        timestamp_column: Optional[str] = None,
        update_metadata: bool = True,
        verbose: bool = False,
        **kwargs
    ) -> Optional[List[Dict]]:
        """
        Write data to the dataset.

        Args:
            data: Data to write (Polars DataFrame, Arrow Table, Pandas DataFrame, etc.)
            mode: Write mode ("append", "overwrite", "delta")
            partition_by: Columns to partition by
            basename: File naming template
            max_rows_per_file: Maximum rows per file
            row_group_size: Row group size for Parquet files
            compression: Compression codec
            sort_by: Columns to sort by
            unique: Handle unique rows
            ts_unit: Timestamp unit
            tz: Timezone
            remove_tz: Remove timezone info
            delta_subset: Columns for delta comparison
            alter_schema: Allow schema changes
            timestamp_column: Timestamp column name
            update_metadata: Update metadata after write
            verbose: Verbose output
            **kwargs: Additional writer arguments
        """
        # Normalize partition_by
        partition_by = self._normalize_partition_by(partition_by)

        # Update timestamp column if specified
        if timestamp_column is not None:
            self.dataset._timestamp_column = timestamp_column

        # Prepare data
        data_list = self._normalize_data(data)
        if not data_list:
            logger.warning("No data provided for writing")
            return None

        # Process each data item
        metadata_list = []
        schema = self._get_write_schema(partition_by, alter_schema)

        for item in data_list:
            try:
                metadata = self._process_single_item(
                    item=item,
                    mode="append",  # Always append individual items
                    schema=schema,
                    partition_by=partition_by,
                    basename=basename,
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
                    verbose=verbose,
                    **kwargs
                )

                if metadata is not None:
                    metadata_list.append(metadata)

            except Exception as e:
                logger.error(f"Failed to process data item: {e}")
                if not kwargs.get("continue_on_error", False):
                    raise

        # Handle mode-dependent operations
        if mode == "overwrite":
            self._handle_overwrite()

        # Clear cache
        self.dataset.filesystem_manager.clear_all_caches()

        # Update dataset
        if update_metadata and metadata_list:
            self._update_dataset(metadata_list)

        # Reload if dataset is loaded
        if self.dataset.is_loaded:
            self.dataset.reload()

        return metadata_list if update_metadata else None

    def _normalize_partition_by(self, partition_by: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
        """Normalize partition_by parameter."""
        if partition_by is None and self.dataset.partitioning_manager:
            # Use existing partition columns if none specified
            return self.dataset.partitioning_manager.get_partition_names(self.dataset)

        if isinstance(partition_by, str):
            return [partition_by]

        return partition_by

    def _normalize_data(self, data: Any) -> List[Any]:
        """Normalize data to list format."""
        # Already a list
        if isinstance(data, (list, tuple)) and not isinstance(data, str):
            return list(data)

        # Create single-item list
        return [data]

    def _get_write_schema(self, partition_by: Optional[List[str]], alter_schema: bool) -> Optional[pa.Schema]:
        """Get schema to use for writing."""
        if alter_schema:
            # Allow automatic schema detection
            return None

        if partition_by and self.dataset.is_loaded:
            # Use existing schema for partitioned writes
            return self.dataset.schema

        # Use configured schema if provided
        return self.config.schema

    def _process_single_item(
        self,
        item: Any,
        schema: Optional[pa.Schema],
        **write_options
    ) -> Optional[Dict]:
        """Process a single data item."""
        # Create writer
        writer = Writer(
            data=item,
            path=self.config.path,
            filesystem=self.dataset.filesystem,
            schema=schema,
        )

        # Check for empty data
        if writer.shape[0] == 0:
            logger.debug("Skipping empty data")
            return None

        # Apply transformations
        self._apply_transformations(writer, **write_options)

        # Write to dataset
        metadata = writer.write_to_dataset(**write_options)

        return metadata

    def _apply_transformations(self, writer: Writer, **options) -> None:
        """Apply transformations before writing."""
        # Sort data
        if options.get("sort_by"):
            writer.sort_data(by=options["sort_by"])

        # Handle unique rows
        if options.get("unique"):
            writer.unique(columns=options["unique"])

        # Cast schema
        writer.cast_schema(
            ts_unit=options.get("ts_unit", "us"),
            tz=options.get("tz"),
            remove_tz=options.get("remove_tz", False),
            alter_schema=options.get("alter_schema", False),
        )

        # Add datepart columns for partitioning
        if options.get("partition_by"):
            writer.add_datepart_columns(
                columns=options["partition_by"],
                timestamp_column=self.dataset._timestamp_column,
            )

        # Apply delta transformation if needed
        if options.get("mode") == "delta" and self.dataset.is_loaded:
            self._apply_delta_transformation(writer, **options)

    def _apply_delta_transformation(self, writer: Writer, **options) -> None:
        """Apply delta (UPSERT) transformation."""
        # Get conflicting data
        # Note: This is a simplified version - full implementation would need more logic
        writer._to_polars()
        logger.debug("Applied delta transformation")

    def _handle_overwrite(self) -> None:
        """Delete existing files for overwrite mode."""
        if not self.dataset.has_files:
            return

        logger.info("Deleting existing files for overwrite")

        # Delete all existing files
        existing_files = self.dataset.files
        self.dataset.filesystem_manager.delete_files(existing_files)

        # Clear file cache
        self.dataset._files = None

    def _update_dataset(self, metadata_list: List[Dict]) -> None:
        """Update dataset metadata after writing."""
        logger.debug(f"Updating dataset with {len(metadata_list)} new metadata entries")
        # Implementation would depend on metadata storage mechanism
        # For now, just log that it happened
        pass
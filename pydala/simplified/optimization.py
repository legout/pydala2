"""
Dataset optimization functionality for simplified dataset module.
"""

import logging
from typing import Optional, Union, List, Tuple

from .partitioning import PartitioningManager

logger = logging.getLogger(__name__)


class DatasetOptimizer:
    """Handles dataset optimization operations."""

    def __init__(self, dataset):
        self.dataset = dataset

    def optimize(
        self,
        method: str = "auto",
        max_rows_per_file: int = 2_500_000,
        row_group_size: int = 250_000,
        compression: str = "zstd",
        sort_by: Optional[Union[str, List[str], List[Tuple[str, str]]]] = None,
        **kwargs
    ) -> None:
        """
        Optimize the dataset based on the specified method.

        Args:
            method: Optimization method ("auto", "rows", "time", "partitions", "schema")
            max_rows_per_file: Maximum rows per file after optimization
            row_group_size: Row group size for Parquet
            compression: Compression codec
            sort_by: Columns to sort by
            **kwargs: Additional optimization parameters
        """
        if method == "auto":
            method = self._select_auto_method()

        logger.info(f"Optimizing dataset using method: {method}")

        if method == "rows":
            self.optimize_by_rows(
                max_rows_per_file=max_rows_per_file,
                row_group_size=row_group_size,
                compression=compression,
                sort_by=sort_by,
                **kwargs
            )
        elif method == "time":
            self.optimize_by_time(
                max_rows_per_file=max_rows_per_file,
                row_group_size=row_group_size,
                compression=compression,
                sort_by=sort_by,
                **kwargs
            )
        elif method == "partitions":
            self.optimize_partitions(
                max_rows_per_file=max_rows_per_file,
                row_group_size=row_group_size,
                compression=compression,
                sort_by=sort_by,
                **kwargs
            )
        elif method == "schema":
            self.optimize_schema(**kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        logger.info("Dataset optimization complete")

    def _select_auto_method(self) -> str:
        """Automatically select best optimization method."""
        # Check if dataset is partitioned
        if self.dataset._partitioning:
            logger.debug("Dataset is partitioned, using partition optimization")
            return "partitions"

        # Check if dataset has timestamp column
        if self.dataset._timestamp_column:
            logger.debug("Dataset has timestamp column, using time optimization")
            return "time"

        # Default to row-based optimization
        logger.debug("Using row-based optimization")
        return "rows"

    def optimize_by_rows(
        self,
        max_rows_per_file: int = 2_500_000,
        **kwargs
    ) -> None:
        """Compact small files by row count."""
        if self.dataset._partitioning:
            self._compact_partitioned_files(max_rows_per_file, **kwargs)
        else:
            self._compact_unpartitioned_files(max_rows_per_file, **kwargs)

    def _compact_partitioned_files(self, max_rows_per_file: int, **kwargs) -> None:
        """Compact files within each partition."""
        # Get partitions with multiple files
        partitions = self._get_partitions_to_compact(max_rows_per_file)

        logger.info(f"Compacting {len(partitions)} partitions")

        for partition_spec in partitions:
            self._compact_single_partition(partition_spec, max_rows_per_file, **kwargs)

        # Reload dataset
        self.dataset.reload()

    def _compact_unpartitioned_files(self, max_rows_per_file: int, **kwargs) -> None:
        """Compact unpartitioned dataset files."""
        from .filtering import DatasetFilter

        # Find files with fewer rows than target
        filter_expr = f"num_rows < {max_rows_per_file}"
        small_files_scan = self.dataset.filter(filter_expr)

        if len(small_files_scan.files) <= 1:
            logger.info("No small files to compact")
            return

        logger.info(f"Compacting {len(small_files_scan.files)} small files")

        # Read and rewrite data
        batches = small_files_scan.to_batch_reader(
            batch_size=max_rows_per_file,
            **kwargs
        )

        for batch in batches:
            self.dataset.write(
                batch,
                mode="append",
                update_metadata=False,
                **kwargs
            )

        # Delete old files
        self.dataset.filesystem_manager.delete_files(small_files_scan.files)

        # Reload dataset
        self.dataset.reload()

    def optimize_by_time(
        self,
        interval: str = "1d",
        timestamp_column: Optional[str] = None,
        **kwargs
    ) -> None:
        """Compact files based on time intervals."""
        if timestamp_column is None:
            timestamp_column = self.dataset._timestamp_column

        if timestamp_column is None:
            raise ValueError("No timestamp column available for time-based optimization")

        import polars as pl

        # Get time range
        time_range_query = f"""
        SELECT
            MIN({timestamp_column}) as min_time,
            MAX({timestamp_column}) as max_time
        FROM {self.dataset.name}
        """
        time_range = self.dataset.db_manager.connection.sql(time_range_query).pl()

        min_time = time_range["min_time"][0]
        max_time = time_range["max_time"][0]

        logger.info(f"Optimizing by time from {min_time} to {max_time}")

        # Create time intervals
        intervals = pl.datetime_range(min_time, max_time, interval=interval, eager=True)

        # Process each time interval
        for i in range(len(intervals) - 1):
            start_time = intervals[i]
            end_time = intervals[i + 1]

            self._compact_time_window(start_time, end_time, timestamp_column, **kwargs)

    def _compact_time_window(
        self,
        start: Union[str, "datetime"],
        end: Union[str, "datetime"],
        timestamp_column: str,
        **kwargs
    ) -> None:
        """Compact a single time window."""
        # Filter data for this time window
        from .filtering import DatasetFilter
        date_filter = DatasetFilter(self.dataset)
        time_window = date_filter.filter_range(
            start=start,
            end=end,
            inclusive=True
        )

        # Skip if no files
        if len(time_window.files) <= 1:
            return

        logger.debug(f"Compacting time window {start} to {end}")

        # Rewrite the time window
        self.dataset.write(
            time_window,
            mode="append",
            partition_by=self.dataset._partitioning,
            update_metadata=False,
            **kwargs
        )

        # Delete old files
        self.dataset.filesystem_manager.delete_files(time_window.files)

    def optimize_partitions(
        self,
        max_files_per_partition: int = 1,
        **kwargs
    ) -> None:
        """Optimize partitioned dataset."""
        if not self.dataset._partitioning:
            logger.warning("Dataset is not partitioned")
            return

        # Get partitions with too many files
        partitions = self._get_partitions_with_excess_files(max_files_per_partition)

        logger.info(f"Optimizing {len(partitions)} partitions")

        for partition in partitions:
            self._repartition_single_group(partition, **kwargs)

        self.dataset.reload()

    def optimize_schema(
        self,
        infer_rows: int = 10_000,
        exclude: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
        strict: bool = True,
        **kwargs
    ) -> None:
        """Optimize data types based on actual values."""
        logger.info("Optimizing schema")

        # Sample data to infer optimal types
        sample = self.dataset.table.pl.head(infer_rows)
        sample = sample.drop(self.dataset.partitioning_manager.get_partition_names(self.dataset))

        # Infer optimal schema
        optimized_df = sample.opt_dtype(strict=strict, exclude=exclude, include=include)
        optimized_schema = optimized_df.schema

        logger.debug(f"Optimized schema has {len(optimized_schema.names)} columns")

        # Process each file with optimized schema
        for file_path in self.dataset.files:
            self._optimize_file_schema(file_path, optimized_schema, **kwargs)

        self.dataset.reload()

    def _optimize_file_schema(
        self,
        file_path: str,
        optimized_schema,
        **kwargs
    ) -> None:
        """Optimize schema for a single file."""
        from ..schema import replace_schema

        logger.debug(f"Optimizing schema for file: {file_path}")

        # Scan single file
        from .filtering import DatasetFilter
        file_filter = DatasetFilter(self.dataset)
        file_scan = file_filter.filter(f"file_path = '{file_path}'")

        # Replace schema
        old_table = file_scan.to_arrow()
        new_table = replace_schema(old_table, optimized_schema)

        # Write back with new schema
        self.dataset.write(
            new_table,
            mode="append",
            update_metadata=False,
            **kwargs
        )

        # Delete old file
        self.dataset.filesystem_manager.delete_files([file_path])

    def _get_partitions_to_compact(self, max_rows_per_file: int) -> List[Dict]:
        """Get partitions that need compaction."""
        # This would query metadata to find partitions with excessive files
        # Simplified version returns empty list
        return []

    def _get_partitions_with_excess_files(self, max_files: int) -> List[Dict]:
        """Get partitions with too many files."""
        # This would query metadata to find partitions with many files
        # Simplified version returns empty list
        return []
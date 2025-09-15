import logging
import posixpath
import typing as t
from fsspec import AbstractFileSystem
import polars as pl
import pyarrow as pa
import pyarrow as pa
import duckdb as _duckdb

from .io import Writer
from .helpers.polars import pl as _pl

logger = logging.getLogger(__name__)


class DatasetWriter:
    """Handles write operations for datasets."""

    def __init__(
        self,
        path: str,
        filesystem: AbstractFileSystem,
        data_processor,
        file_manager,
    ):
        self.path = path
        self.filesystem = filesystem
        self.data_processor = data_processor
        self.file_manager = file_manager

    def write_data(
        self,
        data: (
            _pl.DataFrame
            | _pl.LazyFrame
            | pa.Table
            | pa.RecordBatch
            | pa.RecordBatchReader
            | t.Any
            | list[
                _pl.DataFrame
                | _pl.LazyFrame
                | pa.Table
                | pa.RecordBatch
                | pa.RecordBatchReader
                | t.Any
            ]
        ),
        mode: str = "append",
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
        basename: str | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> list | None:
        """Write data to dataset."""
        # Handle parameter aliases
        if "partitioning_columns" in kwargs:
            partition_by = kwargs.pop("partitioning_columns")

        # Use existing partitioning if none specified
        if not partition_by and self.data_processor.partition_names:
            partition_by = self.data_processor.partition_names

        # Update timestamp column if provided
        if timestamp_column is not None:
            self.data_processor._timestamp_column = timestamp_column

        # Normalize data to list
        if not isinstance(data, (list, tuple)) and not isinstance(data, pa.RecordBatchReader):
            data = [data]

        metadata = []
        schema = self.data_processor.schema if self.data_processor.is_loaded else None

        for data_item in data:
            writer = Writer(
                data=data_item,
                path=self.path,
                filesystem=self.filesystem,
                schema=schema if not alter_schema else None,
            )

            if writer.shape[0] == 0:
                continue

            # Process data
            writer.sort_data(by=sort_by)

            if unique:
                writer.unique(columns=unique)

            writer.cast_schema(
                ts_unit=ts_unit,
                tz=tz,
                remove_tz=remove_tz,
                alter_schema=alter_schema,
            )

            if partition_by:
                writer.add_datepart_columns(
                    columns=partition_by,
                    timestamp_column=self.data_processor.timestamp_column,
                )

            # Handle delta mode
            if mode == "delta" and self.data_processor.is_loaded:
                writer._to_polars()
                other_df = self._get_delta_df(
                    writer.data,
                    filter_columns=delta_subset,
                )
                if other_df is not None:
                    writer.delta(other=other_df, subset=delta_subset)

            # Write to dataset
            metadata_item = writer.write_to_dataset(
                row_group_size=row_group_size,
                compression=compression,
                partitioning_columns=partition_by,
                partitioning_flavor="hive",
                max_rows_per_file=max_rows_per_file,
                basename=basename,
                verbose=verbose,
                **kwargs,
            )

            if metadata_item is not None:
                metadata.extend(metadata_item)

        # Handle overwrite mode
        if mode == "overwrite":
            existing_files = [posixpath.join(self.path, fn) for fn in self.file_manager.files]
            self.file_manager.delete_files(existing_files)

        # Clear cache after writing
        self.file_manager.clear_cache()

        return metadata

    def _get_delta_df(
        self,
        df: _pl.DataFrame | _pl.LazyFrame,
        filter_columns: str | list[str] | None = None,
    ) -> _pl.DataFrame | _pl.LazyFrame:
        """Get dataframe filtered for delta operations."""
        if not self.file_manager.has_files:
            return _pl.DataFrame(schema=df.schema)

        filter_expr = self.data_processor.get_delta_filter_df(
            df=df,
            existing_columns=self.data_processor.columns,
            filter_columns=filter_columns,
        )

        if not filter_expr:
            return _pl.DataFrame(schema=df.schema)

        # Create temporary filter instance
        from .dataset_filter import DatasetFilter
        filter_instance = DatasetFilter(
            arrow_dataset=self.data_processor._arrow_dataset,
            table=self.data_processor.table,
        )

        return filter_instance.filter(" AND ".join(filter_expr)).pl
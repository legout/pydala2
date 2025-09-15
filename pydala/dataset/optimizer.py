import datetime as dt
import logging
import re
import typing as t
from fsspec import AbstractFileSystem
import duckdb as _duckdb
import pyarrow as pa
import pyarrow.dataset as pds
import polars as pl
import tqdm

from .datasets import ParquetDataset
from .helpers.security import (
    escape_sql_identifier,
    escape_sql_literal,
    validate_partition_name,
    validate_partition_value,
    sanitize_filter_expression,
)
from .schema import convert_large_types_to_normal, replace_schema

logger = logging.getLogger(__name__)


class DatasetOptimizer(ParquetDataset):
    """Handles optimization operations for Parquet datasets."""

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
        **caching_options,
    ):
        super().__init__(
            path=path,
            filesystem=filesystem,
            name=name,
            bucket=bucket,
            partitioning=partitioning,
            cached=cached,
            timestamp_column=timestamp_column,
            ddb_con=ddb_con,
            **caching_options,
        )
        self.load()

    def _compact_partition(
        self,
        partition: dict[str, t.Any],
        max_rows_per_file: int | None = 2_500_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        compression: str = "zstd",
        row_group_size: int | None = 250_000,
        unique: bool = True,
        **kwargs,
    ) -> None:
        """Compact files within a single partition."""
        # Build filter expression securely
        filter_parts = []
        for name, value in partition.items():
            if not validate_partition_name(name):
                raise ValueError(f"Invalid partition name: {name}")
            if not validate_partition_value(value):
                raise ValueError(f"Invalid partition value: {value}")

            escaped_name = escape_sql_identifier(name)
            escaped_value = escape_sql_literal(value)
            filter_parts.append(f"{escaped_name}={escaped_value}")

        filter_ = " AND ".join(filter_parts)

        # Scan partition files
        scan = self.scan(filter_)

        # Read and rewrite data in batches
        batches = scan.to_batch_reader(sort_by=sort_by, batch_size=max_rows_per_file)
        for batch in batches:
            self.write_to_dataset(
                pa.table(batch),
                mode="append",
                max_rows_per_file=max_rows_per_file,
                row_group_size=row_group_size,
                compression=compression,
                update_metadata=False,
                unique=unique,
                **kwargs,
            )

        # Delete old files
        self.delete_files(self.scan_files)
        self.reset_scan()

    def compact_partitions(
        self,
        max_rows_per_file: int | None = 2_500_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        compression: str = "zstd",
        row_group_size: int | None = 250_000,
        unique: bool = True,
        **kwargs,
    ) -> None:
        """Compact partitions with multiple small files."""
        # Find partitions that need compaction
        partitions_to_compact = (
            self.metadata_table.pl()
            .group_by(list(self.partition_names), maintain_order=True)
            .agg(
                pl.n_unique("file_path").alias("num_partition_files"),
                pl.sum("num_rows"),
            )
            .filter(pl.col("num_partition_files") > 1)
            .filter(pl.col("num_rows") < max_rows_per_file)
            .select(self.partition_names)
            .to_dicts()
        )

        # Compact each partition
        for partition in tqdm.tqdm(partitions_to_compact):
            self._compact_partition(
                partition=partition,
                max_rows_per_file=max_rows_per_file,
                sort_by=sort_by,
                compression=compression,
                row_group_size=row_group_size,
                unique=unique,
                **kwargs,
            )

        # Refresh dataset
        self.clear_cache()
        self.load(update_metadata=True)
        self.update_metadata_table()

    def compact_small_files(self) -> None:
        """Compact small files - placeholder implementation."""
        pass

    def _compact_by_timeperiod(
        self,
        start_date: dt.datetime,
        end_date: dt.datetime,
        timestamp_column: str | None = None,
        max_rows_per_file: int | None = 2_500_000,
        row_group_size: int | None = 250_000,
        compression: str = "zstd",
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        unique: bool = False,
        **kwargs,
    ) -> list[str]:
        """Compact files within a time period."""
        # Validate timestamp column
        timestamp_column = timestamp_column or self._timestamp_column
        if timestamp_column is None:
            raise ValueError("No timestamp column specified or found")

        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', timestamp_column):
            raise ValueError(f"Invalid timestamp column name: {timestamp_column}")

        # Build time filter securely
        start_date_str = start_date.isoformat()
        end_date_str = end_date.isoformat()
        filter_ = f"{timestamp_column} >= '{start_date_str}' AND {timestamp_column} < '{end_date_str}'"
        filter_ = sanitize_filter_expression(filter_)

        # Scan files in time period
        scan = self.scan(filter_)

        # Check if compaction is needed
        files_to_delete = []
        if len(self.scan_files) > 1:
            files_to_delete = self.scan_files.copy()

            # Read and rewrite data
            batches = (
                scan.to_duckdb(sort_by=sort_by, distinct=unique)
                .filter(filter_)
                .fetch_arrow_reader(batch_size=max_rows_per_file)
            )

            for batch in batches:
                self.write_to_dataset(
                    pa.table(batch),
                    mode="append",
                    max_rows_per_file=max_rows_per_file,
                    row_group_size=row_group_size,
                    compression=compression,
                    update_metadata=False,
                    unique=unique,
                    **kwargs,
                )

        self.reset_scan()
        return files_to_delete

    def compact_by_timeperiod(
        self,
        interval: str | dt.timedelta,
        timestamp_column: str | None = None,
        max_rows_per_file: int | None = 2_500_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        unique: bool = False,
        compression: str = "zstd",
        row_group_size: int | None = 250_000,
        **kwargs,
    ) -> None:
        """Compact files by time periods."""
        timestamp_column = timestamp_column or self._timestamp_column

        # Get time range
        min_max_ts = self.t.sql(
            f"SELECT MIN({timestamp_column}.min) AS min, MAX({timestamp_column}.max) AS max FROM {self.name}_metadata"
        ).pl()

        min_ts = min_max_ts["min"][0]
        max_ts = min_max_ts["max"][0]

        # Generate date ranges
        if isinstance(interval, str):
            dates = pl.datetime_range(
                min_ts, max_ts, interval=interval, eager=True
            ).to_list()
        elif isinstance(interval, dt.timedelta):
            dates = [min_ts]
            ts = min_ts
            while ts < max_ts:
                ts = ts + interval
                dates.append(ts)

        if len(dates) == 1:
            dates.append(max_ts)
        if dates[-1] < max_ts:
            dates.append(max_ts)

        # Compact each time period
        files_to_delete = []
        start_dates = dates[:-1]
        end_dates = dates[1:]

        for start_date, end_date in tqdm.tqdm(list(zip(start_dates, end_dates))):
            files_to_delete_ = self._compact_by_timeperiod(
                start_date=start_date,
                end_date=end_date,
                timestamp_column=timestamp_column,
                max_rows_per_file=max_rows_per_file,
                row_group_size=row_group_size,
                compression=compression,
                sort_by=sort_by,
                unique=unique,
                **kwargs,
            )
            files_to_delete += files_to_delete_

        # Delete old files and refresh
        self.delete_files(files_to_delete)
        self.clear_cache()
        self.load(update_metadata=True)
        self.update_metadata_table()

    def compact_by_rows(
        self,
        max_rows_per_file: int | None = 2_500_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        unique: bool = False,
        compression: str = "zstd",
        row_group_size: int | None = 250_000,
        **kwargs,
    ) -> None:
        """Compact files based on row count."""
        if self._data_processor._partitioning:
            # Use partition-based compaction for partitioned datasets
            self.compact_partitions(
                max_rows_per_file=max_rows_per_file,
                sort_by=sort_by,
                unique=unique,
                compression=compression,
                row_group_size=row_group_size,
                **kwargs,
            )
        else:
            # Compact non-partitioned dataset
            scan = self.scan(f"num_rows!={max_rows_per_file}")

            batches = scan.to_duckdb(
                sort_by=sort_by, distinct=unique
            ).fetch_arrow_reader(batch_size=max_rows_per_file)

            for batch in tqdm.tqdm(batches):
                self.write_to_dataset(
                    pa.table(batch),
                    mode="append",
                    max_rows_per_file=max_rows_per_file,
                    row_group_size=row_group_size,
                    compression=compression,
                    update_metadata=False,
                    unique=unique,
                    **kwargs,
                )

            self.delete_files(self.scan_files)
            self.clear_cache()
            self.load(update_metadata=True)
            self.update_metadata_table()

    def repartition(
        self,
        partitioning_columns: str | list[str] | None = None,
        partitioning_flavor: str = "hive",
        max_rows_per_file: int | None = 2_500_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        unique: bool = False,
        compression: str = "zstd",
        row_group_size: int | None = 250_000,
        **kwargs,
    ) -> None:
        """Repartition the dataset."""
        batches = self.table.to_batch_reader(
            sort_by=sort_by, distinct=unique, batch_size=max_rows_per_file
        )

        for batch in tqdm.tqdm(batches):
            self.write_to_dataset(
                pa.table(batch),
                partition_by=partitioning_columns,
                mode="append",
                max_rows_per_file=max_rows_per_file,
                row_group_size=min(max_rows_per_file, row_group_size),
                compression=compression,
                update_metadata=False,
                unique=unique,
                **kwargs,
            )

        self.delete_files(self.files)
        self.clear_cache()
        self.update()
        self.load()

    def _optimize_dtypes(
        self,
        file_path: str,
        optimized_schema: pa.Schema,
        exclude: str | list[str] | None = None,
        strict: bool = True,
        include: str | list[str] | None = None,
        ts_unit: str | None = None,
        tz: str | None = None,
        **kwargs,
    ) -> None:
        """Optimize data types for a single file."""
        # Safely escape file path
        escaped_file_path = file_path.replace("'", "''")
        scan = self.scan(f"file_path='{escaped_file_path}'")

        # Get schema without partition columns
        schema = pa.schema([
            field
            for field in scan.arrow_dataset.schema
            if field.name not in scan.arrow_dataset.partitioning.schema.names
        ])

        # Optimize if schema differs
        if schema != optimized_schema:
            table = replace_schema(
                scan.pl.opt_dtype(strict=strict, exclude=exclude, include=include)
                .collect(streaming=True)
                .to_arrow(),
                schema=optimized_schema,
                ts_unit=ts_unit,
                tz=tz,
            )

            self.write_to_dataset(
                table,
                mode="append",
                update_metadata=False,
                ts_unit=ts_unit,
                tz=tz,
                alter_schema=True,
                **kwargs,
            )

            self.delete_files(self.scan_files)
            self.update_file_metadata()

        self.reset_scan()

    def optimize_dtypes(
        self,
        exclude: str | list[str] | None = None,
        strict: bool = True,
        include: str | list[str] | None = None,
        ts_unit: str | None = None,
        tz: str | None = None,
        infer_schema_size: int = 10_000,
        **kwargs,
    ) -> None:
        """Optimize data types across all files."""
        # Infer optimized schema
        optimized_schema = (
            self.table.pl.drop(self.partition_names)
            .head(infer_schema_size)
            .opt_dtype(strict=strict, exclude=exclude, include=include)
            .collect()
            .to_arrow()
            .schema
        )

        optimized_schema = convert_large_types_to_normal(optimized_schema)

        # Optimize each file
        for file_path in tqdm.tqdm(self.files):
            self._optimize_dtypes(
                file_path=file_path,
                optimized_schema=optimized_schema,
                exclude=exclude,
                strict=strict,
                include=include,
                ts_unit=ts_unit,
                tz=tz,
                **kwargs,
            )

        # Refresh dataset
        self.clear_cache()
        self._update_metadata()
        self.update_metadata_table()


# Monkey patch optimization methods to ParquetDataset
ParquetDataset.compact_partitions = DatasetOptimizer.compact_partitions
ParquetDataset._compact_partition = DatasetOptimizer._compact_partition
ParquetDataset.compact_by_timeperiod = DatasetOptimizer.compact_by_timeperiod
ParquetDataset._compact_by_timeperiod = DatasetOptimizer._compact_by_timeperiod
ParquetDataset.compact_by_rows = DatasetOptimizer.compact_by_rows
ParquetDataset.repartition = DatasetOptimizer.repartition
ParquetDataset._optimize_dtypes = DatasetOptimizer._optimize_dtypes
ParquetDataset.optimize_dtypes = DatasetOptimizer.optimize_dtypes
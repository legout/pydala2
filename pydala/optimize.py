import datetime as dt

import duckdb as _duckdb
import pyarrow as pa
import tqdm
from fsspec import AbstractFileSystem

from pydala.dataset import ParquetDataset
from pydala.helpers.polars_ext import pl


class Optimize(ParquetDataset):
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
        partition: str | list[str],
        max_rows_per_file: int | None = 2_500_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        compression: str = "zstd",
        row_group_size: int | None = 250_000,
        unique: bool = True,
        **kwargs,
    ):
        if isinstance(partition, str):
            partition = [partition]

        filter_ = " AND ".join([f"{n}='{v}'" for n, v in partition.items()])

        scan = self.scan(filter_)
        if len(self.scan_files) == 1:
            num_rows = (
                self.metadata_table.filter(
                    f"file_path='{self.scan_files[0].replace(self._path,'').lstrip('/')}'"
                )
                .aggregate("sum(num_rows)")
                .fetchone()[0]
            )
        else:
            num_rows = 0

        if (len(self.scan_files) > 1) or (num_rows > max_rows_per_file):
            batches = scan.to_duckdb(
                sort_by=sort_by, distinct=distinct
            ).fetch_arrow_reader(batch_size=max_rows_per_file)
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

            self.delete_files(self.scan_files)

        self.reset_scan()

    def compact_partitions(
        self,
        max_rows_per_file: int | None = 2_500_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        compression: str = "zstd",
        row_group_size: int | None = 250_000,
        unique: bool = True,
        **kwargs,
    ) -> None:
        for partition in tqdm.tqdm(self.partitions.to_pylist()):
            self._compact_partition(
                partition=partition,
                max_rows_per_file=max_rows_per_file,
                sort_by=sort_by,
                distinct=distinct,
                compression=compression,
                row_group_size=row_group_size,
                unique=unique,
                **kwargs,
            )

        self.clear_cache()
        self.load(update_metadata=True)
        self.gen_metadata_table()

    def _compact_by_timeperiod(
        self,
        start_date: dt.datetime,
        end_date: dt.datetime,
        timestamp_column: str | None = None,
        max_rows_per_file: int | None = 2_500_000,
        row_group_size: int | None = 250_000,
        compression: str = "zstd",
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        **kwargs,
    ):
        filter_ = f"{timestamp_column} >= '{start_date}' AND {timestamp_column} < '{end_date}'"

        scan = self.scan(filter_)
        if len(self.scan_files) == 1:
            date_diff = (
                self.metadata_table.filter(
                    f"file_path='{self.scan_files[0].replace(self._path,'').lstrip('/')}'"
                )
                .aggregate("max(AE_DATUM.max) - min(AE_DATUM.min)")
                .fetchone()[0]
            )
        else:
            date_diff = dt.timedelta(0)

        files_to_delete = []
        if (len(self.scan_files) > 1) or (date_diff > end_date - start_date):
            files_to_delete += self.scan_files
            batches = (
                scan.to_duckdb(sort_by=sort_by, distinct=distinct)
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
                    unique=True,
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
        distinct: bool = False,
        compression="zstd",
        row_group_size: int | None = 250_000,
        **kwargs,
    ):
        timestamp_column = timestamp_column or self._timestamp_column

        min_max_ts = self.sql(
            f"SELECT MIN({timestamp_column}.min) AS min, MAX({timestamp_column}.max) AS max FROM {self.name}_metadata"
        ).pl()

        min_ts = min_max_ts["min"][0]
        max_ts = min_max_ts["max"][0]

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

        start_dates = dates[:-1]
        end_dates = dates[1:]

        files_to_delete = []
        for start_date, end_date in tqdm.tqdm(list(zip(start_dates, end_dates))):
            files_to_delete_ = self._compact_by_timeperiod(
                start_date=start_date,
                end_date=end_date,
                timestamp_column=timestamp_column,
                max_rows_per_file=max_rows_per_file,
                row_group_size=row_group_size,
                compression=compression,
                sort_by=sort_by,
                distinct=distinct,
                **kwargs,
            )
            files_to_delete += files_to_delete_

        self.delete_files(files_to_delete)
        self.clear_cache()
        self.load(update_metadata=True)
        self.gen_metadata_table()

    def compact_by_rows(
        self,
        max_rows_per_file: int | None = 2_500_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        compression: str = "zstd",
        row_group_size: int | None = 250_000,
        **kwargs,
    ):
        if self._partitioning:
            self.compact_partitions(
                max_rows_per_file=max_rows_per_file,
                sort_by=sort_by,
                distinct=distinct,
                compression=compression,
                row_group_size=row_group_size,
                **kwargs,
            )
        else:
            scan = self.scan(f"num_rows!={max_rows_per_file}")

            batches = scan.to_duckdb(
                sort_by=sort_by, distinct=distinct
            ).fetch_arrow_reader(batch_size=max_rows_per_file)

            for batch in tqdm.tqdm(batches):
                self.write_to_dataset(
                    pa.table(batch),
                    mode="append",
                    max_rows_per_file=max_rows_per_file,
                    row_group_size=row_group_size,
                    compression=compression,
                    update_metadata=False,
                    unique=True,
                    **kwargs,
                )
            self.delete_files(self.scan_files)
            self.clear_cache()
            self.load(update_metadata=True)
            self.gen_metadata_table()

    def repartition(
        self,
        partitioning_columns: str | list[str] | None = None,
        partitioning_falvor: str = "hive",
        max_rows_per_file: int | None = 2_500_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        compression="zstd",
        row_group_size: int | None = 250_000,
        **kwargs,
    ):
        batches = self.to_duckdb(
            sort_by=sort_by, distintinct=distinct
        ).fetch_arrow_reader(batch_size=max_rows_per_file)

        for batch in tqdm.tqdm(batches):
            self.write_to_dataset(
                pa.table(batch),
                partitioning_columns=partitioning_columns,
                # partitioning_falvor=partitioning_falvor,
                mode="append",
                max_rows_per_file=max_rows_per_file,
                row_group_size=min(max_rows_per_file, row_group_size),
                compression=compression,
                update_metadata=False,
                unique=True,
                **kwargs,
            )
        self.delete_files(self.files)
        self.clear_cache()
        self.update()
        self.load()

    def optimize_dtypes(
        self,
        exclude: str | list[str] | None = None,
        strict: bool = True,
        include: str | list[str] | None = None,
    ):
        # optimizes_schema = self.table.
        pass


#        self.gen_metadata_table()


ParquetDataset.compact_partitions = Optimize.compact_partitions
ParquetDataset._compact_partition = Optimize._compact_partition
ParquetDataset.compact_by_timeperiod = Optimize.compact_by_timeperiod
ParquetDataset._compact_by_timeperiod = Optimize._compact_by_timeperiod
ParquetDataset.compact_by_rows = Optimize.compact_by_rows
ParquetDataset.repartition = Optimize.repartition

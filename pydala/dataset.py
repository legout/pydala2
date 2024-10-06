import datetime as dt
import os
import re

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
from .helpers.polars import pl as _pl
from .io import Writer
from .metadata import ParquetDatasetMetadata, PydalaDatasetMetadata
from .schema import (
    replace_schema,  # from .optimize import Optimize
    shrink_large_string,
)
from .table import PydalaTable


class BaseDataset:
    def __init__(
        self,
        path: str,
        name: str | None = None,
        filesystem: AbstractFileSystem | None = None,
        bucket: str | None = None,
        partitioning: str | list[str] | None = None,
        format: str | None = "parquet",
        cached: bool = False,
        timestamp_column: str | None = None,
        ddb_con: _duckdb.DuckDBPyConnection | None = None,
        **fs_kwargs,
    ):
        self._path = path
        self._bucket = bucket
        self._cached = cached
        self._format = format
        self._base_filesystem = filesystem
        self._filesystem = FileSystem(
            bucket=bucket, fs=filesystem, cached=cached, **fs_kwargs
        )
        self._makedirs()

        if name is None:
            self.name = os.path.basename(path)
        else:
            self.name = name

        if ddb_con is None:
            ddb_con = _duckdb.connect()

        self.ddb_con = ddb_con
        # enable object caching for e.g. parquet metadata
        self.ddb_con.execute(
            f"""PRAGMA enable_object_cache;
                             SET THREADS={psutil.cpu_count()*2};"""
        )
        self._timestamp_column = timestamp_column

        self.load_files()

        if self.has_files:
            if partitioning == "ignore":
                self._partitioning = None
            elif partitioning is None and "=" in self._files[0]:
                self._partitioning = "hive"
            else:
                self._partitioning = partitioning
        else:
            self._partitioning = partitioning

        try:
            self.load()
        except Exception as e:
            _ = e
            pass

    # @abstactmethod
    def load_files(self) -> None:
        """
        Loads the files from the specified path with the specified format.

        This method populates the `_files` attribute with a list of file names
        that match the specified format in the specified path.

        Returns:
            None
        """
        self.clear_cache()
        self._files = [
            fn.replace(self._path, "").lstrip("/")
            for fn in sorted(
                self._filesystem.glob(os.path.join(self._path, f"**/*.{self._format}"))
            )
        ]

    def _makedirs(self):
        if self._filesystem.exists(self._path):
            return
        try:
            self._filesystem.mkdirs(self._path)
        except Exception as e:
            _ = e
            self._filesystem.touch(os.path.join(self._path, "tmp.delete"))
            self._filesystem.rm(os.path.join(self._path, "tmp.delete"))

    @property
    def files(self) -> list:
        """
        Returns a list of files in the dataset.

        If the files have not been loaded yet, this method will call the `load_files` method to load them.

        Returns:
            list: A list of files in the dataset.
        """
        if not hasattr(self, "_files"):
            self.load_files()

        return self._files

    @property
    def path(self):
        return self._path

    @property
    def has_files(self):
        """
        Returns True if the dataset has files, False otherwise.
        """
        return len(self.files) > 0

    def load(self):
        """
        Loads the dataset from the specified path and initializes the PydalaTable.

        Returns:
            None
        """
        if self.has_files:
            self._arrow_dataset = pds.dataset(
                self._path,
                filesystem=self._filesystem,
                format=self._format,
                partitioning=self._partitioning,
            )
            self.table = PydalaTable(result=self._arrow_dataset, ddb_con=self.ddb_con)
            # self.ddb_con.register("arrow__dataset", self._arrow_parquet_dataset)

            if self._timestamp_column is None:
                self._timestamp_columns = get_timestamp_column(self.table.pl.head(10))
                if len(self._timestamp_columns) > 0:
                    self._timestamp_column = self._timestamp_columns[0]

            if self._timestamp_column is not None:
                tz = self.schema.field(self._timestamp_column).type.tz
                self._tz = tz
                if tz is not None:
                    self.ddb_con.execute(f"SET TimeZone='{tz}'")
            else:
                self._tz = None

            self.ddb_con.register(f"{self.name}", self._arrow_dataset)

    def clear_cache(self) -> None:
        """
        Clears the cache for the dataset.

        This method clears the cache for the dataset by calling the `clear_cache` function
        for both the `_filesystem` and `_base_filesystem` attributes.

        Returns:
            None
        """
        if hasattr(self._filesystem, "fs"):
            clear_cache(self._filesystem.fs)
        clear_cache(self._filesystem)
        clear_cache(self._base_filesystem)

    @property
    def schema(self):
        """
        Returns the schema of the dataset.

        If the schema has not been cached, it retrieves the schema from the underlying Arrow dataset.

        Returns:
            pyarrow.Schema: The schema of the dataset.
        """
        if self.is_loaded:
            if not hasattr(self, "_schema"):
                self._schema = self._arrow_dataset.schema
            return self._schema

    @property
    def is_loaded(self) -> bool:
        """
        Returns True if the dataset has been loaded into memory, False otherwise.
        """
        return hasattr(self, "_arrow_dataset")

    @property
    def columns(self) -> list:
        """
        Returns a list of column names for the dataset, including both schema and partitioning columns.

        Returns:
            list: A list of column names.
        """
        if self.is_loaded:
            return self.schema.names

    @property
    def fs(self):
        """
        Returns the filesystem associated with the dataset.

        Returns:
            The filesystem object associated with the dataset.
        """
        return self._filesystem

    @property
    def t(self):
        """
        Returns the table associated with the dataset.

        Returns:
            The table associated with the dataset.
        """
        return self.table

    def count_rows(self) -> int:
        """
        Returns the number of rows in the dataset.

        If the dataset is not loaded, it prints an error message and returns 0.

        Returns:
            int: The number of rows in the dataset.
        """
        if self.is_loaded:
            return self._arrow_dataset.count_rows()
        else:
            # print(f"No dataset loaded yet. Run {self}.load()")
            return 0

    @property
    def num_rows(self) -> int:
        """
        Returns the number of rows in the dataset.

        If the dataset is not loaded, it prints an error message and returns 0.

        Returns:
            int: The number of rows in the dataset.
        """
        return self.count_rows()

    @property
    def num_columns(self) -> int:
        """
        Returns the number of columns in the dataset.

        If the dataset is not loaded, it prints an error message and returns 0.

        Returns:
            int: The number of columns in the dataset.
        """
        if self.is_loaded:
            return len(self.schema.names)
        else:
            # print(f"No dataset loaded yet. Run {self}.load()")
            return 0

    def delete_files(self, files: str | list[str]):
        """
        Deletes the specified files from the dataset.

        Args:
            files (str | list[str], optional): The file(s) to be deleted. If not provided, all files in the dataset
                will be deleted. Defaults to None.
        """

        if self._path not in files[0]:
            files = [os.path.join(self._path, fn) for fn in files]
        self._filesystem.rm(files, recursive=True)
        # self.load(reload=True)

    def vacuum(self):
        """
        Deletes all files in the dataset.
        """
        self.delete_files(self._files)
        self.delete_metadata_files()

    @property
    def partitioning_schema(self) -> pa.Schema:
        """
        Returns the partitioning schema of the dataset.

        If the dataset has files and the partitioning schema has not been
        computed yet, it will be computed and cached. If the dataset is not
        loaded, an empty schema will be returned.

        Returns:
            pa.Schema: The partitioning schema of the dataset.
        """
        if self.is_loaded:
            if hasattr(self._arrow_dataset, "partitioning"):
                if not hasattr(self, "_partitioning_schema"):
                    if self.is_loaded:
                        self._partitioning_schema = (
                            self._arrow_dataset.partitioning.schema
                        )
                    else:
                        return pa.schema([])
            return self._partitioning_schema

    @property
    def partition_names(self) -> list:
        """
        Returns a list of partitioning names.

        Returns:
            list: A list of partitioning names.
        """

        if self.is_loaded:
            if not hasattr(self, "_partition_names") and hasattr(
                self._arrow_dataset, "partitioning"
            ):
                self._partition_names = self._arrow_dataset.partitioning.schema.names

            return self._partition_names

    @property
    def partition_values(self) -> dict:
        """
        Returns a list of partitioning values.

        Returns:
            list: A list of partitioning values.
        """
        if self.is_loaded:
            if hasattr(self._arrow_dataset, "partitioning"):
                self._partition_values = dict(
                    zip(
                        self._arrow_dataset.partitioning.schema.names,
                        [
                            pa_list.to_pylist()
                            for pa_list in self._arrow_dataset.partitioning.dictionaries
                        ],
                    )
                )
            else:
                # print(f"No dataset loaded yet. Run {self}.load()")
                return []
            return self._partition_values

    @property
    def partitions(self) -> pa.Table:
        """
        Returns the partitions of the dataset as a pyarrow Table.

        If the dataset has files and is partitioned, it retrieves the partitions
        based on the partitioning schema. If the dataset is not loaded, an empty
        pyarrow Table is returned.

        Returns:
            pa.Table: The partitions of the dataset as a pyarrow Table.
        """
        if self.is_loaded:
            if self._partitioning:
                if not hasattr(self, "_partitions") and hasattr(
                    self._arrow_dataset, "partitioning"
                ):
                    self._partitions = sorted(
                        {
                            re.findall(
                                r"/".join(
                                    [
                                        f"{name}=([^/]+)"
                                        for name in self._arrow_dataset.partitioning.schema.names
                                    ]
                                ),
                                f,
                            )[0]
                            for f in self.files
                        }
                    )
                    self._partitions = (
                        _pl.DataFrame(
                            self._partitions,
                            schema=self._arrow_dataset.partitioning.schema.names,
                            orient="row",
                        )
                        .to_arrow()
                        .cast(self._arrow_dataset.partitioning.schema)
                    )

                return self._partitions

    def _filter_duckdb(self, filter_expr: str) -> _duckdb.DuckDBPyRelation:
        """
        Filters the DuckDBPyRelation table based on the given filter expression.

        Args:
            filter_expr (str): The filter expression to apply.

        Returns:
            _duckdb.DuckDBPyRelation: The filtered DuckDBPyRelation table.
        """
        return self.table.ddb.filter(filter_expr)

    def _filter_arrow_dataset(
        self, filter_expr: str | pds.Expression
    ) -> pds.FileSystemDataset:
        """
        Filters a PyArrow Parquet dataset based on the provided filter expression.

        Args:
            filter_expr (str | pds.Expression): The filter expression to be applied to the dataset.
                It can be either a string representation of a filter expression or a PyArrow
                Expression object.

        Returns:
            pds.FileSystemDataset: The filtered PyArrow Parquet dataset.
        """
        if self.is_loaded:
            if isinstance(filter_expr, str):
                filter_expr = sql2pyarrow_filter(filter_expr, self.schema)
            return self._arrow_dataset.filter(filter_expr)

    def filter(
        self,
        filter_expr: str | pds.Expression,
        use: str = "auto",
    ) -> pds.FileSystemDataset | pds.Dataset | _duckdb.DuckDBPyRelation:
        """
        Filters the dataset based on the given filter expression.

        Args:
            filter_expr (str | pds.Expression): The filter expression to apply.
            use (str, optional): The method to use for filtering. Defaults to "auto".

        Returns:
            pds.FileSystemDataset | pds.Dataset | _duckdb.DuckDBPyRelation: The filtered dataset.

        Raises:
            Exception: If filtering with PyArrow fails and use is set to "auto".

        Note:
            If the filter expression contains any of the following: "%", "like", "similar to", or "*",
            the method will automatically use DuckDB for filtering.

        """
        if any([s in filter_expr for s in ["%", "like", "similar to", "*", "(", ")"]]):
            use = "duckdb"

        if use == "auto":
            try:
                res = self._filter_arrow_dataset(filter_expr)

            except Exception as e:
                e.add_note("Note: Filtering with PyArrow failed. Using DuckDB.")
                print(e)
                res = self._filter_duckdb(filter_expr)

        elif use == "pyarrow":
            res = self._filter_arrow_dataset(filter_expr)

        elif use == "duckdb":
            res = self._filter_duckdb(filter_expr)

        return PydalaTable(result=res, ddb_con=self.ddb_con)

    @property
    def registered_tables(self) -> list[str]:
        """
        Get a list of registered tables.

        Returns:
            list[str]: A list of table names.
        """
        return self.ddb_con.sql("SHOW TABLES").arrow().column("name").to_pylist()

    def interrupt_duckdb(self):
        """
        Interrupts the DuckDB connection.
        """
        self.ddb_con.interrupt()

    def reset_duckdb(self):
        """
        Resets the DuckDB connection and registers the necessary tables and datasets.

        Returns:
            None
        """
        # self.ddb_con = _duckdb.connect()
        self.interrupt_duckdb()

        if f"{self.name}" not in self.registered_tables:
            if hasattr(self, "_arrow_table"):
                self.ddb_con.register(f"{self.name}", self._arrow_table)

            elif hasattr(self, "_arrow_dataset"):
                self.ddb_con.register(f"{self.name}", self._arrow_dataset)

    def _get_delta_other_df(
        self,
        df: _pl.DataFrame | _pl.LazyFrame,
        filter_columns: str | list[str] | None = None,
    ) -> _pl.DataFrame | _pl.LazyFrame:
        """
        Returns a filtered DataFrame or LazyFrame based on the given input DataFrame and filter columns.

        Args:
            df (DataFrame or LazyFrame): The input DataFrame or LazyFrame to filter.
            filter_columns (str or list[str] or None, optional): The columns to filter on. Defaults to None.

        Returns:
            DataFrame or LazyFrame: The filtered DataFrame or LazyFrame.

        """
        collect = False
        if len(self.files) == 0:
            return _pl.DataFrame(schema=df.schema)
        if isinstance(df, _pl.LazyFrame):
            collect = True

        columns = set(df.columns) & set(self.columns)
        null_columns = df.select(cs.by_dtype(_pl.Null)).collect_schema().names()
        columns = columns - set(null_columns)
        if filter_columns is not None:
            columns = set(columns) & set(filter_columns)
        if len(columns) == 0:
            return _pl.DataFrame(schema=df.schema)

        filter_expr = []
        for col in columns:
            # if df.collect_schema().get(col).is_(_pl.Null):
            #     max_min = df.select(
            #         _pl.first(col).alias("max"), _pl.last(col).alias("min")
            #     )
            # else:
            max_min = df.select(_pl.max(col).alias("max"), _pl.min(col).alias("min"))

            if collect:
                max_min = max_min.collect()
            f_max = max_min["max"][0]
            f_min = max_min["min"][0]

            if isinstance(f_max, str):
                f_max = f_max.strip("'").replace(",", "")
            if isinstance(f_min, str):
                f_min = f_min.strip("'").replace(",", "")

            filter_expr.append(
                f"(({col}<='{f_max}' AND {col}>='{f_min}') OR {col} IS NULL)".replace(
                    "'None'", "NULL"
                )
            )

        return self.filter(" AND ".join(filter_expr)).pl

    def write_to_dataset(
        self,
        data: (
            _pl.DataFrame
            | _pl.LazyFrame
            | pa.Table
            | pa.RecordBatch
            | pd.DataFrame
            | _duckdb.DuckDBPyConnection
            | list[
                _pl.DataFrame
                | _pl.LazyFrame
                | pa.Table
                | pa.RecordBatch
                | pd.DataFrame
                | _duckdb.DuckDBPyConnection
            ]
        ),
        mode: str = "append",  # "delta", "overwrite"
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
        use_large_string: bool = False,
        delta_subset: str | list[str] | None = None,
        alter_schema: bool = False,
        timestamp_column: str | None = None,
        **kwargs,
    ):
        """
        Writes the given data to the dataset.

        Parameters:
        - data: The data to be written to the dataset. It can be one of the following types:
            - _pl.DataFrame
            - _pl.LazyFrame
            - pa.Table
            - pa.RecordBatch
            - pa.RecordBatchReader
            - pd.DataFrame
            - _duckdb.DuckDBPyConnection
            - list of any of the above types
        - mode: The write mode. Possible values are "append", "delta", or "overwrite". Defaults to "append".
        - basename: The template for the basename of the output files. Defaults to None.
        - partition_by: The columns to be used for partitioning the dataset. Can be a string, a list of strings,
            or None. Defaults to None.
        - max_rows_per_file: The maximum number of rows per file. Defaults to 2,500,000.
        - row_group_size: The size of each row group. Defaults to 250,000.
        - compression: The compression algorithm to be used. Defaults to "zstd".
        - sort_by: The column(s) to sort the data by. Can be a string, a list of strings, or a list of tuples (column,
            order). Defaults to None.
        - unique: Whether to keep only unique rows. Can be a bool, a string, or a list of strings. Defaults to False.
        - ts_unit: The unit of the timestamp column. Defaults to "us".
        - tz: The timezone to be used for the timestamp column. Defaults to None.
        - remove_tz: Whether to remove the timezone information from the timestamp column. Defaults to False.
        - use_large_string: Whether to use large string type for string columns. Defaults to False.
        - delta_subset: The subset of columns to consider for delta updates. Can be a string, a list of strings, or
            None. Defaults to None.
        - alter_schema: Whether to alter the schema of the dataset. Defaults to False.
        - timestamp_column: The name of the timestamp column. Defaults to None.
        - **kwargs: Additional keyword arguments to be passed to the writer.

        Returns:
        None
        """

        if "partitioning_columns" in kwargs:
            partition_by = kwargs.pop("partitioning_columns")

        if not partition_by and self.partition_names:
            partition_by = self.partition_names

        if timestamp_column is not None:
            self._timestamp_column = timestamp_column

        if (
            not isinstance(data, list)
            and not isinstance(data, tuple)
            and not isinstance(data, pa.RecordBatchReader)
        ):
            data = [data]

        for data_ in data:
            writer = Writer(
                data=data_,
                path=self._path,
                filesystem=self._filesystem,
                schema=self.schema if not alter_schema else None,
            )
            if writer.shape[0] == 0:
                continue

            writer.sort_data(by=sort_by)

            if unique:
                writer.unique(columns=unique)

            if partition_by:
                writer.add_datepart_columns(
                    columns=partition_by,
                    timestamp_column=self._timestamp_column,
                )

            writer.cast_schema(
                use_large_string=use_large_string,
                ts_unit=ts_unit,
                tz=tz,
                remove_tz=remove_tz,
                alter_schema=alter_schema,
            )

            if mode == "delta" and self.is_loaded:
                writer._to_polars()
                other_df = self._get_delta_other_df(
                    writer.data,
                    filter_columns=delta_subset,
                )
                if other_df is not None:
                    writer.delta(other=other_df, subset=delta_subset)

            writer.write_to_dataset(
                row_group_size=row_group_size,
                compression=compression,
                partitioning_columns=partition_by,
                partitioning_flavor="hive",
                max_rows_per_file=max_rows_per_file,
                basename=basename,
                **kwargs,
            )

        if mode == "overwrite":
            del_files = [os.path.join(self._path, fn) for fn in self.files]
            self.delete_files(del_files)

        self.clear_cache()
        self.load_files()


class ParquetDataset(PydalaDatasetMetadata, BaseDataset):
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
        **fs_kwargs,
    ):
        """
        Initialize a Dataset object.

        Args:
            path (str): The path to the dataset.
            filesystem (AbstractFileSystem, optional): The filesystem to use.
                Defaults to None.
            bucket (str, optional): The bucket to use. Defaults to None.
            partitioning (str, list[str], optional): The partitioning scheme
                to use. Defaults to None.
            cached (bool, optional): Whether to use cached data. Defaults to
                False.
            **cached_options: Additional options for cached data.

        Returns:
            None
        """

        PydalaDatasetMetadata.__init__(
            self,
            path=path,
            filesystem=filesystem,
            bucket=bucket,
            cached=cached,
            partitioning=partitioning,
            ddb_con=ddb_con,
            **fs_kwargs,
        )
        BaseDataset.__init__(
            self,
            path=path,
            name=name,
            filesystem=filesystem,
            bucket=bucket,
            partitioning=partitioning,
            cached=cached,
            timestamp_column=timestamp_column,
            ddb_con=self.ddb_con,
            **fs_kwargs,
        )

        # if self.has_metadata:
        # self.pydala_dataset_metadata = PydalaDatasetMetadata(
        #    metadata=self.metadata, partitioning=partitioning, ddb_con=ddb_con
        # )
        # self.metadata_table.create_view(f"{self.name}_metadata")
        # self.ddb_con.unregister("metadata_table")

        try:
            self.load()
        except Exception as e:
            _ = e
            pass

    def load(
        self,
        update_metadata: bool = False,
        reload_metadata: bool = False,
        schema: pa.Schema | None = None,
        ts_unit: str = "us",
        tz: str | None = None,
        use_large_string: bool = False,
        sort_schema: bool = False,
        format_version: str = "2.6",
        **kwargs,
    ):
        """
        Loads the data from the dataset.

        Args:
            update_metadata (bool, optional): Whether to update the metadata. Defaults to False.
            reload_metadata (bool, optional): Whether to reload the metadata. Defaults to False.
            schema (pa.Schema | None, optional): The schema of the data. Defaults to None.
            ts_unit (str, optional): The unit of the timestamp. Defaults to "us".
            tz (str | None, optional): The timezone. Defaults to None.
            use_large_string (bool, optional): Whether to use large string. Defaults to False.
            sort_schema (bool, optional): Whether to sort the schema. Defaults to False.
            format_version (str, optional): The version of the data format. Defaults to "2.6".
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        if kwargs.pop("update", None):
            update_metadata = True
        if kwargs.pop("reload", None):
            reload_metadata = True
        if self.has_files:
            if update_metadata or reload_metadata or not self.has_metadata_file:
                self.update(
                    reload=reload_metadata,
                    schema=schema,
                    ts_unit=ts_unit,
                    tz=tz,
                    use_large_string=use_large_string,
                    sort=sort_schema,
                    format_version=format_version,
                    **kwargs,
                )
                self.update_metadata_table()

            self._arrow_dataset = pds.parquet_dataset(
                self._metadata_file,
                partitioning=self._partitioning,
                filesystem=self._filesystem,
            )

            self.table = PydalaTable(result=self._arrow_dataset, ddb_con=self.ddb_con)

            if self._timestamp_column is None:
                self._timestamp_columns = get_timestamp_column(self.table.pl.head(10))
                if len(self._timestamp_columns) > 0:
                    self._timestamp_column = self._timestamp_columns[0]
            if self._timestamp_column is not None:
                tz = self.schema.field(self._timestamp_column).type.tz
                self._tz = tz
                if tz is not None:
                    self.ddb_con.execute(f"SET TimeZone='{tz}'")
            else:
                self._tz = None

            self.ddb_con.register(f"{self.name}", self._arrow_dataset)

    # def gen_metadata_table(self):
    #     """
    #     Generate the metadata table for the dataset.

    #     This function calls the `gen_metadata_table` method of the `pydala_dataset_metadata` object
    #     to generate the metadata table for the dataset. It takes two parameters:

    #     - `metadata`: The metadata object containing information about the dataset.
    #     - `_partitioning`: The partitioning object containing information about the dataset partitioning.

    #     This function does not return anything.

    #     Example usage:
    #     ```
    #     self.gen_metadata_table()
    #     ```
    #     """
    #     self.pydala_dataset_metadata.gen_metadata_table(
    #         self.metadata, self._partitioning
    #     )

    def scan(self, filter_expr: str | None = None) -> ParquetDatasetMetadata:
        """
        Scans the Parquet dataset metadata.

        Args:
            filter_expr (str, optional): An optional filter expression to apply to the metadata.

        Returns:
            ParquetDatasetMetadata: The ParquetDatasetMetadata object.
        """
        PydalaDatasetMetadata.scan(self, filter_expr=filter_expr)

        if len(self.scan_files) == 0:
            return PydalaTable(result=self.table.ddb.limit(0))

        return PydalaTable(
            result=pds.dataset(
                [os.path.join(self.path, fn) for fn in self.scan_files],
                filesystem=self._filesystem,
                format=self._format,
                partitioning=self._partitioning,
            ),
            ddb_con=self.ddb_con,
        )

    def __repr__(self):
        return self.table.__repr__()

    def write_to_dataset(
        self,
        data: (
            _pl.DataFrame
            | _pl.LazyFrame
            | pa.Table
            | pa.RecordBatch
            | pa.RecordBatchReader
            | pd.DataFrame
            | _duckdb.DuckDBPyConnection
            | list[
                _pl.DataFrame
                | _pl.LazyFrame
                | pa.Table
                | pa.RecordBatch
                | pd.DataFrame
                | _duckdb.DuckDBPyConnection
            ]
        ),
        mode: str = "append",  # "delta", "overwrite"
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
        use_large_string: bool = False,
        delta_subset: str | list[str] | None = None,
        update_metadata: bool = False,
        alter_schema: bool = False,
        timestamp_column: str | None = None,
        **kwargs,
    ):
        """
        Writes the given data to the dataset.

        Parameters:
        - data: The data to be written to the dataset. It can be one of the following types:
            - _pl.DataFrame
            - _pl.LazyFrame
            - pa.Table
            - pa.RecordBatch
            - pa.RecordBatchReader
            - pd.DataFrame
            - _duckdb.DuckDBPyConnection
            - list of any of the above types
        - mode: The write mode. Possible values are "append", "delta", or "overwrite". Defaults to "append".
        - basename: The template for the basename of the output files. Defaults to None.
        - partition_by: The columns to be used for partitioning the dataset. Can be a string, a list of strings,
            or None. Defaults to None.
        - max_rows_per_file: The maximum number of rows per file. Defaults to 2,500,000.
        - row_group_size: The size of each row group. Defaults to 250,000.
        - compression: The compression algorithm to be used. Defaults to "zstd".
        - sort_by: The column(s) to sort the data by. Can be a string, a list of strings, or a list of tuples (column,
            order). Defaults to None.
        - unique: Whether to keep only unique rows. Can be a bool, a string, or a list of strings. Defaults to False.
        - ts_unit: The unit of the timestamp column. Defaults to "us".
        - tz: The timezone to be used for the timestamp column. Defaults to None.
        - remove_tz: Whether to remove the timezone information from the timestamp column. Defaults to False.
        - use_large_string: Whether to use large string type for string columns. Defaults to False.
        - delta_subset: The subset of columns to consider for delta updates. Can be a string, a list of strings, or
            None. Defaults to None.
        - update_metadata: Whether to update the metadata table after writing. Defaults to False.
        - alter_schema: Whether to alter the schema of the dataset. Defaults to False.
        - timestamp_column: The name of the timestamp column. Defaults to None.
        - **kwargs: Additional keyword arguments to be passed to the writer.

        Returns:
        None
        """
        # if not partitioning_columns and self.partition_names:
        #     partitioning_columns = self.partition_names

        # if timestamp_column is not None:
        #     self._timestamp_column = timestamp_column

        # if (
        #     not isinstance(data, list)
        #     and not isinstance(data, tuple)
        #     and not isinstance(data, pa.RecordBatchReader)
        # ):
        #     data = [data]

        # for data_ in data:
        #     writer = Writer(
        #         data=data_,
        #         path=self._path,
        #         filesystem=self._filesystem,
        #         schema=self.schema if not alter_schema else None,
        #     )
        #     if writer.shape[0] == 0:
        #         continue

        #     writer.sort_data(by=sort_by)

        #     if unique:
        #         writer.unique(columns=unique)

        #     if partitioning_columns:
        #         writer.add_datepart_columns(
        #             columns=partitioning_columns,
        #             timestamp_column=self._timestamp_column,
        #         )

        #     writer.cast_schema(
        #         use_large_string=use_large_string,
        #         ts_unit=ts_unit,
        #         tz=tz,
        #         remove_tz=remove_tz,
        #         alter_schema=alter_schema,
        #     )

        #     if mode == "delta" and self.is_loaded:
        #         writer._to_polars()
        #         other_df = self._get_delta_other_df(
        #             writer.data,
        #             filter_columns=delta_subset,
        #         )
        #         if other_df is not None:
        #             writer.delta(other=other_df, subset=delta_subset)

        #     writer.write_to_dataset(
        #         row_group_size=row_group_size,
        #         compression=compression,
        #         partitioning_columns=partitioning_columns,
        #         partitioning_flavor="hive",
        #         max_rows_per_file=max_rows_per_file,
        #         basename=basename,
        #         **kwargs,
        #     )

        # if mode == "overwrite":
        #     del_files = [os.path.join(self._path, fn) for fn in self.files]
        #     self.delete_files(del_files)

        # self.clear_cache()
        # self.load_files()
        BaseDataset.write_to_dataset(
            self,
            data=data,
            mode=mode,
            basename=basename,
            partition_by=partition_by,
            max_rows_per_file=max_rows_per_file,
            row_group_size=row_group_size,
            compression=compression,
            sort_by=sort_by,
            unique=unique,
            ts_unit=ts_unit,
            tz=tz,
            remove_tz=remove_tz,
            use_large_string=use_large_string,
            delta_subset=delta_subset,
            alter_schema=alter_schema,
            timestamp_column=timestamp_column,
            **kwargs,
        )

        if update_metadata:
            try:
                self.load(update_metadata=True, verbose=False)
            except Exception as e:
                _ = e
                self.load(reload_metadata=True, verbose=False)
            self.update_metadata_table()


class PyarrowDataset(BaseDataset):
    def __init__(
        self,
        path: str,
        name: str | None = None,
        filesystem: AbstractFileSystem | None = None,
        bucket: str | None = None,
        partitioning: str | list[str] | None = None,
        format: str = "parquet",
        cached: bool = False,
        timestamp_column: str | None = None,
        ddb_con: _duckdb.DuckDBPyConnection | None = None,
        **fs_kwargs,
    ):
        super().__init__(
            path=path,
            name=name,
            filesystem=filesystem,
            bucket=bucket,
            partitioning=partitioning,
            format=format,
            cached=cached,
            timestamp_column=timestamp_column,
            ddb_con=ddb_con,
            **fs_kwargs,
        )


class CsvDataset(PyarrowDataset):
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
        **fs_kwargs,
    ):
        super().__init__(
            path=path,
            name=name,
            filesystem=filesystem,
            bucket=bucket,
            partitioning=partitioning,
            format="csv",
            cached=cached,
            timestamp_column=timestamp_column,
            ddb_con=ddb_con,
            **fs_kwargs,
        )


class JsonDataset(BaseDataset):
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
        **fs_kwargs,
    ):
        super().__init__(
            path=path,
            name=name,
            filesystem=filesystem,
            bucket=bucket,
            partitioning=partitioning,
            format="json",
            cached=cached,
            timestamp_column=timestamp_column,
            ddb_con=ddb_con,
            **fs_kwargs,
        )

    def load(self):
        self._arrow_dataset = pds.dataset(
            self._filesystem.read_json_dataset(
                self._path,
            )
            .opt_dtype(strict=False)
            .to_arrow()
        )
        self.table = PydalaTable(result=self._arrow_dataset, ddb_con=self.ddb_con)

        self.ddb_con.register(f"{self.name}", self._arrow_dataset)
        # self.ddb_con.register("arrow__dataset", self._arrow_parquet_dataset)

        if self._timestamp_column is None:
            self._timestamp_columns = get_timestamp_column(self.table.pl.head(10))
            if len(self._timestamp_columns) > 1:
                self._timestamp_column = self._timestamp_columns[0]


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
                sort_by=sort_by, distinct=unique
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
                compression=compression,
                row_group_size=row_group_size,
                unique=unique,
                **kwargs,
            )

        self.clear_cache()
        self.load(update_metadata=True)
        self.update_metadata_table()

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
            dates = _pl.datetime_range(
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
                unique=unique,
                **kwargs,
            )
            files_to_delete += files_to_delete_

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
    ):
        if self._partitioning:
            self.compact_partitions(
                max_rows_per_file=max_rows_per_file,
                sort_by=sort_by,
                unique=unique,
                compression=compression,
                row_group_size=row_group_size,
                **kwargs,
            )
        else:
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
        partitioning_falvor: str = "hive",
        max_rows_per_file: int | None = 2_500_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        unique: bool = False,
        compression="zstd",
        row_group_size: int | None = 250_000,
        **kwargs,
    ):
        batches = self.table.to_batch_reader(
            sort_by=sort_by, distinct=unique, batch_size=max_rows_per_file
        )

        for batch in tqdm.tqdm(batches):
            self.write_to_dataset(
                pa.table(batch),
                partition_by=partitioning_columns,
                # partitioning_falvor=partitioning_falvor,
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
        ts_unit: str | None = None,  # "us",
        tz: str | None = None,
        **kwargs,
    ):
        scan = self.scan(f"file_path='{file_path}'")
        schema = scan.arrow_dataset.schema
        schema = pa.schema(
            [
                field
                for field in scan.arrow_dataset.schema
                if field.name not in scan.arrow_dataset.partitioning.schema.names
            ]
        )

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
                ts_unit=ts_unit,  # "us",
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
        ts_unit: str | None = None,  # "us",
        tz: str | None = None,
        use_large_string: bool = False,
        infer_schema_size: int = 10_000,
        **kwargs,
    ):
        optimized_schema = (
            self.table.pl.drop(self.partition_names)
            .head(infer_schema_size)
            .opt_dtype(strict=strict, exclude=exclude, include=include)
            .collect()
            .to_arrow()
            .schema
        )
        if not use_large_string:
            optimized_schema = shrink_large_string(optimized_schema)

        for file_path in tqdm.tqdm(self.files):
            self._optimize_dtypes(
                file_path=file_path,
                optimized_schema=optimized_schema,
                exclude=exclude,
                strict=strict,
                include=include,
                ts_unit=ts_unit,
                tz=tz,
                use_large_string=use_large_string,
                **kwargs,
            )

        self.clear_cache()

        self._update_metadata_file()
        self.update_metadata_table()


ParquetDataset.compact_partitions = Optimize.compact_partitions
ParquetDataset._compact_partition = Optimize._compact_partition
ParquetDataset.compact_by_timeperiod = Optimize.compact_by_timeperiod
ParquetDataset._compact_by_timeperiod = Optimize._compact_by_timeperiod
ParquetDataset.compact_by_rows = Optimize.compact_by_rows
ParquetDataset.repartition = Optimize.repartition
ParquetDataset._optimize_dtypes = Optimize._optimize_dtypes
ParquetDataset.optimize_dtypes = Optimize.optimize_dtypes

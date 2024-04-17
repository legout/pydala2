import os
import re
import duckdb as _duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
from fsspec import AbstractFileSystem

from .filesystem import FileSystem, clear_cache
from .helpers.io import Writer
from .helpers.misc import get_timestamp_column, str2pyarrow_filter
from .helpers.polars_ext import pl as _pl
from .metadata import ParquetDatasetMetadata, PydalaDatasetMetadata
from .table import PydalaTable
# from .optimize import Optimize


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
        **caching_options,
    ):
        self._path = path
        self._bucket = bucket
        self._cached = cached
        self._format = format
        self._base_filesystem = filesystem
        self._filesystem = FileSystem(
            bucket=bucket, fs=filesystem, cached=cached, **caching_options
        )

        if name is None:
            self.name = os.path.basename(path)
        else:
            self.name = name

        if ddb_con is None:
            ddb_con = _duckdb.connect()

        self.ddb_con = ddb_con
        self._timestamp_column = timestamp_column

        self.load_files()

        if self.has_files:
            if partitioning == "ignore":
                self._partitioning = None
            elif partitioning is None and "=" in self._files[0]:
                self._partitioning = "hive"
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
        self._files = [
            fn.replace(self._path, "").lstrip("/")
            for fn in sorted(
                self._filesystem.glob(os.path.join(self._path, f"**/*.{self._format}"))
            )
        ]

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
        self._arrow_dataset = pds.dataset(
            self._path,
            filesystem=self._filesystem,
            format=self._format,
            partitioning=self._partitioning,
        )
        self.table = PydalaTable(result=self._arrow_dataset, ddb_con=self.ddb_con)

        self.ddb_con.register(f"{self.name}_dataset", self._arrow_dataset)
        # self.ddb_con.register("arrow__dataset", self._arrow_parquet_dataset)

        if self._timestamp_column is None:
            self._timestamp_columns = get_timestamp_column(self.table.pl.head(1))
            if len(self._timestamp_columns) > 1:
                self._timestamp_column = self._timestamp_columns[0]

    def clear_cache(self) -> None:
        """
        Clears the cache for the dataset.

        This method clears the cache for the dataset by calling the `clear_cache` function
        for both the `_filesystem` and `_base_filesystem` attributes.

        Returns:
            None
        """
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
        if not hasattr(self, "_schema"):
            self._schema = self._arrow_dataset.schema
        return self._schema

    @property
    def is_loaded(self) -> bool:
        """
        Returns True if the dataset has been loaded into memory, False otherwise.
        """
        return hasattr(self, "table")

    @property
    def columns(self) -> list:
        """
        Returns a list of column names for the dataset, including both schema and partitioning columns.

        Returns:
            list: A list of column names.
        """
        if self.has_files:
            return self.schema.names

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
        self.count_rows()

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

    def delete_files(self, files: str | list[str] = None):
        """
        Deletes the specified files from the dataset.

        Args:
            files (str | list[str], optional): The file(s) to be deleted. If not provided, all files in the dataset
                will be deleted. Defaults to None.
        """
        if files is None:
            files = self.files
        if self._path not in files[0]:
            files = [os.path.join(self._path, fn) for fn in files]
        self._filesystem.rm(files, recursive=True)
        # self.load(reload=True)

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
        if self.has_files:
            if not hasattr(self, "_partitioning_schema"):
                if self.is_loaded:
                    self._partitioning_schema = self._arrow_dataset.partitioning.schema
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

        if self.has_files:
            if not hasattr(self, "_partition_names"):
                if self.is_loaded:
                    self._partition_names = (
                        self._arrow_dataset.partitioning.schema.names
                    )
                else:
                    # print(f"No dataset loaded yet. Run {self}.load()")
                    return []
            return self._partition_names

    @property
    def partition_values(self) -> dict:
        """
        Returns a list of partitioning values.

        Returns:
            list: A list of partitioning values.
        """
        if self.has_files:
            if not hasattr(self, "_partition_values"):
                if self.is_loaded:
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
        if self.has_files:
            if self._partitioning:
                if not hasattr(self, "_partitions"):
                    if self.is_loaded:
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
                            )
                            .to_arrow()
                            .cast(self._arrow_dataset.partitioning.schema)
                        )
                    else:
                        return pa.Table.from_pylist([])
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
        if self.has_files:
            if isinstance(filter_expr, str):
                filter_expr = str2pyarrow_filter(filter_expr, self.schema)
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

        if f"{self.name}_table" not in self.registered_tables:
            if len(self._table_files):
                self.ddb_con.register(f"{self.name}_table", self._arrow_table)

        if f"{self.name}_dataset" not in self.registered_tables:
            if hasattr(self, "_arrow_dataset"):
                if len(self._arrow_dataset.files):
                    self.ddb_con.register(f"{self.name}_dataset", self._arrow_dataset)

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

        filter_expr = []
        columns = set(df.columns) & set(self.columns)
        if filter_columns is not None:
            columns = set(columns) & set(filter_columns)
        if len(columns) == 0:
            return _pl.DataFrame(schema=df.schema)

        for col in columns:
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

        return self.filter(" AND ".join(filter_expr), use="duckdb").pl

    def write_to_dataset(
        self,
        df: (
            _pl.DataFrame
            | _pl.LazyFrame
            | pa.Table
            | pa.RecordBatch
            | pd.DataFrame
            | _duckdb.DuckDBPyConnection
        ),
        mode: str = "append",  # "delta", "overwrite"
        basename_template: str | None = None,
        partitioning_columns: str | list[str] | None = None,
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
        Writes the given DataFrame to the dataset.

        Args:
            df (DataFrame | LazyFrame | Table | RecordBatch | pd.DataFrame | DuckDBPyConnection):
                The DataFrame to write.
            mode (str, optional): The mode of writing. Can be "append", "delta", or "overwrite".
                Defaults to "append".
            basename_template (str | None, optional): The template for the basename of the files.
                Defaults to None.
            partitioning_columns (str | list[str] | None, optional): The columns used for partitioning.
                Defaults to None.
            max_rows_per_file (int | None, optional): The maximum number of rows per file. Defaults to 2,500,000.
            row_group_size (int | None, optional): The size of each row group. Defaults to 250,000.
            compression (str, optional): The compression algorithm. Defaults to "zstd".
            sort_by (str | list[str] | list[tuple[str, str]] | None, optional): The columns to sort by.
                Defaults to None.
            unique (bool | str | list[str], optional): Whether to remove duplicate rows. Can be True, False,
                or a list of columns. Defaults to False.
            ts_unit (str, optional): The timestamp unit. Defaults to "us".
            tz (str | None, optional): The timezone. Defaults to None.
            remove_tz (bool, optional): Whether to remove the timezone. Defaults to False.
            use_large_string (bool, optional): Whether to use large strings. Defaults to False.
            delta_subset (str | list[str] | None, optional): The subset of columns for delta computation.
                Defaults to None.
            alter_schema (bool, optional): Whether to alter the schema. Defaults to False.
            timestamp_column (str | None, optional): The timestamp column. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        Raises:
            None
        """

        if df.shape[0] == 0:
            return

        if self.partition_names:
            partitioning_columns = self.partition_names.copy()

        if timestamp_column is not None:
            self._timestamp_column = timestamp_column

        writer = Writer(
            data=df,
            path=self._path,
            filesystem=self._filesystem,
            schema=self.schema if not alter_schema else None,
        )
        writer.sort_data(by=sort_by)

        if unique:
            writer.unique(columns=unique)

        if partitioning_columns:
            writer.add_datepart_columns(
                columns=partitioning_columns, timestamp_column=self._timestamp_column
            )

        writer.cast_schema(
            use_large_string=use_large_string,
            ts_unit=ts_unit,
            tz=tz,
            remove_tz=remove_tz,
            alter_schema=alter_schema,
        )

        if mode == "overwrite":
            del_files = [os.path.join(self._path, fn) for fn in self.files]

        elif mode == "delta" and self.has_files:
            writer._to_polars()
            other_df = self._get_delta_other_df(
                writer.data,
                filter_columns=delta_subset,
            )
            if other_df is not None:
                writer.delta(other=other_df, subset=delta_subset)

        if writer.shape[0] == 0:
            return

        writer.write_to_dataset(
            row_group_size=row_group_size,
            compression=compression,
            partitioning_columns=partitioning_columns,
            partitioning_flavor="hive",
            max_rows_per_file=max_rows_per_file,
            basename_template=basename_template,
            **kwargs,
        )
        if mode == "overwrite":
            self.delete_files(del_files)

        self.clear_cache()
        self.load_files()


class ParquetDataset(ParquetDatasetMetadata, BaseDataset):
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

        ParquetDatasetMetadata.__init__(
            self,
            path=path,
            filesystem=filesystem,
            bucket=bucket,
            cached=cached,
            **caching_options,
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
            ddb_con=ddb_con,
            **caching_options,
        )

        if self.has_metadata:
            self.pydala_dataset_metadata = PydalaDatasetMetadata(
                metadata=self.metadata, partitioning=partitioning, ddb_con=ddb_con
            )
            self.metadata_table.create_view(f"{self.name}_metadata")

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

            self.ddb_con.register(f"{self.name}_dataset", self._arrow_dataset)

            if self._timestamp_column is None:
                self._timestamp_columns = get_timestamp_column(self.table.pl.head(1))
                if len(self._timestamp_columns) > 1:
                    self._timestamp_column = self._timestamp_columns[0]

    def gen_metadata_table(self):
        """
        Generate the metadata table for the dataset.

        This function calls the `gen_metadata_table` method of the `pydala_dataset_metadata` object
        to generate the metadata table for the dataset. It takes two parameters:

        - `metadata`: The metadata object containing information about the dataset.
        - `_partitioning`: The partitioning object containing information about the dataset partitioning.

        This function does not return anything.

        Example usage:
        ```
        self.gen_metadata_table()
        ```
        """
        self.pydala_dataset_metadata.gen_metadata_table(
            self.metadata, self._partitioning
        )

    def scan(self, filter_expr: str | None = None) -> ParquetDatasetMetadata:
        """
        Scans the Parquet dataset metadata.

        Args:
            filter_expr (str, optional): An optional filter expression to apply to the metadata.

        Returns:
            ParquetDatasetMetadata: The ParquetDatasetMetadata object.
        """
        self.pydala_dataset_metadata.scan(filter_expr=filter_expr)

        return PydalaTable(
            result=pds.dataset(
                self.scan_files,
                filesystem=self._filesystem,
                partitioning=self._partitioning,
            ),
            ddb_con=self.ddb_con,
        )

    def __repr__(self):
        return self.table.__repr__()

    def reset_scan(self):
        """
        Reset the scan of the dataset metadata.

        This function calls the `reset_scan` method of the `pydala_dataset_metadata` object.

        Args:
            self (object): The instance of the class.

        Returns:
            None
        """
        self.pydala_dataset_metadata.reset_scan()

    def update_metadata_table(self):
        """
        Update the metadata table.

        This function updates the metadata table by creating a new instance of the `PydalaDatasetMetadata` class
        if it does not already exist. The `PydalaDatasetMetadata` class is initialized with the `metadata` and
        `_partitioning` attributes of the current instance. The `gen_metadata_table` method of the
        `PydalaDatasetMetadata` instance is then called with the `metadata` and `_partitioning` attributes as
        arguments to generate the metadata table.


        Returns:
            None
        """
        if not hasattr(self, "pydala_dataset_metadata"):
            self.pydala_dataset_metadata = PydalaDatasetMetadata(
                metadata=self.metadata,
                partitioning=self._partitioning,
            )
        self.pydala_dataset_metadata.gen_metadata_table(
            self.metadata, self._partitioning
        )

    @property
    def metadata_table(self) -> pa.Table:
        """
        A description of the entire function, its parameters, and its return types.

        Returns:
            pa.Table: A pyarrow Table containing the metadata of the dataset.

        """
        return self.pydala_dataset_metadata.metadata_table

    @property
    def metadata_table_scanned(self) -> pa.Table:
        """
        Returns a Polars DataFrame containing information about the filtered files in the dataset.

        Returns:
            _pl.DataFrame: A Pandas DataFrame containing information about the files in the dataset.
        """
        return self.pydala_dataset_metadata._metadata_table_scanned

    @property
    def scan_files(self) -> list[str]:
        """
        Returns a list of files in the dataset.

        Returns:
            list[str]: A list of files in the dataset.
        """
        return sorted(
            [
                os.path.join(self._path, f)
                for f in self.pydala_dataset_metadata.scan_files
            ]
        )

    def _get_delta_other_df(
        self,
        df: _pl.DataFrame | _pl.LazyFrame,
        filter_columns: str | list[str] | None = None,
    ) -> _pl.DataFrame | _pl.LazyFrame:
        """
        Generate the delta dataframe based on the given dataframe, columns, use, and on parameters.

        Parameters:
            df (_pl.DataFrame | _pl.LazyFrame): The input dataframe or lazyframe.
            filter_columns (str | list[str] | None, optional): The columns to consider. Defaults to None.
            use (str, optional): The use parameter. Defaults to "auto".
            on (str, optional): The on parameter. Defaults to "auto".

        Returns:
            _pl.DataFrame: The delta dataframe.
        """
        collect = False
        if len(self.files) == 0:
            return _pl.DataFrame(schema=df.schema)
        if isinstance(df, _pl.LazyFrame):
            collect = True

        filter_expr = []
        columns = set(df.columns) & set(self.columns)
        if filter_columns is not None:
            columns = set(columns) & set(filter_columns)
        if len(columns) == 0:
            return _pl.DataFrame(schema=df.schema)

        for col in columns:
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
                f"{col}<='{f_max}' AND {col}>='{f_min}'".replace("'None'", "NULL")
            )

        return self.scan(" AND ".join(filter_expr)).pl

    def write_to_dataset(
        self,
        df: (
            _pl.DataFrame
            | _pl.LazyFrame
            | pa.Table
            | pa.RecordBatch
            | pd.DataFrame
            | _duckdb.DuckDBPyConnection
        ),
        mode: str = "append",  # "delta", "overwrite"
        basename_template: str | None = None,
        partitioning_columns: str | list[str] | None = None,
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
        Write data to the dataset with specified options and handling for different modes and parameters.

        Args:
            df: DataFrame to be written to the dataset. Can be a pandas DataFrame, PyPolars DataFrame, PyArrow Table,
                PyArrow RecordBatch, DuckDBPyConnection, or LazyFrame.
            mode: Mode of writing to the dataset, either "append", "delta", or "overwrite".
            partitioning_columns: Columns to partition the dataset by. Can be a string, list of strings, or None.
            max_rows_per_file: Maximum number of rows per file, or None.
            row_group_size: Size of row groups, or None.
            compression: Compression algorithm to use, default is "zstd".
            sort_by: Columns to sort the dataset by. Can be a string, list of strings, list of tuples of strings,
                or None.
            unique: Columns to enforce uniqueness on, can be a bool, string, or list of strings.
            ts_unit: Unit for timestamps, default is "us" (microseconds).
            tz: Timezone to use, or None.
            remove_tz: Whether to remove the timezone, default is False.
            use_large_string: Whether to use large string types, default is False.
            delta_subset: Subset of columns to consider for delta mode, can be a string, list of strings, or None.
            other_df_filter_columns: Columns to filter the other DataFrame by, can be a string, list of strings,
                or None.
            use: Library to use for writing, default is "pyarrow".
            on: Format of the dataset, default is "parquet_dataset".
            update_metadata: Whether to update the metadata, default is False.
            alter_schema: Whether to alter the schema, default is False.
            **kwargs: Additional keyword arguments.
        """
        if "n_rows" in kwargs:
            max_rows_per_file = kwargs.pop("n_rows")
        if "num_rows" in kwargs:
            max_rows_per_file = kwargs.pop("num_rows")

        if df.shape[0] == 0:
            return

        if self.partition_names:
            partitioning_columns = self.partition_names.copy()

        if timestamp_column is not None:
            self._timestamp_column = timestamp_column

        writer = Writer(
            data=df,
            path=self._path,
            filesystem=self._filesystem,
            schema=self.schema if not alter_schema else None,
        )
        writer.sort_data(by=sort_by)

        if unique:
            writer.unique(columns=unique)
        if partitioning_columns:
            writer.add_datepart_columns(
                columns=partitioning_columns, timestamp_column=self._timestamp_column
            )

        writer.cast_schema(
            use_large_string=use_large_string,
            ts_unit=ts_unit,
            tz=tz,
            remove_tz=remove_tz,
            alter_schema=alter_schema,
        )

        if mode == "overwrite":
            del_files = [os.path.join(self._path, fn) for fn in self.files]

        elif mode == "delta" and self.has_files:
            writer._to_polars()
            other_df = self._get_delta_other_df(
                writer.data,
                filter_columns=delta_subset,
                # use=use,
                # on=on,
            )
            if other_df is not None:
                writer.delta(other=other_df, subset=delta_subset)

        if writer.shape[0] == 0:
            return

        writer.write_to_dataset(
            row_group_size=row_group_size,
            compression=compression,
            partitioning_columns=partitioning_columns,
            partitioning_flavor="hive",
            max_rows_per_file=max_rows_per_file,
            basename_template=basename_template,
            **kwargs,
        )
        if mode == "overwrite":
            self.delete_files(del_files)
            if update_metadata:
                self.load(update_metadata=True, verbose=False)
                self.update_metada_table()

        else:
            if update_metadata:
                try:
                    self.load(update_metadata=True, verbose=False)
                except Exception as e:
                    _ = e
                    self.load(reload_metadata=True, verbose=False)
                self.update_metada_table()

        self.clear_cache()
        self.load_files()


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
        **caching_options,
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
            **caching_options,
        )


class CSVDataset(PyarrowDataset):
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
            name=name,
            filesystem=filesystem,
            bucket=bucket,
            partitioning=partitioning,
            format="csv",
            cached=cached,
            timestamp_column=timestamp_column,
            ddb_con=ddb_con,
            **caching_options,
        )

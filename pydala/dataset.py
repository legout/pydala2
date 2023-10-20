import datetime as dt
import os
import re
import uuid
import tqdm


import duckdb as _duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.fs as pfs
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem

from .filesystem import clear_cache, get_filesystem
from .helpers.misc import (
    get_timestamp_column,
    humanized_size_to_bytes,
    partition_by,
    run_parallel,
    str2pyarrow_filter,
)
from .helpers.io import read_table, write_table
from .helpers.polars_ext import pl as _pl

from .metadata import ParquetDatasetMetadata, PydalaDatasetMetadata


class ParquetDataset(ParquetDatasetMetadata):
    def __init__(
        self,
        path: str,
        filesystem: AbstractFileSystem | None = None,
        bucket: str | None = None,
        partitioning: str | list[str] | None = None,
        cached: bool = False,
        **caching_options,
    ):
        """
        Initialize a Dataset object.

        Args:
            path (str): The path to the dataset.
            filesystem (AbstractFileSystem, optional): The filesystem to use. Defaults to None.
            bucket (str, optional): The bucket to use. Defaults to None.
            partitioning (str, list[str], optional): The partitioning scheme to use. Defaults to None.
            cached (bool, optional): Whether to use cached data. Defaults to False.
            **cached_options: Additional options for cached data.

        Returns:
            None
        """

        super().__init__(
            path=path,
            filesystem=filesystem,
            bucket=bucket,
            cached=cached,
            **caching_options,
        )

        if self.has_files:
            if partitioning == "ignore":
                self._partitioning = None
            elif partitioning is None and "=" in self._files[0]:
                partitioning = "hive"
            else:
                self._partitioning = partitioning

        else:
            self._partitioning = None

        self.pydala_dataset_metadata = PydalaDatasetMetadata(
            metadata=self.metadata,
            partitioning=partitioning,
        )

        self.ddb_con = _duckdb.connect()

    def load(
        self,
        update_metadata: bool = False,
        reload_metadata: bool = False,
        schema: pa.Schema | None = None,
        ts_unit: str = "us",
        tz: str | None = None,
        use_large_string: bool = False,
        sort: bool = False,
        format_version: str = "1.0",
        **kwargs,
    ):
        """
        Load the dataset.

        Args:
            reload (bool, optional): Reload the dataset even if it is already loaded. Defaults to False.
            update (bool, optional): Update the dataset metadata. Defaults to False.
            schema (pa.Schema | None, optional): Arrow schema to use for the dataset. Defaults to None.
            ts_unit (str, optional): Timestamp unit to use for the dataset. Defaults to "us".
            tz (str | None, optional): Timezone to use for the dataset. Defaults to None.
            use_large_string (bool, optional): Use large string type for the dataset. Defaults to False.
            sort (bool, optional): Sort the dataset. Defaults to False.
            format_version (str, optional): Format version to use for the dataset. Defaults to "1.0".
            **kwargs: Additional keyword arguments to pass to the update method.

        Returns:
            None
        """

        if self.has_files:
            if update_metadata or reload_metadata or not self.has_metadata_file:
                self.update(
                    reload=reload_metadata,
                    schema=schema,
                    ts_unit=ts_unit,
                    tz=tz,
                    use_large_string=use_large_string,
                    sort=sort,
                    format_version=format_version,
                    **kwargs,
                )
                self.update_metadata_table()

            self._pyarrow_parquet_dataset = pds.parquet_dataset(
                self.metadata_file,
                partitioning=self._partitioning,
                filesystem=self._filesystem,
            )

            if len(self._pyarrow_parquet_dataset.files):
                self.ddb_con.register(
                    "pyarrow_parquet_dataset", self._pyarrow_parquet_dataset
                )

                self._timestamp_columns = get_timestamp_column(self.pl().head(1))

    # @property
    def arrow_parquet_dataset(self) -> pds.FileSystemDataset:
        if not hasattr(self, "_pyarrow_parquet_dataset"):
            self.load()
        return self._pyarrow_parquet_dataset

    @property
    def is_loaded(self) -> bool:
        """
        Returns True if the dataset has been loaded into memory, False otherwise.
        """
        return hasattr(self, "pyarrow_parquet_dataset")

    @property
    def columns(self) -> list:
        """
        Returns a list of column names for the dataset, including both schema and partitioning columns.

        Returns:
            list: A list of column names.
        """
        if not hasattr(self, "_columns"):
            self._columns = self.schema.names + self.partitioning_names
        return self._columns

    @property
    def count_rows(self) -> int:
        """
        Returns the number of rows in the dataset.

        If the dataset is not loaded, it prints an error message and returns 0.

        Returns:
            int: The number of rows in the dataset.
        """
        if self.is_loaded:
            return self.arrow_parquet_dataset().count_rows
        else:
            # print(f"No dataset loaded yet. Run {self}.load()")
            return 0

    @property
    def partitioning_schema(self) -> pa.Schema:
        """
        Returns the partitioning schema for the dataset.

        If the dataset has not been loaded yet, an empty schema is returned.

        Returns:
            pa.Schema: The partitioning schema for the dataset.
        """
        if not hasattr(self, "_partitioning_schema"):
            if self.is_loaded:
                self._partitioning_schema = (
                    self.arrow_parquet_dataset().partitioning.schema
                )
            else:
                # print(f"No dataset loaded yet. Run {self}.load()")
                return pa.schema([])
        return self._partitioning_schema

    @property
    def schema(self) -> pa.Schema:
        """
        Returns the schema of the dataset, which is a unified schema
        of the file schema and partitioning schema (if present).
        """
        if not hasattr(self, "_schema"):
            # if self._partitioning is not None and self._partitioning!="ignore":
            self._schema = pa.unify_schemas(
                [self.file_schema, self.partitioning_schema]
            )
        return self._schema

    @property
    def partitioning_names(self) -> list:
        """
        Returns a list of partitioning names for the dataset.

        If the partitioning names have not been loaded yet, this method will attempt to load them.
        If the dataset has not been loaded yet, this method will return an empty list and print a message
        instructing the user to load the dataset first.

        Returns:
            A list of partitioning names for the dataset.
        """
        if not hasattr(self, "_partitioning_names"):
            if self.is_loaded:
                self._partitioning_names = self.partitioning_schema.names
            else:
                # print(f"No dataset loaded yet. Run {self}.load()")
                return []
        return self._partitioning_names

    def gen_metadata_table(self):
        self.pydala_dataset_metadata.gen_metadata_table(
            self.metadata, self._partitioning
        )

    def scan(self, filter_expr: str | None = None) -> ParquetDatasetMetadata:
        self.pydala_dataset_metadata.scan(filter_expr=filter_expr)
        return self

    def to_arrow_dataset(
        self,
    ) -> pds.Dataset:
        """
        Converts the current Pydala object to a PyArrow Dataset object.


        Returns:
            pds.Dataset: The PyArrow Dataset object.
        """
        if hasattr(self, "_pyarrow_dataset"):
            if sorted(self._pyarrow_dataset.files) == sorted(self.scan_files):
                return self._pyarrow_dataset

            self._pyarrow_dataset = pds.dataset(
                self.scan_files,
                partitioning=self._partitioning,
                filesystem=self._filesystem,
            )
        else:
            self._pyarrow_dataset = pds.dataset(
                self.scan_files,
                partitioning=self._partitioning,
                filesystem=self._filesystem,
            )
        if len(self._pyarrow_dataset.files):
            self.ddb_con.register("pyarrow_dataset", self._pyarrow_dataset)

        return self._pyarrow_dataset

    # @property
    def arrow_dataset(self) -> pds.Dataset:
        """
        Returns a PyArrow Dataset object representing the data in this instance.
        """
        # if not hasattr(self, "_pyarrow_dataset"):
        return self.to_arrow_dataset()
        # return self._pyarrow_dataset

    def to_arrow(self, **kwargs) -> pa.Table:
        """
        Converts the dataset to a PyArrow table by reading the data from disk.

        Returns:
            A PyArrow table containing the data from the dataset.

        """
        if hasattr(self, "_pyarrow_table"):
            if sorted(self._table_files) == sorted(self.scan_files):
                return self._pyarrow_table

            else:
                self._pyarrow_table = pa.concat_tables(
                    run_parallel(
                        read_table,
                        self.scan_files,
                        schema=self.schema,
                        filesystem=self._filesystem,
                        partitioning=self._partitioning,
                        **kwargs,
                    )
                )

        else:
            self._pyarrow_table = pa.concat_tables(
                run_parallel(
                    read_table,
                    self.scan_files,
                    schema=self.schema,
                    filesystem=self._filesystem,
                    partitioning=self._partitioning,
                    **kwargs,
                )
            )
        self._table_files = self.scan_files.copy()

        if len(self._table_files):
            self.ddb_con.register("pyarrow_table", self._pyarrow_table)

        return self._pyarrow_table

    # @property
    def arrow(self) -> pa.Table:
        """
        Returns a PyArrow Table representation of the dataset.

        Returns:
            pa.Table: A PyArrow Table representation of the dataset.
        """
        # if not hasattr(self, "_pyarrow_table"):
        return self.to_arrow()
        # return self._pyarrow_table

    def to_duckdb(
        self,
        lazy: bool = True,
    ) -> _duckdb.DuckDBPyRelation:
        """
        Converts the dataset to a DuckDBPyRelation object.

        Args:
            filter_expr (str | None): An optional filter expression to apply to the dataset.
            lazy (bool): If True, the dataset is lazily loaded. If False, the dataset is eagerly loaded.
            from_ (str): The source of the dataset. Can be either "scan_files" or "files". Defaults to "scan_files".

        Returns:
            _duckdb.DuckDBPyRelation: A DuckDBPyRelation object representing the dataset.
        """

        if lazy:
            if sorted(self.files) == sorted(self.scan_files):
                self._ddb = self.ddb_con.from_arrow(self.arrow_parquet_dataset())
            else:
                self._ddb = self.ddb_con.from_arrow(self.arrow_dataset())

        else:
            self._ddb = self.ddb_con.from_arrow(self.arrow())

        return self._ddb

    def duckdb(self, lazy: bool = True) -> _duckdb.DuckDBPyRelation:
        """
        Converts the dataset to a DuckDBPyRelation object.

        Args:
            filter_expr (str | None): An optional filter expression to apply to the dataset.
            lazy (bool): If True, the dataset is lazily loaded. If False, the dataset is eagerly loaded.
            from_ (str): The source of the dataset. Can be either "scan_files" or "files". Defaults to "scan_files".

        Returns:
            _duckdb.DuckDBPyRelation: A DuckDBPyRelation object representing the dataset.
        """
        return self.to_duckdb(lazy=lazy)

    # @property
    def ddb(self) -> _duckdb.DuckDBPyRelation:
        """
        Converts the dataset to a DuckDBPyRelation object.

        Returns:
            _duckdb.DuckDBPyRelation: A DuckDBPyRelation object representing the dataset.
        """
        # if not hasattr(self, "_ddb"):
        return self.to_duckdb()
        # return self._ddb

    def to_polars(
        self,
        lazy: bool = True,
    ) -> _pl.DataFrame:
        """
        Converts the dataset to a Polars DataFrame.

        Args:
            filter_expr (str | None): An optional filter expression to apply to the dataset.
            lazy (bool): If True, the dataset is lazily loaded. If False, the dataset is eagerly loaded.
            from_ (str): The source of the dataset. Can be either "scan_files" or "files". Defaults to "scan_files".

        Returns:
            _pl.DataFrame: A Polars DataFrame representing the dataset.
        """
        if lazy:
            self._pl = _pl.scan_pyarrow_dataset(self.arrow_dataset())
        else:
            self._pl = _pl.from_arrow(self.arrow())

        return self._pl

    def to_pl(self, lazy: bool = True) -> _pl.DataFrame:
        """
        Converts the dataset to a Polars DataFrame.

        Args:
            filter_expr (str | None): An optional filter expression to apply to the dataset.
            lazy (bool): If True, the dataset is lazily loaded. If False, the dataset is eagerly loaded.
            from_ (str): The source of the dataset. Can be either "scan_files" or "files". Defaults to "scan_files".

        Returns:
            _pl.DataFrame: A Polars DataFrame representing the dataset.
        """
        return self.to_polars(lazy=lazy)

    # @property
    def pl(self) -> _pl.DataFrame:
        """
        Convert the dataset to a Polars DataFrame.

        Returns:
            _pl.DataFrame: A Polars DataFrame representing the dataset.
        """
        # if not hasattr(self, "_pl"):
        return self.to_polars()
        # return self._pl

    def to_pandas(
        self,
        lazy: bool = True,
    ) -> pd.DataFrame:
        """
        Convert the dataset to a pandas DataFrame.

        Args:
            filter_expr (str | None): An optional filter expression to apply to the dataset.
            lazy (bool): If True, the dataset is lazily loaded. If False, the dataset is eagerly loaded.
            from_ (str): The source of the dataset. Can be either "scan_files" or "files". Defaults to "scan_files".
        Returns:
            pd.DataFrame: A pandas DataFrame containing the dataset.
        """
        self._df = self.to_duckdb(lazy=lazy).df()
        return self._df

    def to_df(self, lazy: bool = True) -> pd.DataFrame:
        """
        Convert the dataset to a pandas DataFrame.

        Args:
            filter_expr (str | None): An optional filter expression to apply to the dataset.
            lazy (bool): If True, the dataset is lazily loaded. If False, the dataset is eagerly loaded.
            from_ (str): The source of the dataset. Can be either "scan_files" or "files". Defaults to "scan_files".
        Returns:
            pd.DataFrame: A pandas DataFrame containing the dataset.
        """
        return self.to_pandas(lazy=lazy)

    # @property
    def df(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame representation of the dataset.

        Returns:
            pd.DataFrame: A pandas DataFrame representation of the dataset.
        """
        # if not hasattr(self, "_df"):
        return self.to_pandas()
        # return self._df

    def sql(self, sql: str) -> _duckdb.DuckDBPyRelation:
        """
        Executes an SQL query on the DuckDBPy connection.

        Args:
            sql (str): The SQL query to be executed.

        Returns:
            _duckdb.DuckDBPyRelation: The result of the SQL query as a DuckDBPyRelation object.
        """
        return self.ddb_con.sql(sql)

    def _filter_duckdb(self, filter_expr: str) -> _duckdb.DuckDBPyRelation:
        """
        Filter the DuckDBPyRelation based on a given filter expression.

        Parameters:
            filter_expr (str): The filter expression to apply.

        Returns:
            _duckdb.DuckDBPyRelation: The filtered DuckDBPyRelation object.
        """
        return self.ddb().filter(filter_expr)

    def _filter_pyarrow_dataset(self, filter_expr: str | pds.Expression) -> pds.Dataset:
        """
        Filters the pyarrow dataset based on a filter expression.

        Args:
            filter_expr (str | pds.Expression): The filter expression to apply. It can be either a string or a pyarrow expression.

        Returns:
            pds.Dataset: The filtered pyarrow dataset.
        """
        if isinstance(filter_expr, str):
            filter_expr = str2pyarrow_filter(filter_expr, self.schema)
        return self.arrow_dataset().filter(filter_expr)

    def _filter_pyarrow_parquet_dataset(
        self, filter_expr: str | pds.Expression
    ) -> pds.FileSystemDataset:
        """
        Filters a PyArrow Parquet dataset based on the provided filter expression.

        Args:
            filter_expr (str | pds.Expression): The filter expression to be applied to the dataset.
                It can be either a string representation of a filter expression or a PyArrow Expression object.

        Returns:
            pds.FileSystemDataset: The filtered PyArrow Parquet dataset.
        """
        if isinstance(filter_expr, str):
            filter_expr = str2pyarrow_filter(filter_expr, self.schema)
        return self.arrow_parquet_dataset().filter(filter_expr)

    def filter(
        self, filter_expr: str | pds.Expression, use: str = "pyarrow", on: str = "auto"
    ) -> pds.FileSystemDataset | pds.Dataset | _duckdb.DuckDBPyRelation:
        """
        Filters the dataset based on the given filter expression.

        Args:
            filter_expr (str | pds.Expression): The filter expression to apply.
            use (str, optional): The library to use for filtering. Options are "pyarrow" and "duckdb".
                Defaults to "pyarrow".
            on (str, optional): The dataset type to filter on. Options are "parquet_dataset", "dataset" or "auto".
                Defaults to "auto".

        Returns:
            pds.FileSystemDataset | pds.Dataset | _duckdb.DuckDBPyRelation: The filtered dataset based on the specified parameters.
        """

        if use == "pyarrow":
            if on == "auto":
                if sorted(self.files) == sorted(self.scan_files):
                    return self._filter_pyarrow_parquet_dataset(filter_expr)
                else:
                    return self._filter_pyarrow_dataset(filter_expr)
            elif on == "parquet_dataset":
                return self._filter_pyarrow_parquet_dataset(filter_expr)
            elif on == "dataset":
                return self._filter_pyarrow_dataset(filter_expr)

        elif use == "duckdb":
            if on == "parquet_dataset":
                self.reset_scan()
                return self._filter_duckdb(filter_expr)
            else:
                return self._filter_duckdb(filter_expr)

    @property
    def registered_tables(self) -> list[str]:
        """
        Get a list of registered tables.

        Returns:
            list[str]: A list of table names.
        """
        return self.ddb_con.sql("SHOW TABLES").arrow().column("name").to_pylist()

    def reset_scan(self):
        """
        Reset the scan of the dataset metadata.

        This function calls the `reset_scan` method of the `pydala_dataset_metadata` object.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        self.pydala_dataset_metadata.reset_scan()

    def update_metadata_table(self):
        self.pydala_dataset_metadata.gen_metadata_table(
            self.metadata, self._partitioning
        )

    @property
    def metadata_table(self) -> pa.Table:
        """
        Returns a Polars DataFrame containing information about the files in the dataset.

        If the file catalog has not yet been generated, this method will call the `gen_file_catalog` method to generate it.

        Returns:
            _pl.DataFrame: A Pandas DataFrame containing information about the files in the dataset.
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

    def delete_files(self, files: str | list[str] | None = None):
        """
        Deletes the specified files from the dataset.

        Args:
            files (str | list[str] | None, optional): The name(s) of the file(s) to delete. If None, all files in the dataset will be deleted. Defaults to None.
        """
        self._filesystem.rm(files, recursive=True)
        self.load(reload=True)

    def _gen_delta_df(
        self,
        df: _pl.DataFrame | _pl.LazyFrame,
        delta_subset: str | list[str] | None = None,
        use: str = "duckdb",
        on: str = "auto",
    ):
        """
        Generates a delta DataFrame by comparing the input DataFrame with the current
        state of the Dataset object. The delta DataFrame contains only the rows that
        have changed since the last scan.

        Args:
            df: A polars DataFrame or LazyFrame to compare with the current state of the
                Dataset object.
            delta_subset: A string or list of strings representing the subset of columns
                to consider when computing the delta. If None, all columns are used.

        Returns:
            A polars DataFrame containing the rows that have changed since the last scan.
        """

        if isinstance(df, _pl.LazyFrame):
            df = df.collect()

        filter_expr = []
        for col in delta_subset or df.columns:
            f_max = df.select(_pl.col(col).max())[0, 0]
            if isinstance(f_max, str):
                f_max = f_max.strip("'")

            f_min = df.select(_pl.col(col).min())[0, 0]
            if isinstance(f_min, str):
                f_min = f_min.strip("'")

            filter_expr.append(
                f"{col}<='{f_max}' AND {col}>='{f_min}'".replace("'None'", "NULL")
            )

        res = self.scan(" AND ".join(filter_expr)).filter(
            " AND ".join(filter_expr), use=use, on=on
        )

        if isinstance(res, _duckdb.DuckDBPyRelation):
            df0 = res.pl()
        else:
            df0 = _pl.from_arrow(res.to_table())
        self.reset_scan()

        if df0.shape[0] > 0:
            return df.delta(df0, subset=delta_subset, eager=True)

        return df

    def write_to_dataset(
        self,
        df: _pl.DataFrame
        | _pl.LazyFrame
        | pa.Table
        | pd.DataFrame
        | _duckdb.DuckDBPyConnection,
        base_name: str | None = None,
        mode: str = "append",  # "delta", "overwrite"
        num_rows: int | None = 100_000_000,
        row_group_size: int | None = None,
        compression: str = "zstd",
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        auto_optimize_dtypes: bool = True,
        delta_subset: str | list[str] | None = None,
        partitioning_columns: str | list[str] | None = None,
        **kwargs,
    ):
        """
        Write a DataFrame to the dataset.

        Args:
            df: A DataFrame to write to the dataset. Can be a polars DataFrame, Arrow Table, Pandas DataFrame, or DuckDBPyConnection.
            mode: The write mode. Can be "append", "delta", or "overwrite".
            num_rows: The number of rows per partition.
            row_group_size: The size of each row group.
            compression: The compression algorithm to use.
            sort_by: The column(s) to sort by.
            distinct: Whether to write only distinct rows.
            delta_subset: The subset of columns to use for delta updates.
            partitioning_columns: The column(s) to partition by.
            **kwargs: Additional arguments to pass to the write_table function.

        Returns:
            None
        """
        if isinstance(df, pd.DataFrame):
            df = _pl.from_pandas(df)
        elif isinstance(df, pa.Table):
            df = _pl.from_arrow(df)
        elif isinstance(df, _duckdb.DuckDBPyRelation):
            df = df.pl()

        if mode == "overwrite":
            del_files = self._files.copy()

        if self.partitioning_names:
            partitioning_columns = self.partitioning_names.copy()
        if base_name is not None:
            _partitions = [df]
            paths = [base_name.split(".")[0] + ".parquet"]
        else:
            _partitions = partition_by(
                df=df, columns=partitioning_columns, num_rows=num_rows
            )
            paths = [
                os.path.join(
                    self._path,
                    "/".join(
                        (
                            "=".join([k, str(v).lstrip("0")])
                            for k, v in partition[0].items()
                            if k != "row_nr"
                        )
                    ),
                    f"data-{dt.datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]}-{uuid.uuid4().hex[:16]}.parquet",
                )
                for partition in _partitions
            ]
        schema = self.file_schema if self.has_files else None
        partitions = [partition[1] for partition in _partitions]
        file_metadata = []

        for _df, path in zip(partitions, paths):
            if mode == "delta" and self.has_files:
                _df = self._gen_delta_df(df=_df, delta_subset=delta_subset)

            if _df.shape[0]:
                if isinstance(_df, _pl.LazyFrame):
                    _df = _df.collect()
                metadata = write_table(
                    df=_df,
                    path=path,
                    schema=schema,
                    filesystem=self._filesystem,
                    row_group_size=row_group_size,
                    compression=compression,
                    sort_by=sort_by,
                    distinct=distinct,
                    auto_optimize_dtypes=auto_optimize_dtypes,
                    **kwargs,
                )
                file_metadata.append(metadata)

        if len(file_metadata):
            file_metadata = dict(file_metadata)
            for f in file_metadata:
                file_metadata[f].set_file_path(f.split(self._path)[-1].lstrip("/"))

            if hasattr(self, "file_metadata"):
                self.file_metadata.update(file_metadata)
            else:
                self.file_metadata = file_metadata

            try:
                self.load_metadata()
            except:
                self.load_metadata(reload=True)
            self.update_metadata_table()

        if mode == "overwrite":
            self.delete_files(del_files)

        self.load_metadata()
        self.update_metadata_table()
        self.clear_cache()

    def _optimize_by_file_size(
        self,
        target_size: str | int,
        strict: bool = False,
        sort_by_timestamp: bool | str = True,
        filter_expr: str | None = None,
        lazy: bool = True,
        allow_smaller: bool = False,
        **kwargs,
    ):
        """
        Optimize the dataset by file size.

        Args:
            target_size (str | int): The target size of the dataset in bytes or a human-readable string (e.g. '1GB').
            strict (bool, optional): Whether to strictly enforce the target size. Defaults to False.
            sort_by_timestamp (bool | str, optional): Whether to sort the dataset by timestamp. If True, the dataset will be sorted by the first timestamp column. If a string is provided, the dataset will be sorted by the specified column. Defaults to True.
            filter_expr (str | None, optional): An optional filter expression to apply to the dataset before optimizing. Defaults to None.
            lazy (bool, optional): Whether to lazily load the dataset. Defaults to True.
            allow_smaller (bool, optional): Whether to allow the dataset to be smaller than the target size. Defaults to False.
            **kwargs: Additional keyword arguments to pass to `write_to_dataset`.

        Raises:
            ValueError: If `target_size` is not a valid size string.

        Returns:
            None
        """
        if filter_expr is not None:
            self.scan_file_catalog(filter_expr=filter_expr)
            file_catalog = self.file_catalog.filter(
                _pl.col("file_path").is_in(
                    [os.path.basename(f) for f in self._scan_files]
                )
            )

        else:
            file_catalog = self.file_catalog

        if sort_by_timestamp:
            if len(self._timestamp_columns):
                file_catalog.sort(
                    [
                        self._timestamp_columns[0] + "_min",
                        self._timestamp_columns[0] + "_max",
                    ]
                )
            else:
                file_catalog.sort(
                    [sort_by_timestamp + "_min", sort_by_timestamp + "_max"]
                )

        if isinstance(target_size, str):
            target_byte_size = humanized_size_to_bytes(target_size)
        elif isinstance(target_size, int):
            target_byte_size = target_size
        else:
            raise ValueError("Invalid target size")

        del_files = []

        if not strict:
            file_groups = file_catalog.with_columns(
                (_pl.col("total_byte_size").cumsum() // target_byte_size).alias("group")
            ).select(["file_path", "total_byte_size", "group"])

            for file_group in tqdm.tqdm(file_groups.partition_by("group")):
                if (
                    file_group.shape[0] == 1
                    and file_group["total_byte_size"].sum() > target_byte_size
                ):
                    continue
                paths = [
                    os.path.join(self._path, f)
                    for f in file_group["file_path"].to_list()
                ]
                del_files.extend(paths)
                if lazy:
                    df = _pl.scan_pyarrow_dataset(
                        pds.dataset(
                            paths,
                            filesystem=self._filesystem,
                            partitioning=self._partitioning,
                        )
                    )
                else:
                    df = _pl.from_arrow(
                        read_table(
                            paths, self._filesystem, partitioning=self._partitioning
                        )
                    )

                self.write_to_dataset(df=df, mode="append", **kwargs)
            # print(del_files)
            self.delete_files(del_files)

        else:
            target_num_rows = int(
                target_byte_size
                / self.file_catalog.with_columns(
                    (_pl.col("total_byte_size") / _pl.col("num_rows")).alias(
                        "row_byte_size"
                    )
                )["row_byte_size"].mean()
            )

            self._optimize_by_num_rows(
                target_num_rows=target_num_rows,
                strict=True,
                sort_by_timestamp=sort_by_timestamp,
                filter_expr=filter_expr,
                lazy=lazy,
                allow_smaller=allow_smaller**kwargs,
            )

    def _optimize_num_rows(
        self,
        target_num_rows: int,
        strict: bool = False,
        sort_by_timestamp: bool | str = True,
        filter_expr: str | None = None,
        lazy: bool = True,
        allow_smaller: bool = False,
        **kwargs,
    ):
        """
        Optimize the number of rows in the dataset by appending data from files with fewer rows
        to files with more rows, until the target number of rows is reached.

        Args:
            target_num_rows (int): The target number of rows for the dataset.
            strict (bool, optional): If True, only files with exactly target_num_rows rows will be used.
                Defaults to False.
            sort_by_timestamp (bool | str, optional): If True, sort files by timestamp.
                If str, sort files by the specified column. Defaults to True.
            filter_expr (str | None, optional): A filter expression to apply to the dataset before optimizing.
                Defaults to None.
            lazy (bool, optional): If True, use lazy loading when reading files.
                If False, load all files into memory at once. Defaults to True.
            allow_smaller (bool, optional): If True, allow files with fewer rows than target_num_rows to be used.
                Defaults to False.
            **kwargs: Additional keyword arguments to pass to write_to_dataset().

        Returns:
            None
        """
        if filter_expr is not None:
            self.scan_file_catalog(filter_expr=filter_expr)
            file_catalog = self.file_catalog.filter(
                _pl.col("file_path").is_in(
                    [os.path.basename(f) for f in self._scan_files]
                )
            )
        else:
            file_catalog = self.file_catalog

        if sort_by_timestamp:
            if len(self._timestamp_columns):
                file_catalog.sort(
                    [
                        self._timestamp_columns[0] + "_min",
                        self._timestamp_columns[0] + "_max",
                    ]
                )
            else:
                file_catalog.sort(
                    [sort_by_timestamp + "_min", sort_by_timestamp + "_max"]
                )

        del_files = []

        if not strict:
            file_groups = file_catalog.with_columns(
                (_pl.col("num_rows").cumsum() // target_num_rows).alias("group")
            ).select(["file_path", "num_rows", "group"])

            for file_group in tqdm.tqdm(file_groups.partition_by("group")):
                if not allow_smaller and file_group["num_rows"].sum() > target_num_rows:
                    continue
                paths = [
                    os.path.join(self._path, f)
                    for f in file_group["file_path"].to_list()
                ]
                del_files.extend(paths)
                if lazy:
                    df = _pl.scan_pyarrow_dataset(
                        pds.dataset(
                            paths,
                            filesystem=self._filesystem,
                            partitioning=self._partitioning,
                        )
                    )
                else:
                    df = _pl.from_arrow(
                        read_table(
                            paths, self._filesystem, partitioning=self._partitioning
                        )
                    )

                self.write_to_dataset(df=df, mode="append", **kwargs)

        else:
            file_catalog = file_catalog.filter(_pl.col("num_rows") != target_num_rows)
            paths = [
                os.path.join(self._path, f) for f in file_catalog["file_path"].to_list()
            ]
            del_files.extend(paths)
            if lazy:
                df = _pl.scan_pyarrow_dataset(
                    pds.dataset(
                        paths,
                        filesystem=self._filesystem,
                        partitioning=self._partitioning,
                    )
                )
            else:
                df = _pl.from_arrow(
                    read_table(paths, self._filesystem, partitioning=self._partitioning)
                )

            self.write_to_dataset(
                df=df, mode="append", num_rows=target_num_rows, **kwargs
            )

        # print(del_files)
        self.delete_files(del_files)

    def optimize(
        self,
        target_size: str | int | None = None,
        target_num_rows: int | None = None,
        strict: bool = False,
        sort_by_timestamp: bool | str = True,
        filter_expr: str | None = None,
        lazy: bool = True,
        allow_smaller: bool = False,
        **kwargs,
    ):
        """
        Optimize the dataset by either target file size or target number of rows.

        Args:
            target_size (str | int | None): The target file size in bytes or a string with a suffix (e.g. '10MB').
            target_num_rows (int | None): The target number of rows.
            strict (bool): If True, raise an exception if the target size or number of rows cannot be reached.
            sort_by_timestamp (bool | str): If True, sort files by timestamp before optimizing.
                If 'asc', sort files by timestamp in ascending order.
                If 'desc', sort files by timestamp in descending order.
            filter_expr (str | None): A filter expression to apply to the dataset before optimizing.
            lazy (bool): If True, only load the metadata of the files, not the actual data.
            allow_smaller (bool): If True, allow the resulting file to be smaller than the target size.

        Returns:
            None
        """
        if target_size is not None:
            self._optimize_by_file_size(
                target_size=target_size,
                strict=strict,
                sort_by_timestamp=sort_by_timestamp,
                filter_expr=filter_expr,
                lazy=lazy,
                allow_smaller=allow_smaller,
                **kwargs,
            )
        elif target_num_rows is not None:
            self._optimize_num_rows(
                target_num_rows=target_num_rows,
                strict=strict,
                sort_by_timestamp=sort_by_timestamp,
                filter_expr=filter_expr,
                lazy=lazy,
                allow_smaller=allow_smaller,
                **kwargs,
            )

    # def zorder()

import os

import duckdb as _duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
from fsspec import AbstractFileSystem

from .helpers.io import Writer
from .helpers.misc import get_timestamp_column, str2pyarrow_filter
from .helpers.polars_ext import pl as _pl
from .metadata import ParquetDatasetMetadata, PydalaDatasetMetadata
from .table import PydalaTable

# from .optimize import Optimize


class ParquetDataset(ParquetDatasetMetadata):
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

        super().__init__(
            path=path,
            filesystem=filesystem,
            bucket=bucket,
            cached=cached,
            **caching_options,
        )
        if name is None:
            self.name = os.path.basename(path)
        else:
            self.name = name

        if ddb_con is None:
            ddb_con = _duckdb.connect()

        self.ddb_con = ddb_con
        self._timestamp_column = timestamp_column

        if self.has_files:
            if partitioning == "ignore":
                self._partitioning = None
            elif partitioning is None and "=" in self._files[0]:
                self._partitioning = "hive"
            else:
                self._partitioning = partitioning

        else:
            self._partitioning = partitioning

        if self.has_metadata:
            self.pydala_dataset_metadata = PydalaDatasetMetadata(
                metadata=self.metadata, partitioning=partitioning, ddb_con=ddb_con
            )
            self.metadata_table.create_view(f"{self.name}_metadata")

        try:
            self.load()
        except Exception as e:
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

            self._arrow_parquet_dataset = pds.parquet_dataset(
                self._metadata_file,
                partitioning=self._partitioning,
                filesystem=self._filesystem,
            )

            self.table = PydalaTable(
                result=self._arrow_parquet_dataset, ddb_con=self.ddb_con
            )

            self.ddb_con.register(f"{self.name}_dataset", self._arrow_parquet_dataset)
            self.ddb_con.register("arrow_parquet_dataset", self._arrow_parquet_dataset)

            if self._timestamp_column is None:
                self._timestamp_columns = get_timestamp_column(self.pl.head(1))
                if len(self._timestamp_columns) > 1:
                    self._timestamp_column = self._timestamp_columns[0]

    @property
    def arrow_parquet_dataset(self) -> pds.FileSystemDataset:
        """
        Return the arrow parquet dataset if it exists, otherwise load the dataset and return it.

        Returns:
            pds.FileSystemDataset: The arrow parquet dataset.
        """
        if self.has_files:
            if not hasattr(self, "_arrow_parquet_dataset"):
                self.load()
            return self._arrow_parquet_dataset

    @property
    def is_loaded(self) -> bool:
        """
        Returns True if the dataset has been loaded into memory, False otherwise.
        """
        return hasattr(self, "arrow_parquet_dataset")

    @property
    def columns(self) -> list:
        """
        Returns a list of column names for the dataset, including both schema and partitioning columns.

        Returns:
            list: A list of column names.
        """
        if self.has_files:
            return self.schema.names + self.partition_names

    @property
    def count_rows(self) -> int:
        """
        Returns the number of rows in the dataset.

        If the dataset is not loaded, it prints an error message and returns 0.

        Returns:
            int: The number of rows in the dataset.
        """
        if self.is_loaded:
            return self.metadata.num_rows
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
        if self.is_loaded:
            return self.metadata.num_rows
        else:
            # print(f"No dataset loaded yet. Run {self}.load()")
            return 0

    @property
    def num_columns(self) -> int:
        """
        Returns the number of columns in the dataset.

        If the dataset is not loaded, it prints an error message and returns 0.

        Returns:
            int: The number of columns in the dataset.
        """
        if self.is_loaded:
            return self.metadata.num_columns
        else:
            # print(f"No dataset loaded yet. Run {self}.load()")
            return 0

    @property
    def schema(self) -> pa.Schema:
        """
        Returns the schema of the data source.

        Returns:
            pyarrow.Schema: A pyarrow.Schema object representing the schema of the data source.
        """
        if self.has_files:
            if not hasattr(self, "_schema"):
                # if self._partitioning is not None and self._partitioning!="ignore":
                self._schema = pa.unify_schemas(
                    [self.file_schema, self.partitioning_schema]
                )
            return self._schema

    @property
    def partitioning_schema(self) -> pa.Schema:
        """
        Returns the partitioning schema for the dataset.

        If the dataset has not been loaded yet, an empty schema is returned.

        Returns:
            pa.Schema: The partitioning schema for the dataset.
        """
        if self.has_files:
            if not hasattr(self, "_partitioning_schema"):
                if self.is_loaded:
                    self._partitioning_schema = (
                        self.arrow_parquet_dataset.partitioning.schema
                    )
                else:
                    # print(f"No dataset loaded yet. Run {self}.load()")
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
                    self._partition_names = self.partitioning_schema.names
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
                            self.partition_names,
                            [
                                pa_list.to_pylist()
                                for pa_list in self.arrow_parquet_dataset.partitioning.dictionaries
                            ],
                        )
                    )
            else:
                # print(f"No dataset loaded yet. Run {self}.load()")
                return []
            return self._partition_values

    @property
    def partitions(self) -> dict:
        """
        Returns the partitions if the file is loaded and partitioning is enabled, otherwise returns an empty dictionary.
        """

        if self.has_files:
            if self._partitioning:
                if not hasattr(self, "_partitions"):
                    if self.is_loaded:
                        self._partitions = sorted(
                            (self.metadata_table.select(",".join(self.partition_names)))
                            .distinct()
                            .fetchall()
                        )

                    else:
                        return {}
                return self._partitions

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

    def to_arrow_dataset(
        self,
    ) -> pds.Dataset:
        """
        Converts the current object to an Arrow dataset.

        Returns:
            pds.Dataset: The converted Arrow dataset.

        """

        self._arrow_dataset = self.table.to_arrow_dataset()
        self.ddb_con.register("arrow_dataset", self._arrow_dataset)
        return self._arrow_dataset

    @property
    def arrow_dataset(self) -> pds.Dataset:
        """
        Generates a `pds.Dataset` representation of the object.

        Returns:
            pds.Dataset: The `pds.Dataset` representation of the object.
        """

        if not hasattr(self, "_arrow_dataset"):
            self.to_arrow_dataset()
        return self._arrow_dataset

    def to_arrow(
        self,
        columns: str | list[str] | None = None,
        batch_size: int = 100_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        **kwargs,
    ) -> pa.Table:
        """
        Convert the table to Apache Arrow format.

        Args:
            columns (str | list[str] | None): The columns to include in the conversion. If None, all columns are included.
            batch_size (int): The number of rows to include in each converted batch.
            sort_by (str | list[str] | list[tuple[str, str]] | None): The column(s) to sort by. If None, no sorting is applied.
            distinct (bool): Whether to return only distinct rows.
            **kwargs: Additional keyword arguments.

        Returns:
            pa.Table: The table in Apache Arrow format.
        """

        self._arrow_table = self.table.to_arrow(
            columns=columns,
            batch_size=batch_size,
            sort_by=sort_by,
            distinct=distinct,
            **kwargs,
        )
        self.ddb_con.register("arrow_table", self._arrow_table)
        return self._arrow_table

    # @property
    def arrow_table(
        self,
        columns: str | list[str] | None = None,
        batch_size: int = 100_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        **kwargs,
    ) -> pa.Table:
        """
        Returns the table in Apache Arrow format.

        Args:
            columns (str | list[str] | None): The columns to include in the conversion. If None, all columns are included.
            batch_size (int): The number of rows to include in each converted batch.
            sort_by (str | list[str] | list[tuple[str, str]] | None): The column(s) to sort by. If None, no sorting is applied.
            distinct (bool): Whether to return only distinct rows.
            **kwargs: Additional keyword arguments.

        Returns:
            pa.Table: The table in Apache Arrow format.
        """
        return self.to_arrow(
            columns=columns,
            batch_size=batch_size,
            sort_by=sort_by,
            distinct=distinct,
            **kwargs,
        )

    @property
    def arrow(self):
        """
        Returns the table in Apache Arrow format.

        Returns:
            pa.Table: The table in Apache Arrow format.
        """
        if not hasattr(self, "_arrow_table"):
            self.to_arrow()
        return self._arrow_table

    def to_batch_reader(
        self,
        columns: str | list[str] | None = None,
        batch_size: int = 100_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        **kwargs,
    ) -> pa.RecordBatchReader:
        """
        Converts the dataset to a RecordBatch reader that can be used to read data in batches.

        Args:
            columns (str | list[str] | None, optional): The columns to include in the batch reader. Defaults to None.
            batch_size (int, optional): The size of each batch. Defaults to 100_000.
            sort_by (str | list[str] | list[tuple[str, str]] | None, optional): The column(s) to sort the data by. Defaults to None.
            distinct (bool, optional): Whether to return distinct rows. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            pa.RecordBatchReader: The batch reader object.
        """

        return self.table.to_batch_reader(
            columns=columns,
            batch_size=batch_size,
            sort_by=sort_by,
            distinct=distinct,
            **kwargs,
        )

    def to_batches(
        self,
        columns: str | list[str] | None = None,
        batch_size: int = 100_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        **kwargs,
    ) -> pa.RecordBatch:
        """
        Converts the dataset into RecordBatches.

        Args:
            columns (str | list[str] | None, optional): Columns to include in the batches. Defaults to None.
            batch_size (int, optional): Size of each batch. Defaults to 100_000.
            sort_by (str | list[str] | list[tuple[str, str]] | None, optional): Columns to sort the batches by. Defaults to None.
            distinct (bool, optional): Whether to return distinct batches. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            pa.RecordBatch: The batches of records.
        """

        return self.table.to_batches(
            columns=columns,
            batch_size=batch_size,
            sort_by=sort_by,
            distinct=distinct,
            **kwargs,
        )

    def to_duckdb(
        self,
        lazy: bool = True,
        columns: str | list[str] | None = None,
        batch_size: int = 100_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        **kwargs,
    ) -> _duckdb.DuckDBPyRelation:
        """
        A method to convert the current table to a DuckDBPyRelation.

        Args:
            lazy (bool): Whether to load data lazily.
            columns (str | list[str] | None): Columns to include in the conversion.
            batch_size (int): Size of each batch for conversion.
            sort_by (str | list[str] | list[tuple[str, str]] | None): Columns to sort by.
            distinct (bool): Whether to select only distinct rows.
            **kwargs: Additional keyword arguments.

        Returns:
            _duckdb.DuckDBPyRelation: The DuckDBPyRelation representing the converted table.
        """
        return self.table.to_duckdb(
            lazy=lazy,
            columns=columns,
            batch_size=batch_size,
            sort_by=sort_by,
            distinct=distinct,
            **kwargs,
        )

    def duckdb(
        self,
        lazy: bool = True,
        columns: str | list[str] | None = None,
        batch_size: int = 100_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        **kwargs,
    ) -> _duckdb.DuckDBPyRelation:
        """
        A method to convert the current table to a DuckDBPyRelation.

        Args:
            lazy (bool): Whether to load data lazily.
            columns (str | list[str] | None): Columns to include in the conversion.
            batch_size (int): Size of each batch for conversion.
            sort_by (str | list[str] | list[tuple[str, str]] | None): Columns to sort by.
            distinct (bool): Whether to select only distinct rows.
            **kwargs: Additional keyword arguments.

        Returns:
            _duckdb.DuckDBPyRelation: The DuckDBPyRelation representing the converted table.
        """
        return self.to_duckdb(
            lazy=lazy,
            columns=columns,
            batch_size=batch_size,
            sort_by=sort_by,
            distinct=distinct,
            **kwargs,
        )

    @property
    def ddb(self) -> _duckdb.DuckDBPyRelation:
        """
        Returns the DuckDBPyRelation object.

        Returns:
            _duckdb.DuckDBPyRelation: The DuckDBPyRelation object.
        """
        if not hasattr(self, "_ddb"):
            self._ddb = self.to_duckdb()
        return self._ddb

    def to_polars(
        self,
        lazy: bool = True,
        columns: str | list[str] | None = None,
        batch_size: int = 100_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        **kwargs,
    ) -> _pl.DataFrame:
        """
        Convert the table to a Polars DataFrame with specified args.

        Args:
            lazy (bool): Whether to do lazy evaluation or not.
            columns (Union[str, List[str], None]): The columns to include in the result.
            batch_size (int): The size of each batch to process.
            sort_by (Union[str, List[str], List[Tuple[str, str]], None]): The column(s) to sort by.
            distinct (bool): Whether to return distinct rows or not.
            **kwargs: Additional keyword arguments.

        Returns:
            _pl.DataFrame: A Polars DataFrame.
        """
        return self.table.to_polars(
            lazy=lazy,
            columns=columns,
            batch_size=batch_size,
            sort_by=sort_by,
            distinct=distinct,
            **kwargs,
        )

    def to_pl(
        self,
        lazy: bool = True,
        columns: str | list[str] | None = None,
        batch_size: int = 100_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        **kwargs,
    ) -> _pl.DataFrame:
        """
        Convert the table to a Polars DataFrame with specified args.

        Args:
            lazy (bool): Whether to do lazy evaluation or not.
            columns (Union[str, List[str], None]): The columns to include in the result.
            batch_size (int): The size of each batch to process.
            sort_by (Union[str, List[str], List[Tuple[str, str]], None]): The column(s) to sort by.
            distinct (bool): Whether to return distinct rows or not.
            **kwargs: Additional keyword arguments.

        Returns:
            _pl.DataFrame: A Polars DataFrame.
        """

        return self.to_polars(
            lazy=lazy,
            columns=columns,
            batch_size=batch_size,
            sort_by=sort_by,
            distinct=distinct,
            **kwargs,
        )

    @property
    def pl(self) -> _pl.DataFrame:
        """
        Returns the Polars DataFrame representation of the dataset.

        If the dataset has not been converted to Polars DataFrame yet, it will be converted and cached.

        Returns:
            _pl.DataFrame: The Polars DataFrame representation of the dataset.
        """

        if not hasattr(self, "_pl"):
            return self.to_polars()
        return self._pl

    def to_pandas(
        self,
        lazy: bool = True,
        columns: str | list[str] | None = None,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Converts the current table to a pandas DataFrame.

        Args:
            lazy (bool, optional): Whether to lazily load the DataFrame. Defaults to True.
            columns (str | list[str] | None, optional): The columns to include in the DataFrame. Defaults to None.
            sort_by (str | list[str] | list[tuple[str, str]] | None, optional): The column(s) to sort by. Defaults to None.
            distinct (bool, optional): Whether to return only distinct rows. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the underlying to_pandas method.

        Returns:
            pd.DataFrame: The pandas DataFrame representation of the table.
        """

        return self.table.to_pandas(
            lazy=lazy, columns=columns, sort_by=sort_by, distinct=distinct, **kwargs
        )

    def to_df(
        self,
        lazy: bool = True,
        columns: str | list[str] | None = None,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Converts the current table to a pandas DataFrame.

        Args:
            lazy (bool, optional): Whether to lazily load the DataFrame. Defaults to True.
            columns (str | list[str] | None, optional): The columns to include in the DataFrame. Defaults to None.
            sort_by (str | list[str] | list[tuple[str, str]] | None, optional): The column(s) to sort by. Defaults to None.
            distinct (bool, optional): Whether to return only distinct rows. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the underlying to_pandas method.

        Returns:
            pd.DataFrame: The pandas DataFrame representation of the table.
        """

        return self.to_pandas(
            lazy=lazy, columns=columns, sort_by=sort_by, distinct=distinct, **kwargs
        )

    @property
    def df(self) -> pd.DataFrame:
        """
        Returns the pandas DataFrame representation of the dataset.

        If the dataset has not been converted to pandas DataFrame yet, it will be converted and cached.

        Returns:
            pd.DataFrame: The pandas DataFrame representation of the dataset.
        """

        if not hasattr(self, "_df"):
            self._df = self.to_pandas()
        return self._df

    def __repr__(self):
        return self.table.__repr__()

    def sql(self, sql: str) -> _duckdb.DuckDBPyRelation:
        """
        Executes an SQL query on the DuckDBPyRelation object.

        Args:
            sql (str): The SQL query to be executed.

        Returns:
            DuckDBPyRelation: The result of the SQL query execution.
        """

        return self.ddb_con.sql(sql)

    def _filter_duckdb(self, filter_expr: str) -> _duckdb.DuckDBPyRelation:
        """
        Filter the DuckDBPyRelation based on a given filter expression.

        Args:
            filter_expr (str): The filter expression to apply.

        Returns:
            _duckdb.DuckDBPyRelation: The filtered DuckDBPyRelation object.
        """
        return self.ddb.filter(filter_expr)

    def _filter_arrow_dataset(self, filter_expr: str | pds.Expression) -> pds.Dataset:
        """
        Filters the pyarrow dataset based on a filter expression.

        Args:
            filter_expr (str | pds.Expression): The filter expression to apply. It can be either a string or a
                pyarrow expression.

        Returns:
            pds.Dataset: The filtered pyarrow dataset.
        """
        if self.has_files:
            if isinstance(filter_expr, str):
                filter_expr = str2pyarrow_filter(filter_expr, self.schema)
            return self.arrow_dataset.filter(filter_expr)

    def _filter_arrow_parquet_dataset(
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
        if self.has_files:
            if isinstance(filter_expr, str):
                filter_expr = str2pyarrow_filter(filter_expr, self.schema)
            return self.arrow_parquet_dataset.filter(filter_expr)

    def filter(
        self,
        filter_expr: str | pds.Expression,
        use: str = "auto",
        on: str = "auto",
        # return_type: str | None = None,
        # lazy: bool = True,
    ) -> pds.FileSystemDataset | pds.Dataset | _duckdb.DuckDBPyRelation:
        """
        Filters the dataset based on the given filter expression.

        Args:
            filter_expr (str | pds.Expression): The filter expression to apply.
            use (str): The engine to use for filtering. Default is "auto".
            on (str): The type of dataset to filter. Default is "auto".

        Returns:
            pds.FileSystemDataset | pds.Dataset | _duckdb.DuckDBPyRelation: The filtered dataset.

        Note:
            If filtering with PyArrow fails, DuckDB will be used as a fallback.


        """
        if any([s in filter_expr for s in ["%", "like", "similar to", "*"]]):
            use = "duckdb"

        if use == "auto":
            try:
                if on == "auto":
                    if sorted(self.files) == sorted(self.scan_files):
                        res = self._filter_arrow_dataset(filter_expr)
                    else:
                        res = self._filter_arrow_dataset(filter_expr)
                elif on == "parquet_dataset":
                    res = self._filter_arrow_parquet_dataset(filter_expr)
                elif on == "dataset":
                    res = self._filter_arrow_dataset(filter_expr)

            except Exception as e:
                e.add_note("Note: Filtering with PyArrow failed. Using DuckDB.")
                print(e)
                if on == "parquet_dataset":
                    self.reset_scan()
                    res = self._filter_duckdb(filter_expr)
                else:
                    res = self._filter_duckdb(filter_expr)

        elif use == "pyarrow":
            if on == "auto":
                if sorted(self.files) == sorted(self.scan_files):
                    res = self._filter_arrow_parquet_dataset(filter_expr)
                else:
                    res = self._filter_arrow_dataset(filter_expr)
            elif on == "parquet_dataset":
                res = self._filter_arrow_parquet_dataset(filter_expr)
            elif on == "dataset":
                res = self._filter_arrow_dataset(filter_expr)

        elif use == "duckdb":
            if on == "parquet_dataset":
                self.reset_scan()
                res = self._filter_duckdb(filter_expr)
            else:
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
            if hasattr(self, "_arrow_parquet_dataset"):
                if len(self._arrow_parquet_dataset.files):
                    self.ddb_con.register(
                        f"{self.name}_dataset", self._arrow_parquet_dataset
                    )

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

    def delete_files(self, files: str | list[str]):
        """
        Deletes the specified files from the dataset.

        Args:
            files (str | list[str] | None, optional): The name(s) of the file(s) to delete. If None,
                all files in the dataset will be deleted. Defaults to None.
        """
        if not self._path in files[0]:
            files = [os.path.join(self._path, fn) for fn in files]
        self._filesystem.rm(files, recursive=True)
        # self.load(reload=True)

    def _get_delta_other_df(
        self,
        df: _pl.DataFrame | _pl.LazyFrame,
        filter_columns: str | list[str] | None = None,
        use: str = "auto",
        on: str = "auto",
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
        for col in filter_columns or df.columns:
            if col in self.columns and df.columns:
                f_max = df.select(_pl.col(col).max())

                if collect:
                    f_max = f_max.collect()[0, 0]
                else:
                    f_max = f_max[0, 0]

                if isinstance(f_max, str):
                    f_max = f_max.strip("'").replace(",", "")

                f_min = df.select(_pl.col(col).min())

                if collect:
                    f_min = f_min.collect()[0, 0]
                else:
                    f_min = f_min[0, 0]

                if isinstance(f_min, str):
                    f_min = f_min.strip("'").replace(",", "")

                filter_expr.append(
                    f"{col}<='{f_max}' AND {col}>='{f_min}'".replace("'None'", "NULL")
                )
        if filter_expr == []:
            return _pl.DataFrame(schema=df.schema)

        return self.filter(" AND ".join(filter_expr), use=use, on=on).to_polars()

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
        other_df_filter_columns: str | list[str] | None = None,
        use: str = "pyarrow",
        on: str = "parquet_dataset",
        update_metadata: bool = False,
        alter_schema: bool = False,
        timestamp_column: str | None = None,
        **kwargs,
    ):
        """
        Write data to the dataset with specified options and handling for different modes and parameters.

        Args:
            df: DataFrame to be written to the dataset. Can be a pandas DataFrame, PyPolars DataFrame, PyArrow Table, PyArrow RecordBatch, DuckDBPyConnection, or LazyFrame.
            mode: Mode of writing to the dataset, either "append", "delta", or "overwrite".
            partitioning_columns: Columns to partition the dataset by. Can be a string, list of strings, or None.
            max_rows_per_file: Maximum number of rows per file, or None.
            row_group_size: Size of row groups, or None.
            compression: Compression algorithm to use, default is "zstd".
            sort_by: Columns to sort the dataset by. Can be a string, list of strings, list of tuples of strings, or None.
            unique: Columns to enforce uniqueness on, can be a bool, string, or list of strings.
            ts_unit: Unit for timestamps, default is "us" (microseconds).
            tz: Timezone to use, or None.
            remove_tz: Whether to remove the timezone, default is False.
            use_large_string: Whether to use large string types, default is False.
            delta_subset: Subset of columns to consider for delta mode, can be a string, list of strings, or None.
            other_df_filter_columns: Columns to filter the other DataFrame by, can be a string, list of strings, or None.
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
                filter_columns=other_df_filter_columns,
                use=use,
                on=on,
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


#
# ParquetDataset.compact_partitions = Optimize.compact_partitions
# ParquetDataset._compact_partition = Optimize._compact_partition
# ParquetDataset.compact_by_timeperiod = Optimize.compact_by_timeperiod
# ParquetDataset._compact_by_timeperiod = Optimize._compact_by_timeperiod
# ParquetDataset.compact_by_rows = Optimize.compact_by_rows
# ParquetDataset.repartition = Optimize.repartition

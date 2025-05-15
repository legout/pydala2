import datetime as dt
import posixpath
import time
import uuid

import duckdb
import pandas as pd
import polars.selectors as cs
import pyarrow as pa
import pyarrow.dataset as pds

# import pyarrow.dataset as pds
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from fsspec import filesystem as fsspec_filesystem
from loguru import logger

from .filesystem import clear_cache
from .helpers.datetime import get_timestamp_column
from .helpers.polars import pl
from .schema import convert_large_types_to_normal, convert_timestamp, replace_schema
from .table import PydalaTable


def write_table(
    table: pa.Table,
    path: str,
    filesystem: AbstractFileSystem | None = None,
    row_group_size: int | None = None,
    compression: str = "zstd",
    **kwargs,
) -> tuple[str, pq.FileMetaData]:
    """
    Writes a PyArrow table to Parquet format.

    Args:
        table (pa.Table): The PyArrow table to write.
        path (str): The path to write the Parquet file to.
        filesystem (AbstractFileSystem | None, optional): The filesystem to use for writing the file. Defaults to None.
        row_group_size (int | None, optional): The size of each row group in the Parquet file. Defaults to None.
        compression (str, optional): The compression algorithm to use. Defaults to "zstd".
        **kwargs: Additional keyword arguments to pass to `pq.write_table`.

    Returns:
        tuple[str, pq.FileMetaData]: A tuple containing the file path and the metadata of the written Parquet file.
    """
    if not filesystem.exists(posixpath.dirname(path)):
        try:
            filesystem.makedirs(posixpath.dirname(path), exist_ok=True)
        except Exception:
            pass

    if filesystem is None:
        filesystem = fsspec_filesystem("file")

    filesystem.invalidate_cache()

    metadata = []
    pq.write_table(
        table,
        path,
        filesystem=filesystem,
        row_group_size=row_group_size,
        compression=compression,
        metadata_collector=metadata,
        allow_truncated_timestamps=True,
        **kwargs,
    )
    metadata = metadata[0]
    # metadata.set_file_path(path)
    return path, metadata


class Writer:
    def __init__(
        self,
        data: (
            pa.Table
            | pa.RecordBatch
            | pl.DataFrame
            | pl.LazyFrame
            | pd.DataFrame
            | duckdb.DuckDBPyRelation
        ),
        path: str,
        schema: pa.Schema | None,
        filesystem: AbstractFileSystem | None = None,
    ):
        """
        Initialize the object with the given data, path, schema, and filesystem.

        Parameters:
            data (pa.Table | pl.DataFrame | pl.LazyFrame | pd.DataFrame | duckdb.DuckDBPyRelation):
                The input data, which can be one of the following types: pa.Table, pl.DataFrame, pl.LazyFrame,
                pd.DataFrame, duckdb.DuckDBPyRelation.
            path (str): The path of the data.
            schema (pa.Schema | None): The schema of the data, if available.
            filesystem (AbstractFileSystem | None, optional): The filesystem to use, defaults to None.

        Returns:
            None
        """
        self.schema = schema
        self.data = (
            data
            if not isinstance(data, pa.RecordBatch)
            else pa.Table.from_batches([data])
        )
        self.base_path = path
        self.path = None
        self._filesystem = filesystem

    def _to_polars(self):
        """
        Convert the data attribute to a Polars DataFrame.

        This function checks the type of the data attribute and converts it to a Polars DataFrame if it is not
        already one.
        It supports conversion from Arrow tables, Pandas DataFrames, and DuckDBPyRelations.

        """
        if isinstance(self.data, pa.Table):
            self.data = pl.from_arrow(self.data)
        elif isinstance(self.data, pd.DataFrame):
            self.data = pl.from_pandas(self.data)
        elif isinstance(self.data, duckdb.DuckDBPyRelation):
            self.data = self.data.pl()

    def _to_arrow(self):
        """
        Convert the data in the DataFrame to Arrow format.

        This method checks the type of the data and converts it to Arrow format accordingly.
        It supports conversion from Polars DataFrames, Polars LazyFrames, Pandas DataFrames, and DuckDBPyRelations.
        """
        if isinstance(self.data, pl.DataFrame):
            self.data = self.data.to_arrow()
        elif isinstance(self.data, pl.LazyFrame):
            self.data = self.data.collect().to_arrow()
        elif isinstance(self.data, pd.DataFrame):
            self.data = pa.Table.from_pandas(self.data)
        elif isinstance(self.data, duckdb.DuckDBPyRelation):
            self.data = self.data.arrow()

    def _set_schema(self):
        """
        Sets the schema of the DataFrame.

        This private method is called internally to set the schema of the DataFrame. It first converts the DataFrame
        to an Arrow table using the `_to_arrow()` method. Then, it checks if a schema has already been set for the
        DataFrame.
        If not, it assigns the schema of the DataFrame's underlying data to the `schema` attribute.

        """
        self._to_arrow()
        self.schema = self.schema or self.data.schema

    def sort_data(self, by: str | list[str] | list[tuple[str, str]] | None = None):
        """
        Sorts the data in the PydalaTable object based on the specified column(s).

        Args:
            by (str | list[str] | list[tuple[str, str]] | None): The column(s) to sort by.
                If a single string is provided, the data will be sorted in ascending order based on that column.
                If a list of strings is provided, the data will be sorted in ascending order based on each
                    column in the list.
                If a list of tuples is provided, each tuple should contain a column name and a sort order
                    ("ascending" or "descending").
                If None is provided, the data will not be sorted.

        Returns:
            None
        """
        if by is not None:
            self._to_arrow()
            by = PydalaTable._get_sort_by(by, type_="pyarrow")
            self.data = self.data.sort_by(**by)

    def unique(self, columns: bool | str | list[str] = False):
        """
        Generates a unique subset of the DataFrame based on the specified columns.

        Args:
            columns (bool | str | list[str], optional): The columns to use for determining uniqueness.
                If set to False, uniqueness is determined based on all columns.
                If set to a string, uniqueness is determined based on the specified column.
                If set to a list of strings, uniqueness is determined based on the specified columns.
                Defaults to False.

        """
        if columns is not None:
            self._to_polars()
            self.data = self.data.with_columns(cs.by_dtype(pl.Null()).cast(pl.Int64()))
            if isinstance(columns, bool):
                columns = None
            self.data = self.data.unique(columns, maintain_order=True)

    def add_datepart_columns(
        self, columns: list[str], timestamp_column: str | None = None
    ):
        """
        Adds datepart columns to the data.

        Args:
            timestamp_column (str): The name of the timestamp column.
            columns (list[str]): Date part columns to add. The available options are: "year",
                "month", "week", "yearday", monthday", "weekday".

        Returns:
            None
        """
        if columns is None:
            columns = []
        if isinstance(columns, str):
            columns = [columns]

        if timestamp_column is None:
            timestamp_column = get_timestamp_column(self.data)
            timestamp_column = timestamp_column[0] if len(timestamp_column) else None

        if timestamp_column is not None:
            self._set_schema()
            self._to_polars()
            datepart_columns = {
                col: True
                for col in self.schema.names + columns
                if col
                in [
                    "year",
                    "month",
                    "week",
                    "yearday",
                    "monthday",
                    "weekday",
                    "day",
                    "hour",
                    "minute",
                ]
            }

            self.data = self.data.with_datepart_columns(
                timestamp_column=timestamp_column, **datepart_columns
            )
            self._to_arrow()
            for col in datepart_columns:
                if col not in self.schema.names:
                    if col == "weekday":
                        self.schema.append(pa.field(col, pa.string()))
                    self.schema = self.schema.append(pa.field(col, pa.int32()))

    def cast_schema(
        self,
        use_large_string: bool = False,
        tz: str = None,
        ts_unit: str = None,
        remove_tz: bool = False,
        alter_schema: bool = False,
    ):
        """
        Casts the schema of the data object based on the specified parameters.

        Args:
            use_large_string (bool, optional): Whether to use large string type. Defaults to False.
            tz (str, optional): Timezone to convert timestamps to. Defaults to None.
            ts_unit (str, optional): Unit to convert timestamps to. Defaults to None.
            remove_tz (bool, optional): Whether to remove timezone from timestamps. Defaults to False.
            alter_schema (bool, optional): Whether to alter the schema. Defaults to False.
        """

        self._set_schema()
        self._use_large_string = use_large_string
        if not use_large_string:
            self.schema = convert_large_types_to_normal(self.schema)

        if tz is not None or ts_unit is not None or remove_tz:
            self.schema = convert_timestamp(
                self.schema,
                tz=tz,
                unit=ts_unit,
                remove_tz=remove_tz,
            )

        self.data = replace_schema(
            self.data,
            self.schema,
            # ts_unit=None,
            # tz=tz
            alter_schema=alter_schema,
        )
        self.schema = self.data.schema

    def delta(
        self,
        other: pl.DataFrame | pl.LazyFrame,
        subset: str | list[str] | None = None,
    ):
        """
        Computes the difference between the current DataFrame and another DataFrame or LazyFrame.

        Parameters:
            other (DataFrame | LazyFrame): The DataFrame or LazyFrame to compute the difference with.
            subset (str | list[str] | None, optional): The column(s) to compute the difference on. If `None`,
                the difference is computed on all columns. Defaults to `None`.

        """
        self._to_polars()
        self.data = self.data.delta(other, subset=subset)

    @property
    def shape(self):
        if self.data is None:
            return 0
        if isinstance(self.data, pl.LazyFrame):
            self.data = self.data.collect()
        return self.data.shape

    def write_to_dataset(
        self,
        row_group_size: int | None = None,
        compression: str = "zstd",
        partitioning_columns: list[str] | None = None,
        partitioning_flavor: str = "hive",
        max_rows_per_file: int | None = None,
        create_dir: bool = False,
        basename: str | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Writes the data to a dataset in the Parquet format.

        Args:
            row_group_size (int | None, optional): The number of rows per row group. Defaults to None.
            compression (str, optional): The compression algorithm to use. Defaults to "zstd".
            partitioning_columns (list[str] | None, optional): The columns to use for partitioning the dataset.
                Defaults to None.
            partitioning_flavor (str, optional): The partitioning flavor to use. Defaults to "hive".
            max_rows_per_file (int | None, optional): The maximum number of rows per file. Defaults to None.
            create_dir (bool, optional): Whether to create directories for the dataset. Defaults to False.
            basename (str | None, optional): The base name for the output files. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self._to_arrow()

        if basename is None:
            basename_template = (
                "data-"
                f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]}-{uuid.uuid4().hex[:16]}-{{i}}.parquet"
            )
        else:
            basename_template = f"{basename}-{{i}}.parquet"

        file_options = pds.ParquetFileFormat().make_write_options(
            compression=compression
        )

        if hasattr(self._filesystem, "fs"):
            if "local" in self._filesystem.fs.protocol:
                create_dir = True
        else:
            if "local" in self._filesystem.protocol:
                create_dir = True

        metadata = []

        def file_visitor(written_file):
            if verbose:
                logger.info(f"path={written_file.path}")
                logger.info(f"size={written_file.size} bytes")
                logger.info(f"metadata={written_file.metadata}")
            # written_file.metadata.set_file_path(written_file.path)
            metadata.append({written_file.path: written_file.metadata})

        retries = 0
        while retries < 2:
            try:
                pds.write_dataset(
                    self.data,
                    base_dir=self.base_path,
                    filesystem=self._filesystem,
                    file_options=file_options,
                    partitioning=partitioning_columns,
                    partitioning_flavor=partitioning_flavor,
                    basename_template=basename_template,
                    min_rows_per_group=row_group_size,
                    max_rows_per_group=row_group_size,
                    max_rows_per_file=max_rows_per_file,
                    existing_data_behavior="overwrite_or_ignore",
                    create_dir=create_dir,
                    format="parquet",
                    file_visitor=file_visitor,
                    **kwargs,
                )
                break
            except Exception as e:
                retries += 1
                if retries == 2:
                    raise e
                self.clear_cache()
                time.sleep(0.1)
                create_dir = False
        return metadata

    def clear_cache(self) -> None:
        """
        Clears the cache for the dataset's filesystem and base filesystem.

        This method clears the cache for the dataset's filesystem and base filesystem,
        which can be useful if the dataset has been modified and the cache needs to be
        updated accordingly.

        Returns:
            None
        """
        if hasattr(self._filesystem, "fs"):
            clear_cache(self._filesystem.fs)
        clear_cache(self._filesystem)

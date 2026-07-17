# Standard library imports
import datetime as dt
import posixpath
import time
import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING

# Third-party imports
import duckdb
import pandas as pd
import polars.selectors as cs
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from fsspec import filesystem as fsspec_filesystem
from fsspec.core import split_protocol
from loguru import logger

# Local imports
from .filesystem import clear_cache
from .helpers.datetime import get_timestamp_column
from .helpers.polars import pl
from .schema import convert_large_types_to_normal, convert_timestamp, replace_schema
from .table import PydalaTable


def strip_protocol(path: str) -> str:
    """Strips the protocol from a given path.

    Args:
        path (str): The input path which may contain a protocol.
    Returns:
        str: The path without the protocol.
    """
    protocol, path = split_protocol(path)
    return path


class PartialWriteError(RuntimeError):
    """Raised when Parquet files were written but dataset completion failed."""

    def __init__(self, message: str, metadata: list[dict[str, pq.FileMetaData]]):
        super().__init__(message)
        self.metadata = metadata


if TYPE_CHECKING:
    from fsspeckit.core.incremental import MergeResult


class PartialMergeError(RuntimeError):
    """Raised when a merge physically succeeded but dataset state refresh failed.

    The successful fsspeckit ``MergeResult`` is preserved both on the
    ``merge_result`` attribute and on ``ParquetDataset.last_merge_result`` so
    callers can recover (e.g. by retrying ``_refresh_after_rewrite``) or
    inspect what fsspeckit wrote.
    """

    def __init__(self, message: str, merge_result: "MergeResult"):
        super().__init__(message)
        self.merge_result = merge_result


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
    path = strip_protocol(path)
    if filesystem is None:
        filesystem = fsspec_filesystem("file")

    if not filesystem.exists(posixpath.dirname(path)):
        try:
            filesystem.makedirs(posixpath.dirname(path), exist_ok=True)
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to create directory {posixpath.dirname(path)}: {e}")
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


WriterSource = (
    pa.Table
    | pa.RecordBatch
    | pl.DataFrame
    | pl.LazyFrame
    | pd.DataFrame
    | duckdb.DuckDBPyRelation
)
"""Type alias for the source families a :class:`Writer` accepts and prepares."""


class Writer:
    def __init__(
        self,
        data: WriterSource,
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
        self.base_path = strip_protocol(path)
        self.path = None
        self._filesystem = filesystem

    def _to_polars(self) -> None:
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

    def _to_arrow(self) -> None:
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
            arrow_data = self.data.arrow()
            # ``DuckDBPyRelation.arrow()`` returns a ``RecordBatchReader`` in
            # recent duckdb releases; normalize to a concrete ``pa.Table`` so
            # the rest of the pipeline (which expects ``column_names`` /
            # ``schema``) keeps working.
            self.data = (
                arrow_data.read_all()
                if isinstance(arrow_data, pa.RecordBatchReader)
                else arrow_data
            )

    def _set_schema(self) -> None:
        """
        Sets the schema of the DataFrame.

        This private method is called internally to set the schema of the DataFrame. It first converts the DataFrame
        to an Arrow table using the `_to_arrow()` method. Then, it checks if a schema has already been set for the
        DataFrame.
        If not, it assigns the schema of the DataFrame's underlying data to the `schema` attribute.

        """
        self._to_arrow()

        self.schema = self.schema or self.data.schema
        self.schema = pa.schema(
            [
                (field.name, field.type)
                for field in self.schema
                if field.name in self.data.column_names
            ]
        )

    def sort_data(
        self, by: str | list[str] | list[tuple[str, str]] | None = None
    ) -> None:
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

    def unique(self, columns: bool | str | list[str] = False) -> None:
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
    ) -> None:
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
        tz: str | None = None,
        ts_unit: str | None = None,
        remove_tz: bool = False,
        alter_schema: bool = False,
    ) -> None:
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
    ) -> None:
        """
        Computes the difference between the current DataFrame and another DataFrame or LazyFrame.

        Parameters:
            other (DataFrame | LazyFrame): The DataFrame or LazyFrame to compute the difference with.
            subset (str | list[str] | None, optional): The column(s) to compute the difference on. If `None`,
                the difference is computed on all columns. Defaults to `None`.

        """
        self._to_polars()
        self.data = self.data.delta(other, subset=subset)

    def prepare(
        self,
        *,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        unique: bool | str | list[str] = False,
        ts_unit: str = "us",
        tz: str | None = None,
        remove_tz: bool = False,
        alter_schema: bool = False,
        partition_by: str | list[str] | None = None,
        timestamp_column: str | None = None,
    ) -> pa.Table:
        """Run every source-normalization step and return the prepared Arrow table.

        This is the reusable transformation seam shared by ordinary writes and
        the (future) merge path. It performs -- in order -- input
        normalization, sorting, optional uniqueness, schema cast/evolution,
        derived date-part partition columns, and final Arrow conversion, then
        returns the resulting ``pyarrow.Table``.

        ``prepare`` performs *no* filesystem writes and *no* cache mutation:
        it only transforms ``self.data`` in place and returns it.

        Args:
            sort_by: Column(s) to sort by; a list of ``[column, order]``
                pairs controls per-column sort direction.
            unique: When truthy, collapse duplicate rows; a string/list keys
                the de-duplication on the given column(s).
            ts_unit: Target timestamp unit. Defaults to ``"us"``.
            tz: Target timezone for timestamp fields.
            remove_tz: Strip timezone information from timestamps.
            alter_schema: When ``False`` (default), enforce the Writer's
                target schema, dropping columns absent from it; when ``True``
                (and ``schema`` is unset), keep every data column.
            partition_by: Date-part column(s) to derive from the timestamp.
            timestamp_column: Timestamp column used for date-part derivation;
                inferred from the data when omitted.

        Returns:
            The fully transformed ``pyarrow.Table``.
        """
        self.sort_data(by=sort_by)
        if unique:
            self.unique(columns=unique)
        self.cast_schema(
            ts_unit=ts_unit,
            tz=tz,
            remove_tz=remove_tz,
            alter_schema=alter_schema,
        )
        if partition_by:
            self.add_datepart_columns(
                columns=partition_by,
                timestamp_column=timestamp_column,
            )
        self._to_arrow()
        return self.data

    @classmethod
    def prepare_many(
        cls,
        sources: Iterable[WriterSource],
        *,
        path: str,
        schema: pa.Schema | None = None,
        filesystem: AbstractFileSystem | None = None,
        **prepare_kwargs,
    ) -> list[pa.Table]:
        """Prepare each source into an Arrow table without writing anything.

        Mirrors how ``ParquetDataset.write_to_dataset`` fans a list of inputs
        out into one ``Writer`` each, but stops short of any write or cache
        side effect. Useful for callers (e.g. merge) that need the prepared
        tables before deciding how to persist them.

        Args:
            sources: Iterable of supported source objects.
            path: Base dataset path forwarded to each per-source ``Writer``.
            schema: Optional target schema forwarded to each ``Writer``.
            filesystem: Optional filesystem forwarded to each ``Writer``.
            **prepare_kwargs: Keyword arguments forwarded to :meth:`prepare`.

        Returns:
            One fully transformed ``pyarrow.Table`` per source, in input order.
        """
        return [
            cls(src, path=path, schema=schema, filesystem=filesystem).prepare(
                **prepare_kwargs
            )
            for src in sources
        ]

    def execute(
        self,
        *,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        unique: bool | str | list[str] = False,
        ts_unit: str = "us",
        tz: str | None = None,
        remove_tz: bool = False,
        alter_schema: bool = False,
        partition_by: str | list[str] | None = None,
        timestamp_column: str | None = None,
        delta_other=None,
        delta_subset: str | list[str] | None = None,
        **write_options,
    ) -> list[dict[str, pq.FileMetaData]]:
        """Execute the complete write plan behind Writer's interface.

        The transformation phase is delegated to :meth:`prepare`; this method
        only adds the write-specific concerns layered on top: the optional
        delta anti-join, the physical parquet write, and cache invalidation.
        """
        # Preserve the historical empty-source short-circuit: an empty input
        # must return before any transformation runs (so e.g. an irrelevant
        # ``sort_by`` cannot raise on an empty source). ``prepare`` itself
        # still prepares empty sources into empty tables when called directly.
        if self.shape[0] == 0:
            return []
        # Transformation phase: prepare normalizes self.data in place
        # (sorting, uniqueness, schema cast, date-part derivation, Arrow
        # conversion). execute then layers write-only concerns on top.
        self.prepare(
            sort_by=sort_by,
            unique=unique,
            ts_unit=ts_unit,
            tz=tz,
            remove_tz=remove_tz,
            alter_schema=alter_schema,
            partition_by=partition_by,
            timestamp_column=timestamp_column,
        )

        if delta_other is not None:
            self._to_polars()
            other = delta_other(self.data, delta_subset)
            if other is not None:
                self.delta(other=other, subset=delta_subset)

        metadata = self.write_to_dataset(
            partitioning_columns=partition_by,
            partitioning_flavor="hive",
            **write_options,
        )
        try:
            self.clear_cache()
        except Exception as exc:
            raise PartialWriteError(
                "Parquet files were written, but cache completion failed.",
                metadata,
            ) from exc
        return metadata

    @property
    def shape(self) -> tuple[int, int]:
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

        # Generate basename template
        basename_template = self._generate_basename_template(basename)

        # Configure file options
        file_options = pds.ParquetFileFormat().make_write_options(
            compression=compression
        )

        # Determine if directories should be created
        create_dir = self._should_create_dir(create_dir)

        # Track metadata
        metadata = []

        # Create file visitor
        file_visitor = self._create_file_visitor(verbose, metadata)

        # Write dataset with retry logic
        self._write_dataset_with_retry(
            file_options=file_options,
            partitioning_columns=partitioning_columns,
            partitioning_flavor=partitioning_flavor,
            basename_template=basename_template,
            row_group_size=row_group_size,
            max_rows_per_file=max_rows_per_file,
            create_dir=create_dir,
            file_visitor=file_visitor,
            **kwargs,
        )

        return metadata

    def _generate_basename_template(self, basename: str | None) -> str:
        """Generate the basename template for output files."""
        if basename is None:
            return (
                "data-"
                f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]}-{uuid.uuid4().hex[:16]}-{{i}}.parquet"
            )
        return f"{basename}-{{i}}.parquet"

    def _should_create_dir(self, create_dir: bool) -> bool:
        """Determine if directories should be created based on filesystem type."""
        if hasattr(self._filesystem, "fs"):
            return create_dir or "local" in self._filesystem.fs.protocol
        return create_dir or "local" in self._filesystem.protocol

    def _create_file_visitor(self, verbose: bool, metadata: list):
        """Create a file visitor function for tracking written files."""

        def file_visitor(written_file):
            if verbose:
                logger.info(f"path={written_file.path}")
                logger.info(f"size={written_file.size} bytes")
                logger.info(f"metadata={written_file.metadata}")
            metadata.append({written_file.path: written_file.metadata})

        return file_visitor

    def _write_dataset_with_retry(
        self,
        file_options,
        partitioning_columns,
        partitioning_flavor,
        basename_template,
        row_group_size,
        max_rows_per_file,
        create_dir,
        file_visitor,
        **kwargs,
    ):
        """Write the dataset with retry logic."""
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

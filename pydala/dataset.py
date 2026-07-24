# Standard library imports
import dataclasses
import datetime as dt
import enum
import logging
import os
import posixpath
import re
import tempfile
import warnings
import typing as t
from typing import Literal

# Third-party imports
import duckdb as _duckdb
import pandas as pd
import polars.selectors as cs
import psutil
import pyarrow as pa
import pyarrow.dataset as pds
import tqdm
from fsspec import AbstractFileSystem
from fsspec.core import split_protocol
from fsspec.implementations.dirfs import DirFileSystem
from fsspeckit.datasets.pyarrow import collect_dataset_stats_pyarrow
from fsspeckit.core.incremental import MergeResult
from fsspeckit.core.maintenance import CastPolicy, SortKey

# Local imports
from .filesystem import FileSystem, clear_cache
from .helpers.datetime import get_timestamp_column
from .helpers.security import (
    validate_partition_name,
    validate_partition_value,
    sanitize_filter_expression,
    safe_join,
)
from .helpers.polars import pl as _pl
from .io import PartialMergeError, PartialWriteError, Writer, WriterSource
from .metadata import PydalaDatasetMetadata
from .schema import (
    SchemaRepairPlan,
    convert_large_types_to_normal,
    convert_timestamp,
)
from .table import PydalaTable
from .adapters.fsspeckit import FsspeckitParquetAdapter

# Initialize logger
logger = logging.getLogger(__name__)


def _normalize_compaction_sort_by(
    sort_by: str | list[str] | list[tuple[str, str]] | tuple[str, str] | None,
    columns: t.Iterable[str],
) -> list[SortKey]:
    """Convert pydala's public sort syntax to fsspeckit's typed sort keys."""
    if not sort_by:
        return []

    available_columns = set(columns)
    specs: list[tuple[str, str]]
    if isinstance(sort_by, str):
        specs = []
        for spec in sort_by.split(","):
            parts = spec.strip().rsplit(" ", 1)
            if not spec.strip():
                raise ValueError("sort_by must not contain an empty sort column")
            specs.append(
                (parts[0], parts[1].lower() if len(parts) == 2 else "ascending")
            )
    elif isinstance(sort_by, (list, tuple)):
        if not sort_by:
            return []
        if (
            isinstance(sort_by, tuple)
            and len(sort_by) == 2
            and isinstance(sort_by[0], str)
            and isinstance(sort_by[1], str)
            and sort_by[1].lower() in {"asc", "ascending", "desc", "descending"}
        ):
            specs = [(sort_by[0], sort_by[1].lower())]
        elif all(isinstance(spec, str) for spec in sort_by):
            specs = [(spec, "ascending") for spec in sort_by]
        elif all(
            isinstance(spec, (list, tuple))
            and len(spec) == 2
            and all(isinstance(value, str) for value in spec)
            for spec in sort_by
        ):
            specs = [(spec[0], spec[1].lower()) for spec in sort_by]
        else:
            raise ValueError(
                "sort_by must be a string, a list of column names, or "
                "a sequence of (column, direction) pairs."
            )
    else:
        raise ValueError(
            "sort_by must be a string, a list of column names, or "
            "a sequence of (column, direction) pairs."
        )

    sort_keys = []
    for column, direction in specs:
        if column not in available_columns:
            raise ValueError(f"Unknown sort column: {column!r}")
        if direction in {"asc", "ascending"}:
            descending = False
        elif direction in {"desc", "descending"}:
            descending = True
        else:
            raise ValueError(f"Invalid sort direction: {direction!r}")
        sort_keys.append(
            SortKey(column=column, descending=descending, nulls_first=False)
        )
    return sort_keys


def strip_protocol(path: str) -> str:
    """Strips the protocol from a given path.

    Args:
        path (str): The input path which may contain a protocol.
    Returns:
        str: The path without the protocol.
    """
    protocol, path = split_protocol(path)
    return path


def _maintenance_to_plain(value: t.Any) -> t.Any:
    """Recursively convert fsspeckit maintenance objects to plain data.

    fsspeckit 0.25 returns frozen dataclasses (``MaintenanceResult``,
    ``CompactionPlan``, ...) with enum fields from its filesystem maintenance
    methods. pydala's public maintenance contract returns plain dicts, so the
    dataclasses are unrolled recursively and enums are replaced by their
    values. Non-dataclass leaves (e.g. ``pyarrow.Schema``) are kept as-is.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _maintenance_to_plain(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_maintenance_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _maintenance_to_plain(item) for key, item in value.items()}
    return value


def _maintenance_plan_to_dict(plan: t.Any) -> dict[str, t.Any]:
    """Convert an fsspeckit maintenance plan to pydala's dry-run dict shape.

    Adds a ``planned_groups`` convenience key (a list of file-path lists, one
    per planned rewrite group) next to the fully unrolled plan fields.
    """
    result = _maintenance_to_plain(plan)
    groups = (
        getattr(plan, "schema_rewrite_groups", None)
        or getattr(plan, "compaction_groups", None)
        or getattr(plan, "optimization_groups", None)
        or getattr(plan, "ordered_groups", None)
        or getattr(plan, "repartition_groups", None)
        or getattr(plan, "dedup_groups", None)
        or ()
    )
    result["planned_groups"] = [[file.path for file in group.files] for group in groups]
    return result


def _require_successful_maintenance(result: t.Any, operation: str) -> None:
    """Raise with fsspeckit recovery data when a rewrite result failed.

    Time-window and partitioned compaction return nested per-scope result
    dictionaries, while a flat rewrite returns one result. Check both shapes
    before refreshing Pydala's managed metadata.
    """
    result = _maintenance_to_plain(result)
    if not isinstance(result, (dict, list)) and hasattr(result, "succeeded"):
        result = {
            "succeeded": result.succeeded,
            "error": getattr(result, "error", None),
            "recovery": _maintenance_to_plain(getattr(result, "recovery", None)),
        }
    if isinstance(result, list):
        for item in result:
            if isinstance(item, (dict, list)):
                _require_successful_maintenance(item, operation)
        return
    if not isinstance(result, dict):
        raise TypeError(f"Unexpected fsspeckit {operation} result: {result!r}")
    if result.get("succeeded") is False:
        recovery = result.get("recovery")
        raise RuntimeError(
            f"fsspeckit {operation} did not succeed: "
            f"{result.get('error') or 'unknown error'}. "
            f"Live files were not refreshed; recovery details: {recovery!r}"
        )
    if "succeeded" not in result:
        for item in result.values():
            if isinstance(item, (dict, list)):
                _require_successful_maintenance(item, operation)


def _validate_compaction_options(
    row_group_size: int | None, options: dict[str, t.Any]
) -> None:
    """Make compaction options unsupported by fsspeckit explicit to callers."""
    if row_group_size not in (None, 256_000):
        warnings.warn(
            "row_group_size is not supported by fsspeckit compaction and will "
            "be ignored.",
            DeprecationWarning,
            stacklevel=3,
        )
    if options:
        unsupported = ", ".join(sorted(options))
        raise TypeError(f"Unsupported compaction options: {unsupported}")


def _resolve_maintenance_target(
    filesystem: AbstractFileSystem, path: str
) -> tuple[AbstractFileSystem, str]:
    """Resolve the (filesystem, path) pair fsspeckit's maintenance expects.

    fsspeckit's atomic-local maintenance lane reads and renames files through
    plain OS calls (``open``/``os``/``shutil``) instead of the fsspec API, so
    for local backends any ``DirFileSystem`` wrappers must be unwrapped to the
    inner filesystem and ``path`` resolved to the real OS path. Non-local
    backends keep the original pair; their best-effort lane routes all I/O
    through ``filesystem.open``.
    """
    root_segments: list[str] = []
    inner = filesystem
    while isinstance(inner, DirFileSystem):
        root_segments.append(inner.path)
        inner = inner.fs
    protocols = inner.protocol
    if isinstance(protocols, str):
        protocols = (protocols,)
    if not any(protocol in {"local", "file", "os"} for protocol in protocols):
        return filesystem, path
    resolved = os.path.join(*reversed(root_segments), path) if root_segments else path
    return inner, os.path.abspath(resolved)


class _DatasetStorage:
    """Canonical owner of path, filesystem, cache, and directory lifecycle.

    Centralizes storage initialization so every dataset type shares one
    unambiguous owner for normalized path state, filesystem selection,
    cache configuration, directory creation, and cache invalidation.
    """

    def _init_storage(
        self,
        path: str,
        filesystem: AbstractFileSystem | None,
        bucket: str | None,
        cached: bool,
        **fs_kwargs,
    ) -> None:
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")

        self._bucket = bucket
        self._cached = cached
        self._base_filesystem = filesystem
        if cached:
            cache_storage = fs_kwargs.pop(
                "cache_storage", tempfile.mkdtemp(prefix="pydala2_")
            )
        else:
            cache_storage = None
        if filesystem is not None and bucket is None and not cached:
            self._filesystem = filesystem
        else:
            self._filesystem = FileSystem(
                bucket=bucket,
                fs=filesystem,
                cached=cached,
                cache_storage=cache_storage,
                **fs_kwargs,
            )
        self._path = strip_protocol(path)
        self._makedirs()

    def _makedirs(self) -> None:
        if self._filesystem.exists(self._path):
            return
        try:
            self._filesystem.mkdirs(self._path)
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to create directory {self._path}: {e}")
            try:
                self._filesystem.touch(posixpath.join(self._path, "tmp.delete"))
                self._filesystem.rm(posixpath.join(self._path, "tmp.delete"))
            except Exception as e2:
                logger.error(f"Alternative directory creation also failed: {e2}")
                raise

    def clear_cache(self) -> None:
        if hasattr(self._filesystem, "fs"):
            clear_cache(self._filesystem.fs)
        clear_cache(self._filesystem)
        clear_cache(self._base_filesystem)


class BaseDataset(_DatasetStorage):
    """A base class for dataset operations supporting multiple file formats.

    This class provides a unified interface for working with datasets in various formats
    (Parquet, CSV, JSON) with support for partitioning, caching, and advanced filtering.
    It integrates with DuckDB for SQL operations and PyArrow for efficient data handling.

    Attributes:
        path (str): The path to the dataset.
        name (str): The name of the dataset.
        schema (pa.Schema): The schema of the dataset.
        filesystem (FileSystem): The filesystem used for data access.
        table (PydalaTable): The table interface for the dataset.
        ddb_con (duckdb.DuckDBPyConnection): DuckDB connection for SQL operations.
        _timestamp_column (str): The timestamp column for time-based operations.
        _partitioning (str | list[str]): The partitioning scheme.
        _format (str): The file format of the dataset.
    """

    def __init__(
        self,
        path: str,
        name: str | None = None,
        schema: pa.Schema | None = None,
        filesystem: AbstractFileSystem | None = None,
        bucket: str | None = None,
        partitioning: str | list[str] | None = None,
        format: str | None = "parquet",
        cached: bool = False,
        timestamp_column: str | None = None,
        ddb_con: _duckdb.DuckDBPyConnection | None = None,
        **fs_kwargs,
    ) -> None:
        """Initialize a BaseDataset instance.

        Args:
            path: The path to the dataset.
            name: The name of the dataset. If None, uses basename of path.
            schema: The schema for the dataset. If None, inferred from data.
            filesystem: The filesystem to use for data access.
            bucket: The bucket name for cloud storage.
            partitioning: The partitioning scheme ('hive', 'ignore', or list of columns).
            format: The file format ('parquet', 'csv', 'json').
            cached: Whether to use caching for filesystem operations.
            timestamp_column: The column to use as timestamp for time operations.
            ddb_con: Existing DuckDB connection. If None, creates a new one.
            **fs_kwargs: Additional arguments for filesystem configuration.

        Raises:
            ValueError: If path, format, partitioning, or timestamp_column are invalid types.
        """
        # Input validation
        if format is not None and not isinstance(format, str):
            raise ValueError("format must be a string or None")
        if partitioning is not None and not isinstance(partitioning, (str, list)):
            raise ValueError("partitioning must be a string, list, or None")
        if timestamp_column is not None and not isinstance(timestamp_column, str):
            raise ValueError("timestamp_column must be a string or None")

        self._init_storage(path, filesystem, bucket, cached, **fs_kwargs)

        self._schema = schema
        self._format = format
        self.table = None

        if name is None:
            self.name = posixpath.basename(path)
        else:
            self.name = name

        if ddb_con is None:
            ddb_con = _duckdb.connect()

        self.ddb_con = ddb_con
        # enable object caching for e.g. parquet metadata
        self.ddb_con.execute(
            f"""PRAGMA enable_object_cache;
            SET THREADS={psutil.cpu_count() * 2};"""
        )
        self._timestamp_column = timestamp_column

        if partitioning is None:
            # try to infer partitioning
            try:
                if any(["=" in obj for obj in self.fs.ls(self._path)]):
                    partitioning = "hive"
            except FileNotFoundError as e:
                _ = e
                partitioning = None
        else:
            if partitioning == "ignore":
                partitioning = None
        self._partitioning = partitioning

        if type(self) is BaseDataset:
            self._load_eager()

    def _load_eager(self) -> None:
        """Eagerly load the dataset after construction state is complete.

        Errors during loading are swallowed so datasets can be created
        before any files exist (matching the historical contract).
        """
        try:
            self.load()
        except FileNotFoundError:
            logger.debug(f"Dataset path does not exist yet: {self._path}")
        except Exception as e:
            logger.warning(f"Failed to load dataset {self._path}: {e}")

    def load_files(self) -> None:
        """
        Loads the files from the specified path with the specified format.

        This method populates the `_files` attribute with a list of file names
        that match the specified format in the specified path.

        Returns:
            None
        """
        self.clear_cache()
        # Safely construct glob pattern
        glob_pattern = safe_join(self._path, f"**/*.{self._format}")
        self._files = [
            fn.replace(self._path, "").lstrip("/")
            for fn in sorted(self._filesystem.glob(glob_pattern))
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
    def path(self) -> str:
        """Get the path of the dataset.

        Returns:
            The path to the dataset.
        """
        return self._path

    @property
    def has_files(self) -> bool:
        """Check if the dataset has any files.

        Returns:
            True if the dataset has files, False otherwise.
        """
        return len(self.files) > 0

    def load(self) -> None:
        """Load the dataset from the specified path and initialize the PydalaTable.

        This method creates a PyArrow dataset from the files at the specified path,
        initializes a PydalaTable for data operations, and registers the dataset
        with DuckDB for SQL queries. It also attempts to identify timestamp columns
        and configure timezone settings if found.

        Returns:
            None

        Note:
            If the dataset has no files, this method will log a debug message
            but not raise an exception.
        """
        if self.has_files:
            self._arrow_dataset = pds.dataset(
                self._path,
                schema=self._schema,
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
                    # Use parameterized query to prevent SQL injection
                    self.ddb_con.execute("SET TimeZone=?", [str(tz)])
            else:
                self._tz = None

            self.ddb_con.register(f"{self.name}", self._arrow_dataset)

    @property
    def schema(self):
        """
        Returns the schema of the dataset.

        If the schema has not been cached, it retrieves the schema from the underlying Arrow dataset.

        Returns:
            pyarrow.Schema: The schema of the dataset.
        """
        if self.is_loaded:
            if not hasattr(self, "_schema") or self._schema is None:
                self._schema = self._arrow_dataset.schema
            return self._schema

    @property
    def is_loaded(self) -> bool:
        """
        Returns True if the dataset has been loaded into memory, False otherwise.
        """
        return hasattr(self, "_arrow_dataset")

    @property
    def columns(self) -> list[str]:
        """Get a list of column names for the dataset.

        Returns:
            A list of column names in the dataset, including both schema
            and partitioning columns. Returns None if dataset is not loaded.
        """
        if self.is_loaded:
            return self.schema.names

    @property
    def fs(self) -> FileSystem:
        """Get the filesystem associated with the dataset.

        Returns:
            The filesystem object used by the dataset.
        """
        return self._filesystem

    @property
    def t(self) -> PydalaTable | None:
        """Get the table associated with the dataset.

        Returns:
            The PydalaTable object for the dataset, or None if not loaded.
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
            files = [posixpath.join(self._path, fn) for fn in files]
        self._filesystem.rm(files, recursive=True)
        # self.load(reload=True)

    def vacuum(self) -> None:
        """Delete all data files in the dataset.

        Removes all parquet data files and metadata sidecars, preserving
        the directory structure so writes can resume.
        """
        if self.files:
            self.delete_files(self.files)
        self.delete_metadata_files()

        # Reset in-memory metadata state so stale file references are gone.
        self._metadata = None
        self._file_metadata = None
        for attr in ("_files", "_schema", "_file_schema"):
            if hasattr(self, attr):
                delattr(self, attr)

        self.clear_cache()

    @property
    def partitioning_schema(self) -> pa.Schema:
        """

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

    def filter(
        self,
        filter_expr: str | pds.Expression,
        use: str = "auto",
    ) -> PydalaTable:
        if self.table is None:
            if use not in {"fsspeckit", "fsspeckit-pyarrow", "fsspeckit-duckdb"}:
                raise RuntimeError("Dataset must be loaded before filtering.")
            backend = "duckdb" if use == "fsspeckit-duckdb" else "pyarrow"
            return PydalaTable.filter_filesystem(
                filter_expr,
                filesystem=self._filesystem,
                path=self._path,
                backend=backend,
                ddb_con=self.ddb_con,
            )
        return self.table.filter(
            filter_expr,
            use=use,
            filesystem=self._filesystem,
            path=self._path,
        )

    @property
    def registered_tables(self) -> list[str]:
        """
        Get a list of registered tables.

        Returns:
            list[str]: A list of table names.
        """
        return [row[0] for row in self.ddb_con.sql("SHOW TABLES").fetchall()]

    def interrupt_duckdb(self) -> None:
        """Interrupt any ongoing DuckDB operations.

        This method sends an interrupt signal to the DuckDB connection,
        which can be useful for cancelling long-running queries.

        Returns:
            None
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

    # TODO(next-major): remove with the legacy delta compatibility helpers.
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
        max_rows_per_file: int | None = 10_000_000,
        row_group_size: int | None = 256_000,
        compression: str = "zstd",
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        unique: bool | str | list[str] = False,
        ts_unit: str = "us",
        tz: str | None = None,
        remove_tz: bool = False,
        # TODO(next-major): remove delta_subset after the delta migration window.
        delta_subset: str | list[str] | None = None,
        alter_schema: bool = False,
        timestamp_column: str | None = None,
        update_metadata: bool = False,
        verbose: bool = False,
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
        - delta_subset: The subset of columns to consider for delta updates. Can be a string, a list of strings, or
            None. Defaults to None.
        - alter_schema: Whether to alter the schema of the dataset. Defaults to False.
        - timestamp_column: The name of the timestamp column. Defaults to None.
        - update_metadata: Whether to update the metadata table after writing. Defaults to False.
        - verbose: Whether to print verbose output. Defaults to False.
        - **kwargs: Additional keyword arguments to be passed to the writer.

        Returns:
        None
        """

        if mode not in {"append", "overwrite"}:
            raise ValueError(
                f"Unsupported write mode: {mode!r}. "
                "Expected one of ('append', 'overwrite')."
            )
        if "partitioning_columns" in kwargs:
            partition_by = kwargs.pop("partitioning_columns")

        if not partition_by and self.partition_names:
            partition_by = self.partition_names

        if timestamp_column is not None:
            self._timestamp_column = timestamp_column

        if isinstance(data, pa.RecordBatchReader):
            data = data.read_all()

        if (
            not isinstance(data, list)
            and not isinstance(data, tuple)
            and not isinstance(data, pa.RecordBatchReader)
        ):
            data = [data]
        metadata = []
        if not partition_by:
            schema = self.metadata.schema.to_arrow_schema() if self.metadata else None
        else:
            schema = self.schema if self.schema else None

        for data_ in data:
            writer = Writer(
                data=data_,
                path=self._path,
                filesystem=self._filesystem,
                schema=schema if not alter_schema else None,
            )
            metadata.extend(
                writer.execute(
                    sort_by=sort_by,
                    unique=unique,
                    ts_unit=ts_unit,
                    tz=tz,
                    remove_tz=remove_tz,
                    alter_schema=alter_schema,
                    partition_by=partition_by,
                    timestamp_column=self._timestamp_column,
                    row_group_size=row_group_size,
                    compression=compression,
                    max_rows_per_file=max_rows_per_file,
                    basename=basename,
                    verbose=verbose,
                    **kwargs,
                )
            )

        if mode == "overwrite":
            try:
                self.delete_files([posixpath.join(self._path, fn) for fn in self.files])
            except Exception as exc:
                raise PartialWriteError(
                    "Parquet files were written, but overwrite completion failed.",
                    metadata,
                ) from exc

        return metadata

        # if update_metadata:
        return metadata


class ParquetDataset(PydalaDatasetMetadata, BaseDataset):
    """A dataset implementation for Parquet files with metadata support.

    This class extends BaseDataset with Parquet-specific features including
    metadata management, optimization operations, and efficient querying.
    It leverages Parquet dataset metadata for improved performance and
    provides advanced features like partitioning and compression optimization.
    """

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
    ) -> None:
        """Initialize a ParquetDataset instance.

        Args:
            path: The path to the Parquet dataset.
            name: The name of the dataset. If None, uses basename of path.
            filesystem: The filesystem to use for data access.
            bucket: The bucket name for cloud storage.
            partitioning: The partitioning scheme ('hive', 'ignore', or list of columns).
            cached: Whether to use caching for filesystem operations.
            timestamp_column: The column to use as timestamp for time operations.
            ddb_con: Existing DuckDB connection. If None, creates a new one.
            **fs_kwargs: Additional arguments for filesystem configuration.

        Note:
            This class inherits from both PydalaDatasetMetadata for metadata
            management and BaseDataset for core dataset functionality.
        """
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

        # Recovery surface for ``merge``: the last successful fsspeckit
        # ``MergeResult`` is retained here even when the post-merge state
        # refresh raises ``PartialMergeError``. ``None`` until a merge runs.
        self.last_merge_result: MergeResult | None = None

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
        # use_large_types: bool = False,
        format_version: str = "2.6",
        verbose: bool = False,
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
            format_version (str, optional): The version of the data format. Defaults to "2.6".
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        if not self.has_file_metadata_file:
            if len(self.fs.ls(self.path)) == 0:
                return
            else:
                update_metadata = True
        if kwargs.pop("update", None):
            update_metadata = True
        if kwargs.pop("reload", None):
            reload_metadata = True

        if update_metadata or reload_metadata:
            self.update(
                reload=reload_metadata,
                schema=schema,
                ts_unit=ts_unit,
                tz=tz,
                format_version=format_version,
                verbose=verbose,
                **kwargs,
            )
            if not hasattr(self, "_schema"):
                self._schema = self.metadata.schema.to_arrow_schema()
            self.update_metadata_table()

        self._arrow_dataset = pds.parquet_dataset(
            self._metadata_file,
            # schema=self._schema,
            partitioning=self._partitioning,
            filesystem=self._filesystem,
        )

        self.table = PydalaTable(result=self._arrow_dataset, ddb_con=self.ddb_con)

        if self._timestamp_column is None:
            self._timestamp_columns = get_timestamp_column(self.table.pl.head(10))
            if len(self._timestamp_columns) > 0:
                self._timestamp_column = self._timestamp_columns[0]
        if self._timestamp_column is not None:
            ts_type = self.schema.field(self._timestamp_column).type
            if hasattr(ts_type, "tz"):
                tz = ts_type.tz
            else:
                tz = None
            self._tz = tz
            if tz is not None:
                # Use parameterized query to prevent SQL injection
                self.ddb_con.execute("SET TimeZone=?", [str(tz)])
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

    def scan(self, filter_expr: str | None = None) -> PydalaTable:
        """
        Scan the Parquet dataset metadata and return a filtered table.

        Applies the filter expression to per-file statistics to select
        candidate files, then returns a :class:`PydalaTable` backed by
        only those files.

        Args:
            filter_expr: An optional filter expression to apply to the
                metadata (e.g. ``"id >= 5"``).

        Returns:
            A PydalaTable containing only the rows from files that match
            the filter.
        """
        scan_files = self.table.prune_files(
            self.metadata_table,
            filter_expr,
            self.files,
        )

        # Maintain the _metadata_table_scanned invariant so the scan_files
        # property returns the pruned subset, matching the files included
        # in the returned PydalaTable.  Without this, scan_files falls back
        # to self.files and callers (e.g. _optimize_dtypes) that delete
        # scan_files would remove every file in the dataset.
        if scan_files:
            in_values = ", ".join("'" + f.replace("'", "''") + "'" for f in scan_files)
            self._metadata_table_scanned = self._metadata_table.filter(
                f"file_path IN ({in_values})"
            )
        else:
            self._metadata_table_scanned = None

        if len(scan_files) == 0:
            return PydalaTable(result=self.table.ddb.limit(0))

        return PydalaTable(
            result=pds.dataset(
                [posixpath.join(self.path, fn) for fn in scan_files],
                filesystem=self._filesystem,
                format=self._format,
                partitioning=self._partitioning,
            ),
            ddb_con=self.ddb_con,
        )

    def __repr__(self):
        if self.table is not None:
            return self.table.__repr__()
        else:
            return f"{self.path} is empty."

    # TODO(next-major): remove with mode="delta" compatibility.
    def _resolve_deprecated_delta_partition_by(
        self,
        partition_by: str | list[str] | None,
    ) -> str | list[str] | None:
        """Resolve existing hive partition columns without relying on load state."""
        if partition_by:
            return partition_by

        partition_names = list(self.partition_names or [])
        if partition_names:
            return partition_names

        target_files = self._filesystem.glob(
            posixpath.join(self._path, "**", "*.parquet")
        )
        for file_path in target_files:
            for part in file_path.split("/"):
                if "=" not in part:
                    continue
                name = part.split("=", maxsplit=1)[0]
                if name and name not in partition_names:
                    partition_names.append(name)
        return partition_names or None

    # TODO(next-major): remove with mode="delta" compatibility.
    def _infer_deprecated_delta_key_columns(
        self,
        data: WriterSource | list[WriterSource],
        *,
        partition_by: str | list[str] | None,
        sort_by: str | list[str] | list[tuple[str, str]] | None,
        unique: bool | str | list[str],
        ts_unit: str,
        tz: str | None,
        remove_tz: bool,
        alter_schema: bool,
        timestamp_column: str | None,
    ) -> list[str]:
        """Infer legacy exact-row identity keys without relying on load state."""
        sources = list(data) if isinstance(data, (list, tuple)) else [data]
        prepared = Writer.prepare_many(
            sources,
            path=self._path,
            schema=None,
            filesystem=self._filesystem,
            sort_by=sort_by,
            unique=unique,
            ts_unit=ts_unit,
            tz=tz,
            remove_tz=remove_tz,
            alter_schema=True,
            partition_by=partition_by,
            timestamp_column=timestamp_column,
        )
        if not prepared:
            raise ValueError(
                "Cannot infer delta keys from an empty source list; "
                "pass delta_subset explicitly."
            )

        key_columns = [
            column
            for column in prepared[0].column_names
            if all(column in table.column_names for table in prepared)
        ]
        target_files = self._filesystem.glob(
            posixpath.join(self._path, "**", "*.parquet")
        )
        if target_files:
            dataset_options: dict[str, t.Any] = (
                {"partitioning": "hive"} if partition_by else {}
            )
            target = pds.dataset(
                self._path,
                filesystem=self._filesystem,
                format="parquet",
                **dataset_options,
            )
            target_columns = set(target.schema.names)
            key_columns = [column for column in key_columns if column in target_columns]
        if not key_columns:
            raise ValueError(
                "Cannot infer common source/target columns for "
                "mode='delta'; pass delta_subset explicitly."
            )
        return key_columns

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
        max_rows_per_file: int | None = 10_000_000,
        row_group_size: int | None = 256_000,
        compression: str = "zstd",
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        unique: bool | str | list[str] = False,
        ts_unit: str = "us",
        tz: str | None = None,
        remove_tz: bool = False,
        # TODO(next-major): remove delta_subset after the delta migration window.
        delta_subset: str | list[str] | None = None,
        update_metadata: bool = True,
        alter_schema: bool = False,
        timestamp_column: str | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> MergeResult | None:
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
        - delta_subset: The subset of columns to consider for delta updates. Can be a string, a list of strings, or
            None. Defaults to None.
        - update_metadata: Whether to update the metadata table after writing. Defaults to False.
        - alter_schema: Whether to alter the schema of the dataset. Defaults to False.
        - timestamp_column: The name of the timestamp column. Defaults to None.
        - verbose: Whether to print verbose output. Defaults to False.
        - **kwargs: Additional keyword arguments to be passed to the writer.

        Returns:
        None
        """

        if "partitioning_columns" in kwargs:
            partition_by = kwargs.pop("partitioning_columns")

        valid_modes = ("append", "overwrite", "delta")
        if mode not in valid_modes:
            raise ValueError(
                f"Unsupported write mode: {mode!r}. Expected one of {valid_modes}."
            )

        if mode == "delta":
            if isinstance(data, pa.RecordBatchReader):
                data = data.read_all()
            partition_by = self._resolve_deprecated_delta_partition_by(partition_by)

            if delta_subset is None:
                inferred_keys = self._infer_deprecated_delta_key_columns(
                    data,
                    partition_by=partition_by,
                    sort_by=sort_by,
                    unique=unique,
                    ts_unit=ts_unit,
                    tz=tz,
                    remove_tz=remove_tz,
                    alter_schema=alter_schema,
                    timestamp_column=timestamp_column,
                )
                warning = (
                    "delta_subset=None infers key columns from common "
                    "source/target columns for compatibility; explicit keys "
                    "will be required after mode='delta' is removed. "
                )
                key_columns: str | list[str] = inferred_keys
            else:
                warning = ""
                key_columns = delta_subset

            warning += (
                "mode='delta' is deprecated and now delegates to "
                "merge(strategy='insert'). Migration behavior: key equality "
                "is null-safe; duplicate source keys use the last row; "
                "execution is independent of load state; the return value is "
                "a typed MergeResult."
            )
            warnings.warn(warning, DeprecationWarning, stacklevel=2)
            return self.merge(
                data,
                strategy="insert",
                key_columns=key_columns,
                partition_by=partition_by,
                compression=compression,
                max_rows_per_file=max_rows_per_file,
                row_group_size=row_group_size,
                sort_by=sort_by,
                unique=unique,
                ts_unit=ts_unit,
                tz=tz,
                remove_tz=remove_tz,
                alter_schema=alter_schema,
                timestamp_column=timestamp_column,
                **kwargs,
            )
        metadata = BaseDataset.write_to_dataset(
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
            delta_subset=delta_subset,
            alter_schema=alter_schema,
            timestamp_column=timestamp_column,
            update_metadata=update_metadata,
            verbose=verbose,
            **kwargs,
        )

        if update_metadata:
            try:
                for md in metadata:
                    metadata_ = list(md.values())[0]
                    if (
                        self.metadata is not None
                        and metadata_.schema != self.metadata.schema
                    ):
                        continue

                    file_path = list(md.keys())[0].replace(self._path, "").lstrip("/")
                    metadata_.set_file_path(file_path)
                    if self._file_metadata is None:
                        self._file_metadata = {file_path: metadata_}
                    else:
                        self._file_metadata[file_path] = metadata_
                self._update_metadata(verbose=verbose)
            except Exception as exc:
                raise PartialWriteError(
                    "Parquet files were written, but metadata completion failed.",
                    metadata,
                ) from exc
            # self._repair_file_schemas(alter_schema=alter_schema, verbose=verbose)
            # self._update_metadata()
            # try:
            #     self.load(update_metadata=True, verbose=False)
            # except Exception as e:
            #     _ = e
            #     self.load(reload_metadata=True, verbose=False)
            # self.update_metadata_table()

    # ------------------------------------------------------------------ #
    # Merge (insert / update / upsert) through fsspeckit
    # ------------------------------------------------------------------ #
    def _finalize_merge_result(self, result: MergeResult) -> MergeResult:
        """Retain a successful merge result and refresh dataset state."""
        self.last_merge_result = result
        try:
            self._refresh_after_rewrite()
        except Exception as exc:
            raise PartialMergeError(
                "Merge succeeded but dataset state refresh failed; the "
                "successful MergeResult is available on "
                "ParquetDataset.last_merge_result.",
                result,
            ) from exc
        return result

    def merge(
        self,
        data: WriterSource | list[WriterSource],
        *,
        strategy: Literal["insert", "update", "upsert"],
        key_columns: str | list[str] | None = None,
        partition_by: str | list[str] | None = None,
        backend: Literal["pyarrow", "duckdb"] = "pyarrow",
        compression: str = "zstd",
        max_rows_per_file: int | None = None,
        row_group_size: int | None = None,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        unique: bool | str | list[str] = False,
        ts_unit: str = "us",
        tz: str | None = None,
        remove_tz: bool = False,
        alter_schema: bool = False,
        timestamp_column: str | None = None,
        **merge_options: t.Any,
    ) -> MergeResult:
        """Merge ``data`` into the dataset via fsspeckit (insert/update/upsert).

        Normalizes every supported source family through
        :meth:`Writer.prepare`, treats list inputs as one logical source
        batch, and delegates persistence strictly through
        :meth:`FsspeckitParquetAdapter.merge`. After a successful physical
        merge, pydala file/metadata/table/cache state is refreshed through
        :meth:`_refresh_after_rewrite` so the same instance reads the merged
        result immediately.

        Supported source families (mirroring :meth:`write_to_dataset`):
        Polars ``DataFrame``/``LazyFrame``, PyArrow ``Table``/``RecordBatch``,
        pandas ``DataFrame``, and a DuckDB relation. A ``list``/``tuple`` of
        any of these is merged as one logical source batch; duplicate source
        keys resolve last-row-wins as required by fsspeckit.

        Args:
            data: Source(s) to merge; a list is one logical batch.
            strategy: ``"insert"`` (new keys only), ``"update"`` (matched
                keys only), or ``"upsert"`` (both).
            key_columns: Column(s) uniquely identifying rows. When omitted or
                ``None``, every source column that is present in every prepared
                source table is used. Pass explicit keys for stable business-key
                update/upsert semantics; an all-column key treats a changed
                value as a new row. Key equality is null-safe: null matches
                null and differs from every non-null value.
            partition_by: Optional pydala-style partition column(s). Date-part
                names (``year``/``month``/...) are derived from the timestamp
                column; all named columns become hive partition directories.
            backend: ``"pyarrow"`` (default; arbitrary fsspec filesystems) or
                ``"duckdb"`` (limited to filesystems DuckDB can write to
                natively).
            compression: Compression codec forwarded to fsspeckit. Defaults
                to ``"zstd"`` to match :meth:`write_to_dataset`.
            max_rows_per_file / row_group_size: Optional fsspeckit
                file/row-group sizing; ``None`` uses fsspeckit defaults.
            sort_by / unique / ts_unit / tz / remove_tz / alter_schema /
                timestamp_column: Source-preparation options forwarded to
                :meth:`Writer.prepare`.
            **merge_options: Additional fsspeckit merge tuning options
                forwarded verbatim (e.g. ``merge_chunk_size_rows``,
                ``enable_streaming_merge``, ``merge_max_memory_mb``).

        Returns:
            fsspeckit's typed :class:`~fsspeckit.core.incremental.MergeResult`.

        Raises:
            ValueError: If explicit ``key_columns`` is empty, keys cannot be
                inferred from the source, the strategy/backend is unsupported,
                a key column is absent from the source, or the backend cannot
                serve this filesystem.
            PartialMergeError: If the physical merge succeeded but the
                post-merge state refresh failed; the successful
                ``MergeResult`` is retained on :attr:`last_merge_result`
                and ``PartialMergeError.merge_result``.
        """
        # --- key validation -------------------------------------------------
        infer_key_columns = key_columns is None
        if isinstance(key_columns, str):
            key_cols: list[str] = [key_columns]
        else:
            key_cols = list(key_columns) if key_columns is not None else []
        if not infer_key_columns and not key_cols:
            raise ValueError(
                "key_columns must be a non-empty string or list of strings"
            )

        # --- normalize sources to a list -----------------------------------
        sources: list[WriterSource] = (
            list(data) if isinstance(data, (list, tuple)) else [data]
        )
        if not sources:
            if infer_key_columns:
                raise ValueError(
                    "Cannot infer merge key columns from an empty source; "
                    "pass key_columns explicitly."
                )
            valid_strategies = ("insert", "update", "upsert")
            if strategy not in valid_strategies:
                raise ValueError(
                    f"Unsupported merge strategy: {strategy!r}. "
                    f"Expected one of {valid_strategies}."
                )
            valid_backends = ("pyarrow", "duckdb")
            if backend not in valid_backends:
                raise ValueError(
                    f"Unsupported merge backend: {backend!r}. "
                    f"Expected one of {valid_backends}."
                )

            target_files = sorted(
                self._filesystem.glob(posixpath.join(self._path, "**", "*.parquet"))
            )
            target_count = (
                pds.dataset(
                    target_files,
                    filesystem=self._filesystem,
                    format="parquet",
                ).count_rows()
                if target_files
                else 0
            )
            result = MergeResult(
                strategy=strategy,
                source_count=0,
                target_count_before=target_count,
                target_count_after=target_count,
                inserted=0,
                updated=0,
                deleted=0,
                files=[],
                rewritten_files=[],
                inserted_files=[],
                preserved_files=target_files,
            )
            return self._finalize_merge_result(result)

        # --- resolve target schema (mirrors write_to_dataset) --------------
        if timestamp_column is not None:
            self._timestamp_column = timestamp_column
        if not partition_by and self.partition_names:
            partition_by = self.partition_names
        if alter_schema or partition_by:
            # When ``alter_schema`` is set we keep every data column. When
            # ``partition_by`` is set we must NOT enforce the on-disk physical
            # schema either: hive partition columns live in the directory
            # layout, not the file schema, so enforcing the file schema would
            # drop them -- and fsspeckit requires partition columns to be
            # physically present in the source. fsspeckit owns partition-aware
            # schema handling on its side.
            target_schema: pa.Schema | None = None
        else:
            target_schema = (
                self.metadata.schema.to_arrow_schema() if self.metadata else None
            )
        if partition_by:
            # fsspeckit writes hive partition directories; make the instance
            # partition-aware immediately so the post-merge refresh (and any
            # subsequent merge that does not repeat ``partition_by``) rebuilds
            # a partition-aware table. Without this a dataset that started
            # empty would keep ``_partitioning=None`` and read partition
            # columns back as absent.
            self._partitioning = "hive"

        # --- prepare every source through the shared normalization seam ----
        prepared = Writer.prepare_many(
            sources,
            path=self._path,
            schema=target_schema,
            filesystem=self._filesystem,
            sort_by=sort_by,
            unique=unique,
            ts_unit=ts_unit,
            tz=tz,
            remove_tz=remove_tz,
            alter_schema=alter_schema,
            partition_by=partition_by,
            timestamp_column=self._timestamp_column,
        )

        if infer_key_columns:
            if not prepared:
                raise ValueError(
                    "Cannot infer merge key columns from an empty prepared source; "
                    "pass key_columns explicitly."
                )
            key_cols = [
                column
                for column in prepared[0].column_names
                if all(column in table.column_names for table in prepared)
            ]
            if not key_cols:
                raise ValueError(
                    "Cannot infer merge key columns: no source column is "
                    "present in every prepared source table."
                )

        # --- validate key presence before persistence ----------------------
        for table in prepared:
            missing = [col for col in key_cols if col not in table.column_names]
            if missing:
                raise ValueError(
                    f"key column(s) {missing!r} not present in prepared merge source"
                )

        # --- delegate persistence to fsspeckit through the adapter ---------
        # List inputs are one logical batch: fsspeckit concats and dedups
        # source keys (last-row-wins) in a single typed MergeResult.
        source_payload: pa.Table | list[pa.Table] = (
            prepared if len(prepared) > 1 else prepared[0]
        )
        adapter = FsspeckitParquetAdapter(filesystem=self._filesystem)
        try:
            result = adapter.merge(
                source_payload,
                self._path,
                strategy=strategy,
                key_columns=key_cols,
                partition_by=partition_by,
                backend=backend,
                compression=compression,
                max_rows_per_file=max_rows_per_file,
                row_group_size=row_group_size,
                **merge_options,
            )
        finally:
            adapter.close()

        return self._finalize_merge_result(result)

    # ------------------------------------------------------------------ #
    # Maintenance and optimization methods
    #
    # These fsspeckit-backed compaction, repartitioning, and type
    # optimization methods live directly on ParquetDataset per ADR 0001.
    # The legacy Optimize subclass inherits them for compatibility.
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # fsspeckit maintenance adapters
    # ------------------------------------------------------------------ #
    def _partition_prefix(self, file_path: str) -> str:
        """Return the partition-directory prefix for a dataset-relative path."""
        if not self.partition_names:
            return ""
        parts = file_path.split("/")
        prefix_parts = []
        for part in parts:
            if "=" in part:
                prefix_parts.append(part)
            else:
                break
        return "/".join(prefix_parts)

    def _fsspeckit_compact(
        self,
        path: str,
        filesystem: AbstractFileSystem,
        max_rows_per_file: int | None = None,
        compression: str | None = None,
        dry_run: bool = False,
        partition_filter: list[str] | None = None,
        sort_keys: list[SortKey] | None = None,
    ) -> dict[str, t.Any]:
        # fsspeckit >= 0.25 exposes maintenance as filesystem methods: a dry
        # run is the immutable plan, a real run plans and executes in one call.
        if sort_keys:
            ordered_kwargs = {
                "sort_keys": sort_keys,
                "target_rows_per_file": max_rows_per_file,
                "partition_filter": partition_filter,
                "compression": compression,
            }
            if dry_run:
                plan = filesystem.plan_parquet_ordered_compaction(
                    path, **ordered_kwargs
                )
                return _maintenance_plan_to_dict(plan)
            result = filesystem.ordered_compact_parquet_dataset(path, **ordered_kwargs)
            return _maintenance_to_plain(result)

        if dry_run:
            plan = filesystem.plan_parquet_compaction(
                path,
                target_rows_per_file=max_rows_per_file,
                partition_filter=partition_filter,
                compression=compression,
            )
            return _maintenance_plan_to_dict(plan)
        result = filesystem.compact_parquet_dataset(
            path,
            target_rows_per_file=max_rows_per_file,
            partition_filter=partition_filter,
            compression=compression,
        )
        return _maintenance_to_plain(result)

    def _fsspeckit_optimize(
        self,
        path: str,
        filesystem: AbstractFileSystem,
        max_rows_per_file: int | None = None,
        compression: str | None = None,
        unique: bool | list[str] = False,
        dry_run: bool = False,
        partition_filter: list[str] | None = None,
    ) -> dict[str, t.Any]:
        if isinstance(unique, list):
            deduplicate_key_columns: list[str] | None = unique
        elif unique:
            deduplicate_key_columns = list(self.schema.names)
        else:
            deduplicate_key_columns = None

        # Hive partition values belong to directory names rather than the
        # Parquet file schema. Compaction optimizes one physical partition at a
        # time, where each such value is constant anyway, so including it in
        # fsspeckit's dedup keys both changes nothing and makes Arrow fail to
        # resolve the field.
        if deduplicate_key_columns is not None:
            partition_columns = set(self.partition_names or [])
            deduplicate_key_columns = [
                column
                for column in deduplicate_key_columns
                if column not in partition_columns
            ]
            if not deduplicate_key_columns:
                raise ValueError(
                    "Deduplication requires at least one physical Parquet column; "
                    "Hive partition columns are stored only in directory paths."
                )

        if dry_run:
            plan = filesystem.plan_parquet_optimization(
                path,
                deduplicate_key_columns=deduplicate_key_columns,
                target_rows_per_file=max_rows_per_file,
                partition_filter=partition_filter,
                compression=compression,
            )
            return _maintenance_plan_to_dict(plan)

        result = filesystem.optimize_parquet_dataset(
            path,
            deduplicate_key_columns=deduplicate_key_columns,
            target_rows_per_file=max_rows_per_file,
            partition_filter=partition_filter,
            compression=compression,
        )
        return _maintenance_to_plain(result)

    def _run_fsspeckit_rewrite(
        self,
        path: str,
        filesystem: AbstractFileSystem,
        max_rows_per_file: int | None = None,
        compression: str | None = None,
        unique: bool | list[str] = False,
        dry_run: bool = False,
        partition_filter: list[str] | None = None,
        sort_by: str
        | list[str]
        | list[tuple[str, str]]
        | tuple[str, str]
        | None = None,
        sort_keys: list[SortKey] | None = None,
    ) -> dict[str, t.Any]:
        """Run compaction (dry-run or real) or dedup+compaction.

        Ordered compaction is exclusive with fsspeckit's deduplication rewrite:
        rejecting the combination preserves both the requested business order
        and the existing dedup winner semantics.
        """
        if sort_keys is None:
            if sort_by:
                self.load()
                sort_keys = _normalize_compaction_sort_by(sort_by, self.schema.names)
            else:
                sort_keys = []
        if sort_keys and unique:
            raise ValueError(
                "sort_by cannot be combined with unique: fsspeckit does not "
                "support ordered compaction with deduplication."
            )

        fs, resolved_path = _resolve_maintenance_target(filesystem, path)
        if not unique:
            return self._fsspeckit_compact(
                path=resolved_path,
                filesystem=fs,
                max_rows_per_file=max_rows_per_file,
                compression=compression,
                dry_run=dry_run,
                partition_filter=partition_filter,
                sort_keys=sort_keys,
            )
        return self._fsspeckit_optimize(
            path=resolved_path,
            filesystem=fs,
            max_rows_per_file=max_rows_per_file,
            compression=compression,
            unique=unique,
            dry_run=dry_run,
            partition_filter=partition_filter,
        )

    def _refresh_after_rewrite(self) -> None:
        """Invalidate caches and rebuild per-file, aggregate and table state."""
        self.clear_cache()
        self.load(update_metadata=True, reload_metadata=True)

    def repair_schema(
        self,
        schema: pa.Schema | None = None,
        ts_unit: str | None = None,
        tz: str | None = None,
        format_version: str | None = None,
        alter_schema: bool = True,
        dry_run: bool = False,
        verbose: bool = False,
        **kwargs: t.Any,
    ) -> SchemaRepairPlan | None:
        """Repair physical Parquet schemas to pydala's canonical target.

        This is the runtime façade. Per ADR 0001, planning, execution, and the
        file/aggregate metadata sidecar refresh live in the metadata layer
        (:meth:`PydalaDatasetMetadata.repair_schema`); this method rebuilds the
        Arrow dataset/table state, the metadata-table view, and caches after a
        successful real repair.

        Candidate files are selected by comparing each file's *raw* physical
        schema (rediscovered fresh from disk) against the canonical
        large-type-normalized target, so real physical differences -- such as
        ``large_string`` versus ``string`` -- are detected and repaired even
        though both normalize to the same canonical type (issue #36). Casting
        completes fully before a source file is replaced, so a failed cast
        leaves the original intact and raises.

        With ``dry_run=True`` no data or metadata is mutated: a plan
        (``{"files": [...], "schema": schema}``) is returned. The ``files`` are
        dataset-relative paths, consistent with :attr:`files`, and reflect the
        *current* physical files on disk.

        Note:
            :meth:`update` (called automatically on the first load of a managed
            dataset) already converges physical schemas to the canonical target.
            Use this method to plan or force a repair against the current
            physical state, for example after files are written by an external
            tool.

        Args:
            schema: Explicit canonical target. If ``None`` the target is
                derived from the unified raw physical schemas and normalized.
            ts_unit: Timestamp unit applied to the target schema.
            tz: Timezone applied to timestamp fields in the target schema.
            format_version: Parquet format version for repair writes and
                candidate selection. Files whose stored version differs are
                also selected.
            alter_schema: Whether to add missing fields while repairing files.
            dry_run: Return a plan without mutating files or sidecars.
            verbose: Log repair progress.
            **kwargs: Additional options forwarded to the executor, including
                ``n_jobs`` and ``backend``.

        Returns:
            A :class:`~pydala.schema.SchemaRepairPlan` when ``dry_run`` is
            ``True``; otherwise ``None``.
        """
        result = PydalaDatasetMetadata.repair_schema(
            self,
            schema=schema,
            ts_unit=ts_unit,
            tz=tz,
            format_version=format_version,
            alter_schema=alter_schema,
            dry_run=dry_run,
            verbose=verbose,
            **kwargs,
        )
        if not dry_run:
            # Rebuild runtime Arrow dataset/table state, the metadata-table
            # view, and caches from the refreshed sidecars.
            self._refresh_after_rewrite()
        return result

    def collect_stats(
        self, partition_filter: list[str] | None = None
    ) -> dict[str, t.Any]:
        """Collect generic Parquet statistics through fsspeckit."""
        return collect_dataset_stats_pyarrow(
            path=self._path,
            filesystem=self._filesystem,
            partition_filter=partition_filter,
        )

    # ------------------------------------------------------------------ #
    # Compaction methods
    # ------------------------------------------------------------------ #
    def _compact_partition(
        self,
        partition: dict[str, t.Any],
        max_rows_per_file: int | None = 10_000_000,
        sort_keys: list[SortKey] | None = None,
        compression: str = "zstd",
        row_group_size: int | None = 256_000,
        unique: bool = True,
        dry_run: bool = False,
        **kwargs: t.Any,
    ) -> dict[str, t.Any]:
        """Compact every file in one physical partition.

        The dataset-root filesystem and file paths are retained for ordered
        compaction so fsspeckit plans against the complete Hive layout rather
        than a schema-less partition subtree.
        """
        for name, value in partition.items():
            if not validate_partition_name(name):
                raise ValueError(f"Invalid partition name: {name}")
            if not validate_partition_value(value):
                raise ValueError(f"Invalid partition value: {value}")

        partition_prefix = "/".join(f"{k}={v}" for k, v in partition.items())
        partition_files = [
            file_path
            for file_path in self.files
            if file_path.startswith(f"{partition_prefix}/")
        ]
        result = self._run_fsspeckit_rewrite(
            path=self._path,
            filesystem=self._filesystem,
            max_rows_per_file=max_rows_per_file,
            compression=compression,
            unique=unique,
            dry_run=dry_run,
            partition_filter=partition_files,
            sort_keys=sort_keys,
        )
        self.reset_scan()
        return result

    def compact_partitions(
        self,
        max_rows_per_file: int | None = 10_000_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        compression: str = "zstd",
        row_group_size: int | None = 256_000,
        unique: bool = True,
        dry_run: bool = False,
        **kwargs: t.Any,
    ) -> list[dict[str, t.Any]] | None:
        """Compact partition files or fully order every physical partition.

        Without ``sort_by``, only small multi-file partitions are candidates.
        Supplying ``sort_by`` performs a full rewrite of every partition so the
        requested ordering covers all of each partition's output files.

        Args:
            max_rows_per_file: Maximum number of rows per output file.
            sort_by: Column(s) for partition-ordered compaction. This cannot be
                combined with ``unique`` because upstream does not support both.
            compression: Compression algorithm to use.
            row_group_size: Reserved compatibility option. A non-default value
                emits a deprecation warning because fsspeckit manages row groups.
            unique: Whether to remove duplicate rows.
            dry_run: When ``True``, return compaction plans without mutating files.
            **kwargs: Unsupported options raise ``TypeError``.

        Returns:
            List of per-partition fsspeckit result dicts when ``dry_run`` is
            ``True``; otherwise ``None``.
        """
        _validate_compaction_options(row_group_size, kwargs)
        self.load(update_metadata=bool(sort_by))
        sort_keys: list[SortKey] | None = None
        if sort_by:
            sort_keys = _normalize_compaction_sort_by(sort_by, self.schema.names)
            if unique:
                raise ValueError(
                    "sort_by cannot be combined with unique: fsspeckit does not "
                    "support ordered compaction with deduplication."
                )

        partition_groups = (
            self.metadata_table.pl()
            .group_by(list(self.partition_names), maintain_order=True)
            .agg(
                _pl.n_unique("file_path").alias("num_partition_files"),
                _pl.sum("num_rows"),
            )
        )
        if not sort_by:
            partition_groups = partition_groups.filter(
                _pl.col("num_partition_files") > 1
            ).filter(_pl.col("num_rows") < max_rows_per_file)
        partitions_to_compact = partition_groups.select(self.partition_names).to_dicts()
        results: list[dict[str, t.Any]] = []
        for partition in tqdm.tqdm(partitions_to_compact):
            result = self._compact_partition(
                partition=partition,
                max_rows_per_file=max_rows_per_file,
                sort_keys=sort_keys,
                compression=compression,
                row_group_size=row_group_size,
                unique=unique,
                dry_run=dry_run,
                **kwargs,
            )
            if not dry_run:
                _require_successful_maintenance(result, "compaction")
            results.append(result)

        if not dry_run:
            self._refresh_after_rewrite()
        return results if dry_run else None

    def compact_small_files(self) -> None:
        """Placeholder for compacting small files across partitions."""
        pass

    def _compact_by_timeperiod(
        self,
        start_date: dt.datetime,
        end_date: dt.datetime,
        timestamp_column: str | None = None,
        max_rows_per_file: int | None = 10_000_000,
        row_group_size: int | None = 256_000,
        compression: str = "zstd",
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        unique: bool = False,
        dry_run: bool = False,
        **kwargs: t.Any,
    ) -> dict[str, t.Any]:
        """Compact files within a specific time period.

        The timestamp filter is applied through pydala's metadata scan; the
        selected files are then rewritten by ``fsspeckit`` while keeping the
        output inside the original partition directories.
        """
        if timestamp_column is None:
            timestamp_column = self._timestamp_column
        if timestamp_column is None:
            raise ValueError("No timestamp column specified or found")
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", timestamp_column):
            raise ValueError(f"Invalid timestamp column name: {timestamp_column}")

        start_date_str = start_date.isoformat()
        end_date_str = end_date.isoformat()
        filter_ = (
            f"{timestamp_column} >= '{start_date_str}' AND "
            f"{timestamp_column} < '{end_date_str}'"
        )
        filter_ = sanitize_filter_expression(filter_)

        self.scan(filter_)
        files = list(self.scan_files)
        self.reset_scan()

        if not files:
            return {"planned_groups": []}

        # Group selected files by partition prefix so fsspeckit can rewrite each
        # group inside its own partition directory.
        grouped: dict[str, list[str]] = {}
        for file_path in files:
            prefix = self._partition_prefix(file_path)
            rel = file_path[len(prefix) :].lstrip("/") if prefix else file_path
            grouped.setdefault(prefix, []).append(rel)

        period_results: dict[str, t.Any] = {}
        for prefix, rel_files in grouped.items():
            if prefix:
                fs = DirFileSystem(
                    path=posixpath.join(self._path, prefix), fs=self._filesystem
                )
                path = "."
            else:
                fs = self._filesystem
                path = self._path
            period_results[prefix] = self._run_fsspeckit_rewrite(
                path=path,
                filesystem=fs,
                max_rows_per_file=max_rows_per_file,
                compression=compression,
                unique=unique,
                dry_run=dry_run,
                partition_filter=rel_files,
                sort_by=sort_by,
            )
        return period_results

    def compact_by_timeperiod(
        self,
        interval: str | dt.timedelta,
        timestamp_column: str | None = None,
        max_rows_per_file: int | None = 10_000_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        unique: bool = False,
        compression: str = "zstd",
        row_group_size: int | None = 256_000,
        dry_run: bool = False,
        **kwargs: t.Any,
    ) -> list[dict[str, t.Any]] | None:
        """Compact files by time periods.

        Divides the dataset's time range into intervals and delegates each
        interval rewrite to ``fsspeckit``. After a real run the full metadata
        lifecycle is refreshed.
        When ``sort_by`` is supplied, ordering requires every file in a physical
        partition, so this method delegates to ``compact_partitions`` rather
        than applying the time-range filter.
        """

        if sort_by:
            return self.compact_partitions(
                max_rows_per_file=max_rows_per_file,
                sort_by=sort_by,
                compression=compression,
                row_group_size=row_group_size,
                unique=unique,
                dry_run=dry_run,
                **kwargs,
            )
        _validate_compaction_options(row_group_size, kwargs)
        timestamp_column = timestamp_column or self._timestamp_column

        min_max_ts = self.t.sql(
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

        results: list[dict[str, t.Any]] = []
        for start_date, end_date in tqdm.tqdm(list(zip(start_dates, end_dates))):
            result = self._compact_by_timeperiod(
                start_date=start_date,
                end_date=end_date,
                timestamp_column=timestamp_column,
                max_rows_per_file=max_rows_per_file,
                row_group_size=row_group_size,
                compression=compression,
                sort_by=sort_by,
                unique=unique,
                dry_run=dry_run,
                **kwargs,
            )
            results.append(result)
            if not dry_run:
                _require_successful_maintenance(result, "compaction")
                self._refresh_after_rewrite()

        return results if dry_run else None

    def compact_by_rows(
        self,
        max_rows_per_file: int | None = 10_000_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        unique: bool = False,
        compression: str = "zstd",
        row_group_size: int | None = 256_000,
        dry_run: bool = False,
        **kwargs: t.Any,
    ) -> dict[str, t.Any] | None:
        """Compact files based on row count.

        For partitioned datasets this delegates to ``compact_partitions``. For
        non-partitioned datasets the rewrite is delegated directly to
        ``fsspeckit``. After a real run the metadata lifecycle is refreshed.
        """
        if self._partitioning:
            return self.compact_partitions(
                max_rows_per_file=max_rows_per_file,
                sort_by=sort_by,
                unique=unique,
                compression=compression,
                row_group_size=row_group_size,
                dry_run=dry_run,
                **kwargs,
            )

        _validate_compaction_options(row_group_size, kwargs)

        result = self._run_fsspeckit_rewrite(
            path=self._path,
            filesystem=self._filesystem,
            max_rows_per_file=max_rows_per_file,
            compression=compression,
            unique=unique,
            dry_run=dry_run,
            sort_by=sort_by,
        )
        if not dry_run:
            _require_successful_maintenance(result, "compaction")
            self._refresh_after_rewrite()
        return result if dry_run else None

    def repartition(
        self,
        partitioning_columns: str | list[str] | None = None,
        partitioning_falvor: str = "hive",
        max_rows_per_file: int | None = 10_000_000,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        unique: bool = False,
        compression: str = "zstd",
        row_group_size: int | None = 256_000,
        dry_run: bool = False,
        derived_partition_columns: dict[str, tuple[str, ...]] | None = None,
        partition_timezone: str = "UTC",
        target_mb_per_file: int | None = None,
        memory_budget_mb: int | None = None,
        **kwargs: t.Any,
    ) -> dict[str, t.Any]:
        """Rewrite the dataset into a new Hive partition layout.

        Delegates physical publication to fsspeckit's coordinated repartition
        maintenance operation. With ``unique=False`` (the default), every row
        is retained, including exact duplicates. ``unique=True`` is retained as
        a deprecated compatibility mode that explicitly selects fsspeckit's
        global deduplication-and-repartition operation using every data column
        as its deduplication key.

        ``sort_by`` and ``row_group_size`` are not supported by fsspeckit's
        repartition operation. Use ordered compaction after repartitioning when
        physical ordering is required.
        """
        if partitioning_falvor != "hive":
            raise ValueError("fsspeckit repartition supports only Hive partitioning")
        if partitioning_columns is None:
            raise ValueError("partitioning_columns must be provided")
        if sort_by:
            raise ValueError(
                "sort_by is not supported by repartition; run ordered compaction "
                "after the repartition instead."
            )
        if row_group_size not in (None, 256_000):
            warnings.warn(
                "row_group_size is not supported by fsspeckit repartition and "
                "will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported repartition options: {unexpected}")

        filesystem, path = _resolve_maintenance_target(self._filesystem, self._path)
        maintenance_filesystem: t.Any = filesystem
        repartition_kwargs: dict[str, t.Any] = {
            "partition_columns": partitioning_columns,
            "target_rows_per_file": max_rows_per_file,
            "target_mb_per_file": target_mb_per_file,
            "compression": compression,
            "derived_partition_columns": derived_partition_columns,
            "partition_timezone": partition_timezone,
        }
        if unique:
            schema = self.schema
            if schema is None:
                raise RuntimeError("Dataset schema is unavailable for deduplication")
            warnings.warn(
                "unique=True is deprecated; it invokes fsspeckit's explicit "
                "deduplicate-and-repartition operation. Use a named "
                "deduplication step before repartitioning instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            repartition_kwargs["key_columns"] = list(schema.names)
            if dry_run:
                plan = maintenance_filesystem.plan_parquet_global_repartition_deduplication(
                    path, **repartition_kwargs
                )
                return _maintenance_plan_to_dict(plan)
            result = maintenance_filesystem.deduplicate_and_repartition_parquet_dataset(
                path, **repartition_kwargs
            )
        else:
            repartition_kwargs["memory_budget_mb"] = memory_budget_mb
            if dry_run:
                plan = maintenance_filesystem.plan_parquet_repartition(
                    path, **repartition_kwargs
                )
                return _maintenance_plan_to_dict(plan)
            result = maintenance_filesystem.repartition_parquet_dataset(
                path, **repartition_kwargs
            )

        plain_result = _maintenance_to_plain(result)
        _require_successful_maintenance(plain_result, "repartition")
        self._partitioning = "hive"
        self.__dict__.pop("_partition_names", None)
        self.__dict__.pop("_partitioning_schema", None)
        self._refresh_after_rewrite()
        return plain_result

    def _dtype_schema_proposal(
        self,
        exclude: str | list[str] | None = None,
        strict: bool = True,
        include: str | list[str] | None = None,
        ts_unit: str | None = None,
        tz: str | None = None,
        infer_schema_size: int = 10_000,
    ) -> pa.Schema:
        """Infer the approved physical schema before any maintenance mutation."""
        if self.table is None:
            self.load()
        proposal = (
            self.table.pl.drop(self.partition_names)
            .head(infer_schema_size)
            .opt_dtype(
                strict=strict,
                exclude=exclude,
                include=include,
                shrink_numerics=True,
            )
            .collect()
        )
        target_schema = proposal.to_arrow().schema
        target_schema = convert_large_types_to_normal(target_schema)
        if ts_unit is not None or tz is not None:
            target_schema = convert_timestamp(
                target_schema,
                unit=ts_unit,
                tz=tz,
            )
        return target_schema

    @staticmethod
    def _dtype_cast_policy(strict: bool) -> CastPolicy:
        """Map pydala's inference strictness to fsspeckit's cast guarantee."""
        return CastPolicy.SAFE if strict else CastPolicy.LOOSE

    @staticmethod
    def _schema_rewrite_error(result: t.Any) -> str:
        """Describe a failed coordinated rewrite and its retained recovery data."""
        recovery = _maintenance_to_plain(result.recovery)
        return (
            "fsspeckit schema rewrite did not succeed: "
            f"{result.error or 'unknown error'}. "
            f"Live files were not refreshed; recovery details: {recovery!r}"
        )

    def optimize_dtypes(
        self,
        exclude: str | list[str] | None = None,
        strict: bool = True,
        include: str | list[str] | None = None,
        ts_unit: str | None = None,
        tz: str | None = None,
        infer_schema_size: int = 10_000,
        dry_run: bool = False,
        compression: str | None = None,
        target_rows_per_file: int | None = None,
        memory_budget_mb: int | None = None,
        **kwargs: t.Any,
    ) -> dict[str, t.Any] | None:
        """Propose dtypes with Polars, then publish them through one rewrite.

        ``strict=True`` maps to fsspeckit's ``safe`` cast policy: every
        narrowing must be value-preserving across the full physical dataset.
        ``strict=False`` maps to ``loose``; fsspeckit still validates every
        cast before publication and retains recovery artifacts on failure.
        With ``dry_run=True``, return the immutable coordinated rewrite plan
        without changing files or managed metadata.

        The legacy ``max_rows_per_file`` spelling remains supported as an
        alias for ``target_rows_per_file``. Other write-only keyword options
        are rejected because coordinated schema publication cannot apply them
        without reintroducing per-file writes.
        """
        if "max_rows_per_file" in kwargs:
            if target_rows_per_file is not None:
                raise TypeError(
                    "Pass only one of target_rows_per_file or max_rows_per_file"
                )
            target_rows_per_file = kwargs.pop("max_rows_per_file")
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(
                "optimize_dtypes does not support write-only options during "
                f"coordinated schema publication: {unsupported}"
            )

        target_schema = self._dtype_schema_proposal(
            exclude=exclude,
            strict=strict,
            include=include,
            ts_unit=ts_unit,
            tz=tz,
            infer_schema_size=infer_schema_size,
        )
        filesystem, path = _resolve_maintenance_target(self._filesystem, self._path)
        rewrite_kwargs = {
            "target_schema": target_schema,
            "cast_policy": self._dtype_cast_policy(strict),
            "target_rows_per_file": target_rows_per_file,
            "compression": compression,
            "memory_budget_mb": memory_budget_mb,
        }
        if dry_run:
            plan = filesystem.plan_parquet_schema_rewrite(path, **rewrite_kwargs)
            return _maintenance_plan_to_dict(plan)

        result = filesystem.schema_rewrite_parquet_dataset(path, **rewrite_kwargs)
        if not result.succeeded:
            raise RuntimeError(self._schema_rewrite_error(result))
        self._refresh_after_rewrite()
        return _maintenance_to_plain(result)


class PyarrowDataset(BaseDataset):
    """A dataset implementation using PyArrow's native dataset format.

    This class provides a generic PyArrow dataset implementation that can
    work with various formats supported by PyArrow. It serves as a base
    class for format-specific implementations like CsvDataset.
    """

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
    ) -> None:
        """Initialize a PyarrowDataset instance.

        Args:
            path: The path to the dataset.
            name: The name of the dataset. If None, uses basename of path.
            filesystem: The filesystem to use for data access.
            bucket: The bucket name for cloud storage.
            partitioning: The partitioning scheme.
            format: The file format (default: 'parquet').
            cached: Whether to use caching for filesystem operations.
            timestamp_column: The column to use as timestamp for time operations.
            ddb_con: Existing DuckDB connection. If None, creates a new one.
            **fs_kwargs: Additional arguments for filesystem configuration.
        """
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
        self._load_eager()


class CSVDataset(PyarrowDataset):
    """A dataset implementation for CSV files.

    This class provides specialized handling for CSV datasets using
    PyArrow's CSV reading capabilities. It inherits from PyarrowDataset
    and configures it specifically for CSV format.
    """

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
    ) -> None:
        """Initialize a CsvDataset instance.

        Args:
            path: The path to the CSV dataset.
            name: The name of the dataset. If None, uses basename of path.
            filesystem: The filesystem to use for data access.
            bucket: The bucket name for cloud storage.
            partitioning: The partitioning scheme for the CSV files.
            cached: Whether to use caching for filesystem operations.
            timestamp_column: The column to use as timestamp for time operations.
            ddb_con: Existing DuckDB connection. If None, creates a new one.
            **fs_kwargs: Additional arguments for filesystem configuration.
        """
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


class JSONDataset(BaseDataset):
    """A dataset implementation for JSON files.

    This class provides specialized handling for JSON datasets using
    PyArrow's JSON reading capabilities. It supports reading JSON files
    with automatic schema inference and type optimization.
    """

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
    ) -> None:
        """Initialize a JsonDataset instance.

        Args:
            path: The path to the JSON dataset.
            name: The name of the dataset. If None, uses basename of path.
            filesystem: The filesystem to use for data access.
            bucket: The bucket name for cloud storage.
            partitioning: The partitioning scheme for the JSON files.
            cached: Whether to use caching for filesystem operations.
            timestamp_column: The column to use as timestamp for time operations.
            ddb_con: Existing DuckDB connection. If None, creates a new one.
            **fs_kwargs: Additional arguments for filesystem configuration.
        """
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
        self._load_eager()

    def load(self) -> None:
        """Load the JSON dataset.

        This method reads JSON files from the dataset path using PyArrow's
        JSON reader with automatic type inference. It creates an Arrow dataset
        and registers it with DuckDB for SQL operations. It also attempts
        to identify timestamp columns automatically.

        Returns:
            None
        """
        files = self._filesystem.glob(safe_join(self._path, "*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON files found at {self._path!r}")
        data = self._filesystem.read_json(files, as_dataframe=True, concat=True)
        self._arrow_dataset = pds.dataset(data.opt_dtype(strict=False).to_arrow())
        self.table = PydalaTable(result=self._arrow_dataset, ddb_con=self.ddb_con)

        self.ddb_con.register(f"{self.name}", self._arrow_dataset)
        # self.ddb_con.register("arrow__dataset", self._arrow_parquet_dataset)

        if self._timestamp_column is None:
            self._timestamp_columns = get_timestamp_column(self.table.pl.head(10))
            if len(self._timestamp_columns) > 1:
                self._timestamp_column = self._timestamp_columns[0]


class Optimize(ParquetDataset):
    """Compatibility-only subclass for Parquet dataset optimization.

    All optimization and maintenance methods are defined directly on
    :class:`ParquetDataset` per ADR 0001. This class is retained solely
    for backward-compatible imports (``from pydala.dataset import Optimize``).
    """

    pass

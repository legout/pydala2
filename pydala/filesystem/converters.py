"""
Data format converters for different file types.
"""
import datetime as dt
from abc import ABC, abstractmethod
from typing import Any, Union

import duckdb as ddb
import orjson
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem

from ..helpers.misc import read_table, run_parallel
from ..schema import convert_large_types_to_normal


class DataConverter(ABC):
    """Abstract base class for data format converters."""

    @abstractmethod
    def read(self, fs: AbstractFileSystem, path: str, **kwargs) -> Any:
        """Read data from filesystem.

        Args:
            fs: Filesystem instance
            path: File path
            **kwargs: Additional arguments

        Returns:
            Read data
        """
        pass

    @abstractmethod
    def write(self, fs: AbstractFileSystem, data: Any, path: str, **kwargs) -> None:
        """Write data to filesystem.

        Args:
            fs: Filesystem instance
            data: Data to write
            path: File path
            **kwargs: Additional arguments
        """
        pass

    @abstractmethod
    def read_dataset(self, fs: AbstractFileSystem, path: Union[str, list], **kwargs) -> Any:
        """Read dataset from filesystem.

        Args:
            fs: Filesystem instance
            path: File path(s)
            **kwargs: Additional arguments

        Returns:
            Read dataset
        """
        pass


class ParquetConverter(DataConverter):
    """Converter for Parquet format."""

    def read(self, fs: AbstractFileSystem, path: str, filename: bool = False, **kwargs) -> Union[dict[str, pl.DataFrame], pl.DataFrame]:
        """Read Parquet file.

        Args:
            fs: Filesystem instance
            path: File path
            filename: Whether to return dict with filename as key
            **kwargs: Additional arguments

        Returns:
            DataFrame or dict with filename key
        """
        data = pl.from_arrow(read_table(path, filesystem=fs, **kwargs))

        if filename:
            return {path: data}
        return data

    def write(self, fs: AbstractFileSystem, data: Union[pl.DataFrame, pa.Table, pd.DataFrame, ddb.DuckDBPyRelation], path: str, **kwargs) -> None:
        """Write data to Parquet file.

        Args:
            fs: Filesystem instance
            data: Data to write
            path: File path
            **kwargs: Additional arguments
        """
        if isinstance(data, pl.DataFrame):
            data = data.to_arrow()
            data = data.cast(convert_large_types_to_normal(data.schema))
        elif isinstance(data, pd.DataFrame):
            data = pa.Table.from_pandas(data, preserve_index=False)
        elif isinstance(data, ddb.DuckDBPyRelation):
            data = data.arrow()

        pq.write_table(data, path, filesystem=fs, **kwargs)

    def read_dataset(self, fs: AbstractFileSystem, path: Union[str, list], concat: bool = True, filename: bool = False, **kwargs) -> Any:
        """Read Parquet dataset.

        Args:
            fs: Filesystem instance
            path: File path(s)
            concat: Whether to concatenate DataFrames
            filename: Whether to return dict with filename keys
            **kwargs: Additional arguments

        Returns:
            DataFrame(s) or dict with filename keys
        """
        if filename:
            concat = False

        if isinstance(path, str):
            files = fs.glob(f"{path}/*.parquet")
        else:
            files = path
        if isinstance(files, str):
            files = [files]

        dfs = run_parallel(self.read, files, filesystem=fs, filename=filename, **kwargs)

        if concat and isinstance(dfs, list):
            dfs = pl.concat(dfs, how="diagonal_relaxed")

        return dfs[0] if isinstance(dfs, list) and len(dfs) == 1 else dfs

    def read_schema(self, fs: AbstractFileSystem, path: str, **kwargs) -> pa.Schema:
        """Read Parquet schema.

        Args:
            fs: Filesystem instance
            path: File path
            **kwargs: Additional arguments

        Returns:
            Arrow schema
        """
        return pq.read_schema(path, filesystem=fs, **kwargs)

    def read_metadata(self, fs: AbstractFileSystem, path: str, **kwargs) -> pq.FileMetaData:
        """Read Parquet metadata.

        Args:
            fs: Filesystem instance
            path: File path
            **kwargs: Additional arguments

        Returns:
            Parquet file metadata
        """
        return pq.read_metadata(path, filesystem=fs, **kwargs)

    def to_pyarrow_dataset(self, fs: AbstractFileSystem, path: str, **kwargs) -> pds.Dataset:
        """Create PyArrow dataset.

        Args:
            fs: Filesystem instance
            path: Dataset path
            **kwargs: Additional arguments

        Returns:
            PyArrow dataset
        """
        return pds.dataset(
            fs.glob(f"{path}/*.parquet"),
            filesystem=fs,
            **kwargs
        )

    def write_dataset(self,
                      fs: AbstractFileSystem,
                      data: Union[pl.DataFrame, pa.Table, pd.DataFrame, ddb.DuckDBPyRelation, list],
                      path: str,
                      basename: str = None,
                      concat: bool = True,
                      schema: pa.Schema = None,
                      partitioning: Union[str, list, pds.Partitioning] = None,
                      partitioning_flavor: str = "hive",
                      mode: str = "append",
                      format: str = "parquet",
                      **kwargs) -> None:
        """Write PyArrow dataset.

        Args:
            fs: Filesystem instance
            data: Data to write
            path: Base directory
            basename: Filename template
            concat: Whether to concatenate data
            schema: Arrow schema
            partitioning: Partitioning specification
            partitioning_flavor: Partitioning flavor
            mode: Write mode
            format: File format
            **kwargs: Additional arguments
        """
        if not isinstance(data, list):
            data = [data]

        # Convert all data to Arrow tables
        converted_data = []
        for item in data:
            if isinstance(item, pl.DataFrame):
                item = item.to_arrow()
                item = item.cast(convert_large_types_to_normal(item.schema))
            elif isinstance(item, pd.DataFrame):
                item = pa.Table.from_pandas(item, preserve_index=False)
            elif isinstance(item, ddb.DuckDBPyRelation):
                item = item.arrow()
            converted_data.append(item)

        if concat:
            data = pa.concat_tables(converted_data, promote=True)
        else:
            data = converted_data

        # Handle different write modes
        if mode == "delete_matching":
            existing_data_behavior = "delete_matching"
        elif mode == "append":
            existing_data_behavior = "overwrite_or_ignore"
        elif mode == "overwrite":
            fs.rm(path, recursive=True)
            existing_data_behavior = "overwrite_or_ignore"
        else:
            existing_data_behavior = mode

        # Generate default basename if not provided
        if basename is None:
            basename = f"data-{dt.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]}-{{i}}.{format}"

        pds.write_dataset(
            data=data,
            base_dir=path,
            basename_template=basename,
            partitioning=partitioning,
            partitioning_flavor=partitioning_flavor,
            filesystem=fs,
            existing_data_behavior=existing_data_behavior,
            schema=schema,
            format=format,
            **kwargs,
        )


class JsonConverter(DataConverter):
    """Converter for JSON format."""

    def read(self, fs: AbstractFileSystem, path: str, filename: bool = False, as_dataframe: bool = True, flatten: bool = True, **kwargs) -> Union[dict, pl.DataFrame]:
        """Read JSON file.

        Args:
            fs: Filesystem instance
            path: File path
            filename: Whether to return dict with filename as key
            as_dataframe: Whether to convert to DataFrame
            flatten: Whether to flatten nested data
            **kwargs: Additional arguments

        Returns:
            Data or DataFrame
        """
        with fs.open(path) as f:
            data = orjson.loads(f.read())

        if as_dataframe:
            data = pl.from_dicts(data)
            if flatten:
                data = data.explode_all().unnest_all()

        if filename:
            return {path: data}

        return data

    def write(self, fs: AbstractFileSystem, data: Union[dict, pl.DataFrame, pa.Table, pd.DataFrame, ddb.DuckDBPyRelation], path: str) -> None:
        """Write data to JSON file.

        Args:
            fs: Filesystem instance
            data: Data to write
            path: File path
        """
        if isinstance(data, pl.DataFrame):
            data = data.to_arrow()
            data = data.cast(convert_large_types_to_normal(data.schema)).to_pydict()
        elif isinstance(data, pd.DataFrame):
            data = pa.Table.from_pandas(data, preserve_index=False).to_pydict()
        elif isinstance(data, ddb.DuckDBPyRelation):
            data = data.arrow().to_pydict()

        with fs.open(path, "w") as f:
            f.write(orjson.dumps(data))

    def read_dataset(self, fs: AbstractFileSystem, path: Union[str, list], filename: bool = False, as_dataframe: bool = True, flatten: bool = True, concat: bool = True, **kwargs) -> Any:
        """Read JSON dataset.

        Args:
            fs: Filesystem instance
            path: File path(s)
            filename: Whether to return dict with filename keys
            as_dataframe: Whether to convert to DataFrame
            flatten: Whether to flatten nested data
            concat: Whether to concatenate DataFrames
            **kwargs: Additional arguments

        Returns:
            Data or DataFrame(s)
        """
        if filename:
            concat = False

        if isinstance(path, str):
            files = fs.glob(f"{path}/*.json")
        else:
            files = path
        if isinstance(files, str):
            files = [files]

        data = run_parallel(
            self.read,
            files,
            filesystem=fs,
            filename=filename,
            as_dataframe=as_dataframe,
            flatten=flatten,
        )

        if as_dataframe and concat and isinstance(data, list):
            data = pl.concat(data, how="diagonal_relaxed")

        return data[0] if isinstance(data, list) and len(data) == 1 else data


class CsvConverter(DataConverter):
    """Converter for CSV format."""

    def read(self, fs: AbstractFileSystem, path: str, filename: bool = False, **kwargs) -> Union[dict[str, pl.DataFrame], pl.DataFrame]:
        """Read CSV file.

        Args:
            fs: Filesystem instance
            path: File path
            filename: Whether to return dict with filename as key
            **kwargs: Additional arguments

        Returns:
            DataFrame or dict with filename key
        """
        with fs.open(path) as f:
            data = pl.read_csv(f, **kwargs)

        if filename:
            return {path: data}
        return data

    def write(self, fs: AbstractFileSystem, data: Union[pl.DataFrame, pa.Table, pd.DataFrame, ddb.DuckDBPyRelation], path: str, **kwargs) -> None:
        """Write data to CSV file.

        Args:
            fs: Filesystem instance
            data: Data to write
            path: File path
            **kwargs: Additional arguments
        """
        if isinstance(data, pa.Table):
            data = pl.from_arrow(data)
        elif isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)
        elif isinstance(data, ddb.DuckDBPyRelation):
            data = data.pl()

        with fs.open(path, "w") as f:
            data.write_csv(f, **kwargs)

    def read_dataset(self, fs: AbstractFileSystem, path: Union[str, list], concat: bool = True, filename: bool = False, **kwargs) -> Any:
        """Read CSV dataset.

        Args:
            fs: Filesystem instance
            path: File path(s)
            concat: Whether to concatenate DataFrames
            filename: Whether to return dict with filename keys
            **kwargs: Additional arguments

        Returns:
            DataFrame(s) or dict with filename keys
        """
        if filename:
            concat = False

        if isinstance(path, str):
            files = fs.glob(f"{path}/*.csv")
        else:
            files = path
        if isinstance(files, str):
            files = [files]

        dfs = run_parallel(self.read, files, filesystem=fs, filename=filename, **kwargs)

        if concat and isinstance(dfs, list):
            dfs = pl.concat(dfs, how="diagonal_relaxed")

        return dfs[0] if isinstance(dfs, list) and len(dfs) == 1 else dfs


class ConversionService:
    """Service for handling data format conversions."""

    def __init__(self):
        self._converters = {
            "parquet": ParquetConverter(),
            "json": JsonConverter(),
            "csv": CsvConverter(),
        }

    def get_converter(self, format_name: str) -> DataConverter:
        """Get converter for specific format.

        Args:
            format_name: Format name (parquet, json, csv)

        Returns:
            Converter instance

        Raises:
            ValueError: If format is not supported
        """
        converter = self._converters.get(format_name.lower())
        if not converter:
            raise ValueError(f"Unsupported format: {format_name}")
        return converter

    def register_converter(self, format_name: str, converter: DataConverter) -> None:
        """Register a new converter.

        Args:
            format_name: Format name
            converter: Converter instance
        """
        self._converters[format_name.lower()] = converter
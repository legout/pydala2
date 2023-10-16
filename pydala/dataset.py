import datetime as dt
import os
import re
import uuid
import tqdm
from typing import List, Optional, Union

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.fs as pfs
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem

from .filesystem import clear_cache, get_filesystem
from .helpers import (
    collect_metadata,
    get_row_group_stats,
    get_timestamp_column,
    humanized_size_to_bytes,
    partition_by,
    run_parallel,
)
from .io import read_table, write_table
from .polars_ext import pl as _pl
from .schema import repair_schema, unify_schemas


class ParquetDatasetMetadata:
    def __init__(
        self,
        path: str,
        filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
        bucket: str | None = None,
        cached: bool = False,
        **cached_options,
    ) -> None:
        """
        A class representing metadata for a Parquet dataset.

        Args:
            path (str): The path to the dataset.
            filesystem (AbstractFileSystem | pfs.FileSystem | None, optional): The filesystem to use. Defaults to None.
            bucket (str | None, optional): The name of the bucket to use. Defaults to None.
            cached (bool, optional): Whether to use a cached filesystem. Defaults to False.
            **cached_options: Additional options to pass to the cached filesystem.

        Returns:
            None
        """
        self._path = path
        self._bucket = bucket
        self._cached = cached
        self._base_filesystem = filesystem
        self._filesystem = get_filesystem(
            bucket=bucket, fs=filesystem, cached=cached, **cached_options
        )

        self._files = sorted(self._filesystem.glob(os.path.join(path, "**.parquet")))

        self._file = os.path.join(path, "_metadata")
        if self.has_metadata_file:
            self._metadata = pq.read_metadata(
                self.metadata_file, filesystem=self._filesystem
            )

    def collect_file_metadata(self, files: list[str] | None = None, **kwargs) -> None:
        """
        Collects metadata for the specified files and updates the `file_metadata` attribute of the dataset object.

        Args:
            files (list[str] | None): A list of file paths to collect metadata for. If None, metadata will be collected for all files in the dataset.
            **kwargs: Additional keyword arguments to pass to the `collect_metadata` function.

        Returns:
            None
        """
        if files is None:
            files = self._files

        file_metadata = collect_metadata(
            files=files, filesystem=self._filesystem, **kwargs
        )

        if file_metadata:
            for f in file_metadata:
                file_metadata[f].set_file_path(f.split(self._path)[-1].lstrip("/"))

            if hasattr(self, "file_metadata"):
                self.file_metadata.update(file_metadata)
            else:
                self.file_metadata = file_metadata

    def reload_files(self) -> None:
        """
        Reloads the list of files in the dataset directory. This method should be called
        after adding or removing files from the directory to ensure that the dataset object
        has an up-to-date list of files.

        Returns:
            None
        """
        self._files = sorted(
            self._filesystem.glob(os.path.join(self._path, "**.parquet"))
        )

    def update_file_metadata(self, **kwargs) -> None:
        """
        Updates the metadata for files in the dataset.

        This method reloads the files in the dataset and then updates the metadata for any files that have not yet been
        processed. If the dataset already has file metadata, this method will only update the metadata for files that have
        been added since the last time metadata was collected.

        Args:
            **kwargs: Additional keyword arguments to pass to the `collect_file_metadata` method.

        Returns:
            None
        """
        self.reload_files()

        files = list((set(self._files) - set(self.files_in_metadata)))
        if hasattr(self, "file_metadata"):
            files = list(set(files) - set(self.file_metadata.keys()))

        if len(files):
            self.collect_file_metadata(files=files, **kwargs)

    def unify_metadata_schema(
        self,
        format_version: str = None,
        update_file_metadata: bool = True,
        unify_schema_args: dict = {},
        **kwargs,
    ) -> None:
        """
        Unifies the metadata schema of all files in the dataset to a common schema.

        Args:
            format_version (str, optional): The format version to use for the unified schema. Defaults to "1.0".
            update_file_metadata (bool, optional): Whether to update the file metadata before unifying the schemas. Defaults to True.
            unify_schema_args (dict, optional): Additional arguments to pass to the `unify_schemas` function. Defaults to {}.
            **kwargs: Additional keyword arguments to pass to the `update_file_metadata` and `repair_schema` methods.

        Returns:
            None
        """

        if update_file_metadata:
            self.update_file_metadata(**kwargs)

        if hasattr(self, "file_metadata"):
            # self.collect_file_metadata(**kwargs)

            schemas = {
                f: self.file_metadata[f].schema.to_arrow_schema()
                for f in self.file_metadata
            }
            schemas_v = list(schemas.values())
            format_version = (
                format_version or self.file_metadata[self._files[0]].format_version
            )
            if self.has_metadata:
                metadata_schema = self._metadata.schema.to_arrow_schema()
                schemas_v.insert(0, metadata_schema)
                format_version = self._metadata.format_version

            unified_schema, schemas_equal = unify_schemas(
                schemas_v, **unify_schema_args
            )

            files = []
            if not schemas_equal:
                files += [f for f in self.file_metadata if schemas[f] != unified_schema]
            files += [
                f
                for f in files
                if self.file_metadata[f].format_version != format_version
            ]
            files = set(files)

            if len(files):
                # print("repair")
                repair_schema(
                    files=files,
                    schema=unified_schema,
                    filesystem=self._filesystem,
                    version=format_version,
                    **kwargs,
                )
                self.clear_cache()
                self.collect_file_metadata(files=files)

    def load_metadata(
        self,
        reload: bool = False,
        update: bool = True,
        format_version: str = "1.0",
        unify_schema_args: dict = {},
        **kwargs,
    ) -> None:
        """Loads the metadata for the dataset.

        Args:
            reload (bool, optional): Whether to reload the metadata. Defaults to False.
            update (bool, optional): Whether to update the metadata. Defaults to True.
            format_version (str, optional): The format version of the metadata. Defaults to "1.0".
            unify_schema_args (dict, optional): Arguments to unify the metadata schema. Defaults to {}.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """

        if reload:
            update = False
            self.reload_files()
            self.delete_metadata_file()
            if self.has_metadata:
                del self._metadata
                if hasattr(self, "_file_schema"):
                    del self._file_schema
                if hasattr(self, "_file_catalog"):
                    del self._file_catalog

            if hasattr(self, "file_metadata"):
                del self.file_metadata

            self.clear_cache()
            self.collect_file_metadata(files=self._files, **kwargs)

        self.unify_metadata_schema(
            update_file_metadata=update,
            format_version=format_version,
            unify_schema_args=unify_schema_args,
            **kwargs,
        )

        if not hasattr(self, "file_metadata"):
            return

        if not self.has_metadata:
            self._metadata = self.file_metadata[list(self.file_metadata.keys())[0]]
            for f in list(self.file_metadata.keys())[1:]:
                self._metadata.append_row_groups(self.file_metadata[f])
        else:
            files = list(set(self.file_metadata.keys()) - set(self.files_in_metadata))
            for f in files:
                self._metadata.append_row_groups(self.file_metadata[f])

        self.write_metadata_file()

    def write_metadata_file(self) -> None:
        """
        Writes metadata to a file named '_metadata' in the dataset directory.

        Returns:
            None
        """
        with self._filesystem.open(os.path.join(self._path, "_metadata"), "wb") as f:
            self._metadata.write_metadata_file(f)

    def delete_metadata_file(self) -> None:
        """
        Deletes the metadata file associated with the dataset, if it exists.

        Raises:
            None
        """
        if self.has_metadata_file:
            self._filesystem.rm(self.metadata_file)

    def clear_cache(self) -> None:
        """
        Clears the cache for the dataset's filesystem and base filesystem.

        This method clears the cache for the dataset's filesystem and base filesystem,
        which can be useful if the dataset has been modified and the cache needs to be
        updated accordingly.

        Returns:
            None
        """
        clear_cache(self._filesystem)
        clear_cache(self._base_filesystem)

    @property
    def has_metadata(self):
        """
        Returns True if the dataset has metadata, False otherwise.
        """
        return hasattr(self, "_metadata")

    @property
    def metadata(self):
        """
        Returns the metadata associated with the dataset.

        If the metadata has not been loaded yet, it will be loaded before being returned.

        Returns:
            dict: The metadata associated with the dataset.
        """
        if not self.has_metadata:
            self.load_metadata()
        return self._metadata

    @property
    def schema(self):
        """
        Returns the Arrow schema for the dataset.

        If the dataset has metadata, the schema includes both the metadata and the data schema.
        Otherwise, the schema only includes the data schema.

        Returns:
            pyarrow.Schema: The Arrow schema for the dataset.
        """
        if not hasattr(self, "_schema"):
            self._schema = self.metadata.schema.to_arrow_schema()
            if self.has_metadata:
                self._file_schema = self.metadata.schema.to_arrow_schema()
            else:
                self._file_schema = pa.schema([])
        return self._schema

    @property
    def file_schema(self):
        """
        Returns the Arrow schema of the dataset file.
        If metadata is available, it returns the schema from metadata.
        If metadata is not available, it returns an empty schema.
        """
        if not hasattr(self, "_file_schema"):
            if self.has_metadata:
                self._file_schema = self.metadata.schema.to_arrow_schema()
            else:
                self._file_schema = pa.schema([])
        return self._file_schema

    @property
    def metadata_file(self):
        """
        Returns the path to the metadata file for this dataset. If the metadata file
        does not exist, it will be created.

        Returns:
            str: The path to the metadata file.
        """
        if not hasattr(self, "_metadata_file"):
            self._file = os.path.join(self._path, "_metadata")
        return self._file

    @property
    def has_metadata_file(self):
        """
        Returns True if the dataset has a metadata file, False otherwise.
        """
        return self._filesystem.exists(self.metadata_file)

    @property
    def has_files(self):
        """
        Returns True if the dataset has files, False otherwise.
        """
        return len(self._files) > 0

    @property
    def files_in_metadata(self) -> list:
        """
        Returns a list of file paths referenced in the metadata of the dataset.

        Returns:
            A list of file paths referenced in the metadata of the dataset.
        """
        if self.has_metadata:
            return list(
                set(
                    [
                        os.path.join(
                            self._path, self._metadata.row_group(i).column(0).file_path
                        )
                        for i in range(self._metadata.num_row_groups)
                    ]
                )
            )
        else:
            return []


class ParquetDataset(ParquetDatasetMetadata):
    def __init__(
        self,
        path: str,
        filesystem: AbstractFileSystem | None = None,
        bucket: str | None = None,
        partitioning: str | None = None,
        cached: bool = False,
        **cached_options,
    ):
        """
        Initialize a Dataset object.

        Args:
            path (str): The path to the dataset.
            filesystem (AbstractFileSystem, optional): The filesystem to use. Defaults to None.
            bucket (str, optional): The bucket to use. Defaults to None.
            partitioning (str, optional): The partitioning scheme to use. Defaults to None.
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
            **cached_options,
        )
        # self.metadata = ParquetDatasetMetadata(path=path, filesystem=filesystem, bucket=bucket, )
        if self.has_files:
            if partitioning == "ignore":
                self._partitioning = None
            elif partitioning is None and "=" in self._files[0]:
                partitioning = "hive"
            else:
                self._partitioning = partitioning
            self.reset_scan_files()
        else:
            self._partitioning = None

        self._ddb = duckdb.connect()

    def load(
        self,
        reload: bool = False,
        update: bool = True,
        format_version: str = None,
        unify_schema_args: dict = {},
        **kwargs,
    ):
        """
        Load the dataset metadata and create a PyArrow ParquetDataset object.

        Args:
            reload (bool, optional): Whether to reload the metadata file if it already exists. Defaults to False.
            update (bool, optional): Whether to update the metadata file with any new files found. Defaults to True.
            format_version (str, optional): The version of the metadata format to use. Defaults to "1.0".
            **kwargs: Additional keyword arguments to pass to the PyArrow ParquetDataset constructor.

        Returns:
            None
        """
        if self.has_files:
            self.load_metadata(
                reload=reload,
                update=update,
                format_version=format_version,
                unify_schema_args=unify_schema_args,
                **kwargs,
            )

            self._base_dataset = pds.parquet_dataset(
                self.metadata_file,
                # schema=self.schema,
                partitioning=self._partitioning,
                filesystem=self._filesystem,
            )
            self.reset_scan_files()
            self._timestamp_columns = get_timestamp_column(self.pl.head(1))

    @property
    def is_loaded(self):
        """
        Returns True if the dataset has been loaded into memory, False otherwise.
        """
        return hasattr(self, "_base_dataset")

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
            return self._base_dataset.count_rows
        else:
            print(f"No dataset loaded yet. Run {self}.load()")
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
                self._partitioning_schema = self._base_dataset.partitioning.schema
            else:
                print(f"No dataset loaded yet. Run {self}.load()")
                return pa.schema([])
        return self._partitioning_schema

    @property
    def schema(self):
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
                print(f"No dataset loaded yet. Run {self}.load()")
                return []
        return self._partitioning_names

    def gen_file_catalog(self):
        """
        Generates a polars DataFrame with statistics for each row group in the dataset.

        """
        self._file_catalog = _pl.DataFrame(
            [
                get_row_group_stats(
                    self.metadata.row_group(i), partitioning=self._partitioning
                )
                for i in range(self.metadata.num_row_groups)
            ]
        )

    def reset_scan_files(self):
        """
        Resets the list of scanned files to the original list of files.
        """
        self._is_scanned = False
        # self.reload_files()
        # self.gen_file_catalog()
        self._scan_files = self._files.copy()
        self.gen_file_catalog()

    @staticmethod
    def _gen_filter_expr_mod(filter_expr: str, exclude_columns: list[str] = []) -> list:
        # chech if filter_expr is a date string
        filter_expr_mod = []
        is_date = False
        res = re.search("(\d{4}-\d{1,2}-\d{1,2})", filter_expr)
        if res:
            if res.end() + 1 == res.endpos:
                is_date = True
        # print(is_date)
        if ">" in filter_expr:
            if not filter_expr.split(">")[0].lstrip("(") in exclude_columns:
                filter_expr_mod.append(
                    f"({filter_expr.replace('>', '_max::DATE>')} OR {filter_expr.split('>')[0]}_max::DATE IS NULL)"
                ) if is_date else filter_expr_mod.append(
                    f"({filter_expr.replace('>', '_max>')} OR {filter_expr.split('>')[0]}_max IS NULL)"
                )
            else:
                filter_expr_mod.append(filter_expr)
        elif "<" in filter_expr:
            if not filter_expr.split("<")[0].lstrip("(") in exclude_columns:
                filter_expr_mod.append(
                    f"({filter_expr.replace('<', '_min::DATE<')} OR {filter_expr.split('<')[0]}_min::DATE IS NULL)"
                ) if is_date else filter_expr_mod.append(
                    f"({filter_expr.replace('<', '_min<')} OR {filter_expr.split('<')[0]}_min IS NULL)"
                )
            else:
                filter_expr_mod.append(filter_expr)
        elif "=" in filter_expr:
            if not filter_expr.split("=")[0].lstrip("(") in exclude_columns:
                filter_expr_mod.append(
                    f"({filter_expr.replace('=', '_min::DATE<=')} OR {filter_expr.split('=')[0]}_min::DATE IS NULL)"
                ) if is_date else filter_expr_mod.append(
                    f"({filter_expr.replace('=', '_min<=')} OR {filter_expr.split('=')[0]}_min IS NULL)"
                )
                filter_expr_mod.append(
                    f"({filter_expr.replace('=', '_max::DATE>=')} OR {filter_expr.split('=')[0]}_max::DATE IS NULL)"
                ) if is_date else filter_expr_mod.append(
                    f"({filter_expr.replace('=', '_max>=')} OR {filter_expr.split('=')[0]}_max IS NULL)"
                )
        else:
            filter_expr_mod.append(filter_expr)

        return filter_expr_mod

    def scan(self, filter_expr: str | None = None, lazy: bool = True):
        """
        Scans the dataset for files that match the given filter expression.

        Args:
            filter_expr (str | None): A filter expression to apply to the dataset. Defaults to None.
            lazy (bool): Whether to perform the scan lazily or eagerly. Defaults to True.

        Returns:
            None
        """
        self._filter_expr = filter_expr
        if filter_expr is not None:
            filter_expr = [fe.strip() for fe in filter_expr.split("AND")]

            filter_expr_mod = []
            for fe in filter_expr:
                filter_expr_mod += self._gen_filter_expr_mod(
                    fe, exclude_columns=self.file_catalog.columns
                )

            self._filter_expr_mod = " AND ".join(filter_expr_mod)

            self._scan_files = [
                os.path.join(self._path, sf)
                for sf in self._ddb.from_arrow(self.file_catalog.to_arrow())
                .filter(self._filter_expr_mod)
                .pl()["file_path"]
                .to_list()
            ]

        # return self

    @property
    def is_scanned(self):
        """
        Check if all files in the dataset have been scanned.

        Returns:
            bool: True if all files have been scanned, False otherwise.
        """
        return sorted(self._scan_files) == sorted(self._files)

    @property
    def filter_expr(self):
        """
        Returns the filter expression and its module.

        If the filter expression has not been set yet, it will return None for both the expression and its module.
        """
        if not hasattr(self, "_filter_expr"):
            self._filter_expr = None
            self._filter_expr_mod = None
        return self._filter_expr, self._filter_expr_mod

    def to_dataset(
        self, filter_expr: str | None = None, from_: str = "scan_files"
    ) -> pds.Dataset:
        """
        Converts the current Pydala object to a PyArrow Dataset object.

        Args:
            filter_expr (str | None): Optional filter expression to apply to the dataset.
            from_ (str): The source of the dataset. Can be either "scan_files" or "files". Defaults to "scan_files".

        Returns:
            pds.Dataset: The PyArrow Dataset object.
        """
        self.scan(filter_expr=filter_expr, lazy=True)
        if filter_expr is not None:
            from_ = "scan_files"
        if from_ == "scan_files":
            if not hasattr(self, "_scan_files"):
                self._scan_files = self._files.copy()
            files = self._scan_files
        else:
            files = self._files
        if hasattr(self, "_dataset"):
            if sorted(self._dataset.files) == sorted(files):
                return self._dataset
        self._dataset = pds.dataset(
            files,
            partitioning=self._partitioning,
            filesystem=self._filesystem,
        )

        return self._dataset

    @property
    def dataset(self) -> pds.Dataset:
        """
        Returns a PyArrow Dataset object representing the data in this instance.
        """
        return self.to_dataset()

    def to_table(
        self, filter_expr: str | None = None, from_: str = "scan_files", **kwargs
    ) -> pa.Table:
        """
        Converts the dataset to a PyArrow table by reading the data from disk.

        Args:
            filter_expr (str | None): An optional filter expression to apply to the dataset.
            from_ (str): The source of the files to read. Can be either "scan_files" (default) or "files".
            **kwargs: Additional keyword arguments to pass to the `read_table` function.

        Returns:
            A PyArrow table containing the data from the dataset.

        Raises:
            ValueError: If the `from_` argument is not valid.
        """
        self.scan(filter_expr=filter_expr, lazy=False)
        if filter_expr is not None:
            from_ = "scan_files"
        if from_ == "scan_files":
            files = self._scan_files
        else:
            files = self._files
        if hasattr(self, "_table"):
            if sorted(self._table_files) == sorted(files):
                return self._table

        self._table_files = files.copy()
        self._table = pa.concat_tables(
            run_parallel(
                read_table,
                files,
                schema=self.schema,
                # format="parquet",
                filesystem=self._filesystem,
                partitioning=self._partitioning,
                **kwargs,
            )
        )
        return self._table

    @property
    def table(self) -> pa.Table:
        """
        Returns a PyArrow Table representation of the dataset.

        Returns:
            pa.Table: A PyArrow Table representation of the dataset.
        """
        return self.to_table()

    def to_duckdb(
        self,
        filter_expr: str | None = None,
        lazy: bool = True,
        from_="scan_files",
    ) -> duckdb.DuckDBPyRelation:
        """
        Converts the dataset to a DuckDBPyRelation object.

        Args:
            filter_expr (str | None): An optional filter expression to apply to the dataset.
            lazy (bool): If True, the dataset is lazily loaded. If False, the dataset is eagerly loaded.
            from_ (str): The source of the dataset. Can be either "scan_files" or "files". Defaults to "scan_files".

        Returns:
            duckdb.DuckDBPyRelation: A DuckDBPyRelation object representing the dataset.
        """

        if lazy:
            self.to_dataset(filter_expr=filter_expr, from_=from_)
            return self._ddb.from_arrow(self._dataset)
        else:
            self.to_table(filter_expr=filter_expr, from_=from_)
            return self._ddb.from_arrow(self._table)

    @property
    def ddb(self) -> duckdb.DuckDBPyRelation:
        """
        Converts the dataset to a DuckDBPyRelation object.

        Returns:
            duckdb.DuckDBPyRelation: A DuckDBPyRelation object representing the dataset.
        """
        return self.to_duckdb()

    def to_polars(
        self,
        filter_expr: str | None = None,
        lazy: bool = True,
        from_="scan_files",
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
            self.to_dataset(filter_expr=filter_expr, from_=from_)
            return _pl.scan_pyarrow_dataset(self._dataset)
        else:
            self.to_table(filter_expr=filter_expr, from_=from_)
            return _pl.from_arrow(self._table)

    @property
    def pl(self) -> _pl.DataFrame:
        """
        Convert the dataset to a Polars DataFrame.

        Returns:
            _pl.DataFrame: A Polars DataFrame representing the dataset.
        """
        return self.to_polars()

    def to_pandas(
        self,
        filter_expr: str | None = None,
        lazy: bool = True,
        from_="scan_files",
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
        return self.to_duckdb(filter_expr=filter_expr, lazy=lazy, from_=from_).df()

    @property
    def df(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame representation of the dataset.

        Returns:
            pd.DataFrame: A pandas DataFrame representation of the dataset.
        """
        return self.to_pandas()

    @property
    def file_catalog(self) -> _pl.DataFrame:
        """
        Returns a Polars DataFrame containing information about the files in the dataset.

        If the file catalog has not yet been generated, this method will call the `gen_file_catalog` method to generate it.

        Returns:
            _pl.DataFrame: A Pandas DataFrame containing information about the files in the dataset.
        """
        if not hasattr(self, "_file_catalog"):
            self.gen_file_catalog()
        return self._file_catalog

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
    ):
        if isinstance(df, _pl.LazyFrame):
            df = df.collect()

        filter_expr = []
        for col in delta_subset or df.columns:
            f_max = df.select(_pl.col(col).max())[0, 0]
            if isinstance(f_max, str):
                f_max = f_max.replace("'", "")

            f_min = df.select(_pl.col(col).min())[0, 0]
            if isinstance(f_min, str):
                f_min = f_min.replace("'", " ")
            filter_expr.append(
                f"{col}<='{f_max}' AND {col}>='{f_min}'".replace("'None'", "NULL")
            )

        self.scan(" AND ".join(filter_expr))

        df0 = self.pl.collect()
        self.reset_scan_files()
        if df0.shape[0] > 0:
            return df.delta(df0, subset=delta_subset, eager=True)
        return df

    def write_to_dataset(
        self,
        df: _pl.DataFrame
        | _pl.LazyFrame
        | pa.Table
        | pd.DataFrame
        | duckdb.DuckDBPyConnection,
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
        elif isinstance(df, duckdb.DuckDBPyRelation):
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
                self.load_metadata(update=False, reload=False)
            except:
                self.load_metadata(update=False, reload=True)

        if mode == "overwrite":
            self.delete_files(del_files)

        self.load_metadata(update=True)
        self.gen_file_catalog()
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
            self.scan(filter_expr=filter_expr)
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
            self.scan(filter_expr=filter_expr)
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

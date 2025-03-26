import concurrent.futures
import copy
import posixpath
import pickle
import re
import tempfile
from collections import defaultdict

import duckdb
import pyarrow as pa
import pyarrow.fs as pfs
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from loguru import logger

from .filesystem import FileSystem, clear_cache

# from .helpers.metadata import collect_parquet_metadata  # , remove_from_metadata
from .helpers.misc import get_partitions_from_path, run_parallel, unify_schemas_pl
from .schema import (
    convert_large_types_to_normal,  # unify_schemas
    repair_schema,
)


def collect_parquet_metadata(
    files: list[str] | str,
    base_path: str | None = None,
    filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
    n_jobs: int = -1,
    backend: str = "threading",
    verbose: bool = True,
) -> dict[str, pq.FileMetaData]:
    """Collect all metadata information of the given parqoet files.

    Args:
        files (list[str] | str): Parquet files.
        filesystem (AbstractFileSystem | pfs.FileSystem | None, optional): Filesystem. Defaults to None.
        n_jobs (int, optional): n_jobs parameter of joblib.Parallel. Defaults to -1.
        backend (str, optional): backend parameter of joblib.Parallel. Defaults to "threading".
        verbose (bool, optional): Wheter to show the task progress using tqdm or not. Defaults to True.

    Returns:
        dict[str, pq.FileMetaData]: Parquet metadata of the given files.
    """

    def get_metadata(f, base_path, filesystem):
        if base_path is not None:
            path = posixpath.join(base_path, f)
        else:
            path = f
        return {f: pq.read_metadata(path, filesystem=filesystem)}

    metadata = run_parallel(
        get_metadata,
        files,
        base_path=base_path,
        filesystem=filesystem,
        backend=backend,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return {key: value for d in metadata for key, value in d.items()}


def remove_from_metadata(
    metadata: pq.FileMetaData,
    rm_files: list[str] | None = None,
    keep_files: list[str] | None = None,
    base_path: str = None,
) -> pq.FileMetaData:
    """
    Removes row groups from the metadata of the dataset.
    This method removes row groups from the given metadata based on the given files.
    Files in `rm_files` will be removed from the metadata. Files in `keep_files` will be kept in the metadata.


    Args:
        metadata (pq.FileMetaData): The metadata of the dataset.
        rm_files (list[str]): The files to delete from the metadata.
        keep_files (list[str]): The files to keep in the metadata.
    Returns:
        pq.FileMetaData: The updated metadata of the dataset.
    """
    row_groups = []
    if rm_files is not None:
        if base_path is not None:
            rm_files = [f.replace(base_path, "").lstrip("/") for f in rm_files]

        # row_groups to keep
        row_groups += [
            metadata.row_group(i)
            for i in range(metadata.num_row_groups)
            if metadata.row_group(i).column(0).file_path not in rm_files
        ]
    if keep_files is not None:
        if base_path is not None:
            keep_files = [f.replace(base_path, "").lstrip("/") for f in keep_files]

        # row_groups to keep
        row_groups += [
            metadata.row_group(i)
            for i in range(metadata.num_row_groups)
            if metadata.row_group(i).column(0).file_path in keep_files
        ]

    if len(row_groups):
        new_metadata = row_groups[0]
        for rg in row_groups[1:]:
            new_metadata.append_row_groups(rg)

        return new_metadata

    return metadata


def get_file_paths(
    metadata: pq.FileMetaData,
) -> list[str]:
    return sorted(
        set(
            [
                metadata.row_group(i).column(0).file_path.lstrip("../")
                for i in range(metadata.num_row_groups)
            ]
        )
    )


class ParquetDatasetMetadata:
    def __init__(
        self,
        path: str,
        filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
        bucket: str | None = None,
        cached: bool = False,
        update_metadata: bool = False,
        **fs_kwargs,
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
        if cached:
            cache_storage = fs_kwargs.pop(
                "cache_storage", tempfile.mkdtemp(prefix="pydala2_")
            )
            # cache_storage = posixpath.join(cache_storage, path)
        else:
            cache_storage = None
        self._filesystem = FileSystem(
            bucket=bucket,
            fs=filesystem,
            cached=cached,
            cache_storage=cache_storage,
            **fs_kwargs,
        )

        self._makedirs()
        # self.load_files()

        self._fs_kwargs = fs_kwargs

        self._metadata_file = posixpath.join(path, "_metadata")
        self._file_metadata_file = posixpath.join(path, "_file_metadata")
        self._metadata = self._read_metadata()
        self._file_metadata = None  # self._read_file_metadata()
        if update_metadata:
            self.update()

    def _read_metadata(self) -> pq.FileMetaData | None:
        if self.has_metadata_file:
            return pq.read_metadata(self._metadata_file, filesystem=self._filesystem)

    def _read_file_metadata(self) -> dict[str : pq.FileMetaData] | None:
        if self.has_file_metadata_file:
            with self._filesystem.open(self._file_metadata_file, "rb") as f:
                return pickle.load(f)

    def _makedirs(self):
        if self._filesystem.exists(self._path):
            return
        try:
            self._filesystem.mkdir(self._path)
        except Exception as e:
            _ = e
            self._filesystem.touch(posixpath.join(self._path, "tmp.delete"))
            self._filesystem.rm(posixpath.join(self._path, "tmp.delete"))

    def load_files(self) -> None:
        if self.has_metadata:
            self._files = get_file_paths(self._metadata)
        else:
            self.clear_cache()
            self._files = [
                fn.replace(self._path, "").lstrip("/")
                for fn in sorted(
                    self._filesystem.glob(
                        posixpath.join(self._path, "**/*.parquet")  # {self._format}")
                    )
                )
            ]

    def _ls_files(self) -> None:
        """
        Reloads the list of files in the dataset directory. This method should be called
        after adding or removing files from the directory to ensure that the dataset object
        has an up-to-date list of files.

        Returns:
            None
        """
        self.clear_cache()
        return [
            fn.replace(self._path, "").lstrip("/")
            for fn in sorted(
                self._filesystem.glob(posixpath.join(self._path, "**/*.parquet"))
            )
        ]

    def _collect_file_metadata(
        self, files: list[str] | None = None, verbose: bool = False, **kwargs
    ) -> None:
        """
        Collects metadata for the specified files and updates the `file_metadata` attribute of the dataset object.

        Args:
            files (list[str] | None): A list of file paths to collect metadata for. If None, metadata will be
                collected for all files in the dataset.
            **kwargs: Additional keyword arguments to pass to the `collect_parquet_metadata` function.

        Return
            None
        """
        if files is None:
            files = self._ls_files()

        if verbose:
            logger.info(f"Collecting metadata for {len(files)} files.")

        file_metadata = collect_parquet_metadata(
            files=files,
            base_path=self._path,
            filesystem=self._filesystem,
            verbose=verbose,
            **kwargs,
        )

        # if file_metadata:
        for f in file_metadata:
            file_metadata[f].set_file_path(f)

        if self.has_file_metadata:
            self._file_metadata.update(file_metadata)
        else:
            self._file_metadata = file_metadata

    def _rm_file_metadata(self, files: list[str] | None = None) -> None:
        """
        Removes file metadata for files that are no longer in the dataset.
        """
        for f in files:
            self._file_metadata.pop(f)

    def _dump_file_metadata(self):
        """
        Save file metadata to a specified file.
        """
        with self._filesystem.open(self._file_metadata_file, "wb") as f:
            pickle.dump(self._file_metadata, f)

    def update_file_metadata(
        self, files: list[str] | None = None, verbose: bool = False, **kwargs
    ) -> None:
        """
        Updates the metadata for files in the dataset.

        This method reloads the files in the dataset and then updates the metadata for any files that have
        not yet been processed. If the dataset already has file metadata, this method will only update the
        metadata for files that have been added since the last time metadata was collected.

        Args:
            **kwargs: Additional keyword arguments to pass to the `collect_file_metadata` method.

        Returns:
            None
        """
        new_files = files or []
        rm_files = []
        if verbose:
            logger.info("Updating file metadata.")
        if not files:
            files = self._ls_files()

            if self.has_file_metadata:
                new_files += sorted(set(files) - set(self.files_in_file_metadata))
                rm_files += sorted(set(self.files_in_file_metadata) - set(files))
            else:
                new_files += files

        if new_files:
            if verbose:
                logger.info(f"Collecting metadata for {len(new_files)} new files.")
            self._collect_file_metadata(files=new_files, verbose=verbose, **kwargs)
        else:
            if verbose:
                logger.info("No new files to collect metadata for.")

        if rm_files:
            self._rm_file_metadata(files=rm_files)

        if new_files or rm_files:
            self._dump_file_metadata()

    def reset(self):
        """
        Resets the dataset by removing the metadata file and clearing the cache.
        This method removes the metadata file and clears the cache for the dataset's filesystem and base filesystem.
        Returns:
            None
        """
        if self.has_metadata:
            del self._metadata
            self._metadata = None
            self._filesystem.rm(self._metadata_file)
        if hasattr(self, "_file_schema"):
            del self._file_schema
        if hasattr(self, "_schema"):
            del self._schema
        if self.has_file_metadata:
            del self._file_metadata
            self._file_metadata = None
            self._filesystem.rm(self._file_metadata_file)

        self.clear_cache()

    def _get_unified_schema(
        self,
        verbose: bool = False,
    ) -> tuple[pa.Schema, bool]:
        """
        Returns the unified schema for the dataset.
        Returns:
            pyarrow.Schema: The unified schema for the dataset.

        """

        if self.has_file_metadata:
            new_files = sorted(
                (set(self.files_in_file_metadata) - set(self.files_in_metadata))
            )

        if len(new_files):
            if verbose:
                logger.info(
                    f"Get unified schema for number of new files: {len(new_files)}"
                )
            schemas = [self.file_schemas[f] for f in new_files]

            if self.has_metadata:
                schemas.insert(0, self.metadata.schema.to_arrow_schema())
            try:
                unified_schema = convert_large_types_to_normal(
                    pa.unify_schemas(schemas, promote_options="permissive")
                )
            except pa.lib.ArrowTypeError:
                unified_schema = convert_large_types_to_normal(
                    unify_schemas_pl(schemas)
                )

            schemas_equal = all([unified_schema == schema for schema in schemas])
        else:
            unified_schema = convert_large_types_to_normal(
                self.metadata.schema.to_arrow_schema()
            )
            schemas_equal = True
        if verbose:
            logger.info(f"Schema is equal: {schemas_equal}")

        return unified_schema, schemas_equal

    def _repair_file_schemas(
        self,
        schema: pa.Schema | None = None,
        format_version: str | None = None,
        tz: str | None = None,
        ts_unit: str | None = None,
        alter_schema: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Repairs the schemas of files in the metadata.

        Args:
            schema (pa.Schema | None, optional): The schema to use for repairing the files. If None,
                the unified schema will be used. Defaults to None.
            format_version (str | None, optional): The format version to use for repairing the files. If None,
                the format version from the metadata will be used. Defaults to None.
            tz (str | None, optional): The timezone to use for repairing the files. Defaults to None.
            ts_unit (str | None, optional): The timestamp unit to use for repairing the files. Defaults to None.
            alter_schema (bool, optional): Whether to alter the schema of the files. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the repair_schema function.

        Returns:
            None
        """
        # get unified schema
        if schema is None:
            schema, _ = self._get_unified_schema(verbose=verbose)

        files_to_repair = [
            f for f in self.file_metadata if self.file_schemas[f] != schema
        ]

        if format_version is None and self.has_metadata:
            format_version = self.metadata.format_version

        if format_version is not None:
            files_to_repair += [
                f
                for f in self.file_metadata
                if self.file_metadata[f].format_version != format_version
            ]
        # files with different schema

        files_to_repair = sorted(set(files_to_repair))
        file_schemas_to_repair = {f: self.file_schemas[f] for f in files_to_repair}
        # repair schema of files
        if len(files_to_repair):
            if verbose:
                logger.info(
                    f"Repairing schema for number of files: {len(files_to_repair)}"
                )

            repair_schema(
                files=files_to_repair,
                file_schemas=file_schemas_to_repair,
                schema=schema,
                base_path=self._path,
                filesystem=self._filesystem,
                version=format_version,
                ts_unit=ts_unit,
                tz=tz,
                alter_schema=alter_schema,
                **kwargs,
            )
            self.clear_cache()

            # update file metadata
            self.update_file_metadata(
                files=sorted(files_to_repair), verbose=verbose, **kwargs
            )

            # update metadata
            if any([f in self.files_in_metadata for f in files_to_repair]):
                self._update_metadata(
                    reload=True,
                    verbose=verbose,
                )
        else:
            self._update_metadata(reload=False, verbose=verbose)

    def _update_metadata(self, reload: bool = False, verbose: bool = False, **kwargs):
        """
        Update metadata based on the given keyword arguments.

        Parameters:
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        if not self.has_file_metadata:
            self.update_file_metadata(**kwargs)

        if not self.has_file_metadata:
            return

        rm_files = sorted(
            (set(self.files_in_metadata) - set(self.files_in_file_metadata))
        )

        new_files = sorted(
            (set(self.files_in_file_metadata) - set(self.files_in_metadata))
        )
        if verbose:
            logger.info(f"Number of files to remove: {len(rm_files)}")
            logger.info(f"Number of files to add: {len(new_files)}")
        if len(rm_files) or (len(new_files) and not self.has_metadata) or reload:
            if verbose:
                logger.info("Updateing metadata: Rewrite metadata from file metadata")

            self._metadata = copy.copy(
                self.file_metadata[self.files_in_file_metadata[0]]
            )
            for f in self.files_in_file_metadata[1:]:
                self._metadata.append_row_groups(self._file_metadata[f])

        elif len(new_files):
            if verbose:
                logger.info("Updateing metadata: Append new file metadata")

            for f in new_files:
                self._metadata.append_row_groups(self.file_metadata[f])
        else:
            if verbose:
                logger.info("Updateing metadata: No changes")

        self._write_metadata_file()
        self.load_files()

    def update(
        self,
        reload: bool = False,
        schema: pa.Schema | None = None,
        ts_unit: str | None = None,
        tz: str | None = None,
        format_version: str | None = None,
        # sort: bool | list[str] = False,
        alter_schema: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        Update the data source with optional parameters and reload option.

        Args:
            reload (bool): Flag to indicate if the data source should be reloaded.
            schema (pa.Schema | None): The schema of the data source.
            ts_unit (str | None): The unit of the timestamp.
            tz (str | None): The time zone of the data source.
            alter_schema (bool): Flag to indicate if the schema should be altered.
            format_version (str | None): The version of the data format.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """

        if reload:
            self.reset()

        # update file metadata
        self.update_file_metadata(verbose=verbose, **kwargs)

        if len(self.files_in_file_metadata) == 0:
            return

        self._repair_file_schemas(
            schema=schema,
            format_version=format_version,
            tz=tz,
            ts_unit=ts_unit,
            alter_schema=alter_schema,
            verbose=verbose,
            # sort=sort,
        )

        # update metadata file
        if not self.has_metadata_file:
            self._update_metadata(**kwargs)

    def replace_schema(self, schema: pa.Schema, **kwargs) -> None:
        """
        Replaces the schema of the dataset with the given schema.

        Args:
            schema (pa.Schema): The schema to use for the dataset.
            **kwargs: Additional keyword arguments to pass to the `repair_schema` method.

        Returns:
            None
        """
        self.update(schema=schema, **kwargs)

    def load_metadata(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def _write_metadata_file(self) -> None:
        """
        Writes metadata to a file named '_metadata' in the dataset directory.

        Returns:
            None
        """
        with self._filesystem.open(posixpath.join(self._path, "_metadata"), "wb") as f:
            self._metadata.write_metadata_file(f)

    def delete_metadata_files(self) -> None:
        """
        Deletes the metadata file associated with the dataset, if it exists.

        Raises:
            None
        """
        if self.has_metadata_file:
            self._filesystem.rm(self._metadata_file)
        if self.has_file_metadata_file:
            self._filesystem.rm(self._file_metadata_file)

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
        clear_cache(self._base_filesystem)

    @property
    def has_file_metadata_file(self):
        """
        Returns True if the dataset has file metadata, False otherwise.
        """
        return self._filesystem.exists(self._file_metadata_file)

    @property
    def has_file_metadata(self):
        """
        Returns True if the dataset has file metadata, False otherwise.
        """
        if self._file_metadata is None:
            if self.has_file_metadata_file:
                self._file_metadata = self._read_file_metadata()
        return self._file_metadata is not None

    @property
    def file_metadata(self):
        """
        Returns the file metadata associated with the dataset.

        If the file metadata has not been loaded yet, it will be loaded before being returned.

        Returns:
            dict: The file metadata associated with the dataset.
        """
        if not self.has_file_metadata:
            self.update_file_metadata()
        return self._file_metadata

    @property
    def file_schemas(self):
        return {
            f: convert_large_types_to_normal(
                self._file_metadata[f].schema.to_arrow_schema()
            )
            for f in self._file_metadata
        }

    @property
    def files_in_file_metadata(self) -> list:
        """
        Returns a list of file paths in the file metadata of the dataset.

        Returns:
            A list of file paths in the file metadata of the dataset.
        """
        if self.has_file_metadata:
            return sorted(set(self._file_metadata.keys()))
        else:
            return []

    @property
    def has_metadata_file(self):
        """
        Returns True if the dataset has a metadata file, False otherwise.
        """
        return self._filesystem.exists(self._metadata_file)

    @property
    def has_metadata(self):
        """
        Returns True if the dataset has metadata, False otherwise.
        """
        if self._metadata is None:
            if self.has_metadata_file:
                self._metadata = self._read_metadata()
        return self._metadata is not None

    @property
    def metadata(self):
        """
        Returns the metadata associated with the dataset.

        If the metadata has not been loaded yet, it will be loaded before being returned.

        Returns:
            dict: The metadata associated with the dataset.
        """
        if not self.has_metadata:
            self._update_metadata()
        return self._metadata

    @property
    def files_in_metadata(self) -> list:
        """
        Returns a list of file paths referenced in the metadata of the dataset.

        Returns:
            A list of file paths referenced in the metadata of the dataset.
        """
        if self.has_metadata:
            return get_file_paths(self.metadata)
        else:
            return []

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
    def has_files(self):
        """
        Returns True if the dataset has files, False otherwise.
        """
        return len(self.files) > 0

    @property
    def files(self) -> list:
        """
        Returns a list of file paths in the dataset.

        Returns:
            A list of file paths in the dataset.
        """
        if not hasattr(self, "_files"):
            self.load_files()
        return self._files


class PydalaDatasetMetadata(ParquetDatasetMetadata):
    def __init__(
        self,
        path: str,
        filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
        bucket: str | None = None,
        cached: bool = False,
        # metadata: pq.FileMetaData,
        partitioning: None | str | list[str] = None,
        ddb_con: duckdb.DuckDBPyConnection | None = None,
        **fs_kwargs,
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

        # self._parquet_dataset_metadata = parquet_dataset_metadata

        super().__init__(
            path=path,
            filesystem=filesystem,
            bucket=bucket,
            cached=cached,
            **fs_kwargs,
        )
        self.reset_scan()
        self._partitioning = partitioning

        if ddb_con is None:
            self.ddb_con = duckdb.connect()
        else:
            self.ddb_con = ddb_con
        try:
            self.update_metadata_table()

        except Exception as e:
            print(e)

    def reset_scan(self):
        """
        Resets the list of scanned files to the original list of files.
        """
        self._metadata_table_scanned = None

    @staticmethod
    def _gen_metadata_table(
        metadata: pq.FileMetaData | list[pq.FileMetaData],
        partitioning: None | str | list[str] = None,
    ):
        """
        Generates a polars DataFrame with statistics for each row group in the dataset.

        """
        if isinstance(metadata, pq.FileMetaData):
            metadata = [metadata]

        metadata_table = defaultdict(list)

        def process_row_group(metadata_, rg_num):
            row_group = metadata_.row_group(rg_num)
            file_path = row_group.column(0).file_path
            result = {
                "file_path": file_path,
                "num_columns": row_group.num_columns,
                "num_rows": row_group.num_rows,
                "total_byte_size": row_group.total_byte_size,
                "compression": row_group.column(0).compression,
            }

            if "=" in file_path:
                partitioning_ = partitioning or "hive"
            else:
                partitioning_ = partitioning

            if partitioning_ is not None:
                partitions = dict(
                    get_partitions_from_path(file_path, partitioning=partitioning_)
                )
                result.update(partitions)

            for col_num in range(row_group.num_columns):
                rgc = row_group.column(col_num)
                rgc = rgc.to_dict()
                col_name = rgc.pop("path_in_schema")
                rgc.pop("file_path")
                rgc.pop("compression")
                if "statistics" in rgc:
                    if rgc["statistics"] is not None:
                        rgc.update(rgc.pop("statistics"))
                    else:
                        rgc.pop("statistics")
                        rgc.update(
                            {
                                "has_min_max": False,
                                "min": None,
                                "max": None,
                                "null_count": None,
                                "distinct_count": None,
                                "num_values": None,
                                "physical_type": "UNKNOWN",
                            }
                        )
                metadata_table[col_name].append(rgc)

            return result

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for metadata_ in metadata:
                for rg_num in range(metadata_.num_row_groups):
                    futures.append(
                        executor.submit(process_row_group, metadata_, rg_num)
                    )

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                for key, value in result.items():
                    metadata_table[key].append(value)

        return metadata_table

    def update_metadata_table(
        self,
        backend: str = "threading",
        verbose: bool = True,
    ):
        if self.has_metadata:
            metadata_table = self._gen_metadata_table(
                metadata=self.metadata,
                partitioning=self._partitioning,
            )
            # self._metadata_table = pa.Table.from_pydict(metadata_table)
            self._metadata_table = self.ddb_con.from_arrow(
                pa.Table.from_pydict(metadata_table)
            )
            self._metadata_table.create_view("metadata_table")

    @property
    def metadata_table(self):
        return self._metadata_table

    @property
    def metadata_table_scanned(self):
        return self._metadata_table_scanned

    @staticmethod
    def _gen_filter(filter_expr: str) -> list[str]:
        """
        Generate a modified filter expression for a given filter expression and list of excluded columns.

        Args:
            filter_expr (str): The filter expression to modify.
            exclude_columns (list[str], optional): A list of columns to exclude from the modification. Defaults to [].

        Returns:
            list: A list of modified filter expressions.
        """
        # chech if filter_expr is a date string
        filter_expr_mod = []
        is_date = False
        is_timestamp = False
        res = re.findall(
            r"[<>=!]\'[1,2]{1}\d{3}-\d{1,2}-\d{1,2}(?:[\s,T]\d{2}:\d{2}:{0,2}\d{0,2})?\'",
            filter_expr,
        )
        if len(res):
            is_date = len(res[0]) <= 13
            is_timestamp = len(res[0]) > 13

        # print(is_date)
        if ">" in filter_expr:
            filter_expr = f"({filter_expr.replace('>', '.max>')} OR {filter_expr.split('>')[0]}.max IS NULL)"
        elif "<" in filter_expr:
            filter_expr = f"({filter_expr.replace('<', '.min<')} OR {filter_expr.split('<')[0]}.min IS NULL)"
        elif "=" in filter_expr:
            filter_expr = (
                f"({filter_expr.replace('=', '.min<=')} OR {filter_expr.split('=')[0]}.min IS NULL) "
                + f"AND ({filter_expr.replace('=', '.max>=')} OR {filter_expr.split('=')[0]}.max IS NULL)"
            )

        if is_date:
            filter_expr = (
                filter_expr.replace(">", "::DATE>")
                .replace("<", "::DATE<")
                .replace(" IS NULL", "::DATE IS NULL")
            )
        elif is_timestamp:
            filter_expr = (
                filter_expr.replace(">", "::TIMESTAMP>")
                .replace("<", "::TIMESTAMP<")
                .replace(" IS NULL", "::TIMESTAMP IS NULL")
            )

        filter_expr_mod.append(filter_expr)

        return filter_expr_mod

    def scan(self, filter_expr: str | None = None):
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
            filter_expr = re.split(
                r"\s+[a,A][n,N][d,D]\s+",
                filter_expr,
            )

            filter_expr_mod = []

            for fe in filter_expr:
                # if (
                #    re.split("[>=<]", fe)[0].lstrip("(")
                #    in self.metadata_table.column_names
                # ):
                col = re.split("[>=<]", fe)[0].lstrip("(")
                # if not isinstance(
                #    self.metadata_table.schema.field(col).type, pa.StructType
                # ):
                if self.metadata_table.select(col).types[0].id != "struct":
                    filter_expr_mod.append(fe)
                else:
                    filter_expr_mod += self._gen_filter(fe)

            self._filter_expr_mod = " AND ".join(filter_expr_mod)

            # self._metadata_table_scanned = (
            #    self._con.from_arrow(self.metadata_table)
            #    .filter(self._filter_expr_mod)
            #    .arrow()
            # )
            self._metadata_table_scanned = self._metadata_table.filter(
                self._filter_expr_mod
            )

    # def filter(self, filter_expr: str | None = None):
    #     """
    #     Filters the dataset for files that match the given filter expression.

    #     Args:
    #         filter_expr (str | None): A filter expression to apply to the dataset. Defaults to None.
    #         lazy (bool): Whether to perform the scan lazily or eagerly. Defaults to True.

    #     Returns:
    #         None
    #     """
    #     self.scan(filter_expr=filter_expr)

    @property
    def latest_filter_expr(self):
        """
        Returns the filter expression and its module.

        If the filter expression has not been set yet, it will return None for both the expression and its module.
        """
        if not hasattr(self, "_filter_expr"):
            self._filter_expr = None
            self._filter_expr_mod = None
        return self._filter_expr, self._filter_expr_mod

    @property
    def scan_files(self):
        if self.metadata_table_scanned is not None:
            return sorted(
                set(
                    map(
                        lambda x: x[0],
                        self.metadata_table_scanned.select("file_path").fetchall(),
                    )
                )
            )
        else:
            return self.files

    # @property
    # def files(self):
    #    return sorted(
    #        set(map(lambda x: x[0], self.metadata_table.select("file_path").fetchall()))
    #    )

    @property
    def is_scanned(self):
        """
        Check if all files in the dataset have been scanned.

        Returns:
            bool: True if all files have been scanned, False otherwise.
        """
        return self._metadata_table_scanned is not None
        # @staticmethod

import os
import pickle
import re
from collections import defaultdict

import duckdb
import pyarrow as pa
import pyarrow.fs as pfs
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem

from .filesystem import FileSystem, clear_cache
from .helpers.metadata import collect_parquet_metadata  # , remove_from_metadata
from .helpers.misc import get_partitions_from_path
from .schema import repair_schema, unify_schemas
import copy


class ParquetDatasetMetadata:
    def __init__(
        self,
        path: str,
        filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
        bucket: str | None = None,
        cached: bool = False,
        **caching_options,
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
        self._filesystem = FileSystem(
            bucket=bucket, fs=filesystem, cached=cached, **caching_options
        )
        self._caching_options = caching_options
        if not self._filesystem.exists(self._path):
            try:
                self._filesystem.mkdir(self._path)
            except Exception:
                pass  # raise FileNotFoundError(f"Directory {self._path} does not exist.")

        # self._files = sorted(self._filesystem.glob(os.path.join(path, "**/*.parquet")))
        self.reload_files()
        # if not self._filesystem.exists(os.path.join(path, "metadata")):
        #     try:
        #         self._filesystem.mkdir(os.path.join(path, "metadata"))
        #     except Exception:
        #         pass
        #
        self._metadata_file = os.path.join(path, "_metadata")
        self._file_metadata_file = os.path.join(path, "_file_metadata")
        self._metadata = self._read_metadata()
        self._file_metadata = self._load_file_metadata()

    def _read_metadata(self) -> pq.FileMetaData | None:
        if self.has_metadata_file:
            return pq.read_metadata(self._metadata_file, filesystem=self._filesystem)

    def _load_file_metadata(self) -> dict[pq.FileMetaData] | None:
        if self.has_file_metadata_file:
            with self._filesystem.open(self._file_metadata_file, "rb") as f:
                return pickle.load(f)

    def reload_files(self) -> None:
        """
        Reloads the list of files in the dataset directory. This method should be called
        after adding or removing files from the directory to ensure that the dataset object
        has an up-to-date list of files.

        Returns:
            None
        """
        self._files = [
            fn.replace(self._path, "").lstrip("/")
            for fn in sorted(
                self._filesystem.glob(os.path.join(self._path, "**/*.parquet"))
            )
        ]

    def _collect_file_metadata(self, files: list[str] | None = None, **kwargs) -> None:
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
            files = self._files

        file_metadata = collect_parquet_metadata(
            files=files,
            base_path=self._path,
            filesystem=self._filesystem,
            **kwargs,
        )

        # if file_metadata:
        for f in file_metadata:
            file_metadata[f.replace(self._path, "")].set_file_path(
                f.split(self._path)[-1].lstrip("/")
            )

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

    def update_file_metadata(self, files: list[str] | None = None, **kwargs) -> None:
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

        # Add new files to file_metadata
        self.reload_files()

        # if self.has_metadata:
        #    new_files = sorted((set(self._files) - set(self.files_in_metadata)))
        # else:
        new_files = []
        rm_files = []

        if self.has_file_metadata:
            new_files += sorted(set(self.files) - set(self._file_metadata.keys()))
            rm_files += sorted(set(self._file_metadata.keys()) - set(self.files))

        else:
            new_files += sorted(set(new_files + self._files))

        if files is not None:
            new_files = sorted(set(files + new_files))
        print(new_files)
        if new_files:
            self._collect_file_metadata(files=new_files, **kwargs)
        print(rm_files)
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
        ts_unit: str | None = "us",
        tz: str | None = None,
        use_large_string: bool = False,
        sort: bool | list[str] = False,
    ) -> tuple[pa.Schema, bool]:
        """
        Returns the unified schema for the dataset.
        Returns:
            pyarrow.Schema: The unified schema for the dataset.

        """
        if not self.has_file_metadata:
            self.update_file_metadata()

        new_files = sorted((set(self._files) - set(self.files_in_metadata)))

        if len(new_files):
            schemas = [
                self._file_metadata[f].schema.to_arrow_schema() for f in new_files
            ]

            if self.has_metadata:
                schemas.insert(0, self.metadata.schema.to_arrow_schema())

            schema, schemas_equal = unify_schemas(
                schemas,
                ts_unit=ts_unit,
                tz=tz,
                use_large_string=use_large_string,
                sort=sort,
            )
        else:
            schema = self.metadata.schema.to_arrow_schema()
            schemas_equal = True

        return schema, schemas_equal

    def _repair_file_schemas(
        self,
        schema: pa.Schema | None = None,
        format_version: str | None = None,
        tz: str | None = None,
        ts_unit: str | None = "us",
        use_large_string: bool = False,
        sort: bool | list[str] = False,
        **kwargs,
    ):
        """
        Repairs the schemas of the files in the dataset.
        This method repairs the schemas of the files in the dataset to match the given schema. If no schema is given,
        the method will attempt to find the unified schema for the dataset and use that as the target schema.
        Args:
            schema (pa.Schema, optional): The schema to use for the files in the dataset. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the `repair_schema` method.
        Returns:
            None
        """
        # get unified schema
        if schema is None:
            schema, _ = self._get_unified_schema(
                ts_unit=ts_unit, tz=tz, use_large_string=use_large_string, sort=sort
            )

        files_to_repair = [
            f
            for f in self._file_metadata
            if self._file_metadata[f].schema.to_arrow_schema() != schema
        ]

        if format_version is None and self.has_metadata:
            format_version = self._metadata.format_version

        # find files to repair
        # files with different schema or format version
        if format_version is not None:
            files_to_repair += [
                f
                for f in self._file_metadata
                if self._file_metadata[f].format_version != format_version
            ]
        # files with different schema

        files_to_repair = sorted(set(files_to_repair))

        # repair schema of files
        if len(files_to_repair):
            repair_schema(
                files=files_to_repair,
                schema=schema,
                base_path=self._path,
                filesystem=self._filesystem,
                version=format_version,
                **kwargs,
            )
            self.clear_cache()
            # update file metadata
            self.update_file_metadata(files=sorted(files_to_repair), **kwargs)

    def _update_metadata(self, **kwargs):
        """
        Update metadata based on the given keyword arguments.

        Parameters:
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """

        self._metadata_temp = copy.copy(self._metadata)
        # update metadata
        if self.has_file_metadata:
            # if not self.has_metadata:
            self._metadata = self._file_metadata[list(self._file_metadata.keys())[0]]
            for f in list(self._file_metadata.keys())[1:]:
                self._metadata.append_row_groups(self._file_metadata[f])

        if self._metadata_temp != self._metadata:
            self._write_metadata_file()
        del self._metadata_temp

    def update(
        self,
        reload: bool = False,
        schema: pa.Schema | None = None,
        ts_unit: str | None = "us",
        tz: str | None = None,
        use_large_string: bool = False,
        format_version: str | None = None,
        sort: bool | list[str] = False,
        **kwargs,
    ) -> None:
        """
        Update the data source with optional parameters and reload option.

        Args:
            reload (bool): Flag to indicate if the data source should be reloaded.
            schema (pa.Schema | None): The schema of the data source.
            ts_unit (str | None): The unit of the timestamp.
            tz (str | None): The time zone of the data source.
            use_large_string (bool): Flag to indicate whether to use large string type.
            format_version (str | None): The version of the data format.
            sort (bool | list[str]): Flag to indicate if sorting is required, or the list of columns to sort by.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """

        if reload:
            self.reset()

        # update file metadata
        self.update_file_metadata(**kwargs)
        if len(self.files) == 0:
            return

        # return if not file metadata
        # if not hasattr(self, "file_metadata"):
        #    return

        self._repair_file_schemas(
            schema=schema,
            format_version=format_version,
            tz=tz,
            ts_unit=ts_unit,
            use_large_string=use_large_string,
            sort=sort,
        )

        # update metadata
        self._update_metadata(**kwargs)

        # if self._read_metadata() != self._metadata:
        #    # write metadata file
        #    self._write_metadata_file()

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
        with self._filesystem.open(os.path.join(self._path, "_metadata"), "wb") as f:
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
        clear_cache(self._filesystem)
        clear_cache(self._base_filesystem)

    @property
    def has_metadata(self):
        """
        Returns True if the dataset has metadata, False otherwise.
        """
        return self._metadata is not None

    @property
    def has_file_metadata(self):
        """
        Returns True if the dataset has file metadata, False otherwise.
        """
        return self._file_metadata is not None

    @property
    def metadata(self):
        """
        Returns the metadata associated with the dataset.

        If the metadata has not been loaded yet, it will be loaded before being returned.

        Returns:
            dict: The metadata associated with the dataset.
        """
        if not self.has_metadata:
            self.update()
        return self._metadata

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
    def has_metadata_file(self):
        """
        Returns True if the dataset has a metadata file, False otherwise.
        """
        return self._filesystem.exists(self._metadata_file)

    @property
    def has_file_metadata_file(self):
        """
        Returns True if the dataset has file metadata, False otherwise.
        """
        return self._filesystem.exists(self._file_metadata_file)

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
            return sorted(
                set(
                    [
                        self._metadata.row_group(i).column(0).file_path.lstrip("../")
                        for i in range(self._metadata.num_row_groups)
                    ]
                )
            )
        else:
            return []

    @property
    def files(self) -> list:
        """
        Returns a list of file paths in the dataset.

        Returns:
            A list of file paths in the dataset.
        """
        if not hasattr(self, "_files"):
            self._files = sorted(
                self._filesystem.glob(os.path.join(self._path, "**/*.parquet"))
            )
        return self._files


class PydalaDatasetMetadata:
    def __init__(
        self,
        metadata: pq.FileMetaData,
        partitioning: None | str | list[str] = None,
        ddb_con: duckdb.DuckDBPyConnection | None = None,
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
        # self._partitioning = partitioning
        self.reset_scan()
        if ddb_con is None:
            self.ddb_con = duckdb.connect()
        else:
            self.ddb_con = ddb_con
        self.gen_metadata_table(metadata=metadata, partitioning=partitioning)

    def reset_scan(self):
        """
        Resets the list of scanned files to the original list of files.
        """
        self._metadata_table_scanned = None

    def gen_metadata_table(
        self,
        metadata: pq.FileMetaData | list[pq.FileMetaData],
        partitioning: None | str | list[str] = None,
    ):
        """
        Generates a polars DataFrame with statistics for each row group in the dataset.

        """
        if isinstance(metadata, pq.FileMetaData):
            metadata = [metadata]

        metadata_table = defaultdict(list)
        for metadata_ in metadata:
            for rg_num in range(metadata_.num_row_groups):
                row_group = metadata_.row_group(rg_num)
                file_path = row_group.column(0).file_path
                metadata_table["file_path"].append(file_path)
                metadata_table["num_columns"].append(row_group.num_columns)
                metadata_table["num_rows"].append(row_group.num_rows)
                metadata_table["total_byte_size"].append(row_group.total_byte_size)
                metadata_table["compression"].append(row_group.column(0).compression)

                if "=" in file_path:
                    partitioning = partitioning or "hive"

                if partitioning is not None:
                    partitions = dict(
                        get_partitions_from_path(file_path, partitioning=partitioning)
                    )
                    for part in partitions:
                        metadata_table[part].append(partitions[part])

                for col_num in range(row_group.num_columns):
                    rgc = row_group.column(col_num)
                    rgc = rgc.to_dict()
                    col_name = rgc.pop("path_in_schema")
                    rgc.pop("file_path")
                    rgc.pop("compression")
                    if "statistics" in rgc:
                        rgc.update(rgc.pop("statistics"))
                    metadata_table[col_name].append(rgc)

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
                "\s+[a,A][n,N][d,D]\s+",
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

    def filter(self, filter_expr: str | None = None):
        """
        Filters the dataset for files that match the given filter expression.

        Args:
            filter_expr (str | None): A filter expression to apply to the dataset. Defaults to None.
            lazy (bool): Whether to perform the scan lazily or eagerly. Defaults to True.

        Returns:
            None
        """
        self.scan(filter_expr=filter_expr)

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

    @property
    def files(self):
        return sorted(
            set(map(lambda x: x[0], self.metadata_table.select("file_path").fetchall()))
        )

    @property
    def is_scanned(self):
        """
        Check if all files in the dataset have been scanned.

        Returns:
            bool: True if all files have been scanned, False otherwise.
        """
        return self._metadata_table_scanned is not None
        # @staticmethod

import os
import re
from collections import defaultdict

import duckdb
import pyarrow as pa
import pyarrow.fs as pfs
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem

from .filesystem import clear_cache, FileSystem
from .helpers.misc import get_partitions_from_path, run_parallel
from .schema import repair_schema, unify_schemas


def collect_parquet_metadata(
    files: list[str] | str,
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

    def get_metadata(f, filesystem):
        return {f: pq.read_metadata(f, filesystem=filesystem)}

    metadata = run_parallel(
        get_metadata,
        files,
        filesystem=filesystem,
        backend=backend,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return {key: value for d in metadata for key, value in d.items()}


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

        self._files = sorted(self._filesystem.glob(os.path.join(path, "**.parquet")))

        self._file = os.path.join(path, "_metadata")
        if self.has_metadata_file:
            self._metadata = pq.read_metadata(
                self.metadata_file, filesystem=self._filesystem
            )

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

    def _collect_file_metadata(self, files: list[str] | None = None, **kwargs) -> None:
        """
        Collects metadata for the specified files and updates the `file_metadata` attribute of the dataset object.

        Args:
            files (list[str] | None): A list of file paths to collect metadata for. If None, metadata will be
                collected for all files in the dataset.
            **kwargs: Additional keyword arguments to pass to the `collect_metadata` function.

        Returns:
            None
        """
        if files is None:
            files = self._files

        file_metadata = collect_parquet_metadata(
            files=files, filesystem=self._filesystem, **kwargs
        )

        if file_metadata:
            for f in file_metadata:
                file_metadata[f].set_file_path(f.split(self._path)[-1].lstrip("/"))

            if hasattr(self, "file_metadata"):
                self.file_metadata.update(file_metadata)
            else:
                self.file_metadata = file_metadata

    def update_file_metadata(self, **kwargs) -> None:
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
        self.reload_files()

        files = sorted((set(self._files) - set(self.files_in_metadata)))
        if hasattr(self, "file_metadata"):
            files = sorted(set(files) - set(self.file_metadata.keys()))

        if len(files):
            self._collect_file_metadata(files=files, **kwargs)

    def update(
        self,
        reload: bool = False,
        schema: pa.Schema | None = None,
        ts_unit: str | None = "us",
        tz: str | None = None,
        use_large_string: bool = False,
        format_version: str = None,
        sort: bool | list[str] = True,
        **kwargs,
    ) -> None:
        """
        Unifies the metadata schema of all files in the dataset to a common schema.

        Args:
            format_version (str, optional): The format version to use for the unified schema. Defaults to "1.0".
            update_file_metadata (bool, optional): Whether to update the file metadata before unifying the schemas.
                Defaults to True.
            unify_schema_args (dict, optional): Additional arguments to pass to the `unify_schemas` function.
                Defaults to {}.
            **kwargs: Additional keyword arguments to pass to the `update_file_metadata` and `repair_schema` methods.

        Returns:
            None
        """
        if reload:
            self.delete_metadata_file()
            if self.has_metadata:
                del self._metadata
            if hasattr(self, "_file_schema"):
                del self._file_schema
            if hasattr(self, "_schema"):
                del self._schema
            if hasattr(self, "file_metadata"):
                del self.file_metadata

            self.clear_cache()

        # update file metadata
        self.update_file_metadata(**kwargs)

        # return if not file metadata
        if not hasattr(self, "file_metadata"):
            return

        if self.has_metadata:
            metadata_schema = self._metadata.schema.to_arrow_schema()
            format_version = self._metadata.format_version
        # get all file schemas and format versions
        schemas = {
            f: self.file_metadata[f].schema.to_arrow_schema()
            for f in self.file_metadata
        }
        format_version = (
            format_version
            or self.file_metadata[sorted(self.file_metadata.keys())[0]].format_version
        )

        # if schems is None, finde the unified schema
        if schema is None:
            schemas_v = list(schemas.values())

            if self.has_metadata:
                schemas_v.insert(0, metadata_schema)

            schema, _ = unify_schemas(
                schemas_v,
                ts_unit=ts_unit,
                tz=tz,
                use_large_string=use_large_string,
                sort=sort,
            )

        # get files to repair, due to different format version or schema
        files = [
            f for f in schemas if self.file_metadata[f].format_version != format_version
        ]
        files += [f for f in schemas if schemas[f] != schema]
        files = sorted(set(files))

        # repaif schema of files
        if len(files):
            repair_schema(
                files=files,
                schema=schema,
                filesystem=self._filesystem,
                version=format_version,
                **kwargs,
            )
            self.clear_cache()
            # collect new file metadata
            self._collect_file_metadata(files=sorted(files))

        # update metadata
        if self.file_metadata:
            if not self.has_metadata:
                self._metadata = self.file_metadata[list(self.file_metadata.keys())[0]]
                for f in list(self.file_metadata.keys())[1:]:
                    self._metadata.append_row_groups(self.file_metadata[f])
            else:
                files = list(
                    set(self.file_metadata.keys()) - set(self.files_in_metadata)
                )
                for f in sorted(files):
                    self._metadata.append_row_groups(self.file_metadata[f])

            self.write_metadata_file()

    def replace_schema(self, schema: pa.Schema, **kwargs) -> None:
        """
        Replaces the schema of the dataset with the given schema.

        Args:
            schema (pa.Schema): The schema to use for the dataset.
            **kwargs: Additional keyword arguments to pass to the `repair_schema` method.

        Returns:
            None
        """
        self.reload_files()
        self.clear_cache()
        repair_schema(
            files=self._files,
            schema=schema,
            filesystem=self._filesystem,
            **kwargs,
        )
        if hasattr(self, "_schema"):
            del self._schema
        if hasattr(self, "_file_schema"):
            del self._file_schema

        self.load_metadata(reload=True)

    def load_metadata(self, *args, **kwargs):
        self.update(*args, **kwargs)

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

    @property
    def files(self) -> list:
        """
        Returns a list of file paths in the dataset.

        Returns:
            A list of file paths in the dataset.
        """
        if not hasattr(self, "_files"):
            self._files = sorted(
                self._filesystem.glob(os.path.join(self._path, "**.parquet"))
            )
        return self._files


class PydalaDatasetMetadata:
    def __init__(
        self,
        metadata: pq.FileMetaData,
        partitioning: None | str | list[str] = None,
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
        self._con = duckdb.connect()
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
                    rgc.update(rgc.pop("statistics"))
                    metadata_table[col_name].append(rgc)

        self._metadata_table = pa.Table.from_pydict(metadata_table)

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
            r"[1,2]{1}\d{3}-\d{1,2}-\d{1,2}(?:[\s,T]\d{2}:\d{2}:{0,2}\d{0,2})?",
            filter_expr,
        )
        if len(res):
            is_date = len(res[0]) <= 10
            is_timestamp = len(res[0]) > 10

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

        #     filter_expr_mod.append(
        #         f"({filter_expr.replace('>', '.max::DATE>')} OR {filter_expr.split('>')[0]}.max::DATE IS NULL)"
        #     ) if is_date else filter_expr_mod.append(
        #         f"({filter_expr.replace('>', '.max>')} OR {filter_expr.split('>')[0]}.max IS NULL)"
        #     )

        # elif "<" in filter_expr:
        #     filter_expr_mod.append(
        #         f"({filter_expr.replace('<', '.min::DATE<')} OR {filter_expr.split('<')[0]}.min::DATE IS NULL)"
        #     ) if is_date else filter_expr_mod.append(
        #         f"({filter_expr.replace('<', '.min<')} OR {filter_expr.split('<')[0]}.min IS NULL)"
        #     )

        # elif "=" in filter_expr:
        #     filter_expr_mod.append(
        #         f"({filter_expr.replace('=', '.min::DATE<=')} OR {filter_expr.split('=')[0]}.min::DATE IS NULL)"
        #     ) if is_date else filter_expr_mod.append(
        #         f"({filter_expr.replace('=', '.min<=')} OR {filter_expr.split('=')[0]}.min IS NULL)"
        #     )
        #     filter_expr_mod.append(
        #         f"({filter_expr.replace('=', '.max::DATE>=')} OR {filter_expr.split('=')[0]}.max::DATE IS NULL)"
        #     ) if is_date else filter_expr_mod.append(
        #         f"({filter_expr.replace('=', '.max>=')} OR {filter_expr.split('=')[0]}.max IS NULL)"
        #     )
        # else:
        #     filter_expr_mod.append(filter_expr)

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
            filter_expr = [fe.strip() for fe in filter_expr.split("AND")]

            filter_expr_mod = []

            for fe in filter_expr:
                # if (
                #    re.split("[>=<]", fe)[0].lstrip("(")
                #    in self.metadata_table.column_names
                # ):
                col = re.split("[>=<]", fe)[0].lstrip("(")
                if not isinstance(
                    self.metadata_table.schema.field(col).type, pa.StructType
                ):
                    filter_expr_mod.append(fe)
                else:
                    filter_expr_mod += self._gen_filter(fe)

            self._filter_expr_mod = " AND ".join(filter_expr_mod)

            self._metadata_table_scanned = (
                self._con.from_arrow(self.metadata_table)
                .filter(self._filter_expr_mod)
                .arrow()
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
                set(self.metadata_table_scanned.column("file_path").to_pylist())
            )
        else:
            return self.files

    @property
    def files(self):
        return sorted(set(self.metadata_table.column("file_path").to_pylist()))

    @property
    def is_scanned(self):
        """
        Check if all files in the dataset have been scanned.

        Returns:
            bool: True if all files have been scanned, False otherwise.
        """
        return self._metadata_table_scanned is not None
        # @staticmethod

    # def _get_row_group_stats(
    #     row_group: pq.RowGroupMetaData,
    #     partitioning: None | str | list[str] = None,
    # ):
    #     def get_column_stats(row_group_column):
    #         name = row_group_column.path_in_schema
    #         column_stats = {}
    #         column_stats[
    #             name + "_total_compressed_size"
    #         ] = row_group_column.total_compressed_size
    #         column_stats[
    #             name + "_total_uncompressed_size"
    #         ] = row_group_column.total_uncompressed_size

    #         column_stats[name + "_physical_type"] = row_group_column.physical_type
    #         if row_group_column.is_stats_set:
    #             column_stats[name + ".min"] = row_group_column.statistics.min
    #             column_stats[name + ".max"] = row_group_column.statistics.max
    #             column_stats[
    #                 name + "_null_count"
    #             ] = row_group_column.statistics.null_count
    #             column_stats[
    #                 name + "_distinct_count"
    #             ]: row_group_column.statistics.distinct_count
    #         return column_stats

    #     stats = {}
    #     file_path = row_group.column(0).file_path
    #     stats["file_path"] = file_path
    #     if "=" in file_path:
    #         partitioning = partitioning or "hive"
    #     if partitioning is not None:
    #         partitions = get_partitions_from_path(file_path, partitioning=partitioning)
    #         stats.update(dict(partitions))

    #     stats["num_columns"] = row_group.num_columns
    #     stats["num_rows"] = row_group.num_rows
    #     stats["total_byte_size"] = row_group.total_byte_size
    #     stats["compression"] = row_group.column(0).compression

    #     column_stats = [
    #         get_column_stats(row_group.column(i)) for i in range(row_group.num_columns)
    #     ]
    #     # column_stats = [await task for task in tasks]

    #     [stats.update(cs) for cs in column_stats]
    #     return stats

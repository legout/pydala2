import os

import duckdb
import polars as _pl
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.fs as pfs
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from fsspec import filesystem as fsspec_filesystem
from fsspec.implementations.dirfs import DirFileSystem

from .utils import (
    collect_file_schemas,
    collect_metadata,
    get_partitions_from_path,
    read_table,
    repair_schema,
    run_parallel,
    sort_schema,
    unify_schemas,
)


class ParquetDataset:
    """
    Poor man´s data lake.

    PyDala is a python library to manage parquet datasets using parquet´s own header
    information (metadata and schema). Using fsspec as the underlying filesystem
    library allows to use PyDala with several (remote) filesystems like s3, adlfs, r2
    or gcs.

    This class is a wrapper around pyarrow.parquet and pyarrow.dataset.

    Usage:

    ```
    from pydala.dataset import ParquetDataset
    import s3fs

    fs = s3fs.S3FileSystem()

    ds = ParquetDataset("dataset", bucket="myDataLake", filesystem=fs)
    ds.load()

    ds.scan("time>'2023-01-02'")
    ddb_rel = ds.to_duckdb()

    """

    def __init__(
        self,
        path: str,
        filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
        bucket: str | None = None,
        partitioning: str | None = None,
    ):
        self._path = path
        self._base_filesystem = filesystem or fsspec_filesystem("file")
        self.set_bucket(bucket)

        self._files = self._filesystem.glob(os.path.join(path, "**.parquet"))
        self._scanfiles = self._files.copy()
        if partitioning is None and "=" in self._files[0]:
            partitioning = "hive"
        self._partitioning = partitioning

        self._metadata_file = os.path.join(path, "_metadata")
        if self.has_metadata_file:
            if self._filesystem.du(self._metadata_file) == 0:
                self._filesystem.rm(self._metadata_file)

        if self.has_metadata_file:
            self._filesystem.invalidate_cache()
            self._metadata = pq.read_metadata(
                self._metadata_file, filesystem=self._filesystem
            )

        # self.load()

    def load(self):
        self.update_metadata()
        self._filesystem.invalidate_cache()
        self._ds = self._ds = pds.parquet_dataset(
            self._metadata_file,
            filesystem=self._filesystem,
            partitioning=self._partitioning,
        )

    @property
    def has_metadata_file(self):
        return self._filesystem.exists(self._metadata_file)

    @property
    def has_parquet_files(self):
        return len(self._files) > 0

    def set_bucket(self, bucket: str | None):
        self._bucket = bucket
        if bucket is None:
            self._filesystem = self._base_filesystem
        else:
            if isinstance(self._base_filesystem, AbstractFileSystem):
                self._filesystem = DirFileSystem(path=bucket, fs=self._base_filesystem)
            else:
                self._filesystem = pfs.SubTreeFileSystem(
                    base_path=bucket, base_fs=self._base_filesystem
                )

    def write_metadata_file(self):
        with self._filesystem.open(os.path.join(self._path, "_metadata"), "wb") as f:
            self._metadata.write_metadata_file(f)

        self._filesystem.invalidate_cache()
        self._ds = self._ds = pds.parquet_dataset(
            self._metadata_file,
            filesystem=self._filesystem,
            partitioning=self._partitioning,
        )

    def collect_file_schemas(
        self, files: str | None = None, update: bool = True, **kwargs
    ):
        files = files or self._files

        if update:
            files = list(set(files) - set(self.files_in_metadata))
            if hasattr(self, "_file_schema"):
                files = list(set(files) - set(self._file_schema.keys()))

        file_schemas = collect_file_schemas(
            files=files, filesystem=self._filesystem, **kwargs
        )

        if len(file_schemas):
            if hasattr(self, "_file_schemas"):
                self._file_schemas.update(file_schemas)
            else:
                self._file_schemas = file_schemas

            self._unified_file_schema, self._file_schemas_equal = unify_schemas(
                list(self._file_schemas.values())
            )

        else:
            self._unified_file_schema = None
            self._file_schemas_equal = True

    def collect_file_metadata(
        self, files: str | None = None, update: bool = True, **kwargs
    ):
        files = files or self._files

        if update:
            files = set(files) - set(self.files_in_metadata)

        file_metadata = collect_metadata(
            files=files, filesystem=self._filesystem, **kwargs
        )
        if len(file_metadata):
            for f in file_metadata:
                file_metadata[f].set_file_path(f.split(self._path)[-1].lstrip("/"))

            if hasattr(self, "_file_metadata"):
                self._file_metadata.update(file_metadata)
            else:
                self._file_metadata = file_metadata
            file_schemas = {
                f: file_metadata[f].schema.to_arrow_schema() for f in file_metadata
            }
            if hasattr(self, "_file_schemas"):
                self._file_schemas.update(file_schemas)
            else:
                self._file_schemas = file_schemas

            self._unified_file_schema, self._file_schemas_equal = unify_schemas(
                list(self._file_schemas.values())
            )
        else:
            self._unified_file_schema = None
            self._file_schemas_equal = True

    def update_metadata(self, init: bool = False):
        if init:
            if self.has_metadata_file:
                self._filesystem.rm(self._metadata_file)
                del self._metadata

        self.collect_file_metadata(update=True)

        if hasattr(self, "_file_metadata") and len(self._file_metadata):
            self.repair_schema(files=list(self._file_metadata.keys()), sort=True)

            if not self.has_metadata_file:
                self._metadata = self._file_metadata[
                    list(self._file_metadata.keys())[0]
                ]
                for f in list(self._file_metadata.keys())[1:]:
                    self._metadata.append_row_groups(self._file_metadata[f])
            else:
                files = list(
                    set(self._file_metadata.keys()) - set(self.files_in_metadata)
                )
                for f in files:
                    self._metadata.append_row_groups(self._file_metadata[f])

            self.write_metadata_file()

        self._file_schema = self._metadata.schema.to_arrow_schema()

    def repair_schema(
        self,
        files: str | None = None,
        sort: bool = False,
        format_version: str = "1.0",
        **kwargs,
    ):
        if self.has_metadata_file:
            format_version = self.metadata.format_version

        files = files or self._files

        files_to_repair = []

        if not self.file_schemas_equal:
            if sort:
                files_to_repair += [
                    f for f in files if self._file_schemas[f] != self.file_schema
                ]
            else:
                files_to_repair += [
                    f
                    for f in files
                    if sort_schema(self._file_schemas[f])
                    != sort_schema(self.file_schema)
                ]

        if hasattr(self, "_file_metadata"):
            files_to_repair += [
                f
                for f in files
                if self._file_metadata[f].format_version != format_version
            ]

        files_to_repair = list(set(files_to_repair))

        if len(files_to_repair):
            repair_schema(
                files=files_to_repair,
                schema=self._schema,
                filesystem=self._filesystem,
                version=format_version,
                **kwargs,
            )
            self.collect_file_metadata(files=files_to_repair, update=False)

    @property
    def files_in_metadata(self) -> list:
        if hasattr(self, "_metadata"):
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
    def columns(self) -> list:
        if not hasattr(self, "_columns"):
            self._columns = self.schema.names
        return self._columns

    @property
    def count_rows(self) -> int:
        return self._ds.count_rows

    @property
    def partitioning_schema(self) -> pa.Schema:
        if not hasattr(self, "_partitioning_schema"):
            self._partitioning_schema = self._ds.partitioning.schema
        return self._partitioning_schema

    @property
    def partitioning_names(self) -> list:
        if not hasattr(self, "_partitioning_names"):
            self._partitioning_names = self.partitioning_schema.names
        return self._partitioning_names

    @property
    def metadata(self) -> pq.FileMetaData:
        if not hasattr(self, "_metadata"):
            self.update_metadata()
        return self._metadata

    @property
    def file_schema(self) -> pa.Schema:
        if not hasattr(self, "_file_schema"):
            if hasattr(self, "_metadata"):
                self.update_metadata()
            else:
                self._file_schema = self.unified_file_schema
        return self._flle_schema

    @property
    def unified_file_schema(self) -> dict:
        if not hasattr(self, "_file_schema"):
            self.collect_file_schemas()
        return self._unified_file_schema

    @property
    def file_schemas_equal(self):
        if not hasattr(self, "_file_schemas_equal"):
            self.collect_file_schemas()
        return self._file_schemas_equal

    def gen_file_catalog(self):
        def _get_row_group_stats(rg, partitioning: None | str | list[str] = None):
            stats = {}
            file_path = rg.column(0).file_path
            stats["file_path"] = file_path
            if "=" in file_path:
                partitioning = partitioning or "hive"
            if partitioning is not None:
                partitions = get_partitions_from_path(
                    file_path, partitioning=partitioning
                )
                stats.update(dict(partitions))

            stats["num_columns"] = rg.num_columns
            stats["num_rows"] = rg.num_rows
            stats["total_byte_size"] = rg.total_byte_size
            stats["compression"] = rg.column(0).compression

            for i in range(rg.num_columns):
                name = rg.column(i).path_in_schema
                stats[name + "_total_compressed_size"] = rg.column(
                    i
                ).total_compressed_size
                stats[name + "_total_uncompressed_size"] = rg.column(
                    i
                ).total_uncompressed_size

                stats[name + "_physical_type"] = rg.column(i).physical_type
                if rg.column(i).is_stats_set:
                    stats[name + "_min"] = rg.column(i).statistics.min
                    stats[name + "_max"] = rg.column(i).statistics.max
                    stats[name + "_null_count"] = rg.column(i).statistics.null_count
                    stats[name + "_distinct_count"]: rg.column(
                        i
                    ).statistics.distinct_count

            return stats

        self._file_catalog = _pl.DataFrame(
            [
                _get_row_group_stats(
                    self.metadata.row_group(i), partitioning=self._partitioning
                )
                for i in range(self.metadata.num_row_groups)
            ]
        )

    def scan(self, filter_expr: str | None = None, lazy: bool = True, **kwargs):
        if filter_expr is not None:
            filter_expr = [fe.strip() for fe in filter_expr.split("AND")]

            filter_expr_mod = []
            for fe in filter_expr:
                if ">" in fe:
                    filter_expr_mod.append(fe.replace(">", "_max>"))
                elif "<" in fe:
                    filter_expr_mod.append(fe.replace("<", "_min<"))
                elif "=" in fe:
                    filter_expr_mod.append(fe.replace("=", "_min<="))
                    filter_expr_mod.append(fe.replace("=", "_max>="))

            filter_expr_mod = " AND ".join(filter_expr_mod)

            for part_name in self.partitioning_names:
                filter_expr_mod = filter_expr_mod.replace(
                    f"{part_name}_max", part_name
                ).replace(f"{part_name}_min", part_name)

            self._scanfiles = [
                os.path.join(self._path, sf)
                for sf in duckdb.from_arrow(self.file_catalog.to_arrow())
                .filter(filter_expr_mod)
                .pl()["file_path"]
                .to_list()
            ]

            self._filesystem.invalidate_cache()
            self._scands = pds.dataset(
                self._scanfiles,
                filesystem=self._filesystem,
                partitioning=self._partitioning,
                schema=self.schema,
            )

            if not lazy:
                self._scantable = pa.concat_tables(
                    run_parallel(
                        read_table,
                        self._scanfiles,
                        schema=self.schema,
                        format="parquet",
                        filesystem=self._filesystem,
                        partitioning=self._partitioning,
                        **kwargs,
                    )
                )
        else:
            if not hasattr(self, "_scands"):
                self._filesystem.invalidate_cache()
                self._scands = pds.dataset(
                    self._scanfiles,
                    filesystem=self._filesystem,
                    partitioning=self._partitioning,
                    schema=self.schema,
                )
            if not lazy and not hasattr(self, "_scantable"):
                self._scantable = pa.concat_tables(
                    run_parallel(
                        read_table,
                        self._scanfiles,
                        schema=self.schema,
                        format="parquet",
                        filesystem=self._filesystem,
                        partitioning=self._partitioning,
                        **kwargs,
                    )
                )

        return self

    def to_dataset(self, filter_expr: str | None = None) -> pds.Dataset:
        self.scan(filter_expr=filter_expr, lazy=True)

        return self._scands

    @property
    def dataset(self) -> pds.Dataset:
        return self.to_dataset()

    def to_table(self, filter_expr: str | None = None) -> pa.Table:
        self.scan(filter_expr=filter_expr, lazy=False)

        return self._scantable

    @property
    def table(self) -> pa.Table:
        return self.to_table()

    def to_duckdb(
        self, filter_expr: str | None = None, lazy: bool = True
    ) -> duckdb.DuckDBPyRelation:
        self.scan(filter_expr=filter_expr, lazy=lazy)
        if lazy and not hasattr(self, "_scantable"):
            return duckdb.from_arrow(self._scands)
        else:
            return duckdb.from_arrow(self._scantable)

    @property
    def ddb(self) -> duckdb.DuckDBPyRelation:
        return self.to_duckdb()

    def to_polars(
        self, filter_expr: str | None = None, lazy: bool = True
    ) -> _pl.DataFrame:
        self.scan(filter_expr=filter_expr, lazy=lazy)
        if lazy and not hasattr(self, "_scantable"):
            return _pl.scan_pyarrow_dataset(self._scands)
        else:
            return _pl.from_arrow(self._scantable)

    @property
    def pl(self) -> _pl.DataFrame:
        return self.to_polars()

    def to_pandas(
        self, filter_expr: str | None = None, lazy: bool = True
    ) -> _pl.DataFrame:
        return self.to_duckdb(filter_expr=filter_expr, lazy=lazy).df()

    @property
    def df(self):
        return self.to_pandas()

    @property
    def file_catalog(self) -> _pl.DataFrame:
        if not hasattr(self, "_file_catalog"):
            self.gen_file_catalog()
        return self._file_catalog


class ParquetWriter:
    def __init__(self, path: str):
        self._path = path

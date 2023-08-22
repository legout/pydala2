# %%
import os
import uuid

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.fs as pfs
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem

from .helpers import (
    collect_metadata,
    get_filesystem,
    get_row_group_stats,
    partition_by,
    run_parallel,
)

# from fsspec.implementations import dirfs, cached as cachedfs
from .io import read_table, write_table
from .polars_ext import pl as _pl
from .schema import convert_timestamp, repair_schema, shrink_large_string, unify_schemas


class ParquetDatasetMetadata:
    def __init__(
        self,
        path: str,
        filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
        bucket: str | None = None,
        cached: bool = False,
    ):
        self._path = path
        self._bucket = bucket
        self._cached = cached
        self._filesystem = get_filesystem(bucket=bucket, fs=filesystem, cached=cached)

        self._files = sorted(self._filesystem.glob(os.path.join(path, "**.parquet")))

        self._file = os.path.join(path, "_metadata")
        if self.has_metadata_file:
            self._metadata = pq.read_metadata(
                self.metadata_file, filesystem=self._filesystem
            )

    def collect(self, files: list[str] | None = None, **kwargs):
        if files is None:
            files = self._files

        if len(files):
            file_metadata = collect_metadata(
                files=files, filesystem=self._filesystem, **kwargs
            )

            if len(file_metadata):
                for f in file_metadata:
                    file_metadata[f].set_file_path(f.split(self._path)[-1].lstrip("/"))

                if hasattr(self, "file_metadata"):
                    self.file_metadata.update(file_metadata)
                else:
                    self.file_metadata = file_metadata

    def reload(self, **kwargs):
        self._files = sorted(
            self._filesystem.glob(os.path.join(self._path, "**.parquet"))
        )
        self.collect(files=self._files, **kwargs)
        if self.has_metadata_file:
            self._filesystem.rm(self.metadata_file)

    def update(self, **kwargs):
        self._files = sorted(
            self._filesystem.glob(os.path.join(self._path, "**.parquet"))
        )
        files = list(set(self._files) - set(self.files_in_metadata))

        if hasattr(self, "file_metadata"):
            files = list(set(files) - set(self.file_metadata.keys()))

        self.collect(files=files, **kwargs)

    def unify_metadata_schema(
        self,
        format_version: str = "1.0",
        update: bool = True,
        reload: bool = False,
        **kwargs,
    ):
        if reload:
            update = False
            self.reload(**kwargs)
        if update:
            self.update(**kwargs)

        schemas = {
            f: self.file_metadata[f].schema.to_arrow_schema()
            for f in self.file_metadata
        }
        schemas_v = list(schemas.values())
        if self.has_metadata:
            metadata_schema = self._metadata.schema.to_arrow_schema()
            schemas_v.append(metadata_schema)
            format_version = self._metadata.format_version

        unified_schema, schemas_equal = unify_schemas(schemas_v)

        files = []
        if not schemas_equal:
            files += [f for f in self._files if schemas[f] != unified_schema]
        files += [
            f for f in files if self.file_metadata[f].format_version != format_version
        ]

        if len(files):
            repair_schema(
                files=list(set(files)),
                schema=unified_schema,
                filesystem=self._filesystem,
                version=format_version,
                **kwargs,
            )

    def to_metadata(
        self,
        update: bool = True,
        reload: bool = False,
        auto_repair_schema: bool = False,
        format_version: str = "1.0",
        **kwargs,
    ):
        if reload:
            update = False
            self.reload(**kwargs)
        if update:
            self.update(**kwargs)

        if hasattr(self, "file_metadata"):
            try:
                if not self.has_metadata:
                    self._metadata = self.file_metadata[
                        list(self.file_metadata.keys())[0]
                    ]
                    for f in list(self.file_metadata.keys())[1:]:
                        self._metadata.append_row_groups(self.file_metadata[f])
                else:
                    files = list(
                        set(self.file_metadata.keys()) - set(self.files_in_metadata)
                    )
                    for f in files:
                        self._metadata.append_row_groups(self.file_metadata[f])

                self.write_metadata_file()

            except Exception as e:
                if auto_repair_schema:
                    self.unify_metadata_schema(
                        update=False, reload=False, format_version=format_version
                    )
                    self.to_metadata()
                else:
                    # e.message += f"Run `{self}.unify_metadata_schema()
                    # or set `auto_repair_schema=True` and try again.
                    raise e

    def write_metadata_file(self):
        with self._filesystem.open(os.path.join(self._path, "_metadata"), "wb") as f:
            self._metadata.write_metadata_file(f)

    @property
    def has_metadata(self):
        return hasattr(self, "_metadata")

    @property
    def metadata(self):
        if not self.has_metadata:
            self.to_metadata()
        return self._metadata

    @property
    def schema(self):
        if not hasattr(self, "_schema"):
            self._schema = self.metadata.schema.to_arrow_schema()
        return self._schema

    @property
    def metadata_file(self):
        if not hasattr(self, "_metadata_file"):
            self._file = os.path.join(self._path, "_metadata")
        return self._file

    @property
    def has_metadata_file(self):
        return self._filesystem.exists(self.metadata_file)

    @property
    def has_files(self):
        return len(self._files) > 0

    @property
    def files_in_metadata(self) -> list:
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
        filesystem: AbstractFileSystem | None = None,
        bucket: str | None = None,
        partitioning: str | None = None,
        cached: bool = False,
    ):
        super().__init__(path=path, filesystem=filesystem, bucket=bucket, cached=cached)
        #self.metadata = ParquetDatasetMetadata(path=path, filesystem=filesystem, bucket=bucket, )
        if self.has_files:
            if partitioning == "ignore":
                self._partitioning = None
            elif partitioning is None and "=" in self._files[0]:
                partitioning = "hive"
            else:
                self._partitioning = partitioning
        
        self._ddb = duckdb.connect()
        # self.load()

    def load(self, auto_repair_schema: bool = False, **kwargs):
        if self.has_files:
            if self.has_metadata_file:
                if len(self.files_in_metadata) <= len(self._files):
                    self.to_metadata(auto_repair_schema=auto_repair_schema, **kwargs)
                    # self.write_metadata_file()
            else:
                self.to_metadata(auto_repair_schema=auto_repair_schema, **kwargs)
                # self.write_metadata_file()

            self._filesystem.invalidate_cache()
            self.metadata_file
            if self.has_files:
                
                self._base_dataset = pds.parquet_dataset(
                    self.metadata_file,
                    # schema=self.schema,
                    partitioning=self._partitioning,
                    filesystem=self._filesystem,
                )

    @property
    def is_loaded(self):
        return hasattr(self, "_base_dataset")

    @property
    def columns(self) -> list:
        if not hasattr(self, "_columns"):
            self._columns = self.schema.names + self.partitioning_names
        return self._columns

    @property
    def count_rows(self) -> int:
        if self.is_loaded:
            return self._base_dataset.count_rows
        else:
            print(f"No dataset loaded yet. Run {self}.load()")
            return 0

    @property
    def partitioning_schema(self) -> pa.Schema:
        if not hasattr(self, "_partitioning_schema"):
            if self.is_loaded:
                self._partitioning_schema = self._base_dataset.partitioning.schema
            else:
                print(f"No dataset loaded yet. Run {self}.load()")
                return pa.schema([])
        return self._partitioning_schema

    @property
    def schema(self):
        if not hasattr(self, "_schema"):
            # if self._partitioning is not None and self._partitioning!="ignore":
            self._schema = pa.unify_schemas(
                [self.file_schema, self.partitioning_schema]
            )
        return self._schema

    @property
    def file_schema(self):
        if not hasattr(self, "_file_schema"):
            self._file_schema = self.metadata.schema.to_arrow_schema()
        return self._file_schema

    @property
    def partitioning_names(self) -> list:
        if not hasattr(self, "_partitioning_names"):
            if self.is_loaded:
                self._partitioning_names = self.partitioning_schema.names
            else:
                print(f"No dataset loaded yet. Run {self}.load()")
                return []
        return self._partitioning_names

    def gen_file_catalog(self):
        self._file_catalog = _pl.DataFrame(
            [
                get_row_group_stats(
                    self.metadata.row_group(i), partitioning=self._partitioning
                )
                for i in range(self.metadata.num_row_groups)
            ]
        )

    def scan(self, filter_expr: str | None = None, lazy: bool = True):
        if filter_expr is not None:
            filter_expr = [fe.strip() for fe in filter_expr.split("AND")]

            filter_expr_mod = []
            for fe in filter_expr:
                if ">" in fe:
                    if not fe.split(">")[0].lstrip("(") in self.partitioning_names:
                        filter_expr_mod.append(fe.replace(">", "_max>"))
                    else:
                        filter_expr_mod.append(fe)
                elif "<" in fe:
                    if not fe.split("<")[0].lstrip("(") in self.partitioning_names:
                        filter_expr_mod.append(fe.replace("<", "_min<"))
                    else:
                        filter_expr_mod.append(fe)
                elif "=" in fe:
                    if not fe.split("=")[0].lstrip("(") in self.partitioning_names:
                        filter_expr_mod.append(fe.replace("=", "_min<="))
                        filter_expr_mod.append(fe.replace("=", "_max>="))
                    else:
                        filter_expr_mod.append(fe)

            filter_expr_mod = " AND ".join(filter_expr_mod)
            # filter_expr_mod = filter_expr.replace(">", "_max>").replace("<", "_min<")

            # for part_name in self.partitioning_names:
            #     filter_expr_mod = filter_expr_mod.replace(
            #         f"{part_name}_max", part_name
            #     ).replace(f"{part_name}_min", part_name)

            self._scan_files = [
                os.path.join(self._path, sf)
                for sf in self._ddb.from_arrow(self.file_catalog.to_arrow())
                .filter(filter_expr_mod)
                .pl()["file_path"]
                .to_list()
            ]

        return self

    def to_dataset(self, filter_expr: str | None = None) -> pds.Dataset:
        self.scan(filter_expr=filter_expr, lazy=True)
        if hasattr(self, "_dataset"):
            if sorted(self._dataset.files) == sorted(self._scan_files):
                return self._dataset
        self._dataset = pds.dataset(
            self._scan_files,
            partitioning=self._partitioning,
            filesystem=self._filesystem,
        )
        return self._dataset

    @property
    def dataset(self) -> pds.Dataset:
        return self.to_dataset()

    def to_table(self, filter_expr: str | None = None, **kwargs) -> pa.Table:
        self.scan(filter_expr=filter_expr, lazy=False)
        if hasattr(self, "_table"):
            if sorted(self._table_files) == sorted(self._scan_files):
                return self._table

        self._table_files = self._scan_files.copy()
        self._table = pa.concat_tables(
            run_parallel(
                read_table,
                self._scan_files,
                schema=self.schema,
                format="parquet",
                filesystem=self._filesystem,
                partitioning=self._partitioning,
                **kwargs,
            )
        )
        return self._table

    @property
    def table(self) -> pa.Table:
        return self.to_table()

    def to_duckdb(
        self, filter_expr: str | None = None, lazy: bool = True
    ) -> duckdb.DuckDBPyRelation:
        if lazy:
            self.to_dataset(filter_expr=filter_expr)
            return self._ddb.from_arrow(self._dataset)
        else:
            self.to_table(filter_expr=filter_expr)
            return self._ddb.from_arrow(self._table)

    @property
    def ddb(self) -> duckdb.DuckDBPyRelation:
        return self.to_duckdb()

    def to_polars(
        self, filter_expr: str | None = None, lazy: bool = True
    ) -> _pl.DataFrame:
        if lazy:
            self.to_dataset(filter_expr=filter_expr)
            return _pl.scan_pyarrow_dataset(self._dataset)
        else:
            self.to_table(filter_expr=filter_expr)
            return _pl.from_arrow(self._table)

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

    def write_to_dataset(
        self,
        df: _pl.DataFrame
        | _pl.LazyFrame
        | pa.Table
        | pd.DataFrame
        | duckdb.DuckDBPyConnection,
        mode: str = "append",
        num_rows: int | None = None,
        row_group_size: int | None = None,
        compression: str = "zstd",
        **kwrags,
    ):
        if isinstance(df, pd.DataFrame):
            df = _pl.from_pandas(df)
        elif isinstance(df, pa.Table):
            df = _pl.from_arrow(df)
        elif isinstance(df, duckdb.DuckDBPyRelation):
            df = df.pl()

        _partitions = partition_by(
            df=df, columns=self.partitioning_names, num_rows=num_rows
        )
        paths = [
            os.path.join(
                "/".join(
                    (
                        "=".join([k, str(v).lstrip("0")])
                        for k, v in partition[0].items()
                        if k != "row_nr"
                    )
                ),
                f"data-{partition[0].get('row_nr', 0)}-{uuid.uuid4().hex}.parquet",
            )
            for partition in _partitions
        ]

        partitions = [partition[1] for partition in _partitions]

        # run_parllel(write_table, partitions, path, schema=self.file_schema, filesystem=self._filesystem,
        #             row_group_size=row_group_size, compression=compression, backend="threading", n_jobs=-1, **kwargs)

        return _partitions, paths

import datetime as dt
import os
import uuid

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
    partition_by,
    run_parallel,
    humanize_size,
    humanized_size_to_bytes,
    get_timestamp_column,
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
    ):
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

    def collect_file_metadata(self, files: list[str] | None = None, **kwargs):
        if files is None:
            files = self._files

        if len(files):
            # print("collect")
            file_metadata = collect_metadata(
                files=files, filesystem=self._filesystem, **kwargs
            )
            # print(len(file_metadata))

            if len(file_metadata):
                for f in file_metadata:
                    file_metadata[f].set_file_path(f.split(self._path)[-1].lstrip("/"))

                if hasattr(self, "file_metadata"):
                    self.file_metadata.update(file_metadata)
                else:
                    self.file_metadata = file_metadata

    def reload_files(self):
        # print("reload")
        self._files = sorted(
            self._filesystem.glob(os.path.join(self._path, "**.parquet"))
        )

    def update_file_metadata(self, **kwargs):
        self.reload_files()

        files = list((set(self._files) - set(self.files_in_metadata)))
        if hasattr(self, "file_metadata"):
            files = list(set(files) - set(self.file_metadata.keys()))

        if len(files):
            self.collect_file_metadata(files=files, **kwargs)

    def unify_metadata_schema(
        self,
        format_version: str = "1.0",
        update_file_metadata: bool = True,
        # reload: bool = False,
        unify_schema_args:dict= {},
        **kwargs,
    ):
        # if reload:
        #    update = False
        #    self.reload(**kwargs)
        if update_file_metadata:
            self.update_file_metadata(**kwargs)

        if hasattr(self, "file_metadata"):
            # self.collect_file_metadata(**kwargs)

            schemas = {
                f: self.file_metadata[f].schema.to_arrow_schema()
                for f in self.file_metadata
            }
            schemas_v = list(schemas.values())
            if self.has_metadata:
                metadata_schema = self._metadata.schema.to_arrow_schema()
                schemas_v.append(metadata_schema)
                format_version = self._metadata.format_version

            unified_schema, schemas_equal = unify_schemas(schemas_v, **unify_schema_args)

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
        # delete_metadata_file:bool=False,
        unify_schema_args:dict={},
        **kwargs,
    ):
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
            update_file_metadata=update, format_version=format_version, unify_schema_args=unify_schema_args,**kwargs
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

    def write_metadata_file(self):
        with self._filesystem.open(os.path.join(self._path, "_metadata"), "wb") as f:
            self._metadata.write_metadata_file(f)

    def delete_metadata_file(self):
        if self.has_metadata_file:
            self._filesystem.rm(self.metadata_file)

    def clear_cache(self):
        clear_cache(self._filesystem)
        clear_cache(self._base_filesystem)

    @property
    def has_metadata(self):
        return hasattr(self, "_metadata")

    @property
    def metadata(self):
        if not self.has_metadata:
            self.load_metadata()
        return self._metadata

    @property
    def schema(self):
        if not hasattr(self, "_schema"):
            self._schema = self.metadata.schema.to_arrow_schema()
            if self.has_metadata:
                self._file_schema = self.metadata.schema.to_arrow_schema()
            else:
                self._file_schema = pa.schema([])
        return self._schema

    @property
    def file_schema(self):
        if not hasattr(self, "_file_schema"):
            if self.has_metadata:
                self._file_schema = self.metadata.schema.to_arrow_schema()
            else:
                self._file_schema = pa.schema([])
        return self._file_schema

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
        **cached_options,
    ):
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
            self._reset_scan_files()
        else:
            self._partitioning = None

        self._ddb = duckdb.connect()
        # self.load()

    def load(
        self,
        reload: bool = False,
        update: bool = True,
        format_version: str = "1.0",
        **kwargs,
    ):
        if self.has_files:
            self.load_metadata(
                reload=reload, update=update, format_version=format_version, **kwargs
            )

            self._base_dataset = pds.parquet_dataset(
                self.metadata_file,
                # schema=self.schema,
                partitioning=self._partitioning,
                filesystem=self._filesystem,
            )
            self._timestamp_columns = get_timestamp_column(self.pl.head(1))

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

    def _reset_scan_files(self):
        self._is_scanned = False
        self._scan_files = self._files.copy()

    def scan(self, filter_expr: str | None = None, lazy: bool = True):
        self._filter_expr = filter_expr
        if filter_expr is not None:
            filter_expr = [fe.strip() for fe in filter_expr.split("AND")]

            filter_expr_mod = []
            for fe in filter_expr:
                if ">" in fe:
                    if not fe.split(">")[0].lstrip("(") in self.file_catalog.columns:
                        filter_expr_mod.append(fe.replace(">", "_max>"))
                    else:
                        filter_expr_mod.append(fe)
                elif "<" in fe:
                    if not fe.split("<")[0].lstrip("(") in self.file_catalog.columns:
                        filter_expr_mod.append(fe.replace("<", "_min<"))
                    else:
                        filter_expr_mod.append(fe)
                elif "=" in fe:
                    if not fe.split("=")[0].lstrip("(") in self.file_catalog.columns:
                        filter_expr_mod.append(fe.replace("=", "_min<="))
                        filter_expr_mod.append(fe.replace("=", "_max>="))
                    else:
                        filter_expr_mod.append(fe)

            filter_expr_mod = " AND ".join(filter_expr_mod)

            self._scan_files = [
                os.path.join(self._path, sf)
                for sf in self._ddb.from_arrow(self.file_catalog.to_arrow())
                .filter(filter_expr_mod)
                .pl()["file_path"]
                .to_list()
            ]

        return self

    @property
    def is_scanned(self):
        return sorted(self._scan_files) == sorted(self._files)

    @property
    def filter_expr(self):
        if not hasattr(self, "_filter_expr"):
            self._filter_expr = None
        return self._filter_expr

    def to_dataset(
        self, filter_expr: str | None = None, from_="scan_files"
    ) -> pds.Dataset:
        self.scan(filter_expr=filter_expr, lazy=True)
        if filter_expr is not None:
            from_ = "scan_files"
        if from_ == "scan_files":
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
        return self.to_dataset()

    def to_table(
        self, filter_expr: str | None = None, from_="scan_files", **kwargs
    ) -> pa.Table:
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
        self,
        filter_expr: str | None = None,
        lazy: bool = True,
        from_="scan_files",
    ) -> duckdb.DuckDBPyRelation:
        if lazy:
            self.to_dataset(filter_expr=filter_expr, from_=from_)
            return self._ddb.from_arrow(self._dataset)
        else:
            self.to_table(filter_expr=filter_expr, from_=from_)
            return self._ddb.from_arrow(self._table)

    @property
    def ddb(self) -> duckdb.DuckDBPyRelation:
        return self.to_duckdb()

    def to_polars(
        self,
        filter_expr: str | None = None,
        lazy: bool = True,
        from_="scan_files",
    ) -> _pl.DataFrame:
        if lazy:
            self.to_dataset(filter_expr=filter_expr, from_=from_)
            return _pl.scan_pyarrow_dataset(self._dataset)
        else:
            self.to_table(filter_expr=filter_expr, from_=from_)
            return _pl.from_arrow(self._table)

    @property
    def pl(self) -> _pl.DataFrame:
        return self.to_polars()

    def to_pandas(
        self,
        filter_expr: str | None = None,
        lazy: bool = True,
        from_="scan_files",
    ) -> _pl.DataFrame:
        return self.to_duckdb(filter_expr=filter_expr, lazy=lazy, from_=from_).df()

    @property
    def df(self):
        return self.to_pandas()

    @property
    def file_catalog(self) -> _pl.DataFrame:
        if not hasattr(self, "_file_catalog"):
            self.gen_file_catalog()
        return self._file_catalog

    def delete_files(self, files: str | list[str] | None = None):
        self._filesystem.rm(files)
        self.load(reload=True)

    def write_to_dataset(
        self,
        df: _pl.DataFrame
        | _pl.LazyFrame
        | pa.Table
        | pd.DataFrame
        | duckdb.DuckDBPyConnection,
        mode: str = "append",  # "delta", "overwrite"
        num_rows: int | None = 100_000_000,
        row_group_size: int | None = None,
        compression: str = "zstd",
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        distinct: bool = False,
        delta_subset: str | list[str] | None = None,
        partitioning_columns: str | list[str] | None = None,
        **kwargs,
    ):
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
                f"data-{dt.datetime.now().strftime('%Y%m%d_%H%M%S')[:-3]}-{uuid.uuid4().hex[:16]}.parquet",
            )
            for partition in _partitions
        ]

        partitions = [partition[1] for partition in _partitions]
        file_metadata = []

        for _df, path in zip(partitions, paths):
            if mode == "delta":
                if isinstance(_df, _pl.LazyFrame):
                    _df = _df.collect()
                self.scan(
                    " AND ".join(
                        [
                            f"{col}<='{_df.select(_pl.col(col).max())[0,0]}' AND {col}>='{_df.select(_pl.col(col).min())[0,0]}'"
                            for col in _df.columns
                        ]
                    )
                )
                df0 = self.pl

                _df = _df.delta(df0, subset=delta_subset, eager=True)
                
            if _df.shape[0]:
                metadata = write_table(
                    df=_df,
                    path=path,
                    schema=self.file_schema,
                    filesystem=self._filesystem,
                    row_group_size=row_group_size,
                    compression=compression,
                    sort_by=sort_by,
                    distinct=distinct,
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

        self.reload_files()
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
        if filter_expr is not None:
            self.scan(filter_expr=filter_expr)
            file_catalog = self.file_catalog.filter(
                _pl.col("file_path").is_in([os.path.basename(f) for f in self._scan_files])
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

        del_files = []

        if not strict:
            file_groups = file_catalog.with_columns(
                (_pl.col("total_byte_size").cumsum() // target_byte_size).alias("group")
            ).select(["file_path", "total_byte_size", "group"])

            for file_group in file_groups.partition_by("group"):
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
        if filter_expr is not None:
            self.scan(filter_expr=filter_expr)
            file_catalog = self.file_catalog.filter(
                _pl.col("file_path").is_in([os.path.basename(f) for f in self._scan_files])
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

            for file_group in file_groups.partition_by("group"):
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

            self.delete_files(del_files)

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

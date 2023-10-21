import os

from fsspec import AbstractFileSystem
from fsspec import filesystem as fsspec_filesystem
from fsspec.implementations import cached as cachedfs
from fsspec.implementations import dirfs


import datetime as dt


import duckdb as ddb
import msgspec
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from fsspec.implementations.dirfs import DirFileSystem
from .helpers.misc import run_parallel
from .schema import shrink_large_string


def get_new_file_names(src: list[str], dst: list[str]) -> list[str]:
    """
    Returns a list of new file names that are not in the destination list

    Parameters
    ----------
    src : list[str]
        List of source file paths
    dst : list[str]
        List of destination file paths

    Returns
    -------
    list[str]
        List of new file names that are not in the destination list
    """
    if len(dst) == 0:
        return src
    src_file_names = [os.path.basename(f).split(".")[0] for f in src]
    src_ext = os.path.basename(src[0]).split(".")[1]
    dst_file_names = [os.path.basename(f).split(".")[0] for f in dst]
    # dst_ext = os.path.basename(dst[0]).split(".")[1]

    return [
        os.path.join(os.path.dirname(src), f, src_ext)
        for f in src_file_names
        if f not in dst_file_names
    ]


def read_parquet(self, path: str, **kwargs) -> pl.DataFrame:
    with self.open(path) as f:
        return pl.from_arrow(pq.read_table(f, **kwargs))


def read_parquet_schema(self, path: str, **kwargs) -> pa.Schema:
    return pq.read_schema(path, filesystem=self, **kwargs)


def read_paruet_metadata(self, path: str, **kwargs) -> dict:
    return pq.read_metadata(path, filesystem=self, **kwargs)


def read_json(
    self,
    path: str,
    filename: bool = False,
    as_dataframe: bool = True,
    flatten: bool = True,
) -> dict | pl.DataFrame:
    with self.open(path) as f:
        data = msgspec.json.decode(f.read())
        if as_dataframe:
            data = pl.from_dicts(data)
            if flatten:
                data = data.explode_all().unnest_all()

        if filename:
            return {path: data}

        return data


def read_csv(self, path: str, **kwargs) -> pl.DataFrame:
    with self.open(path) as f:
        return pl.from_csv(f, **kwargs)


def read_parquet_dataset(
    self, path: str | list[str], concat: bool = True, **kwargs
) -> pl.DataFrame | list[pl.DataFrame]:
    if isinstance(path, str):
        files = self.glob(os.path.join(path, "*.parquet"))
    else:
        files = path
    if isinstance(files, str):
        files = [files]

    dfs = run_parallel(self.read_parquet, files, **kwargs)

    dfs = pl.concat(dfs, how="diagonal_relaxed") if concat else dfs
    dfs = dfs[0] if len(dfs) == 1 else dfs

    return dfs


def read_json_dataset(
    self,
    path: str | list[str],
    filename: bool = False,
    as_dataframe: bool = True,
    flatten: bool = True,
    concat: bool = True,
) -> dict | list[dict] | pl.DataFrame | list[pl.DataFrame]:
    if isinstance(path, str):
        files = self.glob(os.path.join(path, "*.json"))
    else:
        files = path
    if isinstance(files, str):
        files = [files]
    data = run_parallel(
        self.read_json,
        files,
        filename=filename,
        as_dataframe=as_dataframe,
        flatten=flatten,
    )
    if as_dataframe and concat:
        pl.concat(data, how="diagonal_relaxed")

    data = data[0] if len(data) == 1 else data

    return data


def read_csv_dataset(
    self, path: str | list[str], concat: bool = True, **kwargs
) -> pl.DataFrame | list[pl.DataFrame]:
    if isinstance(path, str):
        files = self.glob(os.path.join(path, "*.csv"))
    else:
        files = path
    if isinstance(files, str):
        files = [files]
    dfs = run_parallel(self.read_csv, files, self=self, **kwargs)

    dfs = pl.concat(dfs, how="diagonal_relaxed") if concat else dfs
    dfs = dfs[0] if len(dfs) == 1 else dfs

    return dfs


def pyarrow_dataset(
    self,
    path: str,
    format="parquet",
    schema: pa.Schema | None = None,
    partitioning: str | list[str] | pds.Partitioning = None,
    **kwargs,
) -> pds.Dataset:
    return pds.dataset(
        self.glob(os.path.join(path, f"*.{format}")),
        filesystem=self,
        partitioning=partitioning,
        schema=schema,
        **kwargs,
    )


# def pydala_dataset(
#     self, path: str, partitioning: str | None = None, **kwargs
# ) -> ParquetDataset:
#     return ParquetDataset(path=path, filesystem=self, **kwargs)


def write_parquet(
    self,
    data: pl.DataFrame | pa.Table | pd.DataFrame | ddb.DuckDBPyRelation,
    path: str,
    **kwargs,
) -> None:
    if isinstance(data, pl.DataFrame):
        data = data.to_arrow()
        data = data.cast(shrink_large_string(data.schema))
    elif isinstance(data, pd.DataFrame):
        data = pa.Table.from_pandas(data, preserve_index=False)
    elif isinstance(data, ddb.DuckDBPyRelation):
        data = data.arrow()

    pq.write_table(data, path, filesystem=self, **kwargs)


def write_json(
    self,
    data: dict | pl.DataFrame | pa.Table | pd.DataFrame | ddb.DuckDBPyRelation,
    path: str,
) -> None:
    if isinstance(data, pl.DataFrame):
        data = data.to_arrow()
        data = data.cast(shrink_large_string(data.schema)).to_pydict()
    elif isinstance(data, pd.DataFrame):
        data = pa.Table.from_pandas(data, preserve_index=False).to_pydict()
    elif isinstance(data, ddb.DuckDBPyRelation):
        data = data.arrow().to_pydict()

    with self.open(path, "w") as f:
        f.write(msgspec.json.encode(data))


def write_csv(
    self,
    data: pl.DataFrame | pa.Table | pd.DataFrame | ddb.DuckDBPyRelation,
    path: str,
    **kwargs,
) -> None:
    if isinstance(data, pa.Table):
        data = pl.from_arrow(data)
    elif isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)
    elif isinstance(data, ddb.DuckDBPyRelation):
        data = data.pl()

    with self.open(path, "w") as f:
        data.write_csv(f, **kwargs)


def write_to_pyarrow_dataset(
    self,
    data: pl.DataFrame
    | pa.Table
    | pd.DataFrame
    | ddb.DuckDBPyRelation
    | list[pl.DataFrame]
    | list[pa.Table]
    | list[pd.DataFrame]
    | list[ddb.DuckDBPyRelation],
    path: str,
    basename: str | None = None,
    concat: bool = True,
    schema: pa.Schema | None = None,
    partitioning: str | list[str] | pds.Partitioning | None = None,
    partitioning_flavor: str = "hive",
    mode: str = "append",
    format: str | None = "parquet",
    **kwargs,
) -> None:
    if not isinstance(data, list):
        data = [data]

    if isinstance(data[0], pl.DataFrame):
        data = [dd.to_arrow() for dd in data]
        data = [dd.cast(shrink_large_string(dd.schema)) for dd in data]

    elif isinstance(data[0], pd.DataFrame):
        data = [pa.Table.from_pandas(dd, preserve_index=False) for dd in data]

    elif isinstance(data, ddb.DuckDBPyRelation):
        data = [dd.arrow() for dd in data]

    if concat:
        data = pa.concat_tables(data, promote=True)

    if mode == "delete_matching":
        existing_data_behavior = "delete_matching"
    elif mode == "append":
        existing_data_behavior = "overwrite_or_ignore"
    elif mode == "overwrite":
        self.rm(path, recursive=True)
        existing_data_behavior = "overwrite_or_ignore"
    else:
        existing_data_behavior = mode

    if basename is None:
        basename = (
            f"data-{dt.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]}"
            + "-{i}"
            + f".{format}"
        )

    pds.write_dataset(
        data=data,
        base_dir=path,
        basename_template=basename,
        partitioning=partitioning,
        partitioning_flavor=partitioning_flavor,
        filesystem=self,
        existing_data_behavior=existing_data_behavior,
        schema=schema,
        format=format,
        **kwargs,
    )




def _json_to_parquet(
    self,
    src: str,
    dst: str,
    flatten: bool = True,
    distinct: bool = True,
    sort_by: str | list[str] | None = None,
    auto_optimize_dtypes: bool = True,
    **kwargs,
):
    data = self.read_json(src, as_dataframe=True, flatten=flatten)

    if auto_optimize_dtypes:
        data = data.opt_dtype()

    if distinct:
        data = data.unique(maintain_order=True)

    if sort_by is not None:
        data = data.sort(sort_by)

    if ".parquet" not in dst:
        dst = os.path.join(dst, f"{os.path.basename(src).replace('.json', '.parquet')}")

    self.write_parquet(data, dst, **kwargs)


def _csv_to_parquet(
    self,
    src: str,
    dst: str,
    distinct: bool = True,
    sort_by: str | list[str] | None = None,
    auto_optimize_dtypes: bool = True,
    **kwargs,
):
    data = self.read_csv(src, self, **kwargs)

    if auto_optimize_dtypes:
        data = data.opt_dtype()

    if distinct:
        data = data.unique(maintain_order=True)

    if sort_by is not None:
        data = data.sort(sort_by)

    if ".parquet" not in dst:
        dst = os.path.join(dst, f"{os.path.basename(src).replace('.csv', '.parquet')})")

    self.write_parquet(data, dst, **kwargs)


def json_to_parquet(
    self,
    src: str,
    dst: str,
    flatten: bool = True,
    sync: bool = True,
    parallel: bool = True,
    distinct: bool = True,
    sort_by: str | list[str] | None = None,
    auto_optimize_dtypes: bool = True,
    **kwargs,
):
    if sync:
        src_files = self.glob(os.path.join(src, "*.json"))
        dst_files = self.glob(os.path.join(dst, "*.parquet"))
        new_src_files = get_new_file_names(src_files, dst_files)

    else:
        new_src_files = self.glob(os.path.join(src, "*.json"))

    kwargs.pop("backend", None)
    if len(new_src_files) == 1:
        parallel = False

    if len(new_src_files) > 0:
        run_parallel(
            self._json_to_parquet,
            new_src_files,
            dst=dst,
            flatten=flatten,
            backend="threading" if parallel else "sequential",
            distinct=distinct,
            sort_by=sort_by,
            auto_optimize_dtypes=auto_optimize_dtypes,
            **kwargs,
        )


def csv_to_parquet(
    self,
    src: str,
    dst: str,
    sync: bool = True,
    parallel: bool = True,
    distinct: bool = True,
    sort_by: str | list[str] | None = None,
    auto_optimize_dtypes: bool = True,
    **kwargs,
):
    if sync:
        src_files = self.glob(os.path.join(src, "*.csv"))
        dst_files = self.glob(os.path.join(dst, "*.parquet"))
        new_src_files = get_new_file_names(src_files, dst_files)

    else:
        new_src_files = self.glob(os.path.join(src, "*.csv"))

    kwargs.pop("backend", None)
    if len(new_src_files) == 1:
        parallel = False
    if len(new_src_files) > 0:
        run_parallel(
            self._csv_to_parquet,
            new_src_files,
            dst=dst,
            backend="threading" if parallel else "sequential",
            distinct=distinct,
            sort_by=sort_by,
            auto_optimize_dtypes=auto_optimize_dtypes,
            **kwargs,
        )


def ls(
    self,
    path,
    recursive: bool = False,
    maxdepth: int | None = None,
    detail: bool = False,
    files_only: bool = False,
    dirs_only: bool = False,
):
    if detail:
        return self.listdir(path)

    if isinstance(path, str):
        if self.isfile(path):
            path = [path]
        else:
            path = path.rstrip("*").rstrip("/")
            if self.exists(path):
                if not recursive:
                    path = self.glob(os.path.join(path, "*"))
                else:
                    if maxdepth is not None:
                        path = self.glob(os.path.join(path, *(["*"] * maxdepth)))
                    else:
                        path = self.glob(os.path.join(path, "**"))
            else:
                path = self.glob(path)

    if files_only:
        files = [f for f in path if "." in os.path.basename(f)]
        files += [f for f in sorted(set(path) - set(files)) if self.isfile(f)]
        return files

    if dirs_only:
        dirs = [f for f in path if "." not in os.path.basename(f)]
        dirs += [f for f in sorted(set(path) - set(dirs)) if self.isdir(f)]
        return dirs

    return path



def sync_folder(
    self, src: str, dst: str, recursive: bool = False, maxdepth: int | None = None
):
    src_ = self.lss(src, recursive=recursive, maxdepth=maxdepth, files_only=True)
    dst_ = self.lss(dst, recursive=recursive, maxdepth=maxdepth, files_only=True)

    if len(src) == 0:
        return

    src_names = [os.path.basename(f) for f in src_]
    dst_names = [os.path.basename(f) for f in dst_]

    new_src = [
        os.path.join(os.path.dirname(src[0]), f)
        for f in sorted(set(src_names) - set(dst_names))
    ]

    if len(new_src):
        self.cp(new_src, dst)


fsspec_filesystem.read_parquet = read_parquet
fsspec_filesystem.read_parquet_dataset = read_parquet_dataset
fsspec_filesystem.write_parquet = write_parquet
# DirFileSystem.write_to_parquet_dataset = write_to_pydala_dataset
DirFileSystem.write_to_dataset = write_to_pyarrow_dataset
fsspec_filesystem.read_parquet_schema = read_parquet_schema
fsspec_filesystem.read_parquet_metadata = read_paruet_metadata

fsspec_filesystem.read_csv = read_csv
fsspec_filesystem.read_csv_dataset = read_csv_dataset
fsspec_filesystem.write_csv = write_csv
fsspec_filesystem._csv_to_parquet = _csv_to_parquet
fsspec_filesystem.csv_to_parquet = csv_to_parquet

fsspec_filesystem.read_json = read_json
fsspec_filesystem.read_json_dataset = read_json_dataset
fsspec_filesystem.write_json = write_json
fsspec_filesystem._json_to_parquet = _json_to_parquet
fsspec_filesystem.json_to_parquet = json_to_parquet

fsspec_filesystem.pyarrow_dataset = pyarrow_dataset
# fsspec_filesystem.pydala_dataset = pydala_dataset

fsspec_filesystem.lss = ls
fsspec_filesystem.ls2 = ls

# fsspec_filesystem.parallel_cp = parallel_cp
# fsspec_filesystem.parallel_mv = parallel_mv
# fsspec_filesystem.parallel_rm = parallel_rm
fsspec_filesystem.sync_folder = sync_folder


def get_filesystem(
    bucket: str | None = None,
    fs=AbstractFileSystem | None,
    cached: bool = False,
    cache_storage="~/.tmp",
    check_files: bool = False,
    cache_check: int = 120,
    expire_time: int = 24 * 60 * 60,
    same_names: bool = False,
    **kwargs,
):
    if fs is None:
        fs = fsspec_filesystem("file")

    if bucket is not None:
        fs = DirFileSystem(path=bucket, fs=fs)

    if cached:
        if "~" in cache_storage:
            cache_storage = os.path.expanduser(cache_storage)

        return cachedfs.SimpleCacheFileSystem(
            cache_storage=cache_storage,
            check_files=check_files,
            cache_check=cache_check,
            expire_time=expire_time,
            fs=fs,
            same_names=same_names,
            **kwargs,
        )
    return fs


def clear_cache(fs: AbstractFileSystem | None):
    if fs is not None:
        if hasattr(fs, "clear_cache"):
            fs.clear_cache()
        fs.invalidate_cache()
        fs.clear_instance_cache()
        if hasattr(fs, "fs"):
            fs.invalidate_cache()
            fs.clear_instance_cache()

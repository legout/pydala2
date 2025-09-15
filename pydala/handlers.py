import posixpath
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
import orjson
from fsspec import AbstractFileSystem

from .helpers.misc import run_parallel
from .schema import convert_large_types_to_normal


class FormatHandler(ABC):
    @abstractmethod
    def read_single(self, fs: AbstractFileSystem, path: str, **kwargs) -> Any:
        pass

    @abstractmethod
    def read_dataset(self, fs: AbstractFileSystem, path: str | List[str], concat: bool = True, filename: bool = False, **kwargs) -> Any:
        pass

    @abstractmethod
    def write_single(self, fs: AbstractFileSystem, data: Any, path: str, **kwargs) -> None:
        pass

    @abstractmethod
    def glob_pattern(self, path: str) -> str:
        pass


def _to_arrow(data: Any) -> pa.Table:
    """Centralized type conversion to Arrow Table."""
    if isinstance(data, pl.DataFrame):
        return data.to_arrow().cast(convert_large_types_to_normal(data.schema))
    elif isinstance(data, pa.Table):
        return data.cast(convert_large_types_to_normal(data.schema))
    elif hasattr(data, 'to_pandas'):  # pd.DataFrame or ddb.DuckDBPyRelation
        if hasattr(data, 'arrow'):  # DuckDB
            return data.arrow().cast(convert_large_types_to_normal(data.schema))
        return pa.Table.from_pandas(data, preserve_index=False).cast(convert_large_types_to_normal(data.schema))
    raise ValueError(f"Unsupported data type: {type(data)}")


class ParquetHandler(FormatHandler):
    def read_single(self, fs: AbstractFileSystem, path: str, filename: bool = False, **kwargs) -> Dict[str, pl.DataFrame] | pl.DataFrame:
        from .helpers.misc import read_table
        data = pl.from_arrow(read_table(path, filesystem=fs, **kwargs))
        return {path: data} if filename else data

    def read_dataset(self, fs: AbstractFileSystem, path: str | List[str], concat: bool = True, filename: bool = False, **kwargs) -> Dict[str, pl.DataFrame] | pl.DataFrame | List:
        if filename:
            concat = False
        if isinstance(path, str):
            files = fs.glob(posixpath.join(path, self.glob_pattern("")))
        else:
            files = path
        if isinstance(files, str):
            files = [files]
        dfs = run_parallel(self.read_single, files, fs=fs, filename=filename, **kwargs)
        if concat:
            dfs = pl.concat(dfs, how="diagonal_relaxed")
        return dfs[0] if len(dfs) == 1 and not filename else dfs

    def write_single(self, fs: AbstractFileSystem, data: Any, path: str, **kwargs) -> None:
        arrow_data = _to_arrow(data)
        pq.write_table(arrow_data, path, filesystem=fs, **kwargs)

    def glob_pattern(self, path: str) -> str:
        return posixpath.join(path, "*.parquet")

    def read_schema(self, fs: AbstractFileSystem, path: str, **kwargs) -> pa.Schema:
        return pq.read_schema(path, filesystem=fs, **kwargs)

    def read_metadata(self, fs: AbstractFileSystem, path: str, **kwargs) -> pq.FileMetaData:
        return pq.read_metadata(path, filesystem=fs, **kwargs)


class JSONHandler(FormatHandler):
    def read_single(self, fs: AbstractFileSystem, path: str, filename: bool = False, as_dataframe: bool = True, flatten: bool = True, **kwargs) -> Dict | pl.DataFrame:
        with fs.open(path) as f:
            data = orjson.loads(f.read())
        if as_dataframe:
            data = pl.from_dicts(data)
            if flatten:
                data = data.explode_all().unnest_all()
        return {path: data} if filename else data

    def read_dataset(self, fs: AbstractFileSystem, path: str | List[str], concat: bool = True, filename: bool = False, as_dataframe: bool = True, flatten: bool = True, **kwargs) -> Dict | pl.DataFrame | List:
        if filename:
            concat = False
        if isinstance(path, str):
            files = fs.glob(posixpath.join(path, self.glob_pattern("")))
        else:
            files = path
        if isinstance(files, str):
            files = [files]
        data = run_parallel(self.read_single, files, fs=fs, filename=filename, as_dataframe=as_dataframe, flatten=flatten, **kwargs)
        if as_dataframe and concat:
            data = pl.concat(data, how="diagonal_relaxed")
        return data[0] if len(data) == 1 else data

    def write_single(self, fs: AbstractFileSystem, data: Any, path: str, **kwargs) -> None:
        if isinstance(data, (pl.DataFrame, pa.Table)):
            data = _to_arrow(data).to_pydict()
        with fs.open(path, "w") as f:
            f.write(orjson.dumps(data))

    def glob_pattern(self, path: str) -> str:
        return posixpath.join(path, "*.json")


class CSVHandler(FormatHandler):
    def read_single(self, fs: AbstractFileSystem, path: str, filename: bool = False, **kwargs) -> Dict[str, pl.DataFrame] | pl.DataFrame:
        with fs.open(path) as f:
            data = pl.read_csv(f, **kwargs)
        return {path: data} if filename else data

    def read_dataset(self, fs: AbstractFileSystem, path: str | List[str], concat: bool = True, filename: bool = False, **kwargs) -> Dict[str, pl.DataFrame] | pl.DataFrame | List:
        if filename:
            concat = False
        if isinstance(path, str):
            files = fs.glob(posixpath.join(path, self.glob_pattern("")))
        else:
            files = path
        if isinstance(files, str):
            files = [files]
        dfs = run_parallel(self.read_single, files, fs=fs, filename=filename, **kwargs)
        if concat:
            dfs = pl.concat(dfs, how="diagonal_relaxed")
        return dfs[0] if len(dfs) == 1 else dfs

    def write_single(self, fs: AbstractFileSystem, data: Any, path: str, **kwargs) -> None:
        if not isinstance(data, pl.DataFrame):
            if isinstance(data, pa.Table):
                data = pl.from_arrow(data)
            elif hasattr(data, 'pl'):  # DuckDB
                data = data.pl()
            else:  # pd
                data = pl.from_pandas(data)
        with fs.open(path, "w") as f:
            data.write_csv(f, **kwargs)

    def glob_pattern(self, path: str) -> str:
        return posixpath.join(path, "*.csv")


FORMAT_HANDLERS = {
    "parquet": ParquetHandler(),
    "json": JSONHandler(),
    "csv": CSVHandler(),
}

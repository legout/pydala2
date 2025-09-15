"""
Extensions to fsspec AbstractFileSystem to add data reading/writing capabilities.
"""
import posixpath
from typing import Any, Union

import pyarrow.dataset as pds
from fsspec import AbstractFileSystem
from fsspec.implementations.dirfs import DirFileSystem

from .utils import (
    list_files, sync_folder,
    batch_json_to_parquet, batch_csv_to_parquet
)


# Monkey patch methods to AbstractFileSystem
def _read_parquet(self, path: str, filename: bool = False, **kwargs) -> Union[dict[str, Any], Any]:
    """Read Parquet file."""
    from .converters import ConversionService
    conversion_service = ConversionService()
    converter = conversion_service.get_converter("parquet")
    return converter.read(self, path, filename=filename, **kwargs)


def _read_parquet_schema(self, path: str, **kwargs):
    """Read Parquet schema."""
    from .converters import ConversionService
    conversion_service = ConversionService()
    converter = conversion_service.get_converter("parquet")
    return converter.read_schema(self, path, **kwargs)


def _read_parquet_metadata(self, path: str, **kwargs):
    """Read Parquet metadata."""
    from .converters import ConversionService
    conversion_service = ConversionService()
    converter = conversion_service.get_converter("parquet")
    return converter.read_metadata(self, path, **kwargs)


def _write_parquet(self, data: Any, path: str, **kwargs) -> None:
    """Write Parquet file."""
    from .converters import ConversionService
    conversion_service = ConversionService()
    converter = conversion_service.get_converter("parquet")
    converter.write(self, data, path, **kwargs)


def _read_parquet_dataset(self, path: Union[str, list], concat: bool = True, filename: bool = False, **kwargs) -> Any:
    """Read Parquet dataset."""
    from .converters import ConversionService
    conversion_service = ConversionService()
    converter = conversion_service.get_converter("parquet")
    return converter.read_dataset(self, path, concat=concat, filename=filename, **kwargs)


def _read_csv(self, path: str, filename: bool = False, **kwargs) -> Union[dict[str, Any], Any]:
    """Read CSV file."""
    from .converters import ConversionService
    conversion_service = ConversionService()
    converter = conversion_service.get_converter("csv")
    return converter.read(self, path, filename=filename, **kwargs)


def _write_csv(self, data: Any, path: str, **kwargs) -> None:
    """Write CSV file."""
    from .converters import ConversionService
    conversion_service = ConversionService()
    converter = conversion_service.get_converter("csv")
    converter.write(self, data, path, **kwargs)


def _read_csv_dataset(self, path: Union[str, list], concat: bool = True, filename: bool = False, **kwargs) -> Any:
    """Read CSV dataset."""
    from .converters import ConversionService
    conversion_service = ConversionService()
    converter = conversion_service.get_converter("csv")
    return converter.read_dataset(self, path, concat=concat, filename=filename, **kwargs)


def _read_json(self, path: str, filename: bool = False, as_dataframe: bool = True, flatten: bool = True, **kwargs) -> Union[dict, Any]:
    """Read JSON file."""
    from .converters import ConversionService
    conversion_service = ConversionService()
    converter = conversion_service.get_converter("json")
    return converter.read(self, path, filename=filename, as_dataframe=as_dataframe, flatten=flatten, **kwargs)


def _write_json(self, data: Any, path: str) -> None:
    """Write JSON file."""
    from .converters import ConversionService
    conversion_service = ConversionService()
    converter = conversion_service.get_converter("json")
    converter.write(self, data, path)


def _read_json_dataset(self, path: Union[str, list], filename: bool = False, as_dataframe: bool = True, flatten: bool = True, concat: bool = True, **kwargs) -> Any:
    """Read JSON dataset."""
    from .converters import ConversionService
    conversion_service = ConversionService()
    converter = conversion_service.get_converter("json")
    return converter.read_dataset(self, path, filename=filename, as_dataframe=as_dataframe, flatten=flatten, concat=concat, **kwargs)


def _pyarrow_dataset(self, path: str, format="parquet", **kwargs) -> pds.Dataset:
    """Create PyArrow dataset."""
    from .converters import ConversionService
    conversion_service = ConversionService()
    converter = conversion_service.get_converter("parquet")
    return converter.to_pyarrow_dataset(self, path, format=format, **kwargs)


def _pyarrow_parquet_dataset(self, path: str, **kwargs) -> pds.FileSystemDataset:
    """Create PyArrow Parquet dataset."""
    return pds.dataset(
        posixpath.join(path, "_metadata"),
        filesystem=self,
        **kwargs,
    )


def _lss(self, path, recursive: bool = False, maxdepth: int = None, detail: bool = False, files_only: bool = False, dirs_only: bool = False):
    """Enhanced list files method."""
    return list_files(
        self,
        path,
        recursive=recursive,
        maxdepth=maxdepth,
        files_only=files_only,
        dirs_only=dirs_only,
    )


# Apply monkey patches
AbstractFileSystem.read_parquet = _read_parquet
AbstractFileSystem.read_parquet_dataset = _read_parquet_dataset
AbstractFileSystem.write_parquet = _write_parquet
AbstractFileSystem.read_parquet_schema = _read_parquet_schema
AbstractFileSystem.read_parquet_metadata = _read_parquet_metadata

AbstractFileSystem.read_csv = _read_csv
AbstractFileSystem.read_csv_dataset = _read_csv_dataset
AbstractFileSystem.write_csv = _write_csv

AbstractFileSystem.read_json = _read_json
AbstractFileSystem.read_json_dataset = _read_json_dataset
AbstractFileSystem.write_json = _write_json

AbstractFileSystem.pyarrow_dataset = _pyarrow_dataset
AbstractFileSystem.pyarrow_parquet_dataset = _pyarrow_parquet_dataset

AbstractFileSystem.lss = _lss
AbstractFileSystem.ls2 = _lss  # Alias

# Add write_to_dataset method to DirFileSystem
def _write_to_dataset(self, data, path: str, **kwargs) -> None:
    """Write dataset using PyArrow."""
    from .converters import ConversionService
    conversion_service = ConversionService()
    converter = conversion_service.get_converter("parquet")
    converter.write_dataset(self, data, path, **kwargs)


DirFileSystem.write_to_dataset = _write_to_dataset


# Add conversion utility methods
def _json_to_parquet(self, src: str, dst: str, **kwargs) -> None:
    """Convert JSON to Parquet."""
    batch_json_to_parquet(self, src, dst, **kwargs)


def _csv_to_parquet(self, src: str, dst: str, **kwargs) -> None:
    """Convert CSV to Parquet."""
    batch_csv_to_parquet(self, src, dst, **kwargs)


AbstractFileSystem.json_to_parquet = _json_to_parquet
AbstractFileSystem.csv_to_parquet = _csv_to_parquet


# Add sync utility method
def _sync_folder(self, src: str, dst: str, recursive: bool = False, maxdepth: int = None) -> None:
    """Sync folder."""
    sync_folder(self, src, dst, recursive=recursive, maxdepth=maxdepth)


AbstractFileSystem.sync_folder = _sync_folder
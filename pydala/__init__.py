"""Manage and query Parquet datasets with PyArrow and DuckDB.

    >>> from tempfile import TemporaryDirectory
    >>> import pyarrow as pa
    >>> from pydala import ParquetDataset
    >>> with TemporaryDirectory() as tmp:
    ...     path = f"{tmp}/sales"
    ...     ParquetDataset(path).write_to_dataset(pa.table({"amount": [10, 20]}))
    ...     ParquetDataset(path).filter("amount >= 20").arrow.to_pylist()
    [{'amount': 20}]
"""

import importlib.metadata

__version__ = importlib.metadata.version("pydala2")

# Import main classes for convenience
from .dataset import BaseDataset, ParquetDataset, CSVDataset, JSONDataset, PyarrowDataset
from .catalog import Catalog
from .table import PydalaTable
from .filesystem import FileSystem
from .metadata import ParquetDatasetMetadata, PydalaDatasetMetadata
from .schema import sort_schema, unify_schemas

__all__ = [
    "BaseDataset",
    "ParquetDataset",
    "CSVDataset",
    "JSONDataset",
    "PyarrowDataset",
    "Catalog",
    "PydalaTable",
    "FileSystem",
    "ParquetDatasetMetadata",
    "PydalaDatasetMetadata",
    "sort_schema",
    "unify_schemas",
]

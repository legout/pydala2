"""
PyDala2 - High-performance Python library for managing Parquet datasets with advanced metadata capabilities.

PyDala2 provides efficient management of Parquet datasets with features including:
- Smart dataset management with metadata optimization
- Multi-format support (Parquet, CSV, JSON)
- Multi-backend integration (Polars, PyArrow, DuckDB, Pandas)
- Advanced querying with predicate pushdown
- Schema management with automatic validation
- Performance optimization with caching and partitioning
- Catalog system for centralized dataset management

Example usage:
    >>> from pydala import ParquetDataset
    >>> dataset = ParquetDataset("data/my_dataset")
    >>> dataset.write(dataframe)
    >>> result = dataset.filter("date > '2023-01-01'").to_pandas()
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

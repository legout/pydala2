"""
Dataset module for pydala.

This module provides classes for working with various dataset formats including Parquet, CSV, and JSON.
"""

from .base import BaseDataset
from .datasets import ParquetDataset, PyarrowDataset, CsvDataset, JsonDataset
from .optimizer import DatasetOptimizer

# Re-export for backward compatibility
from .file_manager import FileManager
from .data_processor import DataProcessor
from .dataset_filter import DatasetFilter
from .dataset_writer import DatasetWriter

__all__ = [
    "BaseDataset",
    "ParquetDataset",
    "PyarrowDataset",
    "CsvDataset",
    "JsonDataset",
    "DatasetOptimizer",
    "FileManager",
    "DataProcessor",
    "DatasetFilter",
    "DatasetWriter",
]
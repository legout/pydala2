"""
Dataset management components for Pydala.

This package provides classes and utilities for reading, writing, and managing datasets,
with support for various formats including Parquet, CSV, and JSON.
"""

from .base import (
    BaseDataset,
    ParquetDataset,
    PyarrowDataset,
    CsvDataset,
    JsonDataset,
    OptimizeMixin,
    Optimize,
)
from .reader import DatasetReader, ReadConfig
from .writer import DatasetWriter, WriteConfig, DeltaConfig

__all__ = [
    "BaseDataset",
    "ParquetDataset",
    "PyarrowDataset",
    "CsvDataset",
    "JsonDataset",
    "OptimizeMixin",
    "Optimize",
    "DatasetReader",
    "ReadConfig",
    "DatasetWriter",
    "WriteConfig",
    "DeltaConfig",
]
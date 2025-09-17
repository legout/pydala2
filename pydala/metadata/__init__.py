"""
Metadata management components for Pydala.

This package provides classes and utilities for managing dataset metadata,
including schema information, partition tracking, and file metadata operations.
"""

from .core import DatasetMetadata, MetadataConfig, FileMetadata, PartitionInfo
from .parquet import ParquetDatasetMetadata, PydalaDatasetMetadata

__all__ = [
    "DatasetMetadata",
    "MetadataConfig",
    "FileMetadata",
    "PartitionInfo",
    "ParquetDatasetMetadata",
    "PydalaDatasetMetadata",
]
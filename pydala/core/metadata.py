"""
Dataset metadata management component.

This module provides a focused, simplified interface for managing dataset metadata,
including schema information, partition tracking, and file metadata operations.
"""

import datetime as dt
import json
import posixpath
import re
import tempfile
import typing as t
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from loguru import logger

from ..filesystem import FileSystem
from ..helpers.misc import get_partitions_from_path, run_parallel
from ..helpers.security import (
    escape_sql_identifier,
    escape_sql_literal,
    validate_partition_name,
    validate_partition_value,
)
from ..schema import convert_large_types_to_normal, repair_schema


@dataclass
class MetadataConfig:
    """Configuration for metadata operations."""
    
    update_on_write: bool = True
    format_version: str = "2.6"
    ts_unit: str = "us"
    tz: str | None = None
    cache_metadata: bool = True
    metadata_file_name: str = "_metadata"
    common_metadata_file_name: str = "_common_metadata"


@dataclass
class PartitionInfo:
    """Information about dataset partitioning."""
    
    names: list[str] = field(default_factory=list)
    schema: pa.Schema | None = None
    values: dict[str, list] = field(default_factory=dict)
    flavor: str = "hive"
    
    def is_partitioned(self) -> bool:
        """Check if dataset is partitioned."""
        return len(self.names) > 0
    
    def get_partition_filter(self, partition_values: dict[str, t.Any]) -> str:
        """Build a SQL filter expression for partition values."""
        filter_parts = []
        for name, value in partition_values.items():
            if not validate_partition_name(name):
                raise ValueError(f"Invalid partition name: {name}")
            if not validate_partition_value(value):
                raise ValueError(f"Invalid partition value: {value}")
            
            escaped_name = escape_sql_identifier(name)
            escaped_value = escape_sql_literal(value)
            filter_parts.append(f"{escaped_name}={escaped_value}")
        
        return " AND ".join(filter_parts)


@dataclass
class FileMetadata:
    """Metadata for a single file in the dataset."""
    
    path: str
    num_rows: int
    num_columns: int
    size_bytes: int
    created_by: str | None = None
    format_version: str | None = None
    row_groups: int = 0
    schema: pa.Schema | None = None
    column_stats: dict | None = None
    partition_values: dict | None = None
    
    @classmethod
    def from_parquet_metadata(cls, path: str, metadata: pq.FileMetaData) -> "FileMetadata":
        """Create FileMetadata from PyArrow FileMetaData."""
        return cls(
            path=path,
            num_rows=metadata.num_rows,
            num_columns=metadata.num_columns,
            size_bytes=metadata.serialized_size,
            created_by=metadata.created_by,
            format_version=metadata.format_version,
            row_groups=metadata.num_row_groups,
            schema=metadata.schema.to_arrow_schema(),
        )


class DatasetMetadata:
    """
    Manages all metadata operations for a dataset.
    
    This class consolidates metadata management including schema tracking,
    partition information, file metadata, and metadata persistence.
    """
    
    def __init__(
        self,
        path: str,
        filesystem: AbstractFileSystem | None = None,
        config: MetadataConfig | None = None,
        ddb_con: duckdb.DuckDBPyConnection | None = None,
    ):
        """
        Initialize DatasetMetadata.
        
        Args:
            path: Path to the dataset
            filesystem: Filesystem to use for I/O operations
            config: Metadata configuration
            ddb_con: DuckDB connection for metadata queries
        """
        self.path = path
        self.filesystem = filesystem or FileSystem()
        self.config = config or MetadataConfig()
        self.ddb_con = ddb_con
        
        # Initialize metadata storage
        self._schema: pa.Schema | None = None
        self._partition_info = PartitionInfo()
        self._file_metadata: dict[str, FileMetadata] = {}
        self._metadata_table_name: str | None = None
        self._timestamp_column: str | None = None
        
        # Metadata file paths
        self._metadata_file = posixpath.join(path, self.config.metadata_file_name)
        self._common_metadata_file = posixpath.join(path, self.config.common_metadata_file_name)
        
    @property
    def schema(self) -> pa.Schema | None:
        """Get the dataset schema."""
        return self._schema
    
    @schema.setter
    def schema(self, value: pa.Schema):
        """Set the dataset schema."""
        self._schema = value
        
    @property
    def partition_info(self) -> PartitionInfo:
        """Get partition information."""
        return self._partition_info
    
    @property
    def file_metadata(self) -> dict[str, FileMetadata]:
        """Get file metadata dictionary."""
        return self._file_metadata
    
    @property
    def has_metadata_file(self) -> bool:
        """Check if metadata file exists."""
        return self.filesystem.exists(self._metadata_file)
    
    def load_metadata(self, reload: bool = False) -> None:
        """
        Load metadata from the dataset.
        
        Args:
            reload: Force reload even if metadata is cached
        """
        if not reload and self._schema is not None and self.config.cache_metadata:
            return
            
        try:
            if self.has_metadata_file:
                self._load_from_metadata_file()
            else:
                self._infer_metadata_from_files()
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            self._infer_metadata_from_files()
    
    def _load_from_metadata_file(self) -> None:
        """Load metadata from _metadata file."""
        try:
            metadata = pq.read_metadata(self._metadata_file, filesystem=self.filesystem)
            self._schema = metadata.schema.to_arrow_schema()
            
            # Extract partition information if available
            if hasattr(metadata, 'metadata') and metadata.metadata:
                meta_dict = json.loads(metadata.metadata.get(b'partitioning', b'{}'))
                if meta_dict:
                    self._partition_info.names = meta_dict.get('names', [])
                    
        except Exception as e:
            logger.warning(f"Failed to load from metadata file: {e}")
            raise
    
    def _infer_metadata_from_files(self) -> None:
        """Infer metadata by scanning dataset files."""
        files = self._get_parquet_files()
        if not files:
            logger.info("No files found in dataset")
            return
            
        # Collect metadata from first few files
        sample_size = min(10, len(files))
        sample_files = files[:sample_size]
        
        schemas = []
        for file_path in sample_files:
            full_path = posixpath.join(self.path, file_path)
            try:
                metadata = pq.read_metadata(full_path, filesystem=self.filesystem)
                schemas.append(metadata.schema.to_arrow_schema())
                
                # Store file metadata
                self._file_metadata[file_path] = FileMetadata.from_parquet_metadata(
                    file_path, metadata
                )
                
                # Infer partitioning from path
                if "=" in file_path:
                    partitions = get_partitions_from_path(file_path)
                    if partitions and not self._partition_info.names:
                        self._partition_info.names = list(partitions.keys())
                        
            except Exception as e:
                logger.warning(f"Failed to read metadata from {file_path}: {e}")
                
        # Unify schemas
        if schemas:
            self._schema = self._unify_schemas(schemas)
    
    def _get_parquet_files(self) -> list[str]:
        """Get list of parquet files in the dataset."""
        try:
            pattern = posixpath.join(self.path, "**/*.parquet")
            files = self.filesystem.glob(pattern)
            return [f.replace(self.path, "").lstrip("/") for f in sorted(files)]
        except Exception as e:
            logger.warning(f"Failed to list files: {e}")
            return []
    
    def _unify_schemas(self, schemas: list[pa.Schema]) -> pa.Schema:
        """Unify multiple schemas into a single schema."""
        if not schemas:
            return pa.schema([])
        if len(schemas) == 1:
            return schemas[0]
            
        # Find common fields and types
        field_map = defaultdict(list)
        for schema in schemas:
            for field in schema:
                field_map[field.name].append(field.type)
        
        # Build unified schema
        unified_fields = []
        for name, types in field_map.items():
            # Use the most common type or the first one
            type_counts = defaultdict(int)
            for t in types:
                type_counts[str(t)] += 1
            most_common_type = types[0]  # Default to first
            
            # Find most frequent type
            max_count = 0
            for t in types:
                if type_counts[str(t)] > max_count:
                    max_count = type_counts[str(t)]
                    most_common_type = t
                    
            unified_fields.append(pa.field(name, most_common_type))
            
        return pa.schema(unified_fields)
    
    def update_file_metadata(
        self,
        file_path: str,
        metadata: FileMetadata | pq.FileMetaData
    ) -> None:
        """
        Update metadata for a specific file.
        
        Args:
            file_path: Path to the file (relative to dataset root)
            metadata: File metadata to store
        """
        if isinstance(metadata, pq.FileMetaData):
            metadata = FileMetadata.from_parquet_metadata(file_path, metadata)
        
        self._file_metadata[file_path] = metadata
        
        # Update partition values if needed
        if "=" in file_path:
            partitions = get_partitions_from_path(file_path)
            if partitions:
                metadata.partition_values = partitions
                
                # Update partition info
                if not self._partition_info.names:
                    self._partition_info.names = list(partitions.keys())
    
    def collect_all_file_metadata(
        self,
        n_jobs: int = -1,
        verbose: bool = False
    ) -> None:
        """
        Collect metadata for all files in the dataset.
        
        Args:
            n_jobs: Number of parallel jobs
            verbose: Show progress
        """
        files = self._get_parquet_files()
        if not files:
            return
            
        def get_metadata(file_path):
            full_path = posixpath.join(self.path, file_path)
            try:
                metadata = pq.read_metadata(full_path, filesystem=self.filesystem)
                return {file_path: FileMetadata.from_parquet_metadata(file_path, metadata)}
            except Exception as e:
                logger.warning(f"Failed to read metadata from {file_path}: {e}")
                return {}
        
        # Collect metadata in parallel
        metadata_list = run_parallel(
            get_metadata,
            files,
            backend="threading",
            n_jobs=n_jobs,
            verbose=verbose,
        )
        
        # Merge results
        self._file_metadata.clear()
        for meta_dict in metadata_list:
            self._file_metadata.update(meta_dict)
            
        # Update schema from collected metadata
        if self._file_metadata:
            schemas = [m.schema for m in self._file_metadata.values() if m.schema]
            if schemas:
                self._schema = self._unify_schemas(schemas)
    
    def write_metadata_file(self) -> None:
        """Write the _metadata file for the dataset."""
        if not self._schema:
            logger.warning("No schema available to write metadata file")
            return
            
        try:
            # Create metadata with partitioning info
            metadata_dict = {}
            if self._partition_info.names:
                metadata_dict[b'partitioning'] = json.dumps({
                    'names': self._partition_info.names,
                    'flavor': self._partition_info.flavor,
                }).encode()
            
            # Write common metadata
            pq.write_metadata(
                self._schema,
                self._common_metadata_file,
                filesystem=self.filesystem,
                metadata=metadata_dict,
            )
            
            # Write full metadata if we have file metadata
            if self._file_metadata:
                # Build complete metadata from file metadata
                # This is a simplified version - full implementation would merge row groups
                pq.write_metadata(
                    self._schema,
                    self._metadata_file,
                    filesystem=self.filesystem,
                    metadata=metadata_dict,
                )
                
            logger.info(f"Wrote metadata files to {self.path}")
            
        except Exception as e:
            logger.error(f"Failed to write metadata file: {e}")
            raise
    
    def create_metadata_table(self, table_name: str = None) -> None:
        """
        Create a DuckDB table with file metadata.
        
        Args:
            table_name: Name for the metadata table
        """
        if not self.ddb_con:
            logger.warning("No DuckDB connection available")
            return
            
        if not self._file_metadata:
            self.collect_all_file_metadata()
            
        if not self._file_metadata:
            logger.warning("No file metadata available")
            return
            
        # Create table name
        if table_name is None:
            table_name = f"{posixpath.basename(self.path)}_metadata"
        self._metadata_table_name = table_name
        
        # Build metadata records
        records = []
        for file_path, metadata in self._file_metadata.items():
            record = {
                'file_path': file_path,
                'num_rows': metadata.num_rows,
                'num_columns': metadata.num_columns,
                'size_bytes': metadata.size_bytes,
                'row_groups': metadata.row_groups,
            }
            
            # Add partition values
            if metadata.partition_values:
                record.update(metadata.partition_values)
                
            records.append(record)
        
        # Create table in DuckDB
        if records:
            import polars as pl
            df = pl.DataFrame(records)
            self.ddb_con.register(table_name, df.to_arrow())
            logger.info(f"Created metadata table '{table_name}' with {len(records)} files")
    
    def query_metadata(self, filter_expr: str = None) -> list[FileMetadata]:
        """
        Query file metadata with optional filtering.
        
        Args:
            filter_expr: SQL filter expression
            
        Returns:
            List of matching FileMetadata objects
        """
        if not self._file_metadata:
            return []
            
        results = []
        for file_path, metadata in self._file_metadata.items():
            # Simple filter implementation - could be enhanced
            if filter_expr:
                # This is a simplified version - full implementation would parse SQL
                if "=" in filter_expr:
                    # Extract simple equality filters
                    parts = filter_expr.split("=")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip().strip("'\"")
                        
                        if metadata.partition_values and key in metadata.partition_values:
                            if str(metadata.partition_values[key]) != value:
                                continue
                                
            results.append(metadata)
            
        return results
    
    def get_partitions_table(self) -> pa.Table | None:
        """
        Get a PyArrow table with all unique partition combinations.
        
        Returns:
            PyArrow table with partition values
        """
        if not self._partition_info.names:
            return None
            
        # Collect unique partition combinations
        partition_sets = set()
        for metadata in self._file_metadata.values():
            if metadata.partition_values:
                values = tuple(metadata.partition_values.get(name) for name in self._partition_info.names)
                partition_sets.add(values)
        
        if not partition_sets:
            return None
            
        # Build table
        import polars as pl
        df = pl.DataFrame(
            list(partition_sets),
            schema=self._partition_info.names,
            orient="row",
        )
        
        return df.to_arrow()
    
    def delete_metadata_files(self) -> None:
        """Delete metadata files from the dataset."""
        try:
            if self.filesystem.exists(self._metadata_file):
                self.filesystem.rm(self._metadata_file)
            if self.filesystem.exists(self._common_metadata_file):
                self.filesystem.rm(self._common_metadata_file)
            logger.info("Deleted metadata files")
        except Exception as e:
            logger.error(f"Failed to delete metadata files: {e}")
    
    def update_schema(
        self,
        new_schema: pa.Schema = None,
        alter: bool = False
    ) -> None:
        """
        Update the dataset schema.
        
        Args:
            new_schema: New schema to use
            alter: Whether to alter existing schema or replace
        """
        if new_schema is None:
            # Re-infer from files
            self._infer_metadata_from_files()
        elif alter and self._schema:
            # Merge schemas
            self._schema = self._unify_schemas([self._schema, new_schema])
        else:
            # Replace schema
            self._schema = new_schema
            
        # Apply transformations
        if self._schema:
            self._schema = convert_large_types_to_normal(self._schema)
            
    def get_timestamp_info(self) -> dict[str, t.Any]:
        """
        Get information about timestamp columns.
        
        Returns:
            Dictionary with timestamp column info
        """
        if not self._schema:
            return {}
            
        timestamp_info = {}
        for field in self._schema:
            if pa.types.is_timestamp(field.type):
                timestamp_info[field.name] = {
                    'unit': field.type.unit,
                    'tz': field.type.tz,
                }
                
        return timestamp_info
    
    def set_timestamp_column(self, column_name: str) -> None:
        """
        Set the primary timestamp column for the dataset.
        
        Args:
            column_name: Name of the timestamp column
        """
        if self._schema and column_name in self._schema.names:
            self._timestamp_column = column_name
        else:
            raise ValueError(f"Column {column_name} not found in schema")
    
    def clear_cache(self) -> None:
        """Clear any cached metadata."""
        if not self.config.cache_metadata:
            self._schema = None
            self._file_metadata.clear()
            logger.debug("Cleared metadata cache")
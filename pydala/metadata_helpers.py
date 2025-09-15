import json
import pickle
import posixpath
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from loguru import logger

from .helpers.misc import get_partitions_from_path


class MetadataStorage:
    """Handles storage operations for metadata files."""

    def __init__(self, filesystem: AbstractFileSystem, base_path: str):
        self.filesystem = filesystem
        self.base_path = base_path
        self.metadata_file = posixpath.join(base_path, "_metadata")
        self.file_metadata_file = posixpath.join(base_path, "_file_metadata")

    def read_metadata(self) -> Optional[pq.FileMetaData]:
        """Read main metadata file if it exists."""
        if self.filesystem.exists(self.metadata_file):
            return pq.read_metadata(self.metadata_file, filesystem=self.filesystem)
        return None

    def read_file_metadata(self) -> Optional[dict[str, pq.FileMetaData]]:
        """Read file metadata with JSON/pickle fallback."""
        if not self.filesystem.exists(self.file_metadata_file):
            return None

        # Try JSON format first
        try:
            with self.filesystem.open(self.file_metadata_file, "r") as f:
                data = json.load(f)
                return deserialize_metadata(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle for backward compatibility
            with self.filesystem.open(self.file_metadata_file, "rb") as f:
                logger.warning(
                    f"Using deprecated pickle format for {self.file_metadata_file}. "
                    "Please consider migrating to JSON format."
                )
                return pickle.load(f)

    def write_metadata(self, metadata: pq.FileMetaData) -> None:
        """Write main metadata file."""
        with self.filesystem.open(self.metadata_file, "wb") as f:
            metadata.write_metadata_file(f)

    def write_file_metadata(self, file_metadata: dict[str, pq.FileMetaData]) -> None:
        """Write file metadata in JSON format."""
        with self.filesystem.open(self.file_metadata_file, "w") as f:
            json.dump(serialize_metadata(file_metadata), f, indent=2)

    def exists_metadata(self) -> bool:
        """Check if metadata file exists."""
        return self.filesystem.exists(self.metadata_file)

    def exists_file_metadata(self) -> bool:
        """Check if file metadata exists."""
        return self.filesystem.exists(self.file_metadata_file)

    def delete_metadata_files(self) -> None:
        """Delete both metadata files."""
        if self.exists_metadata():
            self.filesystem.rm(self.metadata_file)
        if self.exists_file_metadata():
            self.filesystem.rm(self.file_metadata_file)


def serialize_metadata(metadata: dict[str, pq.FileMetaData]) -> dict[str, Any]:
    """Safely serialize metadata to JSON-compatible format."""
    result = {}
    for path, meta in metadata.items():
        result[path] = {
            "serialized_metadata": meta.serialize().to_pybytes(),
            "num_rows": meta.num_rows,
            "num_columns": len(meta.schema),
            "created_by": meta.created_by,
            "format_version": meta.format_version,
        }
    return result


def deserialize_metadata(data: dict[str, Any]) -> dict[str, pq.FileMetaData]:
    """Safely deserialize metadata from JSON-compatible format."""
    result = {}
    for path, meta_data in data.items():
        buf = pa.py_buffer(meta_data["serialized_metadata"])
        result[path] = pq.read_metadata(buf)
    return result


class MetadataValidator:
    """Validation utilities for metadata operations."""

    @staticmethod
    def validate_files_for_update(files_in_metadata: set[str],
                                files_in_file_metadata: set[str]) -> tuple[list[str], list[str]]:
        """Identify files to add and remove based on metadata comparison."""
        files_to_remove = sorted(files_in_metadata - files_in_file_metadata)
        files_to_add = sorted(files_in_file_metadata - files_in_metadata)
        return files_to_remove, files_to_add

    @staticmethod
    def get_files_needing_repair(file_metadata: dict[str, pq.FileMetaData],
                               target_schema: pa.Schema,
                               target_format_version: str = None) -> list[str]:
        """Identify files that need schema repair."""
        files_to_repair = []

        for file_path, metadata in file_metadata.items():
            file_schema = metadata.schema.to_arrow_schema()
            if file_schema != target_schema:
                files_to_repair.append(file_path)

            if (target_format_version and
                metadata.format_version != target_format_version):
                files_to_repair.append(file_path)

        return sorted(set(files_to_repair))


class SchemaManager:
    """Manages schema operations and compatibility."""

    def __init__(self, metadata_storage: MetadataStorage):
        self.storage = metadata_storage

    def get_unified_schema(self,
                          file_metadata: dict[str, pq.FileMetaData],
                          existing_metadata: Optional[pq.FileMetaData] = None,
                          verbose: bool = False) -> tuple[pa.Schema, bool]:
        """Get unified schema from file metadata."""
        from .schema import convert_large_types_to_normal, unify_schemas_pl

        if not file_metadata:
            if existing_metadata:
                schema = existing_metadata.schema.to_arrow_schema()
                return convert_large_types_to_normal(schema), True
            return pa.schema([]), True

        # Collect all file schemas
        schemas = [
            convert_large_types_to_normal(meta.schema.to_arrow_schema())
            for meta in file_metadata.values()
        ]

        # Try to unify schemas
        try:
            unified_schema = pa.unify_schemas(schemas, promote_options="permissive")
            unified_schema = convert_large_types_to_normal(unified_schema)
        except pa.lib.ArrowTypeError:
            # Fallback to custom unification
            unified_schema = convert_large_types_to_normal(unify_schemas_pl(schemas))

        # Check if all schemas are equal
        schemas_equal = all(schema == schemas[0] for schema in schemas)

        if verbose:
            logger.info(f"Schema is equal: {schemas_equal}")

        return unified_schema, schemas_equal


class FileMetadataUpdater:
    """Handles updates to file metadata."""

    def __init__(self, storage: MetadataStorage, filesystem: AbstractFileSystem):
        self.storage = storage
        self.filesystem = filesystem
        self.base_path = storage.base_path

    def identify_changes(self,
                        current_files: list[str],
                        existing_metadata: Optional[dict[str, pq.FileMetaData]] = None) -> tuple[list[str], list[str]]:
        """Identify files to add and remove for metadata update."""
        if not existing_metadata:
            return current_files, []

        existing_files = set(existing_metadata.keys())
        current_file_set = set(current_files)

        new_files = sorted(current_file_set - existing_files)
        removed_files = sorted(existing_files - current_file_set)

        return new_files, removed_files

    def update_file_metadata(self,
                           file_metadata: dict[str, pq.FileMetaData],
                           new_files: list[str],
                           removed_files: list[str]) -> dict[str, pq.FileMetaData]:
        """Update file metadata with new and removed files."""
        # Add metadata for new files
        from .metadata import collect_parquet_metadata

        if new_files:
            new_metadata = collect_parquet_metadata(
                files=new_files,
                base_path=self.base_path,
                filesystem=self.filesystem,
                verbose=False
            )

            # Set file paths in metadata
            for file_path, meta in new_metadata.items():
                meta.set_file_path(file_path)

            file_metadata.update(new_metadata)

        # Remove metadata for deleted files
        for file_path in removed_files:
            file_metadata.pop(file_path, None)

        return file_metadata
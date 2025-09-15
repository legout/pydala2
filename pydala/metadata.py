import concurrent.futures
import copy
import json
import pickle  # Only for backward compatibility
import posixpath
import re
import tempfile
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Protocol, Optional

import duckdb
import pyarrow as pa
import pyarrow.fs as pfs
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from loguru import logger

from .filesystem import FileSystem, clear_cache
from .helpers.misc import get_partitions_from_path, run_parallel, unify_schemas_pl
from .schema import (
    convert_large_types_to_normal,
    repair_schema,
)


def serialize_metadata(metadata: dict[str, pq.FileMetaData]) -> dict[str, Any]:
    """Safely serialize metadata to JSON-compatible format."""
    result = {}
    for path, meta in metadata.items():
        # Convert FileMetaData to serializable format
        result[path] = {
            "serialized_metadata": meta.serialize(),
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
        # Reconstruct FileMetaData from serialized data
        buf = pa.py_buffer(meta_data["serialized_metadata"])
        result[path] = pq.read_metadata(buf)
    return result


def collect_parquet_metadata(
    files: list[str] | str,
    base_path: str | None = None,
    filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
    n_jobs: int = -1,
    backend: str = "threading",
    verbose: bool = True,
) -> dict[str, pq.FileMetaData]:
    """Collect all metadata information of the given parquet files.

    Args:
        files (list[str] | str): Parquet files.
        filesystem (AbstractFileSystem | pfs.FileSystem | None, optional): Filesystem. Defaults to None.
        n_jobs (int, optional): n_jobs parameter of joblib.Parallel. Defaults to -1.
        backend (str, optional): backend parameter of joblib.Parallel. Defaults to "threading".
        verbose (bool, optional): Whether to show the task progress using tqdm or not. Defaults to True.

    Returns:
        dict[str, pq.FileMetaData]: Parquet metadata of the given files.
    """

    def get_metadata(f, base_path, filesystem):
        if base_path is not None:
            path = posixpath.join(base_path, f)
        else:
            path = f
        return {f: pq.read_metadata(path, filesystem=filesystem)}

    metadata = run_parallel(
        get_metadata,
        files,
        base_path=base_path,
        filesystem=filesystem,
        backend=backend,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return {key: value for d in metadata for key, value in d.items()}


def remove_from_metadata(
    metadata: pq.FileMetaData,
    rm_files: list[str] | None = None,
    keep_files: list[str] | None = None,
    base_path: str = None,
) -> pq.FileMetaData:
    """
    Removes row groups from the metadata of the dataset.
    This method removes row groups from the given metadata based on the given files.
    Files in `rm_files` will be removed from the metadata. Files in `keep_files` will be kept in the metadata.


    Args:
        metadata (pq.FileMetaData): The metadata of the dataset.
        rm_files (list[str]): The files to delete from the metadata.
        keep_files (list[str]): The files to keep in the metadata.
    Returns:
        pq.FileMetaData: The updated metadata of the dataset.
    """
    row_groups = []
    if rm_files is not None:
        if base_path is not None:
            rm_files = [f.replace(base_path, "").lstrip("/") for f in rm_files]

        # row_groups to keep
        row_groups += [
            metadata.row_group(i)
            for i in range(metadata.num_row_groups)
            if metadata.row_group(i).column(0).file_path not in rm_files
        ]
    if keep_files is not None:
        if base_path is not None:
            keep_files = [f.replace(base_path, "").lstrip("/") for f in keep_files]

        # row_groups to keep
        row_groups += [
            metadata.row_group(i)
            for i in range(metadata.num_row_groups)
            if metadata.row_group(i).column(0).file_path in keep_files
        ]

    if len(row_groups):
        new_metadata = row_groups[0]
        for rg in row_groups[1:]:
            new_metadata.append_row_groups(rg)

        return new_metadata

    return metadata


def get_file_paths(
    metadata: pq.FileMetaData,
) -> list[str]:
    return sorted(
        set(
            [
                metadata.row_group(i).column(0).file_path.lstrip("../")
                for i in range(metadata.num_row_groups)
            ]
        )
    )


class MetadataProcessor(ABC):
    """Abstract base class for metadata processing operations."""

    @abstractmethod
    def process(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Process metadata."""
        pass


class FileMetadataProcessor(MetadataProcessor):
    """Handles file metadata collection and storage."""

    def __init__(self, filesystem: FileSystem, base_path: str):
        self.filesystem = filesystem
        self.base_path = base_path

    def collect_metadata(self, files: list[str], **kwargs) -> dict[str, pq.FileMetaData]:
        """Collect metadata for specified files."""
        file_metadata = collect_parquet_metadata(
            files=files,
            base_path=self.base_path,
            filesystem=self.filesystem,
            **kwargs,
        )

        # Set file paths in metadata
        for f in file_metadata:
            file_metadata[f].set_file_path(f)

        return file_metadata

    def save_metadata(self, metadata: dict[str, pq.FileMetaData], file_path: str) -> None:
        """Save metadata to file in JSON format."""
        with self.filesystem.open(file_path, "w") as f:
            json.dump(serialize_metadata(metadata), f, indent=2)

    def load_metadata(self, file_path: str) -> dict[str, pq.FileMetaData] | None:
        """Load metadata from file, supporting both JSON and legacy pickle format."""
        if not self.filesystem.exists(file_path):
            return None

        # Try JSON format first
        try:
            with self.filesystem.open(file_path, "r") as f:
                data = json.load(f)
                return deserialize_metadata(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to binary pickle format for backward compatibility
            with self.filesystem.open(file_path, "rb") as f:
                logger.warning(
                    f"Using deprecated pickle format for {file_path}. "
                    "Please consider migrating to JSON format."
                )
                # Security note: pickle is insecure, but maintained for backward compatibility
                return pickle.load(f)

    def process(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Process metadata - no transformation needed for this processor."""
        return metadata


class SchemaManager:
    """Manages schema operations including unification and repair."""

    def __init__(self, filesystem: FileSystem, base_path: str):
        self.filesystem = filesystem
        self.base_path = base_path

    def unify_schemas(self, schemas: list[pa.Schema]) -> pa.Schema:
        """Unify multiple schemas into one."""
        try:
            return convert_large_types_to_normal(
                pa.unify_schemas(schemas, promote_options="permissive")
            )
        except pa.lib.ArrowTypeError:
            return convert_large_types_to_normal(unify_schemas_pl(schemas))

    def get_unified_schema(
        self,
        existing_schemas: dict[str, pa.Schema],
        metadata_schema: pa.Schema | None = None,
        new_files: list[str] = None,
        verbose: bool = False,
    ) -> tuple[pa.Schema, bool]:
        """Get unified schema from existing schemas and new files."""
        if not new_files:
            if metadata_schema is not None:
                return convert_large_types_to_normal(metadata_schema), True
            return pa.schema([]), True

        schemas = [existing_schemas[f] for f in new_files]
        if metadata_schema is not None:
            schemas.insert(0, metadata_schema)

        unified_schema = self.unify_schemas(schemas)
        schemas_equal = all(unified_schema == schema for schema in schemas)

        if verbose:
            logger.info(f"Schema is equal: {schemas_equal}")

        return unified_schema, schemas_equal

    def repair_file_schemas(
        self,
        files: list[str],
        file_schemas: dict[str, pa.Schema],
        target_schema: pa.Schema,
        format_version: str | None = None,
        ts_unit: str | None = None,
        tz: str | None = None,
        alter_schema: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Repair schemas for specified files."""
        if verbose:
            logger.info(f"Repairing schema for number of files: {len(files)}")

        repair_schema(
            files=files,
            file_schemas={f: file_schemas[f] for f in files},
            schema=target_schema,
            base_path=self.base_path,
            filesystem=self.filesystem,
            version=format_version,
            ts_unit=ts_unit,
            tz=tz,
            alter_schema=alter_schema,
            **kwargs,
        )


class FileTracker:
    """Tracks file changes and manages file operations."""

    def __init__(self, filesystem: FileSystem, base_path: str):
        self.filesystem = filesystem
        self.base_path = base_path

    def list_parquet_files(self) -> list[str]:
        """List all parquet files in the dataset."""
        return [
            fn.replace(self.base_path, "").lstrip("/")
            for fn in sorted(
                self.filesystem.glob(posixpath.join(self.base_path, "**/*.parquet"))
            )
        ]

    def get_file_changes(
        self,
        current_files: list[str],
        existing_files: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Determine which files are new or removed."""
        if not existing_files:
            return current_files, []

        new_files = sorted(set(current_files) - set(existing_files))
        removed_files = sorted(set(existing_files) - set(current_files))

        return new_files, removed_files


class MetadataUpdater:
    """Coordinates metadata updates across different sources."""

    def __init__(
        self,
        file_processor: FileMetadataProcessor,
        schema_manager: SchemaManager,
        file_tracker: FileTracker,
    ):
        self.file_processor = file_processor
        self.schema_manager = schema_manager
        self.file_tracker = file_tracker

    def update_file_metadata(
        self,
        existing_metadata: dict[str, pq.FileMetaData] | None,
        files: list[str] | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> dict[str, pq.FileMetaData]:
        """Update file metadata with new or specified files."""
        if files is None:
            files = self.file_tracker.list_parquet_files()

        new_files = files
        if existing_metadata:
            existing_file_list = list(existing_metadata.keys())
            new_files, removed_files = self.file_tracker.get_file_changes(files, existing_file_list)

            if removed_files and verbose:
                logger.info(f"Removing metadata for {len(removed_files)} files")
                for f in removed_files:
                    existing_metadata.pop(f, None)

            if not new_files:
                if verbose:
                    logger.info("No new files to process")
                return existing_metadata or {}

        if new_files and verbose:
            logger.info(f"Collecting metadata for {len(new_files)} files")

        new_metadata = self.file_processor.collect_metadata(new_files, **kwargs)

        if existing_metadata:
            existing_metadata.update(new_metadata)
            return existing_metadata
        return new_metadata

    def rebuild_metadata_from_files(
        self,
        file_metadata: dict[str, pq.FileMetaData],
        file_list: list[str],
    ) -> pq.FileMetaData:
        """Rebuild metadata from individual file metadata."""
        if not file_list:
            raise ValueError("Cannot rebuild metadata from empty file list")

        metadata = copy.copy(file_metadata[file_list[0]])
        for f in file_list[1:]:
            metadata.append_row_groups(file_metadata[f])

        return metadata

    def update_metadata(
        self,
        existing_metadata: pq.FileMetaData | None,
        file_metadata: dict[str, pq.FileMetaData],
        force_rebuild: bool = False,
        verbose: bool = False,
    ) -> pq.FileMetaData | None:
        """Update combined metadata from file metadata."""
        if not file_metadata:
            return existing_metadata

        file_list = sorted(file_metadata.keys())

        if existing_metadata and not force_rebuild:
            metadata_files = get_file_paths(existing_metadata)
            new_files = sorted(set(file_list) - set(metadata_files))
            removed_files = sorted(set(metadata_files) - set(file_list))

            if verbose:
                logger.info(f"Files to add: {len(new_files)}, Files to remove: {len(removed_files)}")

            if removed_files or (new_files and not existing_metadata) or force_rebuild:
                # Full rebuild needed
                if verbose:
                    logger.info("Rebuilding metadata from file metadata")
                return self.rebuild_metadata_from_files(file_metadata, file_list)
            elif new_files:
                # Append new files
                if verbose:
                    logger.info("Appending new file metadata")
                for f in new_files:
                    existing_metadata.append_row_groups(file_metadata[f])
                return existing_metadata
            else:
                if verbose:
                    logger.info("No changes to metadata")
                return existing_metadata
        else:
            # Full rebuild
            if verbose:
                logger.info("Rebuilding metadata from file metadata")
            return self.rebuild_metadata_from_files(file_metadata, file_list)


class MetadataWriter:
    """Handles writing metadata to files."""

    def __init__(self, filesystem: FileSystem, base_path: str):
        self.filesystem = filesystem
        self.base_path = base_path
        self.metadata_file = posixpath.join(base_path, "_metadata")

    def write_metadata(self, metadata: pq.FileMetaData) -> None:
        """Write metadata to _metadata file."""
        with self.filesystem.open(self.metadata_file, "wb") as f:
            metadata.write_metadata_file(f)

    def remove_metadata_files(self) -> None:
        """Remove metadata files if they exist."""
        file_metadata_file = posixpath.join(self.base_path, "_file_metadata")

        if self.filesystem.exists(self.metadata_file):
            self.filesystem.rm(self.metadata_file)
        if self.filesystem.exists(file_metadata_file):
            self.filesystem.rm(file_metadata_file)


class ParquetDatasetMetadata:
    def __init__(
        self,
        path: str,
        filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
        bucket: str | None = None,
        cached: bool = False,
        update_metadata: bool = False,
        **fs_kwargs,
    ) -> None:
        """
        A class representing metadata for a Parquet dataset.

        Args:
            path (str): The path to the dataset.
            filesystem (AbstractFileSystem | pfs.FileSystem | None, optional): The filesystem to use. Defaults to None.
            bucket (str | None, optional): The name of the bucket to use. Defaults to None.
            cached (bool, optional): Whether to use a cached filesystem. Defaults to False.
            **cached_options: Additional options to pass to the cached filesystem.

        Returns:
            None
        """
        self._path = path
        self._bucket = bucket
        self._cached = cached
        self._base_filesystem = filesystem

        # Initialize filesystem
        if cached:
            cache_storage = fs_kwargs.pop(
                "cache_storage", tempfile.mkdtemp(prefix="pydala2_")
            )
        else:
            cache_storage = None
        self._filesystem = FileSystem(
            bucket=bucket,
            fs=filesystem,
            cached=cached,
            cache_storage=cache_storage,
            **fs_kwargs,
        )

        self._makedirs()
        self._fs_kwargs = fs_kwargs

        # Initialize components
        self.file_processor = FileMetadataProcessor(self._filesystem, path)
        self.schema_manager = SchemaManager(self._filesystem, path)
        self.file_tracker = FileTracker(self._filesystem, path)
        self.metadata_updater = MetadataUpdater(
            self.file_processor, self.schema_manager, self.file_tracker
        )
        self.metadata_writer = MetadataWriter(self._filesystem, path)

        # Initialize state
        self._metadata = self._read_metadata()
        self._file_metadata = None
        self._files = None

        if update_metadata:
            self.update()

    def _makedirs(self):
        """Create directory if it doesn't exist."""
        if self._filesystem.exists(self._path):
            return
        try:
            self._filesystem.mkdir(self._path)
        except Exception:
            # Fallback method
            self._filesystem.touch(posixpath.join(self._path, "tmp.delete"))
            self._filesystem.rm(posixpath.join(self._path, "tmp.delete"))

    def _read_metadata(self) -> pq.FileMetaData | None:
        """Read metadata from _metadata file."""
        metadata_file = posixpath.join(self._path, "_metadata")
        if self._filesystem.exists(metadata_file):
            return pq.read_metadata(metadata_file, filesystem=self._filesystem)
        return None

    def load_files(self) -> None:
        """Load list of parquet files in the dataset."""
        if self.has_metadata:
            self._files = get_file_paths(self._metadata)
        else:
            self.clear_cache()
            self._files = self.file_tracker.list_parquet_files()

    def update_file_metadata(
        self, files: list[str] | None = None, verbose: bool = False, **kwargs
    ) -> None:
        """Update file metadata for the dataset."""
        new_metadata = self.metadata_updater.update_file_metadata(
            self._file_metadata, files, verbose, **kwargs
        )

        # Check if we need to save
        if new_metadata != self._file_metadata:
            self._file_metadata = new_metadata
            self._dump_file_metadata()

    def _dump_file_metadata(self):
        """Save file metadata to disk."""
        file_metadata_file = posixpath.join(self._path, "_file_metadata")
        self.file_processor.save_metadata(self._file_metadata, file_metadata_file)

    def _get_unified_schema(
        self,
        verbose: bool = False,
    ) -> tuple[pa.Schema, bool]:
        """Get unified schema for the dataset."""
        if not self.has_file_metadata:
            return pa.schema([]), True

        new_files = sorted(
            set(self.files_in_file_metadata) - set(self.files_in_metadata)
        )

        metadata_schema = self.metadata.schema.to_arrow_schema() if self.has_metadata else None

        return self.schema_manager.get_unified_schema(
            self.file_schemas,
            metadata_schema,
            new_files,
            verbose,
        )

    def _repair_file_schemas(
        self,
        schema: pa.Schema | None = None,
        format_version: str | None = None,
        tz: str | None = None,
        ts_unit: str | None = None,
        alter_schema: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        """Repair schemas of files that don't match the target schema."""
        # Get target schema
        if schema is None:
            schema, _ = self._get_unified_schema(verbose=verbose)

        if not self.has_file_metadata:
            return

        # Find files that need repair
        files_to_repair = [
            f for f in self.file_metadata if self.file_schemas[f] != schema
        ]

        # Check format version
        if format_version is None and self.has_metadata:
            format_version = self.metadata.format_version

        if format_version is not None:
            files_with_wrong_version = [
                f for f in self.file_metadata
                if self.file_metadata[f].format_version != format_version
            ]
            files_to_repair = sorted(set(files_to_repair + files_with_wrong_version))

        if files_to_repair:
            self.schema_manager.repair_file_schemas(
                files_to_repair,
                self.file_schemas,
                schema,
                format_version,
                ts_unit,
                tz,
                alter_schema,
                verbose,
                **kwargs,
            )

            # Update metadata after repair
            self.clear_cache()
            self.update_file_metadata(files=files_to_repair, verbose=verbose, **kwargs)

            if any(f in self.files_in_metadata for f in files_to_repair):
                self._update_metadata(reload=True, verbose=verbose)
        else:
            self._update_metadata(reload=False, verbose=verbose)

    def _update_metadata(self, reload: bool = False, verbose: bool = False, **kwargs):
        """Update the combined metadata file."""
        if not self.has_file_metadata:
            self.update_file_metadata(**kwargs)

        if not self.has_file_metadata:
            return

        self._metadata = self.metadata_updater.update_metadata(
            self._metadata,
            self._file_metadata,
            force_rebuild=reload,
            verbose=verbose,
        )

        if self._metadata:
            self.metadata_writer.write_metadata(self._metadata)
            self.load_files()

    def update(
        self,
        reload: bool = False,
        schema: pa.Schema | None = None,
        ts_unit: str | None = None,
        tz: str | None = None,
        format_version: str | None = None,
        alter_schema: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Update the dataset metadata."""
        if reload:
            self.reset()

        # Update file metadata
        self.update_file_metadata(verbose=verbose, **kwargs)

        if not self.has_file_metadata:
            return

        # Repair schemas if needed
        self._repair_file_schemas(
            schema=schema,
            format_version=format_version,
            tz=tz,
            ts_unit=ts_unit,
            alter_schema=alter_schema,
            verbose=verbose,
        )

        # Create metadata file if it doesn't exist
        if not self.has_metadata_file:
            self._update_metadata(**kwargs)

    def reset(self):
        """Reset the dataset by removing metadata files and clearing cache."""
        if self.has_metadata:
            self._metadata = None
            self.metadata_writer.remove_metadata_files()

        # Clear cached properties
        for attr in ['_file_schema', '_schema']:
            if hasattr(self, attr):
                delattr(self, attr)

        if self.has_file_metadata:
            self._file_metadata = None
            file_metadata_file = posixpath.join(self._path, "_file_metadata")
            if self._filesystem.exists(file_metadata_file):
                self._filesystem.rm(file_metadata_file)

        self.clear_cache()

    def clear_cache(self) -> None:
        """Clear filesystem caches."""
        if hasattr(self._filesystem, "fs"):
            clear_cache(self._filesystem.fs)
        clear_cache(self._filesystem)
        clear_cache(self._base_filesystem)

    # Property methods
    @property
    def has_metadata_file(self):
        """Check if _metadata file exists."""
        return self._filesystem.exists(posixpath.join(self._path, "_metadata"))

    @property
    def has_file_metadata_file(self):
        """Check if _file_metadata file exists."""
        return self._filesystem.exists(posixpath.join(self._path, "_file_metadata"))

    @property
    def has_file_metadata(self):
        """Check if file metadata is loaded or can be loaded."""
        if self._file_metadata is None:
            if self.has_file_metadata_file:
                self._file_metadata = self.file_processor.load_metadata(
                    posixpath.join(self._path, "_file_metadata")
                )
        return self._file_metadata is not None

    @property
    def file_metadata(self):
        """Get file metadata, loading if necessary."""
        if not self.has_file_metadata:
            self.update_file_metadata()
        return self._file_metadata

    @property
    def file_schemas(self):
        """Get schemas for all files."""
        return {
            f: convert_large_types_to_normal(
                self._file_metadata[f].schema.to_arrow_schema()
            )
            for f in self._file_metadata
        }

    @property
    def files_in_file_metadata(self) -> list:
        """Get list of files in file metadata."""
        if self.has_file_metadata:
            return sorted(set(self._file_metadata.keys()))
        return []

    @property
    def has_metadata(self):
        """Check if metadata is loaded or can be loaded."""
        if self._metadata is None:
            self._metadata = self._read_metadata()
        return self._metadata is not None

    @property
    def metadata(self):
        """Get metadata, updating if necessary."""
        if not self.has_metadata:
            self._update_metadata()
        return self._metadata

    @property
    def files_in_metadata(self) -> list:
        """Get list of files in combined metadata."""
        if self.has_metadata:
            return get_file_paths(self.metadata)
        return []

    @property
    def file_schema(self):
        """Get the schema of the dataset."""
        if not hasattr(self, "_file_schema"):
            if self.has_metadata:
                self._file_schema = self.metadata.schema.to_arrow_schema()
            else:
                self._file_schema = pa.schema([])
        return self._file_schema

    @property
    def has_files(self):
        """Check if dataset has any files."""
        return len(self.files) > 0

    @property
    def files(self) -> list:
        """Get list of files in dataset."""
        if self._files is None:
            self.load_files()
        return self._files

    # Legacy method aliases
    def replace_schema(self, schema: pa.Schema, **kwargs) -> None:
        """Replace dataset schema."""
        self.update(schema=schema, **kwargs)

    def load_metadata(self, *args, **kwargs):
        """Alias for update method."""
        self.update(*args, **kwargs)

    def delete_metadata_files(self) -> None:
        """Delete metadata files."""
        self.metadata_writer.remove_metadata_files()


class MetadataTableGenerator:
    """Generates metadata tables for Pydala datasets."""

    @staticmethod
    def process_row_group(
        metadata: pq.FileMetaData,
        rg_num: int,
        partitioning: None | str | list[str] = None,
    ):
        """Process a single row group and extract metadata."""
        row_group = metadata.row_group(rg_num)
        file_path = row_group.column(0).file_path

        result = {
            "file_path": file_path,
            "num_columns": row_group.num_columns,
            "num_rows": row_group.num_rows,
            "total_byte_size": row_group.total_byte_size,
            "compression": row_group.column(0).compression,
        }

        # Extract partition information
        if "=" in file_path:
            partitioning_ = partitioning or "hive"
        else:
            partitioning_ = partitioning

        if partitioning_ is not None:
            partitions = dict(
                get_partitions_from_path(file_path, partitioning=partitioning_)
            )
            result.update(partitions)

        # Process column metadata
        for col_num in range(row_group.num_columns):
            rgc = row_group.column(col_num)
            rgc_dict = rgc.to_dict()
            col_name = rgc_dict.pop("path_in_schema")
            rgc_dict.pop("file_path")
            rgc_dict.pop("compression")

            # Handle statistics
            if "statistics" in rgc_dict:
                if rgc_dict["statistics"] is not None:
                    rgc_dict.update(rgc_dict.pop("statistics"))
                else:
                    rgc_dict.pop("statistics")
                    rgc_dict.update({
                        "has_min_max": False,
                        "min": None,
                        "max": None,
                        "null_count": None,
                        "distinct_count": None,
                        "num_values": None,
                        "physical_type": "UNKNOWN",
                    })

            return col_name, rgc_dict

        return result

    @staticmethod
    def generate_metadata_table(
        metadata: pq.FileMetaData | list[pq.FileMetaData],
        partitioning: None | str | list[str] = None,
    ):
        """Generate a metadata table from file metadata."""
        if isinstance(metadata, pq.FileMetaData):
            metadata = [metadata]

        metadata_table = defaultdict(list)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for metadata_ in metadata:
                for rg_num in range(metadata_.num_row_groups):
                    futures.append(
                        executor.submit(
                            MetadataTableGenerator.process_row_group,
                            metadata_,
                            rg_num,
                            partitioning,
                        )
                    )

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if isinstance(result, tuple):
                    # Column metadata
                    col_name, col_data = result
                    metadata_table[col_name].append(col_data)
                else:
                    # Row group metadata
                    for key, value in result.items():
                        metadata_table[key].append(value)

        return metadata_table


class FilterExpressionProcessor:
    """Processes filter expressions for metadata table filtering."""

    @staticmethod
    def detect_date_type(filter_expr: str) -> tuple[bool, bool]:
        """Detect if filter expression contains date or timestamp."""
        res = re.findall(
            r"[<>=!]\'[1,2]{1}\d{3}-\d{1,2}-\d{1,2}(?:[\s,T]\d{2}:\d{2}:{0,2}\d{0,2})?\'",
            filter_expr,
        )
        if len(res):
            is_date = len(res[0]) <= 13
            is_timestamp = len(res[0]) > 13
            return is_date, is_timestamp
        return False, False

    @staticmethod
    def transform_filter_expression(
        filter_expr: str,
        is_date: bool = False,
        is_timestamp: bool = False,
    ) -> str:
        """Transform filter expression for metadata table."""
        # Transform operators
        if ">" in filter_expr:
            filter_expr = (
                f"({filter_expr.replace('>', '.max>')} OR "
                f"{filter_expr.split('>')[0]}.max IS NULL)"
            )
        elif "<" in filter_expr:
            filter_expr = (
                f"({filter_expr.replace('<', '.min<')} OR "
                f"{filter_expr.split('<')[0]}.min IS NULL)"
            )
        elif "=" in filter_expr:
            col_name = filter_expr.split('=')[0]
            filter_expr = (
                f"({filter_expr.replace('=', '.min<=')} OR {col_name}.min IS NULL) "
                f"AND ({filter_expr.replace('=', '.max>=')} OR {col_name}.max IS NULL)"
            )

        # Add type casting for dates/timestamps
        if is_date:
            filter_expr = (
                filter_expr.replace(">", "::DATE>")
                .replace("<", "::DATE<")
                .replace(" IS NULL", "::DATE IS NULL")
            )
        elif is_timestamp:
            filter_expr = (
                filter_expr.replace(">", "::TIMESTAMP>")
                .replace("<", "::TIMESTAMP<")
                .replace(" IS NULL", "::TIMESTAMP IS NULL")
            )

        return filter_expr

    @staticmethod
    def generate_filter(filter_expr: str) -> list[str]:
        """Generate modified filter expressions."""
        filter_expr_mod = []
        is_date, is_timestamp = FilterExpressionProcessor.detect_date_type(filter_expr)

        transformed_expr = FilterExpressionProcessor.transform_filter_expression(
            filter_expr, is_date, is_timestamp
        )

        filter_expr_mod.append(transformed_expr)
        return filter_expr_mod


class PydalaDatasetMetadata(ParquetDatasetMetadata):
    def __init__(
        self,
        path: str,
        filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
        bucket: str | None = None,
        cached: bool = False,
        partitioning: None | str | list[str] = None,
        ddb_con: duckdb.DuckDBPyConnection | None = None,
        **fs_kwargs,
    ) -> None:
        """Initialize Pydala dataset metadata with DuckDB integration."""
        super().__init__(
            path=path,
            filesystem=filesystem,
            bucket=bucket,
            cached=cached,
            **fs_kwargs,
        )

        self.reset_scan()
        self._partitioning = partitioning

        # Initialize DuckDB connection
        self.ddb_con = ddb_con or duckdb.connect()

        # Initialize metadata table
        try:
            self.update_metadata_table()
        except Exception as e:
            logger.error(f"Failed to update metadata table: {e}")

    def reset_scan(self):
        """Reset scan state."""
        self._metadata_table_scanned = None
        self._filter_expr = None
        self._filter_expr_mod = None

    def update_metadata_table(
        self,
        backend: str = "threading",
        verbose: bool = True,
    ):
        """Update the metadata table using current metadata."""
        if self.has_metadata:
            metadata_table = MetadataTableGenerator.generate_metadata_table(
                metadata=self.metadata,
                partitioning=self._partitioning,
            )

            # Create DuckDB table
            arrow_table = pa.Table.from_pydict(metadata_table)
            self._metadata_table = self.ddb_con.from_arrow(arrow_table)
            self._metadata_table.create_view("metadata_table")

            if verbose:
                logger.info(f"Metadata table updated with {len(arrow_table)} rows")

    def scan(self, filter_expr: str | None = None):
        """Scan dataset with optional filter expression."""
        self._filter_expr = filter_expr

        if filter_expr is not None:
            # Split AND conditions
            filter_parts = re.split(r"\s+[a,A][n,N][d,D]\s+", filter_expr)

            filter_expr_mod = []

            for fe in filter_parts:
                col = re.split("[>=<]", fe)[0].lstrip("(")

                # Check if column exists in metadata table
                try:
                    col_type = self.metadata_table.select(col).types[0].id

                    if col_type != "struct":
                        # Regular column - use as-is
                        filter_expr_mod.append(fe)
                    else:
                        # Struct column - needs special handling
                        filter_expr_mod.extend(FilterExpressionProcessor.generate_filter(fe))
                except Exception:
                    # Column not found - skip this filter
                    continue

            self._filter_expr_mod = " AND ".join(filter_expr_mod)

            # Apply filter
            if self._filter_expr_mod:
                self._metadata_table_scanned = self._metadata_table.filter(
                    self._filter_expr_mod
                )
            else:
                self._metadata_table_scanned = self._metadata_table

    # Properties
    @property
    def metadata_table(self):
        """Get the metadata table."""
        if not hasattr(self, "_metadata_table"):
            self.update_metadata_table()
        return self._metadata_table

    @property
    def metadata_table_scanned(self):
        """Get the scanned metadata table."""
        return self._metadata_table_scanned

    @property
    def latest_filter_expr(self):
        """Get the latest filter expression."""
        if not hasattr(self, "_filter_expr"):
            self._filter_expr = None
            self._filter_expr_mod = None
        return self._filter_expr, self._filter_expr_mod

    @property
    def scan_files(self):
        """Get files from scan result."""
        if self.metadata_table_scanned is not None:
            return sorted(
                set(
                    row[0]
                    for row in self.metadata_table_scanned.select("file_path").fetchall()
                )
            )
        return self.files

    @property
    def is_scanned(self):
        """Check if scan has been performed."""
        return self._metadata_table_scanned is not None
import concurrent.futures
import copy
import posixpath
import re
import tempfile
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional, Union

import duckdb
import pyarrow as pa
import pyarrow.fs as pfs
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from loguru import logger

from .filesystem import FileSystem, clear_cache
from .helpers.misc import get_partitions_from_path, run_parallel, unify_schemas_pl
from .metadata_helpers import (
    MetadataStorage,
    MetadataValidator,
    SchemaManager,
    FileMetadataUpdater,
)
from .schema import convert_large_types_to_normal, repair_schema


class ParquetDatasetMetadata:
    """Simplified metadata management for Parquet datasets."""

    def __init__(
        self,
        path: str,
        filesystem: Union[AbstractFileSystem, pfs.FileSystem, None] = None,
        bucket: Optional[str] = None,
        cached: bool = False,
        update_metadata: bool = False,
        **fs_kwargs,
    ) -> None:
        """Initialize dataset metadata handler.

        Args:
            path: Path to the dataset
            filesystem: Filesystem to use (defaults to auto-detected)
            bucket: S3 bucket name if using S3
            cached: Whether to use cached filesystem
            update_metadata: Whether to update metadata on initialization
            **fs_kwargs: Additional filesystem arguments
        """
        self._path = path
        self._bucket = bucket
        self._cached = cached
        self._base_filesystem = filesystem
        self._fs_kwargs = fs_kwargs

        # Initialize filesystem
        self._setup_filesystem(cache_storage=fs_kwargs.pop("cache_storage", None))

        # Initialize storage manager
        self.storage = MetadataStorage(self._filesystem, self._path)

        # Load or initialize metadata
        self._metadata = self.storage.read_metadata()
        self._file_metadata = self.storage.read_file_metadata() if self.storage.exists_file_metadata() else {}

        # Initialize helper components
        self.schema_manager = SchemaManager(self.storage)
        self.file_updater = FileMetadataUpdater(self.storage, self._filesystem)
        self.validator = MetadataValidator()

        # Ensure directory exists
        self._ensure_directory_exists()

        # Update metadata if requested
        if update_metadata:
            self.update()

    def _setup_filesystem(self, cache_storage: Optional[str] = None) -> None:
        """Initialize filesystem with appropriate configuration."""
        if cache_storage is None and self._cached:
            cache_storage = tempfile.mkdtemp(prefix="pydala2_")

        self._filesystem = FileSystem(
            bucket=self._bucket,
            fs=self._base_filesystem,
            cached=self._cached,
            cache_storage=cache_storage,
            **self._fs_kwargs,
        )

    def _ensure_directory_exists(self) -> None:
        """Ensure the dataset directory exists."""
        if self._filesystem.exists(self._path):
            return

        try:
            self._filesystem.mkdir(self._path)
        except Exception:
            # Fallback: create and remove a temporary file
            self._filesystem.touch(posixpath.join(self._path, "tmp.delete"))
            self._filesystem.rm(posixpath.join(self._path, "tmp.delete"))

    # Property-based interfaces
    @property
    def has_metadata(self) -> bool:
        """Check if dataset has metadata."""
        return self._metadata is not None

    @property
    def has_metadata_file(self) -> bool:
        """Check if metadata file exists."""
        return self.storage.exists_metadata()

    @property
    def has_file_metadata(self) -> bool:
        """Check if file metadata exists."""
        return bool(self._file_metadata)

    @property
    def metadata(self) -> pq.FileMetaData:
        """Get dataset metadata, loading if necessary."""
        if not self.has_metadata:
            self._rebuild_metadata()
        return self._metadata

    @property
    def file_metadata(self) -> dict[str, pq.FileMetaData]:
        """Get file metadata, ensuring it's up to date."""
        if self._file_metadata is None:
            self._file_metadata = self._load_or_build_file_metadata()
        return self._file_metadata

    @property
    def files(self) -> list[str]:
        """Get list of all parquet files in dataset."""
        if not hasattr(self, "_files"):
            self._files = self._scan_for_parquet_files()
        return self._files

    @property
    def files_in_metadata(self) -> list[str]:
        """Get files referenced in main metadata."""
        if not self.has_metadata:
            return []
        return resolve_file_paths(self.metadata)

    @property
    def files_in_file_metadata(self) -> list[str]:
        """Get files in file metadata cache."""
        if not self.file_metadata:
            return []
        return sorted(self._file_metadata.keys())

    # Core update methods
    def update(
        self,
        reload: bool = False,
        schema: Optional[pa.Schema] = None,
        ts_unit: Optional[str] = None,
        tz: Optional[str] = None,
        format_version: Optional[str] = None,
        alter_schema: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Update dataset metadata with optional schema repair.

        Args:
            reload: Whether to reload metadata from scratch
            schema: Target schema (uses unified if None)
            ts_unit: Timestamp unit for repairs
            tz: Timezone for repairs
            format_version: Target format version
            alter_schema: Whether to alter file schemas
            verbose: Enable verbose logging
            **kwargs: Additional arguments for metadata collection
        """
        if reload:
            self.reset()

        # Update file metadata
        new_metadata = self._update_file_metadata(verbose=verbose, **kwargs)

        if not new_metadata and not self.file_metadata:
            if verbose:
                logger.info("No files found, skipping schema repair")
            return

        # Repair schemas if needed
        self._repair_schemas_if_needed(
            target_schema=schema,
            format_version=format_version,
            ts_unit=ts_unit,
            tz=tz,
            alter_schema=alter_schema,
            verbose=verbose,
        )

        # Build main metadata if needed
        if not self.has_metadata or new_metadata:
            self._rebuild_metadata()

    def _update_file_metadata(
        self,
        verbose: bool = False,
        **kwargs
    ) -> bool:
        """Update file metadata to reflect current state.

        Returns:
            True if any files were added or removed
        """
        # Scan for current files
        current_files = self._scan_for_parquet_files()

        # Identify changes
        new_files, removed_files = self.file_updater.identify_changes(
            current_files, self._file_metadata
        )

        if verbose:
            logger.info(f"Found {len(new_files)} new files, {len(removed_files)} removed files")

        # Update file metadata
        if new_files or removed_files:
            self._file_metadata = self.file_updater.update_file_metadata(
                self._file_metadata or {}, new_files, removed_files
            )
            self.storage.write_file_metadata(self._file_metadata)
            return True

        return False

    def _scan_for_parquet_files(self) -> list[str]:
        """Scan dataset directory for parquet files."""
        self._clear_caches()
        pattern = posixpath.join(self._path, "**/*.parquet")
        return [
            fn.replace(self._path, "").lstrip("/")
            for fn in sorted(self._filesystem.glob(pattern))
        ]

    def _repair_schemas_if_needed(
        self,
        target_schema: Optional[pa.Schema] = None,
        format_version: Optional[str] = None,
        ts_unit: Optional[str] = None,
        tz: Optional[str] = None,
        alter_schema: bool = True,
        verbose: bool = False,
    ) -> None:
        """Repair file schemas if they don't match target."""
        # Get target schema
        if target_schema is None:
            target_schema, _ = self.schema_manager.get_unified_schema(
                self.file_metadata, verbose=verbose
            )

        # Get format version
        format_version = format_version or (
            self._metadata.format_version if self.has_metadata else None
        )

        # Identify files needing repair
        files_to_repair = self.validator.get_files_needing_repair(
            self.file_metadata, target_schema, format_version
        )

        if not files_to_repair:
            if verbose:
                logger.info("No files need schema repair")
            return

        # Perform repair
        if verbose:
            logger.info(f"Repairing {len(files_to_repair)} files")

        file_schemas = {
            f: self.file_metadata[f].schema.to_arrow_schema()
            for f in files_to_repair
        }

        repair_schema(
            files=files_to_repair,
            file_schemas=file_schemas,
            schema=target_schema,
            base_path=self._path,
            filesystem=self._filesystem,
            version=format_version,
            ts_unit=ts_unit,
            tz=tz,
            alter_schema=alter_schema,
        )

        # Update file metadata for repaired files
        self._update_specific_file_metadata(files_to_repair, verbose)

        # Rebuild main metadata if any repaired files were in it
        if any(f in self.files_in_metadata for f in files_to_repair):
            self._rebuild_metadata()

    def _update_specific_file_metadata(
        self, files: list[str], verbose: bool = False
    ) -> None:
        """Update metadata for specific files."""
        from .metadata import collect_parquet_metadata

        new_metadata = collect_parquet_metadata(
            files=files,
            base_path=self._path,
            filesystem=self._filesystem,
            verbose=verbose,
        )

        for path, meta in new_metadata.items():
            meta.set_file_path(path)
            self._file_metadata[path] = meta

        self.storage.write_file_metadata(self._file_metadata)

    def _rebuild_metadata(self, verbose: bool = False) -> None:
        """Rebuild main metadata from file metadata."""
        if not self.file_metadata:
            self._metadata = None
            return

        if verbose:
            logger.info("Rebuilding metadata from file metadata")

        files = sorted(self.file_metadata.keys())
        self._metadata = copy.copy(self.file_metadata[files[0]])

        for file_path in files[1:]:
            self._metadata.append_row_groups(self.file_metadata[file_path])

        self.storage.write_metadata(self._metadata)

    def _clear_caches(self) -> None:
        """Clear all filesystem caches."""
        for fs in [getattr(self._filesystem, 'fs', None), self._filesystem, self._base_filesystem]:
            if fs:
                clear_cache(fs)

    # Public interface
    def reset(self) -> None:
        """Reset metadata state."""
        self._metadata = None
        self._file_metadata = {}
        self.storage.delete_metadata_files()
        self._clear_caches()

        # Clear cached properties
        for attr in ["_files", "_file_schema", "_schema"]:
            if hasattr(self, attr):
                delattr(self, attr)

    def delete_metadata_files(self) -> None:
        """Delete metadata files."""
        self.storage.delete_metadata_files()

    def load_files(self) -> None:
        """Load list of files (for backward compatibility)."""
        self._files = self._scan_for_parquet_files()


def resolve_file_paths(metadata: pq.FileMetaData) -> list[str]:
    """Extract unique file paths from metadata."""
    return sorted(
        set(
            metadata.row_group(i).column(0).file_path.lstrip("../")
            for i in range(metadata.num_row_groups)
        )
    )


class PydalaDatasetMetadata(ParquetDatasetMetadata):
    """Extended metadata management with DuckDB support."""

    def __init__(
        self,
        path: str,
        filesystem: Union[AbstractFileSystem, pfs.FileSystem, None] = None,
        bucket: Optional[str] = None,
        cached: bool = False,
        partitioning: Optional[Union[str, list[str]]] = None,
        ddb_con: Optional[duckdb.DuckDBPyConnection] = None,
        **fs_kwargs,
    ) -> None:
        """Initialize dataset with MongDB-style query support.

        Args:
            path: Dataset path
            filesystem: Filesystem implementation
            bucket: S3 bucket name
            cached: Use caching layer
            partitioning: Partition strategy (hive, etc.)
            ddb_con: DuckDB connection
            **fs_kwargs: Additional filesystem args
        """
        super().__init__(
            path=path,
            filesystem=filesystem,
            bucket=bucket,
            cached=cached,
            **fs_kwargs,
        )

        self._partitioning = partitioning
        self._ddb_con = ddb_con or duckdb.connect()
        self._metadata_table = None
        self._filter_expr = None
        self._filter_expr_mod = None
        self._scanned_files = None

        try:
            self._update_metadata_table()
        except Exception as e:
            logger.warning(f"Failed to initialize metadata table: {e}")

    # Filter expression handlers
    def _handle_date_filter(self, filter_expr: str) -> str:
        """Add date type casting to filter expression."""
        date_pattern = r"[<>=]\s*'\d{4}-\d{1,2}-\d{1,2}'"
        dates = re.findall(date_pattern, filter_expr)

        if dates:
            filter_expr = (
                filter_expr.replace(">", "::DATE>")
                .replace("<", "::DATE<")
                .replace(" IS NULL", "::DATE IS NULL")
            )

        return filter_expr

    def _handle_timestamp_filter(self, filter_expr: str) -> str:
        """Add timestamp type casting to filter expression."""
        ts_pattern = r"[<>=]\s*'\d{4}-\d{1,2}-\d{1,2}[\s,T]\d{2}:\d{2}:\d{2}'"
        timestamps = re.findall(ts_pattern, filter_expr)

        if timestamps:
            filter_expr = (
                filter_expr.replace(">", "::TIMESTAMP>")
                .replace("<", "::TIMESTAMP<")
                .replace(" IS NULL", "::TIMESTAMP IS NULL")
            )

        return filter_expr

    def _rewrite_comparison_operators(self, filter_expr: str) -> str:
        """Rewrite comparison operators for min/max statistics."""
        # Extract column name
        col_match = re.match(r"^[^(]+", filter_expr.strip())
        if not col_match:
            return filter_expr

        col = col_match.group().strip()
        operators = {
            ">": f"({col}.max{filter_expr[1:]} OR {col}.max IS NULL)",
            "<": f"({col}.min{filter_expr[1:]} OR {col}.min IS NULL)",
            "=": (
                f"({col}.min<={filter_expr[2:]} OR {col}.min IS NULL) "
                f"AND ({col}.max>={filter_expr[2:]} OR {col}.max IS NULL)"
            ),
        }

        for op, rewrite in operators.items():
            if op in filter_expr[:2]:  # Check first two characters
                return rewrite

        return filter_expr

    def _generate_filter_conditions(self, filter_expr: str) -> list[list[str]]:
        """Generate filter conditions suitable for DuckDB query.

        Returns list of conditions grouped by AND/OR precedence.
        """
        # Split by AND/OR, preserving operator precedence
        and_groups = re.split(r"\s+(?:AND|and)\s+", filter_expr)
        conditions = []

        for group in and_groups:
            or_conditions = re.split(r"\s+(?:OR|or)\s+", group)
            group_conditions = []

            for condition in or_conditions:
                condition = condition.strip()
                if not condition:
                    continue

                # Rewrite comparison for statistics
                condition = self._rewrite_comparison_operators(condition)

                # Add type casting if needed
                condition = self._handle_date_filter(condition)
                condition = self._handle_timestamp_filter(condition)

                group_conditions.append(condition)

            if group_conditions:
                conditions.append(group_conditions)

        return conditions

    def _build_duckdb_query(self) -> Optional[str]:
        """Build DuckDB query from filter expression."""
        if not hasattr(self, "_filter_conditions") or not self._filter_conditions:
            return None

        query_parts = []
        for group in self._filter_conditions:
            if len(group) > 1:
                # OR conditions within group
                query_parts.append(f"({') OR ('.join(group)})")
            else:
                query_parts.append(group[0])

        return " AND ".join(query_parts) if query_parts else None

    def scan(self, filter_expr: Optional[str] = None) -> None:
        """Scan dataset with optional filtering.

        Args:
            filter_expr: DuckDB-compatible filter expression
        """
        self._filter_expr = filter_expr

        if filter_expr is None:
            self._scanned_files = None
            return

        # Parse and prepare filter
        self._filter_conditions = self._generate_filter_conditions(filter_expr)
        query = self._build_duckdb_query()

        if query and self._metadata_table is not None:
            self._scanned_files = self._metadata_table.filter(query)
        else:
            self._scanned_files = self._metadata_table

    def _update_metadata_table(self, backend: str = "threading", verbose: bool = True) -> None:
        """Update the DuckDB metadata table from dataset metadata."""
        if not self.has_metadata:
            return

        table_data = self._extract_metadata_table_data()
        arrow_table = pa.Table.from_pydict(table_data)

        # Update DuckDB table
        if self._metadata_table is not None:
            self._metadata_table = self._ddb_con.from_arrow(arrow_table)
        else:
            self._metadata_table = self._ddb_con.from_arrow(arrow_table)

        self._metadata_table.create_view("metadata_table", replace=True)

        if verbose:
            logger.info(f"Updated metadata table with {len(table_data.get('file_path', []))} rows")

    def _extract_metadata_table_data(self) -> dict:
        """Extract row group statistics into table format."""
        metadata_table = defaultdict(list)

        def process_row_group(rg_metadata, path_col, stat_col, cache):
            """Process a single row group."""
            row_group = rg_metadata.row_group(rg_num)
            file_path = row_group.column(0).file_path

            # Check cache
            cache_key = (file_path, rg_metadata)
            if cache_key in cache:
                return cache[cache_key]

            result = {
                "file_path": file_path,
                "num_columns": row_group.num_columns,
                "num_rows": row_group.num_rows,
                "total_byte_size": row_group.total_byte_size,
            }

            # Add partition info
            if self._partitioning is not None:
                partitions = dict(get_partitions_from_path(file_path, self._partitioning))
                result.update(partitions)

            # Process column statistics
            for col_num in range(row_group.num_columns):
                col_metadata = row_group.column(col_num).to_dict()
                col_name = col_metadata.pop("path_in_schema")

                # Filter and format statistics
                if "statistics" in col_metadata and col_metadata["statistics"] is not None:
                    stats = col_metadata.pop("statistics")
                    col_metadata.update(stats)
                else:
                    col_metadata.update({
                        "has_min_max": False,
                        "min": None,
                        "max": None,
                        "null_count": None,
                        "distinct_count": None,
                    })

                metadata_table[col_name].append(col_metadata)

            cache[cache_key] = result
            return result

        # Process all row groups
        cache = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            metadata_list = [self.metadata] if self.has_metadata else []

            for metadata in metadata_list:
                for rg_num in range(metadata.num_row_groups):
                    futures.append(
                        executor.submit(process_row_group, metadata, rg_num, cache)
                    )

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                for key, value in result.items():
                    metadata_table[key].append(value)

        return metadata_table

    @property
    def metadata_table(self):
        """Get DuckDB metadata table."""
        return self._metadata_table

    @property
    def scan_files(self) -> list[str]:
        """Get files matching current scan/filter."""
        if self._scanned_files is not None:
            files = self._scanned_files.select("file_path").fetchall()
            return sorted(set(f[0] for f in files))
        return self.files

    @property
    def is_scanned(self) -> bool:
        """Check if dataset has active scan results."""
        return self._scanned_files is not None

    def reset_scan(self) -> None:
        """Clear scan results."""
        self._scanned_files = None
        self._filter_expr = None
"""Migration utilities for transitioning to simplified metadata API."""

from typing import Optional

from .metadata import (
    ParquetDatasetMetadata as LegacyParquetDatasetMetadata,
    PydalaDatasetMetadata as LegacyPydalaDatasetMetadata,
    collect_parquet_metadata as legacy_collect_parquet_metadata,
    remove_from_metadata as legacy_remove_from_metadata,
    get_file_paths as legacy_get_file_paths,
)
from .metadata_simplified import (
    ParquetDatasetMetadata,
    PydalaDatasetMetadata,
    resolve_file_paths,
)
from .metadata_helpers import (
    serialize_metadata,
    deserialize_metadata,
    MetadataStorage,
    MetadataValidator,
    SchemaManager,
    FileMetadataUpdater,
)


def migrate_to_simplified_parquet_metadata(
    path: str,
    filesystem=None,
    bucket: Optional[str] = None,
    cached: bool = False,
    update_metadata: bool = False,
    **fs_kwargs
):
    """Migrator to create simplified ParquetDatasetMetadata from legacy API.

    Args:
        path: Dataset path
        filesystem: Filesystem to use
        bucket: S3 bucket name if using S3
        cached: Whether to use cached filesystem
        update_metadata: Whether to update metadata on initialization
        **fs_kwargs: Additional filesystem arguments

    Returns:
        Simplified ParquetDatasetMetadata instance
    """
    return ParquetDatasetMetadata(
        path=path,
        filesystem=filesystem,
        bucket=bucket,
        cached=cached,
        update_metadata=update_metadata,
        **fs_kwargs
    )


def migrate_to_simplified_pydala_metadata(
    path: str,
    filesystem=None,
    bucket: Optional[str] = None,
    cached: bool = False,
    partitioning=None,
    ddb_con=None,
    **fs_kwargs
):
    """Migrator to create simplified PydalaDatasetMetadata from legacy API.

    Args:
        path: Dataset path
        filesystem: Filesystem to use
        bucket: S3 bucket name if using S3
        cached: Whether to use cached filesystem
        partitioning: Partition strategy (hive, etc.)
        ddb_con: DuckDB connection
        **fs_kwargs: Additional filesystem arguments

    Returns:
        Simplified PydalaDatasetMetadata instance
    """
    return PydalaDatasetMetadata(
        path=path,
        filesystem=filesystem,
        bucket=bucket,
        cached=cached,
        partitioning=partitioning,
        ddb_con=ddb_con,
        **fs_kwargs
    )


# Frequently used migration patterns
class MetadataMigrator:
    """Helper class for common migration scenarios."""

    def __init__(self, legacy_metadata: LegacyParquetDatasetMetadata):
        """Initialize with existing legacy metadata.

        Args:
            legacy_metadata: Existing legacy metadata object
        """
        self.legacy = legacy_metadata
        self.path = legacy_metadata._path
        self.filesystem = legacy_metadata._filesystem
        self.bucket = legacy_metadata._bucket
        self.cached = legacy_metadata._cached

    def to_simplified(self, update_metadata: bool = True):
        """Create simplified version of metadata.

        Args:
            update_metadata: Whether to update metadata during migration

        Returns:
            Simplified ParquetDatasetMetadata instance
        """
        return migrate_to_simplified_parquet_metadata(
            path=self.path,
            filesystem=self.filesystem,
            bucket=self.bucket,
            cached=self.cached,
            update_metadata=update_metadata
        )

    def to_pydala(self, partitioning=None, ddb_con=None):
        """Create simplified PydalaDatasetMetadata version.

        Args:
            partitioning: Partition strategy
            ddb_con: DuckDB connection

        Returns:
            Simplified PydalaDatasetMetadata instance
        """
        return migrate_to_simplified_pydala_metadata(
            path=self.path,
            filesystem=self.filesystem,
            bucket=self.bucket,
            cached=self.cached,
            partitioning=partitioning,
            ddb_con=ddb_con
        )

    def compare_apis(self):
        """Compare API differences between legacy and simplified versions.

        Returns:
            Dict with API differences
        """
        return {
            "method_changes": {
                "__init__": "Simplified - extracted setup logic into helpers",
                "update": "Simplified - broken into focused methods",
                "_repair_file_schemas": "Simplified - extracted validation and logic",
                "_update_metadata": "Simplified - uses strategy pattern",
            },
            "new_components": {
                "MetadataStorage": "Handles all storage operations",
                "MetadataValidator": "Validates metadata operations",
                "SchemaManager": "Manages schema operations",
                "FileMetadataUpdater": "Handles file metadata updates",
            },
            "benefits": [
                "Functions reduced to <50 lines",
                "Minimal nesting through early returns",
                "Separated concerns for maintainability",
                "Reusable SQL operations",
                "Configuration objects for grouped parameters",
                "Data transformation separated from persistence",
            ]
        }
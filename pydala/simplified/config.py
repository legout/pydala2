"""
Configuration management for simplified dataset module.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Any

import pyarrow as pa
import duckdb
from fsspec import AbstractFileSystem


@dataclass
class DatasetConfig:
    """Configuration for dataset initialization."""

    # Required parameters
    path: str

    # Optional parameters with defaults
    name: Optional[str] = None
    schema: Optional[pa.Schema] = None
    filesystem: Optional[AbstractFileSystem] = None
    bucket: Optional[str] = None
    partitioning: Optional[Union[str, List[str]]] = None
    format: str = "parquet"
    cached: bool = False
    timestamp_column: Optional[str] = None
    ddb_con: Optional[duckdb.DuckDBPyConnection] = None

    # Additional filesystem kwargs
    fs_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation and processing."""
        # Generate name from path if not provided
        if self.name is None:
            import posixpath
            self.name = posixpath.basename(self.path)

        # Validate format
        supported_formats = ["parquet", "csv", "json"]
        if self.format not in supported_formats:
            raise ValueError(
                f"Unsupported format '{self.format}'. "
                f"Supported formats: {', '.join(supported_formats)}"
            )

        # Normalize partitioning
        if isinstance(self.partitioning, str) and self.partitioning != "hive":
            self.partitioning = [self.partitioning]

    def get_cache_storage(self) -> Optional[str]:
        """Get cache storage path if caching is enabled."""
        if not self.cached:
            return None

        import tempfile
        return self.fs_kwargs.pop(
            "cache_storage",
            tempfile.mkdtemp(prefix="pydala2_")
        )
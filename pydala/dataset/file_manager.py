import posixpath
import tempfile
import logging
from typing import List, Optional, Union, Any
from fsspec import AbstractFileSystem

from .filesystem import FileSystem, clear_cache
from .helpers.security import safe_join

logger = logging.getLogger(__name__)


class FileManager:
    """Handles file operations for datasets including listing, loading, and managing files."""

    def __init__(
        self,
        path: str,
        filesystem: AbstractFileSystem | None = None,
        bucket: str | None = None,
        cached: bool = False,
        format: str = "parquet",
        **fs_kwargs,
    ):
        self._path = path
        self._format = format
        self._base_filesystem = filesystem

        # Setup filesystem with caching if enabled
        cache_storage = None
        if cached:
            cache_storage = fs_kwargs.pop(
                "cache_storage", tempfile.mkdtemp(prefix="pydala2_")
            )

        self._filesystem = FileSystem(
            bucket=bucket,
            fs=filesystem,
            cached=cached,
            cache_storage=cache_storage,
            **fs_kwargs,
        )

        self._files = []
        self._ensure_directory_exists()

    def _ensure_directory_exists(self):
        """Create the directory if it doesn't exist."""
        if self._filesystem.exists(self._path):
            return

        try:
            self._filesystem.mkdirs(self._path)
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to create directory {self._path}: {e}")
            # Try alternative approach
            try:
                self._filesystem.touch(posixpath.join(self._path, "tmp.delete"))
                self._filesystem.rm(posixpath.join(self._path, "tmp.delete"))
            except Exception as e2:
                logger.error(f"Alternative directory creation also failed: {e2}")
                raise

    def load_files(self) -> None:
        """Load files from the dataset path matching the format."""
        self.clear_cache()
        # Safely construct glob pattern
        glob_pattern = safe_join(self._path, f"**/*.{self._format}")
        self._files = [
            fn.replace(self._path, "").lstrip("/")
            for fn in sorted(self._filesystem.glob(glob_pattern))
        ]

    @property
    def files(self) -> List[str]:
        """Get list of files in the dataset."""
        if not self._files:
            self.load_files()
        return self._files

    @property
    def has_files(self) -> bool:
        """Check if dataset has any files."""
        return len(self.files) > 0

    def delete_files(self, files: Union[str, List[str]]) -> None:
        """Delete specified files from the dataset."""
        if isinstance(files, str):
            files = [files]

        # Ensure paths are absolute
        if files and self._path not in files[0]:
            files = [posixpath.join(self._path, fn) for fn in files]

        self._filesystem.rm(files, recursive=True)

    def clear_cache(self) -> None:
        """Clear filesystem cache."""
        if hasattr(self._filesystem, "fs"):
            clear_cache(self._filesystem.fs)
        clear_cache(self._filesystem)
        clear_cache(self._base_filesystem)

    @property
    def filesystem(self) -> FileSystem:
        """Get the filesystem instance."""
        return self._filesystem

    @property
    def path(self) -> str:
        """Get the dataset path."""
        return self._path

    @property
    def format(self) -> str:
        """Get the dataset format."""
        return self._format
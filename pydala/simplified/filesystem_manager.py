"""
Filesystem management for simplified dataset module.
"""

import logging
import posixpath
from typing import List, Optional

from ..filesystem import FileSystem, clear_cache
from ..helpers.security import safe_join

logger = logging.getLogger(__name__)


class FilesystemManager:
    """Manages filesystem operations for datasets."""

    def __init__(self, config):
        self.config = config
        self.filesystem = self._create_filesystem()

    def _create_filesystem(self) -> FileSystem:
        """Create and configure filesystem instance."""
        return FileSystem(
            bucket=self.config.bucket,
            fs=self.config.filesystem,
            cached=self.config.cached,
            cache_storage=self.config.get_cache_storage(),
            **self.config.fs_kwargs,
        )

    def load_files(self, path: str, format: str) -> List[str]:
        """Load list of files matching the specified format."""
        # Always clear cache before loading files
        self.clear_all_caches()

        # Construct safe glob pattern
        glob_pattern = safe_join(path, f"**/*.{format}")

        try:
            files = [
                fn.replace(path, "").lstrip("/")
                for fn in sorted(self.filesystem.glob(glob_pattern))
            ]
            return files
        except Exception as e:
            logger.error(f"Failed to load files from {path}: {e}")
            return []

    def create_directory(self, path: str) -> None:
        """Create directory with proper error handling."""
        if self.filesystem.exists(path):
            return

        try:
            self.filesystem.mkdirs(path)
            logger.debug(f"Created directory: {path}")
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to create directory {path}: {e}")
            self._try_alternative_creation(path)

    def _try_alternative_creation(self, path: str) -> None:
        """Try alternative method for creating directories."""
        try:
            # Create and immediately remove a temporary file to create parent dirs
            tmp_path = posixpath.join(path, "tmp.delete")
            self.filesystem.touch(tmp_path)
            self.filesystem.rm(tmp_path)
            logger.debug(f"Created directory via alternative method: {path}")
        except Exception as e:
            logger.error(f"Alternative directory creation failed for {path}: {e}")
            raise

    def delete_files(self, files: List[str], recursive: bool = True) -> None:
        """Delete specified files."""
        if not files:
            return

        try:
            self.filesystem.rm(files, recursive=recursive)
            logger.info(f"Deleted {len(files)} files")
        except Exception as e:
            logger.error(f"Failed to delete files: {e}")
            raise

    def clear_all_caches(self) -> None:
        """Clear all filesystem caches."""
        if hasattr(self.filesystem, "fs"):
            clear_cache(self.filesystem.fs)
        clear_cache(self.filesystem)
        if self.config.filesystem is not None:
            clear_cache(self.config.filesystem)
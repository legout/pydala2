"""
Core filesystem functionality with clean abstractions.
"""
import posixpath
from typing import Optional, Union

import s3fs
import pyarrow.fs as pfs
from fsspec import AbstractFileSystem, filesystem
from fsspec.implementations.dirfs import DirFileSystem

from .cache import CacheManager
from .converters import ConversionService
from .monitoring import MonitoringService
from ..helpers.misc import run_parallel


def get_credentials_from_fsspec(fs: AbstractFileSystem, redact_secrets: bool = True) -> dict:
    """Safely extract credentials from fsspec filesystem.

    Args:
        fs: The fsspec filesystem object
        redact_secrets: If True, redact sensitive values from returned dict

    Returns:
        Dictionary with credential information (secrets redacted if redact_secrets=True)
    """
    if "s3" in fs.protocol:
        credentials = fs.s3._get_credentials()

        if redact_secrets:
            return {
                "access_key": f"REDACTED({len(credentials.access_key) if credentials.access_key else 0} chars)",
                "secret_key": f"REDACTED({len(credentials.secret_key) if credentials.secret_key else 0} chars)",
                "session_token": f"REDACTED({len(credentials.token) if credentials.token else 0} chars)" if credentials.token else None,
                "endpoint_override": fs.s3._endpoint.host if fs.s3._endpoint else None,
            }
        else:
            return {
                "access_key": credentials.access_key,
                "secret_key": credentials.secret_key,
                "session_token": credentials.token,
                "endpoint_override": fs.s3._endpoint.host if fs.s3._endpoint else None,
            }

    return {}


class FileSystemService:
    """Main service for filesystem operations."""

    def __init__(self,
                 cache_manager: Optional[CacheManager] = None,
                 conversion_service: Optional[ConversionService] = None,
                 monitoring_service: Optional[MonitoringService] = None):
        """Initialize filesystem service.

        Args:
            cache_manager: Cache manager instance
            conversion_service: Conversion service instance
            monitoring_service: Monitoring service instance
        """
        self.cache_manager = cache_manager or CacheManager()
        self.conversion_service = conversion_service or ConversionService()
        self.monitoring_service = monitoring_service or MonitoringService()

    def create_filesystem(self,
                         bucket: Optional[str] = None,
                         fs: Optional[AbstractFileSystem] = None,
                         profile: Optional[str] = None,
                         key: Optional[str] = None,
                         endpoint_url: Optional[str] = None,
                         secret: Optional[str] = None,
                         token: Optional[str] = None,
                         protocol: Optional[str] = None,
                         cached: bool = False,
                         **kwargs) -> AbstractFileSystem:
        """Create a filesystem instance.

        Args:
            bucket: Bucket/root directory
            fs: Existing filesystem to wrap
            profile: AWS profile
            key: Access key
            endpoint_url: Endpoint URL
            secret: Secret key
            token: Session token
            protocol: Protocol (s3, file, etc.)
            cached: Whether to use caching
            **kwargs: Additional arguments

        Returns:
            Filesystem instance
        """
        if protocol is None and fs is None:
            protocol = "file"

        if all([fs, profile, key, endpoint_url, secret, token, protocol]):
            fs = filesystem("file", use_listings_cache=False)
        elif fs is None:
            if "client_kwargs" in kwargs:
                fs = s3fs.S3FileSystem(
                    profile=profile,
                    key=key,
                    endpoint_url=endpoint_url,
                    secret=secret,
                    token=token,
                    **kwargs,
                )
            else:
                fs = filesystem(
                    protocol=protocol,
                    profile=profile,
                    key=key,
                    endpoint_url=endpoint_url,
                    secret=secret,
                    token=token,
                    use_listings_cache=False,
                )

        if bucket is not None:
            if protocol in ["file", "local"]:
                bucket = posixpath.abspath(bucket)
            fs = DirFileSystem(path=bucket, fs=fs)

        if cached:
            return self.cache_manager.create_cached_filesystem(fs)

        return fs

    def create_pyarrow_filesystem(self,
                                bucket: Optional[str] = None,
                                fs: Optional[AbstractFileSystem] = None,
                                access_key: Optional[str] = None,
                                secret_key: Optional[str] = None,
                                session_token: Optional[str] = None,
                                endpoint_override: Optional[str] = None,
                                protocol: Optional[str] = None) -> pfs.FileSystem:
        """Create a PyArrow filesystem instance.

        Args:
            bucket: Bucket/root directory
            fs: Existing fsspec filesystem
            access_key: Access key
            secret_key: Secret key
            session_token: Session token
            endpoint_override: Endpoint URL
            protocol: Protocol

        Returns:
            PyArrow filesystem instance
        """
        credentials = None
        if fs is not None:
            protocol = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol

            if protocol == "dir":
                bucket = fs.path
                fs = fs.fs
                protocol = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol

            if protocol == "s3":
                credentials = get_credentials_from_fsspec(fs, redact_secrets=False)

        if credentials is None:
            credentials = {
                "access_key": access_key,
                "secret_key": secret_key,
                "session_token": session_token,
                "endpoint_override": endpoint_override,
            }

        if protocol == "s3":
            fs = pfs.S3FileSystem(**credentials)
        elif protocol in ("file", "local", None):
            fs = pfs.LocalFileSystem()
        else:
            fs = pfs.LocalFileSystem()

        if bucket is not None:
            if protocol in ["file", "local", None]:
                bucket = posixpath.abspath(bucket)
            fs = pfs.SubTreeFileSystem(base_fs=fs, base_path=bucket)

        return fs

    def read_data(self,
                  fs: AbstractFileSystem,
                  path: str,
                  format: str,
                  **kwargs) -> any:
        """Read data in specified format.

        Args:
            fs: Filesystem instance
            path: File path
            format: Data format (parquet, json, csv)
            **kwargs: Additional arguments

        Returns:
            Read data
        """
        converter = self.conversion_service.get_converter(format)
        return converter.read(fs, path, **kwargs)

    def write_data(self,
                   fs: AbstractFileSystem,
                   data: any,
                   path: str,
                   format: str,
                   **kwargs) -> None:
        """Write data in specified format.

        Args:
            fs: Filesystem instance
            data: Data to write
            path: File path
            format: Data format (parquet, json, csv)
            **kwargs: Additional arguments
        """
        converter = self.conversion_service.get_converter(format)
        converter.write(fs, data, path, **kwargs)

    def convert_format(self,
                       fs: AbstractFileSystem,
                       src: str,
                       dst: str,
                       src_format: str,
                       dst_format: str,
                       **kwargs) -> None:
        """Convert data between formats.

        Args:
            fs: Filesystem instance
            src: Source path
            dst: Destination path
            src_format: Source format
            dst_format: Destination format
            **kwargs: Additional arguments
        """
        # Read data
        data = self.read_data(fs, src, src_format)

        # Write in new format
        self.write_data(fs, data, dst, dst_format, **kwargs)

    def get_disk_usage(self, storage: str) -> str:
        """Get disk usage information.

        Args:
            storage: Storage path

        Returns:
            Formatted disk usage message
        """
        return self.monitoring_service.get_disk_usage(storage)

    def clear_cache(self, fs: Optional[AbstractFileSystem] = None) -> None:
        """Clear filesystem cache.

        Args:
            fs: Filesystem instance
        """
        self.cache_manager.clear_cache(fs)


# Global instance for backward compatibility
_default_filesystem_service = FileSystemService()


def FileSystem(bucket: Optional[str] = None,
               fs: Optional[AbstractFileSystem] = None,
               profile: Optional[str] = None,
               key: Optional[str] = None,
               endpoint_url: Optional[str] = None,
               secret: Optional[str] = None,
               token: Optional[str] = None,
               protocol: Optional[str] = None,
               cached: bool = False,
               **kwargs) -> AbstractFileSystem:
    """Create a filesystem instance (legacy function).

    Args:
        bucket: Bucket/root directory
        fs: Existing filesystem to wrap
        profile: AWS profile
        key: Access key
        endpoint_url: Endpoint URL
        secret: Secret key
        token: Session token
        protocol: Protocol (s3, file, etc.)
        cached: Whether to use caching
        **kwargs: Additional arguments

    Returns:
        Filesystem instance
    """
    return _default_filesystem_service.create_filesystem(
        bucket=bucket,
        fs=fs,
        profile=profile,
        key=key,
        endpoint_url=endpoint_url,
        secret=secret,
        token=token,
        protocol=protocol,
        cached=cached,
        **kwargs
    )


def PyArrowFileSystem(bucket: Optional[str] = None,
                      fs: Optional[AbstractFileSystem] = None,
                      access_key: Optional[str] = None,
                      secret_key: Optional[str] = None,
                      session_token: Optional[str] = None,
                      endpoint_override: Optional[str] = None,
                      protocol: Optional[str] = None) -> pfs.FileSystem:
    """Create a PyArrow filesystem instance (legacy function).

    Args:
        bucket: Bucket/root directory
        fs: Existing fsspec filesystem
        access_key: Access key
        secret_key: Secret key
        session_token: Session token
        endpoint_override: Endpoint URL
        protocol: Protocol

    Returns:
        PyArrow filesystem instance
    """
    return _default_filesystem_service.create_pyarrow_filesystem(
        bucket=bucket,
        fs=fs,
        access_key=access_key,
        secret_key=secret_key,
        session_token=session_token,
        endpoint_override=endpoint_override,
        protocol=protocol
    )


def clear_cache(fs: Optional[AbstractFileSystem] = None) -> None:
    """Clear filesystem cache (legacy function).

    Args:
        fs: Filesystem instance
    """
    _default_filesystem_service.clear_cache(fs)
"""
Filesystem module with refactored and simplified components.

This module provides:
- Clean separation of concerns (caching, monitoring, conversions)
- SOLID principles with dependency injection
- Extensible architecture with clear interfaces
- Reduced code duplication
- Improved maintainability and testability

Example usage:
    from pydala.filesystem import FileSystemService, CacheManager, MonitoringService

    # Create services with dependency injection
    cache_manager = CacheManager(cache_storage="/tmp/cache")
    monitoring_service = MonitoringService()
    fs_service = FileSystemService(cache_manager=cache_manager, monitoring_service=monitoring_service)

    # Create filesystem
    fs = fs_service.create_filesystem(protocol="s3", bucket="my-bucket", cached=True)

    # Read/write data
    data = fs_service.read_data(fs, "path/to/data.parquet", format="parquet")
    fs_service.write_data(fs, data, "path/to/output.json", format="json")
"""

from .cache import CacheManager, MonitoredSimpleCacheFileSystem
from .core import FileSystemService, FileSystem, PyArrowFileSystem, clear_cache
from .converters import (
    DataConverter, ParquetConverter, JsonConverter, CsvConverter, ConversionService
)
from .monitoring import MonitoringService, sizeof_fmt, get_total_directory_size, DiskUsageTracker
from .utils import (
    get_new_file_names, list_files, sync_folder,
    json_to_parquet_single, csv_to_parquet_single,
    batch_json_to_parquet, batch_csv_to_parquet
)

# Import extensions to monkey-patch fsspec
from . import extensions

__all__ = [
    # Core services
    "FileSystemService",
    "FileSystem",
    "PyArrowFileSystem",
    "clear_cache",

    # Caching
    "CacheManager",
    "MonitoredSimpleCacheFileSystem",

    # Data converters
    "DataConverter",
    "ParquetConverter",
    "JsonConverter",
    "CsvConverter",
    "ConversionService",

    # Monitoring
    "MonitoringService",
    "sizeof_fmt",
    "get_total_directory_size",
    "DiskUsageTracker",

    # Utilities
    "get_new_file_names",
    "list_files",
    "sync_folder",
    "json_to_parquet_single",
    "csv_to_parquet_single",
    "batch_json_to_parquet",
    "batch_csv_to_parquet",
]
"""
Constants for Pydala package configuration.

This module contains centralized configuration constants used throughout
the Pydala codebase for performance tuning and operational parameters.
"""

from dataclasses import dataclass
from typing import Optional

import pyarrow as pa
import pyarrow.dataset as pds


@dataclass
class ScannerConfig:
    """Configuration for scanner operations.

    This dataclass centralizes scanner configuration parameters that are
    used across multiple methods in the codebase.
    """
    # Batch configuration
    batch_size: int = 131072
    batch_readahead: int = 16

    # Fragment configuration
    fragment_readahead: int = 4

    # Threading and performance
    use_threads: bool = True
    prefetch: int = 2
    num_threads: int = 4

    # Memory management
    memory_pool: Optional[pa.MemoryPool] = None

    # Scan options
    fragment_scan_options: Optional[pds.FragmentScanOptions] = None


# Performance tuning constants
DEFAULT_BATCH_SIZE = 131072
DEFAULT_BUFFER_SIZE = 65536
DEFAULT_PREFETCH_COUNT = 2
DEFAULT_THREAD_COUNT = 4

# Validation constants
MAX_COLUMN_NAME_LENGTH = 255
MIN_PARTITION_SIZE = 1024

# Filesystem constants
DEFAULT_BLOCK_SIZE = 16 * 1024 * 1024  # 16MB
DEFAULT_CACHE_SIZE = 128 * 1024 * 1024  # 128MB
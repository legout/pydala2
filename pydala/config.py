"""
Configuration classes for Pydala table operations.
"""

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.dataset as pds


@dataclass
class ScanConfig:
    """Configuration for table scanning operations."""
    columns: str | list[str] | None = None
    filter: pds.Expression | None = None
    batch_size: int = 131072
    sort_by: str | list[str] | list[tuple[str, str]] | None = None
    batch_readahead: int = 16
    fragment_readahead: int = 4
    fragment_scan_options: pds.FragmentScanOptions | None = None
    use_threads: bool = True
    memory_pool: pa.MemoryPool | None = None
    distinct: bool = False
    lazy: bool = True

    def __post_init__(self):
        if isinstance(self.columns, str):
            self.columns = [self.columns]


@dataclass
class ConversionConfig:
    """Configuration for data conversion operations."""
    columns: str | list[str] | None = None
    sort_by: str | list[str] | list[tuple[str, str]] | None = None
    distinct: bool = False
    lazy: bool = True
    batch_size: int = 131072
    filter: Any = None
    batch_readahead: int = 16
    fragment_readahead: int = 4
    fragment_scan_options: Any = None
    use_threads: bool = True
    memory_pool: Any = None

    def __post_init__(self):
        if isinstance(self.columns, str):
            self.columns = [self.columns]


@dataclass
class TableMetadata:
    """Metadata about the table."""
    table_type: str  # "pyarrow" or "duckdb"
    dataset: pds.Dataset
    ddb_relation: Any
    ddb_connection: Any
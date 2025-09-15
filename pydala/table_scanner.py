"""
Table scanner class to handle scanning operations.
"""

import pyarrow as pa
import pyarrow.dataset as pds

from .config import ScanConfig, TableMetadata
from .converters import ArrowConverter


class TableScanner:
    """Handles table scanning operations with various configurations."""

    def __init__(self, metadata: TableMetadata):
        self.metadata = metadata
        self.arrow_converter = ArrowConverter(metadata)

    def create_scanner(self, config: ScanConfig) -> pds.Scanner:
        """Create a scanner with the given configuration."""
        if self.metadata.table_type != "pyarrow":
            raise ValueError("Arrow scanner is only available for pyarrow datasets")
        return self.arrow_converter.to_scanner(config)

    def to_arrow_scanner(
        self,
        columns: str | list[str] | None = None,
        filter: pds.Expression | None = None,
        batch_size: int = 131072,
        sort_by: str | list[str] | list[tuple[str, str]] | None = None,
        batch_readahead: int = 16,
        fragment_readahead: int = 4,
        fragment_scan_options: pds.FragmentScanOptions | None = None,
        use_threads: bool = True,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pds.Scanner:
        """
        Create an Arrow scanner with the specified parameters.

        This method provides backward compatibility with the original interface.
        """
        config = ScanConfig(
            columns=columns,
            filter=filter,
            batch_size=batch_size,
            sort_by=sort_by,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            fragment_scan_options=fragment_scan_options,
            use_threads=use_threads,
            memory_pool=memory_pool,
        )
        return self.create_scanner(config)

    # Alias methods for backward compatibility
    def to_scanner(self, **kwargs) -> pds.Scanner:
        """Alias for to_arrow_scanner."""
        return self.to_arrow_scanner(**kwargs)

    def scanner(self, **kwargs) -> pds.Scanner:
        """Alias for to_arrow_scanner."""
        return self.to_arrow_scanner(**kwargs)
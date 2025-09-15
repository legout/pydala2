"""
Converter classes for different data formats.
"""

from abc import ABC, abstractmethod
from typing import Any, Union

import duckdb as _duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds

from .config import ConversionConfig, ScanConfig, TableMetadata
from .helpers.polars import pl as _pl
from .sort_handler import SortHandler


class BaseConverter(ABC):
    """Base class for data converters."""

    def __init__(self, metadata: TableMetadata):
        self.metadata = metadata

    @abstractmethod
    def convert(self, config: ConversionConfig) -> Any:
        """Convert data using the provided configuration."""
        pass


class ArrowConverter(BaseConverter):
    """Converts data to PyArrow formats."""

    def to_dataset(self) -> pds.Dataset:
        """Return the underlying PyArrow dataset."""
        return self.metadata.dataset

    def to_scanner(self, scan_config: ScanConfig) -> pds.Scanner:
        """Create an Arrow scanner from the dataset."""
        if self.metadata.table_type != "pyarrow":
            raise ValueError("Arrow scanner is only available for pyarrow datasets")

        scanner_kwargs = {
            "columns": scan_config.columns,
            "filter": scan_config.filter,
            "batch_size": scan_config.batch_size,
            "batch_readahead": scan_config.batch_readahead,
            "fragment_readahead": scan_config.fragment_readahead,
            "fragment_scan_options": scan_config.fragment_scan_options,
            "use_threads": scan_config.use_threads,
            "memory_pool": scan_config.memory_pool,
        }

        if scan_config.sort_by is not None:
            sort_by = SortHandler.get_sort_by(scan_config.sort_by, "pyarrow")
            return self.metadata.dataset.sort_by(**sort_by).scanner(**scanner_kwargs)
        else:
            return self.metadata.dataset.scanner(**scanner_kwargs)

    def to_table(self, scan_config: ScanConfig) -> pa.Table:
        """Convert to Arrow table."""
        if self.metadata.table_type == "pyarrow":
            scanner = self.to_scanner(scan_config)
            table = scanner.to_table()

            if scan_config.distinct:
                return _pl.from_arrow(table).unique(maintain_order=True).to_arrow()
            return table
        else:
            # Use DuckDB conversion
            duckdb_converter = DuckDBConverter(self.metadata)
            ddb_relation = duckdb_converter.convert(
                ConversionConfig(
                    columns=scan_config.columns,
                    sort_by=scan_config.sort_by,
                    distinct=scan_config.distinct,
                    lazy=False,
                    **{k: v for k, v in scan_config.__dict__.items()
                       if k not in ['columns', 'sort_by', 'distinct', 'lazy']}
                )
            )
            return ddb_relation.to_arrow_table()


class DuckDBConverter(BaseConverter):
    """Converts data to DuckDB formats."""

    def convert(self, config: ConversionConfig) -> _duckdb.DuckDBPyRelation:
        """Convert to DuckDB relation."""
        columns_str = "*" if config.columns is None else ", ".join(config.columns)

        if config.lazy:
            relation = self.metadata.ddb_relation.select(columns_str)

            if config.sort_by is not None:
                sort_by = SortHandler.get_sort_by(config.sort_by, "duckdb")
                relation = relation.order(sort_by)

            return relation.distinct() if config.distinct else relation
        else:
            # Materialize the data
            arrow_converter = ArrowConverter(self.metadata)
            table = arrow_converter.to_table(
                ScanConfig(
                    columns=config.columns,
                    sort_by=config.sort_by,
                    distinct=config.distinct
                )
            )
            return self.metadata.ddb_connection.from_arrow(table)


class PolarsConverter(BaseConverter):
    """Converts data to Polars formats."""

    def convert(self, config: ConversionConfig) -> Union[_pl.DataFrame, _pl.LazyFrame]:
        """Convert to Polars DataFrame or LazyFrame."""
        if self.metadata.table_type == "pyarrow":
            if config.lazy:
                # Use lazy scanning
                df = _pl.scan_pyarrow_dataset(
                    self.metadata.dataset,
                    batch_size=config.kwargs.get("batch_size", 131072)
                )

                if config.columns is not None:
                    df = df.select(config.columns)

                if config.sort_by is not None:
                    sort_by = SortHandler.get_sort_by(config.sort_by, "polars")
                    df = df.sort(**sort_by)

                if config.distinct:
                    df = df.unique(maintain_order=True)
            else:
                # Materialize the data
                arrow_converter = ArrowConverter(self.metadata)
                batches = arrow_converter.to_scanner(
                    ScanConfig(
                        columns=config.columns,
                        sort_by=config.sort_by,
                        distinct=config.distinct,
                        **config.kwargs
                    )
                ).to_reader().read_all()
                df = _pl.from_arrow(batches)
        else:
            # Use DuckDB as intermediate
            duckdb_converter = DuckDBConverter(self.metadata)
            ddb_relation = duckdb_converter.convert(config)
            df = ddb_relation.pl(
                batch_size=config.kwargs.get("batch_size", 131072)
            )

        return df


class PandasConverter(BaseConverter):
    """Converts data to Pandas DataFrame."""

    def convert(self, config: ConversionConfig) -> pd.DataFrame:
        """Convert to Pandas DataFrame."""
        arrow_converter = ArrowConverter(self.metadata)
        table = arrow_converter.to_table(
            ScanConfig(
                columns=config.columns,
                sort_by=config.sort_by,
                distinct=config.distinct,
                **config.kwargs
            )
        )
        return table.to_pandas()


class BatchReaderConverter(BaseConverter):
    """Converts data to Arrow RecordBatchReader."""

    def convert(self, config: ConversionConfig) -> pa.RecordBatchReader:
        """Convert to RecordBatchReader."""
        arrow_converter = ArrowConverter(self.metadata)

        # Optimize path for simple cases with PyArrow dataset
        if (self.metadata.table_type == "pyarrow" and
            config.sort_by is None and
            not config.distinct):
            scan_config = ScanConfig(**config.kwargs)
            scanner = arrow_converter.to_scanner(scan_config)
            return scanner.to_reader()

        # Use DuckDB for complex cases
        duckdb_converter = DuckDBConverter(self.metadata)
        ddb_relation = duckdb_converter.convert(
            ConversionConfig(
                columns=config.columns,
                lazy=config.lazy,
                sort_by=config.sort_by,
                distinct=config.distinct
            )
        )
        return ddb_relation.fetch_arrow_reader(
            batch_size=config.kwargs.get("batch_size", 131072)
        )
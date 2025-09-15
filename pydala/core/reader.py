"""
Dataset reader component.

This module provides a focused interface for reading data from datasets
with simplified filtering and scanning capabilities.
"""

import posixpath
import typing as t
from dataclasses import dataclass

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.dataset as pds
from fsspec import AbstractFileSystem
from loguru import logger

from ..filesystem import FileSystem
from ..helpers.misc import sql2pyarrow_filter
from ..helpers.security import sanitize_filter_expression, safe_join
from ..table import PydalaTable
from .metadata import DatasetMetadata, FileMetadata


@dataclass
class ReadConfig:
    """Configuration for read operations."""
    
    # Column selection
    columns: list[str] | None = None
    
    # Filtering
    filter_expr: str | pds.Expression | None = None
    use_engine: str = "auto"  # "auto", "pyarrow", "duckdb"
    
    # Batching
    batch_size: int | None = None
    
    # Sorting
    sort_by: str | list[str] | list[tuple[str, str]] | None = None
    
    # Limiting
    limit: int | None = None
    offset: int | None = None
    
    # Format
    output_format: str = "table"  # "table", "polars", "pandas", "arrow", "duckdb"


class DatasetReader:
    """
    Handles all read operations for a dataset.
    
    This class provides a simplified interface for reading data from datasets,
    with support for filtering, column selection, and various output formats.
    """
    
    def __init__(
        self,
        path: str,
        filesystem: AbstractFileSystem | None = None,
        metadata: DatasetMetadata | None = None,
        format: str = "parquet",
        partitioning: str | list[str] | None = None,
        ddb_con: duckdb.DuckDBPyConnection | None = None,
    ):
        """
        Initialize DatasetReader.
        
        Args:
            path: Path to the dataset
            filesystem: Filesystem to use for I/O operations
            metadata: Dataset metadata manager
            format: File format (parquet, csv, json)
            partitioning: Partitioning scheme
            ddb_con: DuckDB connection for SQL operations
        """
        self.path = path
        self.filesystem = filesystem or FileSystem()
        self.metadata = metadata
        self.format = format
        self.partitioning = partitioning
        self.ddb_con = ddb_con or duckdb.connect()
        
        self._dataset: pds.Dataset | None = None
        self._table_name: str | None = None
        
    @property
    def dataset(self) -> pds.Dataset:
        """Get the PyArrow dataset, loading if necessary."""
        if self._dataset is None:
            self._load_dataset()
        return self._dataset
    
    def _load_dataset(self) -> None:
        """Load the PyArrow dataset."""
        try:
            # Try to load using metadata file if available
            if self.metadata and self.metadata.has_metadata_file:
                metadata_file = posixpath.join(self.path, "_metadata")
                self._dataset = pds.parquet_dataset(
                    metadata_file,
                    partitioning=self.partitioning,
                    filesystem=self.filesystem,
                )
            else:
                # Load dataset by scanning files
                self._dataset = pds.dataset(
                    self.path,
                    format=self.format,
                    partitioning=self.partitioning,
                    filesystem=self.filesystem,
                    schema=self.metadata.schema if self.metadata else None,
                )
                
            # Register with DuckDB
            if self._dataset and self.ddb_con:
                table_name = posixpath.basename(self.path)
                self.ddb_con.register(table_name, self._dataset)
                self._table_name = table_name
                
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def read(
        self,
        config: ReadConfig | None = None,
    ) -> PydalaTable:
        """
        Read data from the dataset.
        
        Args:
            config: Read configuration
            
        Returns:
            PydalaTable with the requested data
        """
        config = config or ReadConfig()
        
        # Apply filtering
        if config.filter_expr:
            result = self._filter(config.filter_expr, config.use_engine)
        else:
            result = self.dataset
            
        # Convert to PydalaTable for consistent interface
        table = PydalaTable(result=result, ddb_con=self.ddb_con)
        
        # Apply additional operations
        if config.columns:
            table = self._select_columns(table, config.columns)
            
        if config.sort_by:
            table = self._sort(table, config.sort_by)
            
        if config.limit or config.offset:
            table = self._limit(table, config.limit, config.offset)
            
        # Convert to requested format
        return self._convert_format(table, config.output_format)
    
    def _filter(
        self,
        filter_expr: str | pds.Expression,
        use_engine: str = "auto",
    ) -> pds.Dataset | duckdb.DuckDBPyRelation:
        """Apply filtering to the dataset."""
        # Sanitize filter expression if it's a string
        if isinstance(filter_expr, str):
            filter_expr = sanitize_filter_expression(filter_expr)
            
        # Check if we need to use DuckDB for complex filters
        if isinstance(filter_expr, str):
            if any(s in filter_expr for s in ["%", "like", "similar to", "*", "(", ")"]):
                use_engine = "duckdb"
                
        if use_engine == "duckdb" or (use_engine == "auto" and self._should_use_duckdb(filter_expr)):
            return self._filter_duckdb(filter_expr)
        else:
            return self._filter_pyarrow(filter_expr)
    
    def _filter_pyarrow(
        self,
        filter_expr: str | pds.Expression,
    ) -> pds.Dataset:
        """Filter using PyArrow."""
        if isinstance(filter_expr, str):
            # Convert SQL-like expression to PyArrow expression
            schema = self.metadata.schema if self.metadata else self.dataset.schema
            filter_expr = sql2pyarrow_filter(filter_expr, schema)
            
        return self.dataset.filter(filter_expr)
    
    def _filter_duckdb(self, filter_expr: str) -> duckdb.DuckDBPyRelation:
        """Filter using DuckDB."""
        if not self._table_name:
            self._load_dataset()
            
        # Build safe SQL query
        query = f"SELECT * FROM {self._table_name} WHERE {filter_expr}"
        return self.ddb_con.sql(query)
    
    def _should_use_duckdb(self, filter_expr: t.Any) -> bool:
        """Determine if DuckDB should be used for filtering."""
        if not isinstance(filter_expr, str):
            return False
            
        # Use DuckDB for complex SQL expressions
        complex_indicators = ["like", "similar to", "in (", "exists", "case when"]
        return any(ind in filter_expr.lower() for ind in complex_indicators)
    
    def _select_columns(self, table: PydalaTable, columns: list[str]) -> PydalaTable:
        """Select specific columns from the table."""
        if hasattr(table.result, 'select'):
            # PyArrow dataset
            result = table.result.select(columns)
        else:
            # DuckDB relation
            col_str = ", ".join(columns)
            result = table.ddb.select(col_str)
            
        return PydalaTable(result=result, ddb_con=self.ddb_con)
    
    def _sort(
        self,
        table: PydalaTable,
        sort_by: str | list[str] | list[tuple[str, str]],
    ) -> PydalaTable:
        """Sort the table."""
        # Convert to standardized format
        if isinstance(sort_by, str):
            sort_by = [sort_by]
            
        # Use the table's built-in sorting if available
        if hasattr(table, 'order'):
            return table.order(sort_by)
        else:
            # Fall back to SQL
            order_clauses = []
            for item in sort_by:
                if isinstance(item, tuple):
                    col, order = item
                    order_clauses.append(f"{col} {order.upper()}")
                else:
                    order_clauses.append(f"{item} ASC")
                    
            order_str = ", ".join(order_clauses)
            result = table.ddb.order(order_str)
            return PydalaTable(result=result, ddb_con=self.ddb_con)
    
    def _limit(
        self,
        table: PydalaTable,
        limit: int | None,
        offset: int | None,
    ) -> PydalaTable:
        """Apply limit and offset to the table."""
        if hasattr(table.result, 'head'):
            # PyArrow dataset - convert to table first
            arrow_table = table.to_arrow()
            if offset:
                arrow_table = arrow_table.slice(offset, limit)
            elif limit:
                arrow_table = arrow_table.slice(0, limit)
            result = arrow_table
        else:
            # DuckDB relation
            result = table.ddb
            if limit:
                result = result.limit(limit)
            if offset:
                result = result.offset(offset)
                
        return PydalaTable(result=result, ddb_con=self.ddb_con)
    
    def _convert_format(self, table: PydalaTable, format: str) -> t.Any:
        """Convert table to requested format."""
        if format == "table":
            return table
        elif format == "polars":
            return table.pl
        elif format == "pandas":
            return table.pd
        elif format == "arrow":
            return table.to_arrow()
        elif format == "duckdb":
            return table.ddb
        else:
            return table
    
    def scan(
        self,
        filter_expr: str | None = None,
        columns: list[str] | None = None,
    ) -> pds.Dataset:
        """
        Scan the dataset with optional filtering.
        
        This is a lightweight operation that returns a dataset
        without loading data into memory.
        
        Args:
            filter_expr: Filter expression
            columns: Columns to select
            
        Returns:
            Filtered/projected PyArrow dataset
        """
        dataset = self.dataset
        
        if filter_expr:
            filter_expr = sanitize_filter_expression(filter_expr)
            if isinstance(filter_expr, str):
                schema = self.metadata.schema if self.metadata else dataset.schema
                filter_expr = sql2pyarrow_filter(filter_expr, schema)
            dataset = dataset.filter(filter_expr)
            
        if columns:
            dataset = dataset.select(columns)
            
        return dataset
    
    def scan_files(
        self,
        filter_expr: str | None = None,
    ) -> list[str]:
        """
        Get list of files matching the filter.
        
        Args:
            filter_expr: Filter expression
            
        Returns:
            List of file paths matching the filter
        """
        if not filter_expr and self.metadata:
            # Return all files
            return list(self.metadata.file_metadata.keys())
            
        # Query metadata to find matching files
        if self.metadata:
            matching_metadata = self.metadata.query_metadata(filter_expr)
            return [m.path for m in matching_metadata]
        else:
            # Fall back to scanning dataset
            dataset = self.scan(filter_expr)
            # This would need to extract file paths from fragments
            # Simplified version:
            return self._get_all_files()
    
    def _get_all_files(self) -> list[str]:
        """Get all files in the dataset."""
        pattern = safe_join(self.path, f"**/*.{self.format}")
        files = self.filesystem.glob(pattern)
        return [f.replace(self.path, "").lstrip("/") for f in sorted(files)]
    
    def to_batches(
        self,
        batch_size: int = 1_000_000,
        config: ReadConfig | None = None,
    ) -> t.Iterator[pa.RecordBatch]:
        """
        Read data as an iterator of record batches.
        
        Args:
            batch_size: Number of rows per batch
            config: Read configuration
            
        Yields:
            PyArrow RecordBatch objects
        """
        config = config or ReadConfig()
        config.batch_size = batch_size
        
        # Apply filtering and column selection
        dataset = self.scan(config.filter_expr, config.columns)
        
        # Create scanner
        scanner = dataset.scanner(batch_size=batch_size)
        
        # Yield batches
        for batch in scanner.to_batches():
            yield batch
    
    def count_rows(self, filter_expr: str | None = None) -> int:
        """
        Count rows in the dataset.
        
        Args:
            filter_expr: Optional filter expression
            
        Returns:
            Number of rows
        """
        if filter_expr:
            dataset = self.scan(filter_expr)
        else:
            dataset = self.dataset
            
        return dataset.count_rows()
    
    def get_schema(self) -> pa.Schema:
        """Get the dataset schema."""
        if self.metadata and self.metadata.schema:
            return self.metadata.schema
        return self.dataset.schema
    
    def get_columns(self) -> list[str]:
        """Get list of column names."""
        schema = self.get_schema()
        return schema.names if schema else []
    
    def get_partitions(self) -> pa.Table | None:
        """Get partition information as a table."""
        if self.metadata:
            return self.metadata.get_partitions_table()
            
        # Try to get from dataset
        if hasattr(self.dataset, 'partitioning') and self.dataset.partitioning:
            # Build partition table from dataset
            # This is a simplified version
            return None
            
        return None
    
    def head(
        self,
        n: int = 10,
        columns: list[str] | None = None,
    ) -> pa.Table:
        """
        Get first n rows from the dataset.
        
        Args:
            n: Number of rows to return
            columns: Columns to select
            
        Returns:
            PyArrow table with first n rows
        """
        config = ReadConfig(
            columns=columns,
            limit=n,
            output_format="arrow",
        )
        return self.read(config)
    
    def sample(
        self,
        n: int = 100,
        fraction: float | None = None,
        seed: int | None = None,
    ) -> pa.Table:
        """
        Get a random sample from the dataset.
        
        Args:
            n: Number of rows to sample
            fraction: Fraction of dataset to sample (alternative to n)
            seed: Random seed for reproducibility
            
        Returns:
            PyArrow table with sampled rows
        """
        # Use DuckDB for efficient sampling
        if not self._table_name:
            self._load_dataset()
            
        if fraction:
            query = f"SELECT * FROM {self._table_name} USING SAMPLE {fraction*100}%"
        else:
            query = f"SELECT * FROM {self._table_name} USING SAMPLE {n}"
            
        if seed:
            query += f" REPEATABLE ({seed})"
            
        result = self.ddb_con.sql(query)
        return result.arrow()
    
    def to_lazy_polars(
        self,
        config: ReadConfig | None = None,
    ) -> pl.LazyFrame:
        """
        Get a lazy Polars DataFrame for the dataset.
        
        Args:
            config: Read configuration
            
        Returns:
            Polars LazyFrame
        """
        # Scan the dataset
        dataset = self.scan(
            config.filter_expr if config else None,
            config.columns if config else None,
        )
        
        # Create lazy frame
        return pl.scan_pyarrow_dataset(dataset)
    
    def explain(
        self,
        config: ReadConfig | None = None,
    ) -> str:
        """
        Explain the query plan for a read operation.
        
        Args:
            config: Read configuration
            
        Returns:
            Query plan explanation
        """
        config = config or ReadConfig()
        
        if config.use_engine == "duckdb" or self._should_use_duckdb(config.filter_expr):
            # Get DuckDB query plan
            if not self._table_name:
                self._load_dataset()
                
            query = f"EXPLAIN SELECT * FROM {self._table_name}"
            if config.filter_expr:
                query += f" WHERE {config.filter_expr}"
                
            result = self.ddb_con.sql(query)
            return str(result.fetchall())
        else:
            # PyArrow doesn't have a direct explain, show dataset info
            dataset = self.scan(config.filter_expr, config.columns)
            info = [
                f"Dataset path: {self.path}",
                f"Format: {self.format}",
                f"Schema: {dataset.schema}",
                f"Partitioning: {self.partitioning}",
            ]
            if config.filter_expr:
                info.append(f"Filter: {config.filter_expr}")
            if config.columns:
                info.append(f"Columns: {config.columns}")
                
            return "\n".join(info)
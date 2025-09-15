"""
Database management for simplified dataset module.
"""

import logging
import psutil
from typing import Optional

import duckdb

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages DuckDB connections and operations."""

    def __init__(self, config):
        self.config = config
        self.connection = self._create_connection()
        self._configure_connection()

    def _create_connection(self) -> duckdb.DuckDBPyConnection:
        """Create or use existing DuckDB connection."""
        if self.config.ddb_con is None:
            conn = duckdb.connect()
            logger.debug("Created new DuckDB connection")
            return conn
        else:
            logger.debug("Using provided DuckDB connection")
            return self.config.ddb_con

    def _configure_connection(self) -> None:
        """Configure DuckDB connection settings."""
        try:
            self.connection.execute(
                f"""PRAGMA enable_object_cache;
                SET THREADS={psutil.cpu_count() * 2};"""
            )
            logger.debug("Configured DuckDB connection")
        except Exception as e:
            logger.warning(f"Failed to configure DuckDB connection: {e}")

    def register_dataset(self, name: str, dataset) -> None:
        """Register a dataset with DuckDB."""
        try:
            self.connection.register(name, dataset)
            logger.debug(f"Registered dataset '{name}' with DuckDB")
        except Exception as e:
            logger.error(f"Failed to register dataset '{name}': {e}")
            raise

    def set_timezone(self, timezone: str) -> None:
        """Set timezone for DuckDB connection."""
        try:
            self.connection.execute("SET TimeZone=?", [str(timezone)])
            logger.debug(f"Set timezone to {timezone}")
        except Exception as e:
            logger.error(f"Failed to set timezone to {timezone}: {e}")
            raise

    def get_tables(self) -> list:
        """Get list of registered tables."""
        try:
            result = self.connection.sql("SHOW TABLES").arrow()
            return result.column("name").to_pylist()
        except Exception as e:
            logger.error(f"Failed to get table list: {e}")
            return []

    def interrupt(self) -> None:
        """Interrupt current DuckDB operation."""
        try:
            self.connection.interrupt()
            logger.debug("Interrupted DuckDB operation")
        except Exception as e:
            logger.warning(f"Failed to interrupt DuckDB: {e}")

    def close(self) -> None:
        """Close the database connection if we created it."""
        if self.config.ddb_con is None:
            try:
                self.connection.close()
                logger.debug("Closed DuckDB connection")
            except Exception as e:
                logger.warning(f"Error closing DuckDB connection: {e}")
from dataclasses import dataclass, field
from typing import Dict, Optional

import duckdb
import yaml
from fsspec import AbstractFileSystem

from .filesystem import FileSystem


@dataclass
class Filesystem:
    """Configuration for a filesystem connection.

    This dataclass stores configuration parameters for various filesystem types
    including local filesystem, S3, and other cloud storage systems.

    Attributes:
        bucket: Bucket name for cloud storage (None for local filesystem).
        fs: Optional pre-configured filesystem instance.
        profile: AWS profile name for S3.
        key: Access key for authentication.
        endpoint_url: Custom endpoint URL for S3-compatible storage.
        secret: Secret key for authentication.
        token: Authentication token.
        protocol: Filesystem protocol (e.g., 'file', 's3', 'gcs').
        cached: Whether to enable caching.
        cache_storage: Directory path for cache storage.
        check_files: Whether to check file existence before operations.
        cache_check: Cache check interval in seconds.
        expire_time: Cache expiration time in seconds.
        same_names: Whether to enforce same naming conventions.
        kwargs: Additional filesystem-specific parameters.
    """
    bucket: str | None = None
    fs: AbstractFileSystem | None = None
    profile: str | None = None
    key: str | None = None
    endpoint_url: str | None = None
    secret: str | None = None
    token: str | None = None
    protocol: str | None = None
    cached: bool = False
    cache_storage = "~/.tmp"
    check_files: bool = False
    cache_check: int = 120
    expire_time: int = 24 * 60 * 60
    same_names: bool = False
    kwargs: dict | None = None

    def connect(self) -> FileSystem:
        """Create a FileSystem instance from this configuration.

        Returns:
            FileSystem: A configured filesystem instance ready for use.
        """
        return FileSystem(**self.__dict__)


@dataclass
class Table:
    """Configuration for a data table.

    This dataclass stores metadata about a data table including its location,
    format, and filesystem configuration.

    Attributes:
        name: Name of the table.
        path: Path to the table data.
        type: Type of the table (e.g., 'dataset', 'table').
        format: Data format (e.g., 'parquet', 'csv', 'json').
        filesystem: Name of the filesystem configuration to use.
        partitioning: Partitioning scheme for the data.
        timestamp_column: Name of the timestamp column for time-series data.
        write_args: Additional arguments for writing operations.
    """
    name: str
    path: str
    type: str
    format: str
    filesystem: str
    partitioning: str | None = None
    # partitioning_flavor: str = "hive" # hive, directory, filename
    timestamp_column: str | None = None
    write_args: dict = field(default_factory=dict)

    def _load_parquet(self) -> None:
        """Load parquet data from this table's path.

        This method is a placeholder for parquet loading functionality.

        Note:
            This method is not yet implemented.
        """
        pass


@dataclass
class Catalog:
    """A catalog for managing filesystems and tables.

    This class provides a centralized way to manage multiple filesystem configurations
    and table metadata, with support for persistence to YAML files and DuckDB integration.

    Attributes:
        filesystems: Dictionary mapping filesystem names to configurations.
        tables: Dictionary mapping table names to configurations.
        ddb_con: DuckDB connection instance for SQL operations.
        _catalog_path: Path to the catalog YAML file.
    """
    filesystems: Dict[str, Filesystem] = field(default_factory=dict)
    tables: Dict[str, Table] = field(default_factory=dict)
    ddb_con: Optional[duckdb.DuckDBPyConnection] = None
    _catalog_path: str = "catalog.yaml"

    def __post_init__(self) -> None:
        """Initialize the catalog after dataclass creation.

        Creates a default DuckDB connection if none is provided.
        """
        if self.ddb_con is None:
            self.ddb_con = duckdb.connect()

    def load_catalog(self) -> None:
        """Load catalog configuration from YAML file.

        Reads the catalog configuration from the YAML file specified by _catalog_path
        and populates the filesystems and tables dictionaries.

        Raises:
            FileNotFoundError: If the catalog file does not exist.
            yaml.YAMLError: If the catalog file is invalid YAML.
        """
        with open(self._catalog_path, "r") as f:
            data = yaml.safe_load(f)

        self.filesystems = {
            name: Filesystem(**fs) for name, fs in data["filesystems"].items()
        }
        self.tables = {
            name: Table(name=name, **table) for name, table in data["tables"].items()
        }

    def save_catalog(self) -> None:
        """Save catalog configuration to YAML file.

        Writes the current state of filesystems and tables to the YAML file
        specified by _catalog_path.

        Raises:
            IOError: If unable to write to the catalog file.
        """
        data = {
            "filesystems": {name: vars(fs) for name, fs in self.filesystems.items()},
            "tables": {
                name: {k: v for k, v in vars(table).items() if k != "name"}
                for name, table in self.tables.items()
            },
        }
        with open(self._catalog_path, "w") as f:
            yaml.dump(data, f)

    def create_filesystem(self, name: str, protocol: str, bucket: str, path: str) -> None:
        """Create a new filesystem configuration.

        Args:
            name: Name to identify this filesystem configuration.
            protocol: Filesystem protocol (e.g., 'file', 's3', 'gcs').
            bucket: Bucket name for cloud storage.
            path: Base path for the filesystem.
        """
        self.filesystems[name] = Filesystem(protocol=protocol, bucket=bucket, path=path)
        self.save_catalog()

    def create_table(
        self, name: str, path: str, format: str, filesystem: str, **kwargs
    ) -> None:
        """Create a new table configuration.

        Args:
            name: Name to identify this table.
            path: Path to the table data.
            format: Data format (e.g., 'parquet', 'csv', 'json').
            filesystem: Name of the filesystem configuration to use.
            **kwargs: Additional table configuration parameters.
        """
        self.tables[name] = Table(
            name=name, path=path, format=format, filesystem=filesystem, **kwargs
        )
        self.save_catalog()

    def delete_table(self, name: str) -> None:
        """Delete a table configuration.

        Args:
            name: Name of the table to delete.

        Raises:
            KeyError: If the table does not exist.
        """
        del self.tables[name]
        self.save_catalog()

    def write_table(self, table_name: str, data, **kwargs) -> None:
        """Write data to a table.

        Args:
            table_name: Name of the table to write to.
            data: Data to write (format depends on table type).
            **kwargs: Additional writing parameters.

        Note:
            This method is a placeholder and needs to be implemented
            with actual writing logic based on the table format.
        """
        table = self.tables[table_name]
        fs = self.filesystems[table.filesystem]
        # Implement writing logic here based on the format and filesystem
        # You may need to add more methods or use external libraries for actual writing

    def read_table(self, table_name: str, **kwargs):
        """Read data from a table.

        Args:
            table_name: Name of the table to read from.
            **kwargs: Additional reading parameters.

        Returns:
            Data from the table (format depends on table type).

        Note:
            This method is a placeholder and needs to be implemented
            with actual reading logic based on the table format.
        """
        table = self.tables[table_name]
        fs = self.filesystems[table.filesystem]
        # Implement reading logic here based on the format and filesystem
        # You may need to add more methods or use external libraries for actual reading

    def sql(self, query: str):
        """Execute a SQL query using DuckDB.

        Registers all tables in the catalog with DuckDB and executes the query.

        Args:
            query: SQL query string to execute.

        Returns:
            DuckDB result object containing the query results.
        """
        # Register tables if not already registered
        for table_name, table in self.tables.items():
            if table_name not in self.ddb_con.execute("SHOW TABLES").fetchall():
                self.ddb_con.register(table_name, self.read_table(table_name))

        return self.ddb_con.execute(query)


# Usage example:
if __name__ == "__main__":
    # %%
    catalog = Catalog()
    catalog.load_catalog()

    catalog.create_filesystem("s3", "s3", "my-bucket", "s3://my-bucket/data")
    catalog.create_table(
        "my_table", "s3://my-bucket/data/my_table.parquet", "parquet", "s3"
    )

    # Write data to the table (you need to implement the actual writing logic)
    data = ...  # Your data here
    catalog.write_table("my_table", data)

    # Read data from the table (you need to implement the actual reading logic)
    read_data = catalog.read_table("my_table")

    # Execute SQL query
    result = catalog.sql("SELECT * FROM my_table LIMIT 10")

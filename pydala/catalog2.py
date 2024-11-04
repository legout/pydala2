from dataclasses import dataclass, field

import duckdb
import yaml
from fsspec import AbstractFileSystem

from .filesystem import FileSystem


@dataclass
class Filesystem:
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

    def connect(self):
        return FileSystem(**self.__dict__)


@dataclass
class Table:
    path: str
    type: str
    format: str
    filesystem: str
    partitioning: str | None = None
    # partitioning_flavor: str = "hive" # hive, directory, filename
    timestamp_column: str | None = None
    write_args: dict = field(default_factory=dict)

    def _load_parquet(self):
        pass


@dataclass
class Catalog:
    filesystems: Dict[str, Filesystem] = field(default_factory=dict)
    tables: Dict[str, Table] = field(default_factory=dict)
    ddb_con: Optional[duckdb.DuckDBPyConnection] = None
    _catalog_path: str = "catalog.yaml"

    def __post_init__(self):
        if self.ddb_con is None:
            self.ddb_con = duckdb.connect()

    def load_catalog(self):
        with open(self._catalog_path, "r") as f:
            data = yaml.safe_load(f)

        self.filesystems = {
            name: Filesystem(**fs) for name, fs in data["filesystems"].items()
        }
        self.tables = {
            name: Table(name=name, **table) for name, table in data["tables"].items()
        }

    def save_catalog(self):
        data = {
            "filesystems": {name: vars(fs) for name, fs in self.filesystems.items()},
            "tables": {
                name: {k: v for k, v in vars(table).items() if k != "name"}
                for name, table in self.tables.items()
            },
        }
        with open(self._catalog_path, "w") as f:
            yaml.dump(data, f)

    def create_filesystem(self, name: str, protocol: str, bucket: str, path: str):
        self.filesystems[name] = Filesystem(protocol=protocol, bucket=bucket, path=path)
        self.save_catalog()

    def create_table(
        self, name: str, path: str, format: str, filesystem: str, **kwargs
    ):
        self.tables[name] = Table(
            name=name, path=path, format=format, filesystem=filesystem, **kwargs
        )
        self.save_catalog()

    def delete_table(self, name: str):
        del self.tables[name]
        self.save_catalog()

    def write_table(self, table_name: str, data, **kwargs):
        table = self.tables[table_name]
        fs = self.filesystems[table.filesystem]
        # Implement writing logic here based on the format and filesystem
        # You may need to add more methods or use external libraries for actual writing

    def read_table(self, table_name: str, **kwargs):
        table = self.tables[table_name]
        fs = self.filesystems[table.filesystem]
        # Implement reading logic here based on the format and filesystem
        # You may need to add more methods or use external libraries for actual reading

    def sql(self, query: str):
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

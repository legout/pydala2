from dataclasses import dataclass

from munch import Munch

from .dataset import ParquetDataset
from .filesystem import FileSystem
import duckdb


@dataclass
class Catalog:
    items: Munch
    ddb_con: duckdb.DuckDBPyConnection | None

    def __post_init__(self):
        if self.ddb_con is None:
            self.ddb_con = duckdb.connect()

        self.load_filesystems()

    def load_filesystems(self):
        if hasattr(self.items, "filesystem"):
            self.filesystems = Munch()
            for name in self.items.filesystem:
                fs = FileSystem(**self.items.filesystem[name])
                type(fs).protocol = name
                if self.item.filesystem[name].type != "file":
                    self.ddb_con.register_filesystem(fs)

                self.filesystems[name] = fs
        else:
            self.filesystems = None

    def load_parquet(self, name: Munch, **kwargs):
        params = self.items.tables[name]

        if "parquet" not in params.type.lower():
            return

        if "file" in params.type.lower():
            return self.filesystems[params.filesystem].read_parquet(
                params.path, **kwargs
            )

        return self.filesystems[params.filesystem].read_parquet_dataset(
            params.path, **kwargs
        )

    def load_csv(self, name: Munch, **kwargs):
        params = self.items.tables[name]

        if "csv" not in params.type.lower():
            return
        if "file" in params.type.lower():
            return self.filesystems[params.filesystem].read_csv(params.path, **kwargs)
        return self.filesystems[params.filesystem].read_csv_dataset(
            params.path, **kwargs
        )

    def load_json(self, name: Munch, **kwargs):
        params = self.items.tables[name]

        if "json" not in params.type.lower():
            return

        if "file" in params.type.lower():
            return self.filesystems[params.filesystem].read_json(params.path, **kwargs)
        return self.filesystems[params.filesystem].read_json_dataset(
            params.path, **kwargs
        )

    def load_pyarrow_dataset(self, name: Munch, **kwargs):
        params = self.items.tables[name]

        if (
            "parquet" not in params.type.lower()
            and "csv" not in params.type.lower()
            and "arrow" not in params.type.lower()
        ):
            return

        return self.filesystems[params.filesystem].pyarrow_dataset(
            params.path, **kwargs
        )

    def load_pydala_dataset(self, name: Munch, **kwargs):
        params = self.items.tables[name]

        if (
            "parquet" not in params.type.lower()
            and "pyarrow" not in params.type.lower()
            and "pydala" not in params.type.lower()
        ):
            return

        return ParquetDataset(
            params.path,
            filesystem=self.filesystems[params.filesystem],
            name=name,
            ddb_con=self.ddb_con,
            **kwargs,
        )

    def load(self, name: Munch, **kwargs):
        if "parquet" in self.items.tables[name].type.lower():
            return self.load_parquet(name, **kwargs)
        elif "csv" in self.items.tables[name].type.lower():
            return self.load_csv(name, **kwargs)
        elif "json" in self.items.tables[name].type.lower():
            return self.load_json(name, **kwargs)
        elif "pyarrow" in self.items.tables[name].type.lower():
            return self.load_pyarrow_dataset(name, **kwargs)
        elif "pydala" in self.items.tables[name].type.lower():
            return self.load_pydala_dataset(name, **kwargs)

        return None

    @property
    def table_names(self):
        return sorted(self.items.tables.keys())

from dataclasses import dataclass
import re
from munch import Munch, munchify, toYAML, from_yaml

from .dataset import ParquetDataset, PyarrowDataset, CsvDataset, JsonDataset
from .filesystem import FileSystem
from .helpers.sql import get_table_names, replace_table_names_with_file_paths
import duckdb
import yaml
from fsspec import AbstractFileSystem


# @dataclass
class Catalog:
    def __init__(
        self,
        path: str,
        ddb_con: duckdb.DuckDBPyConnection | None = None,
        filesystem: AbstractFileSystem | None = None,
        bucket: str | None = None,
        **fs_kwargs,
    ):
        self._catalog_filesystem = FileSystem(bucket=bucket, fs=filesystem, **fs_kwargs)

        with self._catalog_filesystem.open(path, "r") as f:
            self._catalog = munchify(yaml.full_load(f))

        self.ddb_con = ddb_con
        if self.ddb_con is None:
            self.ddb_con = duckdb.connect()

        self._load_filesystems()

    def _load_filesystems(self):
        if hasattr(self._catalog, "filesystem"):
            self.filesystem = Munch()

            for name in self._catalog.filesystem:
                fs = FileSystem(**self._catalog.filesystem[name])
                type(fs).protocol = name

                if self._catalog.filesystem[name].protocol != "file":
                    self.ddb_con.register_filesystem(fs)

                self.filesystem[name] = fs
        else:
            self.filesystem = None

    @property
    def list_tables(self):
        return sorted(self._catalog.table.keys())

    @property
    def table(self):
        return self._catalog.table

    def show_table(self, name: str):
        print(toYAML(self.table[name]))

    @property
    def list_filesystems(self):
        return sorted(self._catalog.filesystem.keys())

    @property
    def name(self):
        return self._catalog.name

    def load_parquet(
        self, name: Munch, as_dataset=True, with_metadata: bool = True, **kwargs
    ):
        params = self._catalog.table[name]

        if "parquet" not in params.format.lower():
            return
        if not as_dataset:
            if params.path.endswith(".parquet"):
                df = self.filesystem[params.filesystem].read_parquet(
                    params.path, **kwargs
                )
                self.ddb_con.register(df, name)
                return df

            df = self.filesystem[params.filesystem].read_parquet_dataset(
                params.path, **kwargs
            )
            self.ddb_con.register(df, name)
            return df

        if with_metadata:
            return ParquetDataset(
                params.path,
                filesystem=self.filesystem[params.filesystem],
                name=name,
                ddb_con=self.ddb_con,
                **kwargs,
            )

        return PyarrowDataset(
            params.path,
            filesystem=self.filesystem[params.filesystem],
            name=name,
            ddb_con=self.ddb_con,
            **kwargs,
        )

    def load_csv(self, name: Munch, as_dataset: bool = True, **kwargs):
        params = self._catalog.table[name]

        if "csv" not in params.format.lower():
            return
        if not as_dataset:
            if params.path.endswith(".csv"):
                df = self.filesystem[params.filesystem].read_parquet(
                    params.path, **kwargs
                )
                self.ddb_con.register(df, name)
                return df

            df = self.filesystem[params.filesystem].read_parquet_dataset(
                params.path, **kwargs
            )
            self.ddb_con.register(df, name)
            return df

        return CsvDataset(
            params.path,
            filesystem=self.filesystem[params.filesystem],
            name=name,
            ddb_con=self.ddb_con,
            **kwargs,
        )

    def load_json(self, name: Munch, as_dataset: bool = True, **kwargs):
        params = self._catalog.table[name]

        if "json" not in params.format.lower():
            return
        if not as_dataset:
            if params.path.endswith(".json"):
                df = self.filesystem[params.filesystem].read_json(params.path, **kwargs)
                self.ddb_con.register(df, name)
                return df

            df = self.filesystem[params.filesystem].read_json_dataset(
                params.path, **kwargs
            )
            self.ddb_con.register(df, name)
            return df
        return JsonDataset(
            params.path,
            filesystem=self.filesystem[params.filesystem],
            name=name,
            ddb_con=self.ddb_con,
            **kwargs,
        )

    def load(
        self, name: str, as_dataset: bool = True, with_metadata: bool = True, **kwargs
    ):
        params = self._catalog.table[name]

        if params.format.lower() == "parquet":
            return self.load_parquet(
                name, as_dataset=as_dataset, with_metadata=with_metadata, **kwargs
            )

        elif params.format.lower() == "csv":
            return self.load_csv(name, as_dataset=as_dataset, **kwargs)

        elif params.format.lower() == "json":
            return self.load_json(name, **kwargs)

        return None

    def _ddb_table_mapping(self, name: str):
        params = self._catalog.table[name]

        return {name: [params.path, params.format, params.hive_partitioning]}

    def sql(
        self, sql: str, as_dataset: bool = True, with_metadata: bool = True, **kwargs
    ):
        table_names = get_table_names(sql)

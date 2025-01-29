import posixpath

import duckdb
import pandas as pd
import pyarrow as pa
import yaml
from fsspec import AbstractFileSystem
from munch import Munch, munchify, toYAML, unmunchify

from pydala.helpers.polars import pl

from .dataset import CsvDataset, JsonDataset, ParquetDataset, PyarrowDataset
from .filesystem import FileSystem
from .helpers.misc import delattr_rec, get_nested_keys, getattr_rec, setattr_rec
from .helpers.sql import get_table_names
from .table import PydalaTable


# @dataclass
class Catalog:
    def __init__(
        self,
        path: str,
        namespace: str | None = None,
        ddb_con: duckdb.DuckDBPyConnection | None = None,
        filesystem: AbstractFileSystem | None = None,
        bucket: str | None = None,
        **fs_kwargs,
    ):
        self._catalog_filesystem = FileSystem(bucket=bucket, fs=filesystem, **fs_kwargs)
        self._catalog_path = self._catalog_filesystem.expand_path(path)[0]
        self._namespace = namespace
        self.load_catalog(namespace=namespace)

        self.ddb_con = ddb_con
        if self.ddb_con is None:
            self.ddb_con = duckdb.connect()

        self._load_filesystems()
        self.table = {}

    def load_catalog(self, namespace: str | None = None) -> Munch:
        with self._catalog_filesystem.open(self._catalog_path, "r") as f:
            catalog = munchify(yaml.full_load(f))

        for ns in catalog.tables.keys():
            if catalog.tables[ns] is None:
                catalog.tables[ns] = Munch()

        self.params = catalog

        if namespace is not None:
            self.params.tables = self.params.tables[namespace]
        else:
            self.params.tables = self.params.tables

    def _write_catalog(self, delete_table: str = None):
        with self._catalog_filesystem.open(self._catalog_path, "r") as f:
            catalog = munchify(yaml.full_load(f))
        if delete_table is not None:
            delattr_rec(catalog, delete_table)

        if self._namespace is not None:
            self.params.tables[self._namespace].update(self.params.tables)
        else:
            self.params.tables.update(self.params.tables)

        catalog.update(self.params)

        with self._catalog_filesystem.open(self._catalog_path, "w") as f:
            yaml.dump(unmunchify(catalog), f)

    def _load_filesystems(self):
        if hasattr(self.params, "filesystem"):
            self.fs = Munch()

            for name in self.params.filesystem:
                if self.params.filesystem[name].protocol in ["file", "local"]:
                    self.params.filesystem[name].bucket = posixpath.join(
                        posixpath.dirname(self._catalog_path),
                        self.params.filesystem[name].bucket,
                    )
                fs = FileSystem(**self.params.filesystem[name])
                type(fs).protocol = name

                if self.params.filesystem[name].protocol != "file":
                    self.ddb_con.register_filesystem(fs)

                self.fs[name] = fs
        else:
            self.fs = None

    # def _get_table_from_table_name(self, table_name: str) -> list[str]:
    #    table_name_items = table_name.split(".")

    # @staticmethod
    # def _get_table_from_table_name(self, table_name: str) -> str:
    #     table_name_items = table_name.split(".")

    #     if table_name_items[0] in self.list_namespaces:
    #         table_name_items.insert(1, "tables")
    #         return ".".join(table_name_items)

    #     if self._namespace is not None:
    #         table_name_items = [self._namespace, "tables"] + table_name_items
    #         return ".".join(table_name_items)

    #     return ".".join(table_name_items)

    def _get_table_params(self, table_name: str) -> Munch:
        return getattr_rec(self.params.tables, table_name)

    def _set_table_params(self, table_name: str, **fields):
        if table_name in self.all_tables:
            getattr_rec(
                self.params.tables,
                table_name,
            ).update(munchify(fields))
        else:
            setattr_rec(
                self.params.tables,
                table_name,
                munchify(fields),
            )

    @property
    def list_namespaces(self) -> list[str]:
        if self._namespace is not None:
            return [self._namespace]
        else:
            return list(self.params.tables.keys())

    @property
    def namespace(self) -> str | None:
        return self._namespace

    @property
    def all_tables(self) -> list[str]:
        return [
            t.split(".path")[0]
            for t in get_nested_keys(self.params.tables)
            if "path" in t
        ]

    def show(self, table_name: str) -> None:
        print(toYAML(self.get(table_name)))

    def get(self, table_name: str) -> Munch:
        return getattr_rec(self.params.tables, table_name)

    @property
    def all_filesystems(self) -> list[str]:
        return sorted(self.params.filesystem.keys())

    def show_filesystem(self, table_name: str) -> None:
        print(toYAML(self.params.filesystem[table_name]))

    def files(self, table_name: str) -> list[str]:
        params = self._get_table_params(table_name=table_name)
        return sorted(
            self.fs[params.filesystem].glob(params.path + f"/**/*.{params.format}")
        )

    def load_parquet(
        self, table_name: str, as_dataset=True, with_metadata: bool = True, **kwargs
    ) -> ParquetDataset | PyarrowDataset | pl.DataFrame | None:
        params = self._get_table_params(table_name=table_name)

        if "parquet" not in params.format.lower():
            return
        if not as_dataset:
            if params.path.endswith(".parquet"):
                df = self.fs[params.filesystem].read_parquet(params.path, **kwargs)
                self.ddb_con.register(table_name, df)
                return df

            df = self.fs[params.filesystem].read_parquet_dataset(params.path, **kwargs)
            self.ddb_con.register(table_name, df)
            return df

        if with_metadata:
            return ParquetDataset(
                params.path,
                filesystem=self.fs[params.filesystem],
                name=table_name,
                ddb_con=self.ddb_con,
                **kwargs,
            )

        return PyarrowDataset(
            params.path,
            filesystem=self.fs[params.filesystem],
            name=table_name,
            ddb_con=self.ddb_con,
            **kwargs,
        )

    def load_csv(
        self, table_name: str, as_dataset: bool = True, **kwargs
    ) -> CsvDataset | pl.DataFrame | None:
        params = self._get_table_params(table_name=table_name)

        if "csv" not in params.format.lower():
            return
        if not as_dataset:
            if params.path.endswith(".csv"):
                df = self.fs[params.filesystem].read_parquet(params.path, **kwargs)
                self.ddb_con.register(table_name, df)
                return df

            df = self.fs[params.filesystem].read_parquet_dataset(params.path, **kwargs)
            self.ddb_con.register(table_name, df)
            return df

        return CsvDataset(
            params.path,
            filesystem=self.fs[params.filesystem],
            name=table_name,
            ddb_con=self.ddb_con,
            **kwargs,
        )

    def load_json(
        self, table_name: str, as_dataset: bool = True, **kwargs
    ) -> JsonDataset | pl.DataFrame | None:
        params = self._get_table_params(table_name=table_name)

        if "json" not in params.format.lower():
            return
        if not as_dataset:
            if params.path.endswith(".json"):
                df = self.fs[params.filesystem].read_json(params.path, **kwargs)
                self.ddb_con.register(table_name, df)
                return df

            df = self.fs[params.filesystem].read_json_dataset(params.path, **kwargs)
            self.ddb_con.register(table_name, df)
            return df
        return JsonDataset(
            params.path,
            filesystem=self.fs[params.filesystem],
            name=table_name,
            ddb_con=self.ddb_con,
            **kwargs,
        )

    def load(
        self,
        table_name: str,
        as_dataset: bool = True,
        with_metadata: bool = True,
        reload: bool = False,
        **kwargs,
    ):
        params = self._get_table_params(table_name=table_name)

        if params.format.lower() == "parquet":
            if table_name not in self.table and not reload:
                self.table[table_name] = self.load_parquet(
                    table_name,
                    as_dataset=as_dataset,
                    with_metadata=with_metadata,
                    **kwargs,
                )
            return self.table[table_name]

        elif params.format.lower() == "csv":
            if table_name not in self.table and not reload:
                self.table[table_name] = self.load_csv(
                    table_name, as_dataset=as_dataset, **kwargs
                )
            return self.table[table_name]

        elif params.format.lower() == "json":
            if table_name not in self.table and not reload:
                self.table[table_name] = self.load_json(table_name, **kwargs)
            return self.table[table_name]

        # return None

    # def _ddb_table_mapping(self, table_name: str):
    #     params = getattr_rec(self._catalog, self._get_table_from_table_name(table_name=table_name))

    #     return {table_name: [params.path, params.format, params.hive_partitioning]}

    @property
    def registered_tables(self):
        return self.ddb_con.sql("SHOW TABLES")

    def sql(
        self, sql: str, as_dataset: bool = True, with_metadata: bool = True, **kwargs
    ):
        table_names = get_table_names(sql)
        registerd_tables = list(map(lambda x: x[0], self.registered_tables.fetchall()))

        for name in table_names:
            if name not in registerd_tables:
                self.load(
                    name, as_dataset=as_dataset, with_metadata=with_metadata, **kwargs
                )

        return self.ddb_con.sql(sql)

    def create_namespace(self, name: str):
        if name in self.params.tables:
            self.params.tables[name] = Munch()

    def create_table(
        self,
        data: pl.DataFrame
        | pa.Table
        | pa.RecordBatch
        | pa.RecordBatchReader
        | pd.DataFrame
        | duckdb.DuckDBPyRelation
        | ParquetDataset
        | PydalaTable
        | CsvDataset
        | JsonDataset
        | PyarrowDataset
        | None = None,
        table_name: str | None = None,
        namespace: str | None = None,
        overwrite: bool = False,
        write_catalog: bool = True,
        write_args: dict | None = None,
        **fields,
    ):
        if isinstance(data, ParquetDataset | PyarrowDataset | CsvDataset | JsonDataset):
            if table_name is None:
                table_name = data.name

            fields["timestamp_column"] = fields.get(
                "timestamp_column", data._timestamp_column
            )
            fields["path"] = fields.get("path", data._path)
            fields["format"] = fields.get("format", data._format)
            fields["partitioning"] = fields.get("partitioning", data._partitioning)
            fields["partitioning_columns"] = fields.get(
                "partitioning_columns", data._partitioning_columns
            )

        if table_name is None:
            raise Exception("table_name is required")

        if namespace is not None and namespace not in table_name:
            table_name = ".".join(
                namespace.split(".") + table_name.split(".")
            )  # f"{namespace}.{table_name}"

        if table_name in self.all_tables:
            if not overwrite:
                raise Exception(
                    f"Table {table_name} already exists. Use overwrite=True to overwrite or use `write_table` to write "
                    "new data (mode='append' or 'delta') to the table."
                )

        if write_args is not None:
            fields["write_args"] = write_args

        self._set_table_params(table_name=table_name, **fields)

        if write_catalog:
            self._write_catalog()

        if data is not None:
            if isinstance(
                data, ParquetDataset | PyarrowDataset | CsvDataset | JsonDataset
            ):
                if fields.get("path") == data._path:
                    return
            self.write_table(data, table_name, **write_args)

    def delete_table(
        self,
        table_name: str | None = None,
        namespace: str | None = None,
        write_catalog: bool = True,
        vacuum: bool = False,
    ):
        if namespace is not None:
            if table_name is None:
                table_name = namespace
            else:
                if namespace not in table_name:
                    table_name = f"{namespace}.{table_name}"

        if table_name in self.all_tables:
            if vacuum:
                self.load(table_name).vacuum()

            delattr_rec(self.params.tables, table_name)
            if self._namespace is not None:
                delattr_rec(self.params[self._namespace].tables, table_name)
            else:
                delattr_rec(self.params.tables, table_name)

            if write_catalog:
                self._write_catalog(delete_table=table_name)

    def update(
        self,
        table_name: str,
        write_catalog: bool = True,
        write_args: dict | None = None,
        **fields,
    ):
        self.load_catalog()
        if write_args is not None:
            fields["write_args"] = write_args
        self._set_table_params(table_name=table_name, **fields)

        if write_catalog:
            self._write_catalog()

    def write_table(
        self,
        data: pl.DataFrame
        | pa.Table
        | pa.RecordBatch
        | pa.RecordBatchReader
        | pd.DataFrame
        | duckdb.DuckDBPyRelation
        | ParquetDataset
        | PydalaTable
        | CsvDataset
        | JsonDataset
        | PyarrowDataset,
        table_name: str,
        as_dataset: bool = True,
        with_metadata: bool = True,
        update_catalog: bool = False,
        **kwargs,
    ):
        params = self._get_table_params(table_name=table_name)

        if kwargs:
            if "write_args" in params:
                params.write_args.update(kwargs)
            else:
                params.write_args = kwargs
            if update_catalog:
                self.update(table_name=table_name, write_args=params.write_args)

        if as_dataset:
            if table_name not in self.table:
                self.load(table_name=table_name, with_metadata=with_metadata)

            self.table[table_name].write_to_dataset(data, **params.write_args)
        else:
            # raise("Not implemented yet. You can use ")
            if params.format.lower() == "parquet":
                self.fs[params.filesystem].write_parquet(data, params.path, **kwargs)
            elif params.format.lower() == "csv":
                self.fs[params.filesystem].write_csv(data, params.path, **kwargs)
            elif params.format.lower() == "json":
                self.fs[params.filesystem].write_json(data, params.path, **kwargs)

    def schema(self, table_name: str):
        self.load(table_name).schema

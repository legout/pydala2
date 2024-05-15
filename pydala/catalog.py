import duckdb
import yaml
from fsspec import AbstractFileSystem
from munch import Munch, munchify, toYAML, unmunchify

from pydala.helpers.polars_ext import pl

from .dataset import CsvDataset, JsonDataset, ParquetDataset, PyarrowDataset
from .filesystem import FileSystem
from .helpers.misc import delattr_rec, get_nested_keys, getattr_rec, setattr_rec
from .helpers.sql import get_table_names


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
        self._catalog_path = path
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

        self._catalog = catalog

        if namespace is not None:
            self._tables = self._catalog.tables[namespace]
        else:
            self._tables = self._catalog.tables

    def _write_catalog(self, delete_table: str = None):
        with self._catalog_filesystem.open(self._catalog_path, "r") as f:
            catalog = munchify(yaml.full_load(f))
        if delete_table is not None:
            delattr_rec(catalog, delete_table)

        if self._namespace is not None:
            self._catalog.tables[self._namespace].update(self._tables)
        else:
            self._catalog.tables.update(self._tables)

        catalog.update(self._catalog)

        with self._catalog_filesystem.open(self._catalog_path, "w") as f:
            yaml.dump(unmunchify(catalog), f)

    def _load_filesystems(self):
        if hasattr(self._catalog, "filesystem"):
            self._filesystem = Munch()

            for name in self._catalog.filesystem:
                fs = FileSystem(**self._catalog.filesystem[name])
                type(fs).protocol = name

                if self._catalog.filesystem[name].protocol != "file":
                    self.ddb_con.register_filesystem(fs)

                self._filesystem[name] = fs
        else:
            self._filesystem = None

    # @staticmethod
    # def _get_table_from_identifier(self, identifier: str) -> str:
    #     identifier_items = identifier.split(".")

    #     if identifier_items[0] in self.list_namespaces:
    #         identifier_items.insert(1, "tables")
    #         return ".".join(identifier_items)

    #     if self._namespace is not None:
    #         identifier_items = [self._namespace, "tables"] + identifier_items
    #         return ".".join(identifier_items)

    #     return ".".join(identifier_items)

    def _get_table_params(self, identifier: str) -> Munch:
        return getattr_rec(self._tables, identifier)

    def _set_table_params(self, identifier: str, **fields):
        if identifier in self.all_tables:
            getattr_rec(
                self._tables,
                identifier,
            ).update(munchify(fields))
        else:
            setattr_rec(
                self._tables,
                identifier,
                munchify(fields),
            )

    @property
    def list_namespaces(self) -> list[str]:
        if self._namespace is not None:
            return [self._namespace]
        else:
            return list(self._tables.keys())

    @property
    def namespace(self) -> str | None:
        return self._namespace

    @property
    def all_tables(self) -> list[str]:
        return [
            t.split(".path")[0] for t in get_nested_keys(self._tables) if "path" in t
        ]

    def show_table(self, identifier: str) -> None:
        print(toYAML(eval(f"self._tables.{identifier}")))

    @property
    def all_filesystems(self) -> list[str]:
        return sorted(self._catalog.filesystem.keys())

    def show_filesystem(self, identifier: str) -> None:
        print(toYAML(self._catalog.filesystem[identifier]))

    def files(self, identifier: str) -> list[str]:
        params = self._get_table_params(identifier=identifier)
        return sorted(
            self._filesystem[params.filesystem].glob(
                params.path + f"/**/*.{params.format}"
            )
        )

    def load_parquet(
        self, identifier: str, as_dataset=True, with_metadata: bool = True, **kwargs
    ) -> ParquetDataset | PyarrowDataset | pl.DataFrame | None:
        params = self._get_table_params(identifier=identifier)

        if "parquet" not in params.format.lower():
            return
        if not as_dataset:
            if params.path.endswith(".parquet"):
                df = self._filesystem[params.filesystem].read_parquet(
                    params.path, **kwargs
                )
                self.ddb_con.register(identifier, df)
                return df

            df = self._filesystem[params.filesystem].read_parquet_dataset(
                params.path, **kwargs
            )
            self.ddb_con.register(identifier, df)
            return df

        if with_metadata:
            return ParquetDataset(
                params.path,
                filesystem=self._filesystem[params.filesystem],
                name=identifier,
                ddb_con=self.ddb_con,
                **kwargs,
            )

        return PyarrowDataset(
            params.path,
            filesystem=self._filesystem[params.filesystem],
            name=identifier,
            ddb_con=self.ddb_con,
            **kwargs,
        )

    def load_csv(
        self, identifier: str, as_dataset: bool = True, **kwargs
    ) -> CsvDataset | pl.DataFrame | None:
        params = self._get_table_params(identifier=identifier)

        if "csv" not in params.format.lower():
            return
        if not as_dataset:
            if params.path.endswith(".csv"):
                df = self._filesystem[params.filesystem].read_parquet(
                    params.path, **kwargs
                )
                self.ddb_con.register(identifier, df)
                return df

            df = self._filesystem[params.filesystem].read_parquet_dataset(
                params.path, **kwargs
            )
            self.ddb_con.register(identifier, df)
            return df

        return CsvDataset(
            params.path,
            filesystem=self._filesystem[params.filesystem],
            name=identifier,
            ddb_con=self.ddb_con,
            **kwargs,
        )

    def load_json(
        self, identifier: str, as_dataset: bool = True, **kwargs
    ) -> JsonDataset | pl.DataFrame | None:
        params = self._get_table_params(identifier=identifier)

        if "json" not in params.format.lower():
            return
        if not as_dataset:
            if params.path.endswith(".json"):
                df = self._filesystem[params.filesystem].read_json(
                    params.path, **kwargs
                )
                self.ddb_con.register(identifier, df)
                return df

            df = self._filesystem[params.filesystem].read_json_dataset(
                params.path, **kwargs
            )
            self.ddb_con.register(identifier, df)
            return df
        return JsonDataset(
            params.path,
            filesystem=self._filesystem[params.filesystem],
            name=identifier,
            ddb_con=self.ddb_con,
            **kwargs,
        )

    def load(
        self,
        identifier: str,
        as_dataset: bool = True,
        with_metadata: bool = True,
        reload: bool = False,
        **kwargs,
    ):
        params = self._get_table_params(identifier=identifier)

        if params.format.lower() == "parquet":
            if identifier not in self.table and not reload:
                self.table[identifier] = self.load_parquet(
                    identifier,
                    as_dataset=as_dataset,
                    with_metadata=with_metadata,
                    **kwargs,
                )
            return self.table[identifier]

        elif params.format.lower() == "csv":
            if identifier not in self.table and not reload:
                self.table[identifier] = self.load_csv(
                    identifier, as_dataset=as_dataset, **kwargs
                )
            return self.table[identifier]

        elif params.format.lower() == "json":
            if identifier not in self.table and not reload:
                self.table[identifier] = self.load_json(identifier, **kwargs)
            return self.table[identifier]

        # return None

    # def _ddb_table_mapping(self, identifier: str):
    #     params = getattr_rec(self._catalog, self._get_table_from_identifier(identifier=identifier))

    #     return {identifier: [params.path, params.format, params.hive_partitioning]}

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
        if name in self._tables:
            self._tables[name] = Munch()

    def create_table(
        self,
        identifier: str,
        namespace: str | None = None,
        overwrite: bool = False,
        write_catalog: bool = True,
        **fields,
    ):
        if namespace is not None and namespace not in identifier:
            identifier = f"{namespace}.{identifier}"
        if identifier in self.all_tables:
            if not overwrite:
                raise Exception(
                    f"Table {identifier} already exists. Use overwrite=True to overwrite"
                )
        self._set_table_params(identifier=identifier, **fields)

        if write_catalog:
            self._write_catalog()

    def delete_table(
        self,
        identifier: str | None = None,
        namespace: str | None = None,
        write_catalog: bool = True,
        vacuum: bool = False,
    ):
        if namespace is not None:
            if identifier is None:
                identifier = namespace
            else:
                if namespace not in identifier:
                    identifier = f"{namespace}.{identifier}"

        if identifier in self.all_tables:
            if vacuum:
                self.load(identifier).vacuum()

            delattr_rec(self._tables, identifier)
            if self._namespace is not None:
                delattr_rec(self._catalog[self._namespace].tables, identifier)
            else:
                delattr_rec(self._catalog.tables, identifier)

            if write_catalog:
                self._write_catalog(delete_table=identifier)

    def update_table(
        self,
        identifier: str,
        write_catalog: bool = True,
        **fields,
    ):
        self.load_catalog()
        self._set_table_params(identifier=identifier, **fields)

        if write_catalog:
            self._write_catalog()

    def write_to_table(self, identifier: str, df, **kwargs):
        params = self._get_table_params(identifier=identifier)

        if kwargs:
            if "write_args" in params:
                params.write_args.update(kwargs)
            else:
                params.write_args = kwargs
            self.update_table(identifier=identifier, write_args=params.write_args)

        if identifier not in self.table:
            self.load(identifier=identifier)

        self.table[identifier].write_to_dataset(df, **params.write_args)

    def schema(self, name: str):
        self.load(name).schema

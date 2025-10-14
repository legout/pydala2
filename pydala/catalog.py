# Standard library imports
import posixpath

# Third-party imports
import duckdb
import pandas as pd
import pyarrow as pa
import yaml
from fsspec import AbstractFileSystem
from munch import Munch, munchify, toYAML, unmunchify

# Local imports
from pydala.helpers.polars import pl
from .dataset import CSVDataset, JSONDataset, ParquetDataset, PyarrowDataset
from .filesystem import FileSystem
from .helpers.misc import delattr_rec, get_nested_keys, getattr_rec, setattr_rec
from .helpers.sql import get_table_names
from .table import PydalaTable


# @dataclass
class Catalog:
    """A catalog for managing datasets and their configurations.

    This class provides a centralized way to manage multiple datasets across
    different namespaces and filesystems. It maintains a YAML configuration
    file that tracks table locations, formats, and other metadata.

    Attributes:
        _catalog_filesystem: Filesystem for catalog operations.
        _catalog_path: Path to the catalog configuration file.
        _namespace: Current namespace for table operations.
        params: Catalog configuration parameters.
        ddb_con: DuckDB connection for SQL operations.
        fs: Available filesystems.
        table: Loaded table instances.
    """

    def __init__(
        self,
        path: str,
        namespace: str | None = None,
        ddb_con: duckdb.DuckDBPyConnection | None = None,
        filesystem: AbstractFileSystem | None = None,
        bucket: str | None = None,
        **fs_kwargs,
    ) -> None:
        """Initialize a Catalog instance.

        Args:
            path: Path to the catalog configuration file (YAML).
            namespace: Namespace to scope table operations to.
            ddb_con: Existing DuckDB connection. If None, creates a new one.
            filesystem: Default filesystem for catalog operations.
            bucket: Bucket name for cloud storage.
            **fs_kwargs: Additional filesystem configuration.
        """
        self._catalog_filesystem = FileSystem(bucket=bucket, fs=filesystem, **fs_kwargs)
        self._catalog_path = self._catalog_filesystem._strip_protocol(
            self._catalog_filesystem.expand_path(path)[0]
        )
        self._namespace = namespace
        self.load_catalog(namespace=namespace)

        self.ddb_con = ddb_con
        if self.ddb_con is None:
            self.ddb_con = duckdb.connect()

        self._load_filesystems()
        self.table = {}

    def load_catalog(self, namespace: str | None = None) -> Munch:
        """Load catalog configuration from YAML file.

        Args:
            namespace: If specified, loads only tables from this namespace.

        Returns:
            The loaded catalog configuration as a Munch object.

        Raises:
            FileNotFoundError: If the catalog file doesn't exist.
            yaml.YAMLError: If the catalog file is invalid YAML.

        Note:
            The catalog structure should follow the format:
                tables:
                    namespace1:
                        table1:
                            path: /path/to/table1
                            format: parquet
                            options: {...}
                    namespace2:
                        ...
        """
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

    def _write_catalog(self, delete_table: str = None) -> None:
        """Write the catalog configuration to disk.

        This method loads the existing catalog from disk, applies any updates
        (including table deletions), and writes it back to the filesystem.

        Args:
            delete_table: If specified, removes this table from the catalog
                before writing. Format should be 'namespace.table_name'.

        Raises:
            OSError: If there's an error writing to the filesystem.
            yaml.YAMLError: If there's an error serializing the catalog.
        """
        with self._catalog_filesystem.open(self._catalog_path, "r") as f:
            catalog = munchify(yaml.full_load(f))

        if delete_table is not None:
            delattr_rec(catalog, delete_table)

        # Update the catalog with current parameters.
        # This ensures any runtime changes to table configurations are persisted.
        # Note: self.params may contain modifications made during the session.
        catalog.update(self.params)

        with self._catalog_filesystem.open(self._catalog_path, "w") as f:
            yaml.dump(unmunchify(catalog), f)

    def _load_filesystems(self) -> None:
        """Load and configure filesystems from catalog settings.

        This method reads filesystem configurations from the catalog,
        creates FileSystem instances, and registers them with DuckDB
        for cloud storage access.

        Returns:
            None
        """
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
        """Get parameters for a specific table.

        Args:
            table_name: Name of the table to get parameters for.

        Returns:
            Table configuration parameters as a Munch object.
        """
        return getattr_rec(self.params.tables, table_name)

    def _set_table_params(self, table_name: str, **fields) -> None:
        """Set or update parameters for a specific table.

        Args:
            table_name: Name of the table to update.
            **fields: Key-value pairs of parameters to set.

        Returns:
            None
        """
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
        """Get list of available namespaces.

        Returns:
            List of namespace names. If a namespace is set for the
            catalog instance, returns only that namespace.
        """
        if self._namespace is not None:
            return [self._namespace]
        else:
            return list(self.params.tables.keys())

    @property
    def namespace(self) -> str | None:
        """Get the current namespace.

        Returns:
            Current namespace name, or None if no namespace is set.
        """
        return self._namespace

    @property
    def all_tables(self) -> list[str]:
        """Get list of all table names across all namespaces.

        Returns:
            List of table names that have paths defined.
        """
        return [
            t.split(".path")[0]
            for t in get_nested_keys(self.params.tables)
            if "path" in t
        ]

    def show(self, table_name: str) -> None:
        """Display table configuration in YAML format.

        Args:
            table_name: Name of the table to display.

        Returns:
            None
        """
        print(toYAML(self.get(table_name)))

    def get(self, table_name: str) -> Munch:
        """Get table configuration.

        Args:
            table_name: Name of the table to get configuration for.

        Returns:
            Table configuration as a Munch object.
        """
        return getattr_rec(self.params.tables, table_name)

    @property
    def all_filesystems(self) -> list[str]:
        """Get list of all configured filesystem names.

        Returns:
            Sorted list of filesystem names from the catalog configuration.
        """
        return sorted(self.params.filesystem.keys())

    def show_filesystem(self, table_name: str) -> None:
        """Display filesystem configuration in YAML format.

        Args:
            table_name: Name of the filesystem configuration to display.

        Returns:
            None
        """
        print(toYAML(self.params.filesystem[table_name]))

    def files(self, table_name: str) -> list[str]:
        """Get list of files for a table.

        Args:
            table_name: Name of the table to get files for.

        Returns:
            Sorted list of file paths matching the table's format.
        """
        params = self._get_table_params(table_name=table_name)
        return sorted(
            self.fs[params.filesystem].glob(params.path + f"/**/*.{params.format}")
        )

    def load_parquet(
        self, table_name: str, as_dataset=True, with_metadata: bool = True, **kwargs
    ) -> ParquetDataset | PyarrowDataset | pl.DataFrame | None:
        """Load a Parquet table from the catalog.

        Args:
            table_name: Name of the table to load.
            as_dataset: If True, returns a Dataset object. If False, returns a DataFrame.
            with_metadata: If True and as_dataset=True, uses ParquetDataset with metadata.
            **kwargs: Additional arguments passed to the dataset constructor.

        Returns:
            ParquetDataset, PyarrowDataset, or DataFrame depending on parameters,
            or None if table is not in Parquet format.
        """
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
    ) -> CSVDataset | pl.DataFrame | None:
        """Load a CSV table from the catalog.

        Args:
            table_name: Name of the table to load.
            as_dataset: If True, returns a CSVDataset. If False, returns a DataFrame.
            **kwargs: Additional arguments passed to the dataset constructor.

        Returns:
            CSVDataset or DataFrame depending on as_dataset parameter,
            or None if table is not in CSV format.
        """
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

        return CSVDataset(
            params.path,
            filesystem=self.fs[params.filesystem],
            name=table_name,
            ddb_con=self.ddb_con,
            **kwargs,
        )

    def load_json(
        self, table_name: str, as_dataset: bool = True, **kwargs
    ) -> JSONDataset | pl.DataFrame | None:
        """Load a JSON table from the catalog.

        Args:
            table_name: Name of the table to load.
            as_dataset: If True, returns a JSONDataset. If False, returns a DataFrame.
            **kwargs: Additional arguments passed to the dataset constructor.

        Returns:
            JSONDataset or DataFrame depending on as_dataset parameter,
            or None if table is not in JSON format.
        """
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
        return JSONDataset(
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
    ) -> ParquetDataset | CSVDataset | JSONDataset | pl.DataFrame | None:
        """Load a table from the catalog, automatically detecting format.

        This method caches loaded tables to avoid repeated loading. Tables
        are loaded based on their format specified in the catalog.

        Args:
            table_name: Name of the table to load.
            as_dataset: If True, returns a Dataset object. If False, returns a DataFrame.
            with_metadata: For Parquet tables, whether to use metadata-aware dataset.
            reload: If True, forces reload even if table is already cached.
            **kwargs: Additional arguments passed to the dataset constructor.

        Returns:
            Dataset or DataFrame object depending on parameters and table format,
            or None if table format is unsupported.
        """
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
    def registered_tables(self) -> duckdb.DuckDBPyRelation:
        """Get list of tables registered with DuckDB.

        Returns:
            DuckDB relation containing information about registered tables.
        """
        return self.ddb_con.sql("SHOW TABLES")

    def sql(
        self, sql: str, as_dataset: bool = True, with_metadata: bool = True, **kwargs
    ) -> duckdb.DuckDBPyRelation:
        """Execute SQL query with automatic table loading.

        This method parses the SQL query to identify table names,
        loads any referenced tables that aren't already registered,
        then executes the query.

        Args:
            sql: SQL query to execute.
            as_dataset: Whether to load tables as datasets.
            with_metadata: For Parquet tables, whether to use metadata.
            **kwargs: Additional arguments passed to load method.

        Returns:
            DuckDB relation containing the query results.
        """
        table_names = get_table_names(sql)
        registerd_tables = list(map(lambda x: x[0], self.registered_tables.fetchall()))

        for name in table_names:
            if name not in registerd_tables:
                self.load(
                    name, as_dataset=as_dataset, with_metadata=with_metadata, **kwargs
                )

        return self.ddb_con.sql(sql)

    def create_namespace(self, name: str) -> None:
        """Create a new namespace in the catalog.

        Args:
            name: Name of the namespace to create.

        Returns:
            None
        """
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
        | CSVDataset
        | JSONDataset
        | PyarrowDataset
        | None = None,
        table_name: str | None = None,
        namespace: str | None = None,
        overwrite: bool = False,
        write_catalog: bool = True,
        write_args: dict | None = None,
        **fields,
    ) -> None:
        """Create a new table entry in the catalog and optionally write data.

        This method can create a table entry from existing data or as a
        placeholder for future data. It supports multiple data formats
        and can automatically extract metadata from Dataset objects.

        Args:
            data: Data to write. Can be None to create a placeholder.
            table_name: Name for the table. If None and data is a Dataset,
                       uses the dataset's name.
            namespace: Namespace to create the table in.
            overwrite: Whether to overwrite an existing table.
            write_catalog: Whether to persist changes to catalog file.
            write_args: Arguments for writing data (passed to write_to_dataset).
            **fields: Additional table configuration fields.

        Raises:
            Exception: If table_name is required but not provided.
            Exception: If table exists and overwrite is False.
        """
        if isinstance(data, ParquetDataset | PyarrowDataset | CSVDataset | JSONDataset):
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
                data, ParquetDataset | PyarrowDataset | CSVDataset | JSONDataset
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
    ) -> None:
        """Delete a table from the catalog and optionally remove its data.

        Args:
            table_name: Name of the table to delete.
            namespace: Namespace containing the table.
            write_catalog: Whether to persist changes to catalog file.
            vacuum: Whether to delete all data files for the table.

        Returns:
            None
        """
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
    ) -> None:
        """Update table configuration in the catalog.

        This method reloads the catalog to ensure fresh data, then
        updates the specified table's configuration with new field values.

        Args:
            table_name: Name of the table to update.
            write_catalog: Whether to persist changes to catalog file.
            write_args: New write arguments to merge with existing ones.
            **fields: Table configuration fields to update.

        Returns:
            None
        """
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
        | CSVDataset
        | JSONDataset
        | PyarrowDataset,
        table_name: str,
        as_dataset: bool = True,
        with_metadata: bool = True,
        update_catalog: bool = False,
        **kwargs,
    ) -> None:
        """Write data to a table defined in the catalog.

        This method writes data using the table's configuration from
        the catalog. It can update write arguments in the catalog
        and supports both dataset and direct file writing.

        Args:
            data: Data to write to the table.
            table_name: Name of the destination table.
            as_dataset: Whether to load as dataset for writing.
            with_metadata: For Parquet, whether to use metadata features.
            update_catalog: Whether to update catalog with new write args.
            **kwargs: Additional write arguments.

        Returns:
            None
        """
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

    def schema(self, table_name: str) -> pa.Schema:
        """Get the schema of a table.

        Args:
            table_name: Name of the table to get schema for.

        Returns:
            PyArrow schema of the table.
        """
        return self.load(table_name).schema

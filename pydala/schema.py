import os

import pyarrow as pa
import pyarrow.fs as pfs
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem

from .helpers.misc import run_parallel

# from .io import read_table, write_table


def sort_schema(schema: pa.Schema, names: list[str] | None = None) -> pa.Schema:
    """Sort fields of a pyarrow schema in alphabetical order.

    Args:
        schema (pa.Schema): Pyarrow schema.

    Returns:
        pa.Schema: Sorted pyarrow schema.
    """
    if names is not None:
        names = names + [name for name in schema.names if name not in names]
        names = [name for name in names if name in schema.names]
    else:
        names = schema.names

    return pa.schema(
        [pa.field(name, type_) for name, type_ in sorted(zip(names, schema.types))]
    )


def shrink_large_string(schema: pa.Schema) -> pa.Schema:
    """Convert all large_string types to string in a pyarrow.schema.

    Args:
        schema (pa.Schema): pyarrow schema

    Returns:
        pa.Schema: converted pyarrow.schema
    """
    return pa.schema(
        [
            (n, pa.utf8()) if t == pa.large_string() else (n, t)
            for n, t in list(zip(schema.names, schema.types))
        ]
    )


def convert_timestamp(
    schema: pa.Schema,
    timestamp_fields: str | list[str] | None = None,
    unit: str | None = "us",
    tz: str | None = None,
    remove_tz: bool = False,
) -> pa.Schema:
    """Convert timestamp based on given unit and timezone.

    Args:
        schema (pa.Schema): Pyarrow schema.
        timestamp_fields (str | list[str] | None, optional): timestamp fields. Defaults to None.
        unit (str | None): timestamp unit. Defaults to "us".
        tz (None, optional): timezone. Defaults to None.
        remove_tz (str | None): Wheter to remove the timezone from the timestamp or not. Defaults to False.

    Returns:
        pa.Schema: Pyarrow schema with converted timestamp(s).
    """
    if timestamp_fields is None:
        timestamp_fields = [
            f
            for f in schema.names
            if isinstance(schema.field(f).type, pa.TimestampType)
        ]

    if isinstance(timestamp_fields, str):
        timestamp_fields = [timestamp_fields]

    for timestamp_field in timestamp_fields:
        timestamp_field_idx = schema.get_field_index(timestamp_field)
        unit = unit or schema.field(timestamp_field).type.unit
        tz = tz or schema.field(timestamp_field).type.tz
        if remove_tz:
            tz = None
        schema = schema.remove(timestamp_field_idx).insert(
            timestamp_field_idx,
            pa.field(timestamp_field, pa.timestamp(unit=unit, tz=tz)),
        )

    return schema


def replace_field(schema: pa.Schema, field: str, dtype: pa.DataType) -> pa.Schema:
    """Replace the dtype of a field in a pyarrow schema.

    Args:
        schema (pa.Schema): Pyarrow schema.
        field (str): Field name.
        dtype (pa.DataType): Pyarrow data type.

    Returns:
        pa.Schema: Pyarrow schema with replaced dtype.
    """
    if field in schema.names:
        field_idx = schema.get_field_index(field)
        return schema.remove(field_idx).insert(field_idx, pa.field(field, dtype))


def append_field(schema: pa.Schema, field: str, dtype: pa.DataType) -> pa.Schema:
    """
    Append a new field to the given schema.

    Args:
        schema (pa.Schema): The schema to append the field to.
        field (str): The name of the field to append.
        dtype (pa.DataType): The data type of the field to append.

    Returns:
        pa.Schema: The updated schema with the new field appended.
    """
    if field not in schema.names:
        return schema.append(pa.field(field, dtype))
    return schema


def modify_field(schema: pa.Schema, field: str, dtype: pa.DataType) -> pa.Schema:
    """
    Modify a field in the given schema.

    Args:
        schema (pa.Schema): The schema to modify the field in.
        field (str): The name of the field to modify.
        dtype (pa.DataType): The data type of the field to modify.

    Returns:
        pa.Schema: The updated schema with the field modified.
    """
    if field in schema.names:
        return replace_field(schema, field, dtype)
    return append_field(schema, field, dtype)


def replace_schema(
    table: pa.Table,
    schema: pa.Schema | None = None,
    field_dtype: dict | None = None,
    # exact: bool = False,
    alter_schema: bool = False,
) -> pa.Table:
    """
    Replaces the schema of a PyArrow table with the specified schema or modifies the existing schema.

    Args:
        table (pa.Table): The PyArrow table to modify.
        schema (pa.Schema | None, optional): The new schema to replace the existing schema. Defaults to None.
        field_dtype (dict | None, optional): A dictionary mapping field names to their new data types. Defaults to None.
        alter_schema (bool, optional): If True, adds missing fields to the table based on the new schema. 
            If False, drops fields from the table that are not present in the new schema. Defaults to False.

    Returns:
        pa.Table: The modified table with the replaced or modified schema.
    """
    
    if field_dtype is not None:
        schema = schema or table.schema
        for field, dtype in field_dtype.items():
            schema = modify_field(schema=schema, field=field, dtype=dtype)

    schema = schema or table.schema

    if alter_schema:
        missing_fields = [
            field for field in schema.names if field not in table.column_names
        ]

        for field in missing_fields:
            table = table.append_column(
                field, pa.array([None] * len(table), type=schema.field(field).type)
            )
    else:
        table = table.drop(
            [field for field in table.column_names if field not in schema.names]
        )

    return table.select(schema.names).cast(schema)


def _unify_schemas(
    schema1: pa.Schema,
    schema2: pa.Schema,
    ts_unit: str | None = "us",
    tz: str | None = None,
    use_large_string: bool = False,
    sort: bool | list[str] | str = False,
) -> tuple[dict, bool]:
    """Returns a unified pyarrow schema.

    Args:
        schema1 (pa.Schema): Pyarrow schema 1.
        schema2 (pa.Schema): Pyarrow schema 2.
        ts_unit (str|None): timestamp unit.
        tz (str|None): timezone for timestamp fields. Defaults to "UTC".
        use_large_string (bool): Convert pyarrow.large_string() to pyarrow.string().
            Defaults to False.
        sort (bool | list[str]): Use this, to define the field(column) order. If it is
            True, the fields are sorted alphanumerically, if it is False, the field
            order of firt schema is used. Otherwise, the order if the given field names is
            applied.
    Returns:
        Tuple[dict, bool]: Unified pyarrow schema, bool value if schemas were equal.
    """

    dtypes = [
        pa.null(),
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
        pa.float16(),
        pa.float32(),
        pa.float64(),
        pa.string(),
        pa.utf8(),
        pa.large_string(),
        pa.large_utf8(),
    ]
    timestamp_units = ["ns", "us", "ms", "s"]
    # string_dtypes = [pa.string(), pa.utf8(), pa.large_string(), pa.large_utf8()]

    # check for equal columns and column order
    if schema1.names == schema2.names:
        # if schema1.types == schema2.types:
        #    return schema1, True

        all_names = schema1.names
        file_schemas_equal = True

    elif sorted(schema1.names) == sorted(schema2.names):
        all_names = schema1.names
        file_schemas_equal = False

    else:
        all_names = list(set(schema1.names + schema2.names))
        file_schemas_equal = False

    if sort:
        if isinstance(sort, bool):
            all_names = sorted(all_names)
        elif isinstance(sort, list[str]):
            sort = [name for name in sort if name in all_names]
            all_names = sort + [name for name in all_names if name not in sort]

    schema = []
    for name in all_names:
        if name in schema1.names:
            type1 = schema1.field(name).type
        else:
            type1 = schema2.field(name).type
        if name in schema2.names:
            type2 = schema2.field(name).type
        else:
            type2 = schema1.field(name).type

        if type1 != type2:
            if type1 in dtypes:
                rank1 = dtypes.index(type1) if type1 in dtypes else 0
                rank2 = dtypes.index(type2) if type2 in dtypes else 0

                if type1.num_buffers == 2 and type2.num_buffers == 2:
                    bit_width1 = type1.bit_width
                    bit_width2 = type2.bit_width
                    if bit_width1 > bit_width2:
                        if rank1 > rank2:
                            type_ = type1
                        else:
                            type_ = pa.float64()
                        # (
                        #     pa.float16()
                        #     if bit_width1 == 16
                        #     else pa.float32()
                        #     if bit_width1 == 32
                        #     else pa.float64()
                        # )
                    else:
                        if rank2 > rank1:
                            type_ = type2
                        else:
                            type_ = pa.float64()
                else:
                    type_ = type1 if rank1 > rank2 else type2

            elif isinstance(type1, pa.TimestampType):
                rank1 = (
                    timestamp_units.index(type1.unit)
                    if isinstance(type1, pa.TimestampType)
                    else 0
                )
                rank2 = (
                    timestamp_units.index(type2.unit)
                    if isinstance(type1, pa.TimestampType)
                    else 0
                )
                type_ = type1 if rank1 > rank2 else type2
            else:
                rank1 = 1
                rank2 = 0
                type_ = type1 if rank1 > rank2 else type2
            schema.append(pa.field(name, type_))

        else:
            schema.append(pa.field(name, type1))

    schema = pa.schema(schema)

    if ts_unit is not None or tz is not None:
        schema = convert_timestamp(schema, unit=ts_unit, tz=tz)

    if not use_large_string:
        schema = shrink_large_string(schema)

    if schema != schema1 or schema != schema2:
        file_schemas_equal = False

    return schema, file_schemas_equal


def unify_schemas(
    schemas: list[pa.Schema],
    ts_unit: str | None = "us",
    tz: str | None = None,
    use_large_string: bool = False,
    sort: bool | list[str] = False,
) -> tuple[pa.Schema, bool]:
    """Get the unified pyarrow schema for a list of schemas.

    Args:
        schemas (list[pa.Schema]): Pyarrow schemas.
        ts_unit (str|None): timestamp unit.
        tz (str|None): timezone for timestamp fields. Defaults to "UTC".
        use_large_string (bool): Convert pyarrow.large_string() to pyarrow.string().
            Defaults to False.
        sort (bool | list[str]): Use this, to define the field(column) order. If it is
            True, the fields are sorted alphanumerically, if it is False, the field
            order of firt schema is used. Otherwise, the order if the given field names is
            applied.
    Returns:
        tuple[pa.Schema, bool]: Unified pyarrow schema.
    """

    schemas_equal = True
    unified_schema = schemas[0]
    for schema in schemas[1:]:
        unified_schema, schemas_equal_ = _unify_schemas(
            unified_schema,
            schema,
            ts_unit=ts_unit,
            tz=tz,
            use_large_string=use_large_string,
            sort=sort,
        )

        schemas_equal *= schemas_equal_

    return unified_schema, bool(schemas_equal)


def repair_schema(
    files: list[str],
    base_path: str | None = None,
    schema: pa.Schema | None = None,
    filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
    n_jobs: int = -1,
    backend: str = "threading",
    verbose: bool = True,
    ts_unit: str | None = "us",
    tz: str | None = None,
    use_large_string: bool = False,
    sort: bool | list[str] = False,
    **kwargs,
):
    """Repairs the pyarrow schema of a parquet or arrow dataset

    Args:
        files (list[str]): Files of the dataset.
        schema (pa.Schema | None, optional): Unified pyarrow schema.
            If None, the unified schema is generated from the given files of the dataset. Defaults to None.
        filesystem (AbstractFileSystem | pfs.FileSystem | None, optional): Filesystem. Defaults to None.
        n_jobs (int, optional): n_jobs parameter of joblib.Parallel. Defaults to -1.
        backend (str, optional): backend parameter of joblib.Parallel. Defaults to "threading".
        verbose (bool, optional): Wheter to show the task progress using tqdm or not. Defaults to True.
        ts_unit (str|None): timestamp unit.
        tz (str|None): timezone for timestamp fields. Defaults to "UTC".
        use_large_string (bool): Convert pyarrow.large_string() to pyarrow.string().
            Defaults to False.
        sort (bool | list[str]): Use this, to define the field(column) order. If it is
            True, the fields are sorted alphanumerically, if it is False, the field
            order of firt schema is used. Otherwise, the order if the given field names is
            applied.
    """
    if base_path is not None:
        files = [os.path.join(base_path, f) for f in files]
    if schema is None:
        schemas = collect_file_schemas(
            files=files,
            filesystem=filesystem,
            n_jobs=n_jobs,
            backend=backend,
            verbose=verbose,
        )
        schema, schemas_equal = unify_schemas(
            schemas=list(schemas.values()),
            ts_unit=ts_unit,
            tz=tz,
            use_large_string=use_large_string,
            sort=sort,
        )

        files = [f for f in files if schemas[f] != schema]

    if ts_unit is not None or tz is not None:
        schema = convert_timestamp(schema, unit=ts_unit, tz=tz)

    if not use_large_string:
        schema = shrink_large_string(schema)

    def _repair_schema(f, schema, filesystem):
        table = pq.read_table(f, schema=schema, filesystem=filesystem)
        pq.write_table(table, f, filesystem=filesystem, **kwargs)

    _ = run_parallel(
        _repair_schema,
        files,
        schema=schema,
        filesystem=filesystem,
        backend=backend,
        n_jobs=n_jobs,
        verbose=verbose,
    )


def collect_file_schemas(
    files: list[str] | str,
    filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
    n_jobs: int = -1,
    backend: str = "threading",
    verbose: bool = True,
) -> dict[str, pa.Schema]:
    """Collect all pyarrow schemas for the given parquet files.

    Args:
        files (list[str] | str): Parquet files.
        filesystem (AbstractFileSystem | pfs.FileSystem | None, optional): Filesystem. Defaults to None.
        n_jobs (int, optional): n_jobs parameter of joblib.Parallel. Defaults to -1.
        backend (str, optional): backend parameter of joblib.Parallel. Defaults to "threading".
        verbose (bool, optional): Wheter to show the task progress using tqdm or not. Defaults to True.

    Returns:
        dict[str, pa.Schema]: Pyarrow schemas of the given files.
    """

    def get_schema(f, filesystem):
        return {f: pq.read_schema(f, filesystem=filesystem)}

    schemas = run_parallel(
        get_schema,
        files,
        filesystem=filesystem,
        backend=backend,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return {key: value for d in schemas for key, value in d.items()}

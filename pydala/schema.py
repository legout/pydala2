import posixpath

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pfs
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem

from .helpers.misc import read_table, run_parallel, unify_schemas_pl


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


# def shrink_large_string(schema: pa.Schema) -> pa.Schema:
#     """Convert all large_string types to string in a pyarrow.schema.

#     Args:
#         schema (pa.Schema): pyarrow schema

#     Returns:
#         pa.Schema: converted pyarrow.schema
#     """
#     return pa.schema(
#         [
#             (n, pa.utf8()) if t == pa.large_string() else (n, t)
#             for n, t in list(zip(schema.names, schema.types))
#         ]
#     )


def convert_large_types_to_normal(schema: pa.Schema) -> pa.Schema:
    # Define mapping of large types to standard types
    type_mapping = {
        pa.large_string(): pa.string(),
        pa.large_binary(): pa.binary(),
        pa.large_utf8(): pa.utf8(),
        pa.large_list(pa.null()): pa.list_(pa.null()),
        pa.large_list_view(pa.null()): pa.list_view(pa.null()),
    }
    # Convert fields
    new_fields = []
    for field in schema:
        field_type = field.type
        # Check if type exists in mapping
        if field_type in type_mapping:
            new_field = pa.field(
                name=field.name,
                type=type_mapping[field_type],
                nullable=field.nullable,
                metadata=field.metadata,
            )
            new_fields.append(new_field)
        # Handle large lists with nested types
        elif isinstance(field_type, pa.LargeListType):
            new_field = pa.field(
                name=field.name,
                type=pa.list_(
                    type_mapping[field_type.value_type]
                    if field_type.value_type in type_mapping
                    else field_type.value_type
                ),
                nullable=field.nullable,
                metadata=field.metadata,
            )
            new_fields.append(new_field)
        # Handle dictionary with large_string, large_utf8, or large_binary values
        elif isinstance(field_type, pa.DictionaryType):
            new_field = pa.field(
                name=field.name,
                type=pa.dictionary(
                    field_type.index_type,
                    type_mapping[field_type.value_type]
                    if field_type.value_type in type_mapping
                    else field_type.value_type,
                    field_type.ordered,
                ),
                # nullable=field.nullable,
                metadata=field.metadata,
            )
            new_fields.append(new_field)
        else:
            new_fields.append(field)

    return pa.schema(new_fields)


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


def drop_field(schema: pa.Schema, field: str) -> pa.Schema:
    """
    Drop a field from the given schema.

    Args:
        schema (pa.Schema): The schema to drop the field from.
        field (str): The name of the field to drop.

    Returns:
        pa.Schema: The updated schema with the field dropped.
    """
    if field in schema.names:
        return schema.remove(schema.get_field_index(field))
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


def cast_int2timestamp(
    table: pa.Table, column: str | list[str], unit: str = "us", tz: str | None = None
) -> pa.Table:
    column = [column] if isinstance(column, str) else column
    columns = table.schema.names
    for col in column:
        arr = table.column(col)
        arr_new = pa.array(arr.to_pylist(), pa.timestamp(unit=unit, tz=tz))
        table = table.drop(col).append_column(col, arr_new)

    return table.select(columns)


def cast_str2bool(
    table: pa.Table,
    schema: pa.Schema | None = None,
    column: str | list[str] | None = None,
    true_values: list[str] = [
        "true",
        "wahr",
        "1",
        "1.0",
        "yes",
        "ja",
        "ok",
        "o.k",
        "okay",
    ],
    skip_nulls: bool = False,
) -> pa.Table:
    if column is None and schema is not None:
        column = [
            col
            for col in table.schema.names
            if pa.types.is_boolean(schema.field(col).type)
            and pa.types.is_string(table.schema.field(col).type)
        ]
    column = [column] if isinstance(column, str) else column
    columns = table.schema.names
    for col in column:
        arr = table.column(col)
        arr_is_in = pc.is_in(
            pc.utf8_lower(table.column(col)),
            pa.array(true_values),
            skip_nulls=skip_nulls,
        )
        arr_is_null = pc.is_null(arr)
        arr_new = pc.if_else(arr_is_null, None, arr_is_in)
        table = table.drop(col).append_column(col, arr_new)
    return table.select(columns)


def replace_schema(
    table: pa.Table,
    schema: pa.Schema | None = None,
    field_dtype: dict | None = None,
    ts_unit: str | None = "us",
    tz: str | None = None,
    alter_schema: bool = True,
) -> pa.Table:
    """
    Replaces the schema of a PyArrow table with the specified schema or modifies the existing schema.

    Args:
        table (pa.Table): The PyArrow table to modify.
        schema (pa.Schema | None, optional): The new schema to replace the existing schema. Defaults to None.
        field_dtype (dict | None, optional): A dictionary mapping field names to their new data types. Defaults to None.
        alter_schema (bool, optional): If True, adds missing fields to the table based on the new schema.
            If False, drops fields from the table that are not present in the new schema. Defaults to True.

    Returns:
        pa.Table: The modified table with the replaced or modified schema.
    """
    schema_org = table.schema
    if field_dtype is not None:
        schema = schema or table.schema
        for field, dtype in field_dtype.items():
            schema = modify_field(schema=schema, field=field, dtype=dtype)

    schema = schema or table.schema

    if schema == schema_org:
        return table

    # add missing fields to the table
    missing_fields = [
        field for field in schema.names if field not in table.column_names
    ]

    for field in missing_fields:
        table = table.append_column(
            field, pa.array([None] * len(table), type=schema.field(field).type)
        )

    if not alter_schema:
        table = table.drop(
            [field for field in table.column_names if field not in schema.names]
        )

    int2timestamp_columns = [
        col
        for col in schema.names
        if pa.types.is_timestamp(schema.field(col).type)
        and (
            pa.types.is_integer(schema_org.field(col).type)
            if col in schema_org.names
            else False
        )
    ]
    table = cast_int2timestamp(table, int2timestamp_columns, unit=ts_unit, tz=tz)
    # cast str to bool
    table = cast_str2bool(table, schema=schema)

    return table.select(schema.names).cast(schema)

    # return table.select(schema.names)


def _unify_schemas(
    schema1: pa.Schema,
    schema2: pa.Schema,
) -> tuple[dict, bool]:
    """Returns a unified pyarrow schema.

    Args:
        schema1 (pa.Schema): Pyarrow schema 1.
        schema2 (pa.Schema): Pyarrow schema 2.
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
        all_names = schema1.names
        file_schemas_equal = True

    elif sorted(schema1.names) == sorted(schema2.names):
        all_names = schema1.names
        file_schemas_equal = False

    else:
        # all_names = list(set(schema1.names + schema2.names))
        all_names = schema1.names + [
            name for name in schema2.names if name not in schema1.names
        ]
        file_schemas_equal = False

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
                if isinstance(type2, pa.TimestampType):
                    rank2 = timestamp_units.index(type2.unit)
                else:
                    rank2 = 0
                type_ = type1 if rank1 > rank2 else type2
            else:
                rank1 = 1
                rank2 = 0
                type_ = type1 if rank1 > rank2 else type2
            schema.append(pa.field(name, type_))

        else:
            schema.append(pa.field(name, type1))

    schema = pa.schema(schema)

    if schema != schema1 or schema != schema2:
        file_schemas_equal = False

    return schema, file_schemas_equal


def unify_schemas(
    schemas: list[pa.Schema],
) -> tuple[pa.Schema, bool]:
    """Get the unified pyarrow schema for a list of schemas.

    Args:
        schemas (list[pa.Schema]): Pyarrow schemas.

    Returns:
        tuple[pa.Schema, bool]: Unified pyarrow schema.
    """

    schemas_equal = True
    unified_schema = schemas[0]
    for schema in schemas[1:]:
        unified_schema, schemas_equal_ = _unify_schemas(
            unified_schema,
            schema,
        )

        schemas_equal *= schemas_equal_

    return unified_schema, bool(schemas_equal)


def repair_schema(
    files: list[str] | None = None,
    file_schemas: dict[str, pa.Schema] | None = None,
    base_path: str | None = None,
    schema: pa.Schema | None = None,
    filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
    n_jobs: int = -1,
    backend: str = "threading",
    verbose: bool = True,
    ts_unit: str | None = None,  # "us",
    tz: str | None = None,
    # use_large_types: bool = False,
    # sort: bool | list[str] = False,
    alter_schema: bool = True,
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
        **kwargs: Additional keyword arguments for pyarrow.parquet.write_table.
    """
    if files is None:
        if file_schemas is not None:
            files = list(file_schemas.keys())
        else:
            if base_path is not None:
                files = filesystem.glob(base_path + "/**/*.parquet")
            else:
                raise ValueError(
                    "Either files or file_schemas or base_path must be provided."
                )

    if file_schemas is None:
        file_schemas = collect_file_schemas(
            files=files,
            base_path=base_path,
            filesystem=filesystem,
            n_jobs=n_jobs,
            backend=backend,
            verbose=verbose,
        )
    if base_path is not None:
        files = [posixpath.join(base_path, f) for f in files]
        file_schemas = {
            posixpath.join(base_path, f): s for f, s in file_schemas.items()
        }

    if schema is None:
        try:
            schema = pa.unify_schemas(
                schemas=list(file_schemas.values()), promote_options="permissive"
            )
        except pa.lib.ArrowTypeError:
            schema = unify_schemas_pl(list(file_schemas.values()))

    if ts_unit is not None or tz is not None:
        schema = convert_timestamp(schema, unit=ts_unit, tz=tz)

    # if not use_large_types:
    schema = convert_large_types_to_normal(schema)

    schemas_equal = all([schema == schemas_ for schemas_ in file_schemas.values()])

    if not schemas_equal:
        files = [f for f in files if file_schemas[f] != schema]

    def _repair_schema(f, schema, filesystem):
        table = replace_schema(
            read_table(f, filesystem=filesystem, partitioning=None),
            schema=schema,
            ts_unit=ts_unit,
            tz=tz,
            alter_schema=alter_schema,
        )
        pq.write_table(
            table,
            f,
            filesystem=filesystem,
            coerce_timestamps=ts_unit,
            allow_truncated_timestamps=True,
            **kwargs,
        )

    if len(files):
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
    base_path: str | None = None,
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
    if base_path is not None:
        files = [posixpath.join(base_path, f) for f in files]

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

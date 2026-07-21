import functools
import posixpath

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pfs
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from fsspec.core import split_protocol
from fsspeckit.datasets.schema import (
    cast_schema as _fsspeckit_cast_schema,
    convert_large_types_to_normal,
    unify_schemas as _fsspeckit_unify_schemas,
)
from fsspeckit.common.parallel import run_parallel as _fsspeckit_run_parallel
from .helpers.misc import read_table, unify_schemas_pl


def strip_protocol(path: str) -> str:
    """Strips the protocol from a given path.

    Args:
        path (str): The input path which may contain a protocol.
    Returns:
        str: The path without the protocol.
    """
    protocol, path = split_protocol(path)
    return path


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
        field = schema.field(timestamp_field)
        field_type = field.type
        field_tz = None if remove_tz else (tz if tz is not None else field_type.tz)
        schema = schema.remove(timestamp_field_idx).insert(
            timestamp_field_idx,
            field.with_type(pa.timestamp(unit=unit or field_type.unit, tz=field_tz)),
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


_LEGACY_PROMOTION_TYPES_LIST = (
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
)
_LEGACY_PROMOTION_TYPES = frozenset(_LEGACY_PROMOTION_TYPES_LIST)
_TIMESTAMP_UNIT_ORDER = {"ns": 0, "us": 1, "ms": 2, "s": 3}


def _pydala_legacy_type(source_types: list[pa.DataType]) -> pa.DataType:
    """Resolve field types using pydala's pre-fsspeckit conflict policy."""
    field_type = source_types[0]
    for candidate in source_types[1:]:
        if field_type == candidate:
            continue
        if field_type in _LEGACY_PROMOTION_TYPES:
            rank1 = _LEGACY_PROMOTION_TYPES_LIST.index(field_type)
            rank2 = (
                _LEGACY_PROMOTION_TYPES_LIST.index(candidate)
                if candidate in _LEGACY_PROMOTION_TYPES
                else 0
            )
            if field_type.num_buffers == 2 and candidate.num_buffers == 2:
                if field_type.bit_width > candidate.bit_width:
                    field_type = field_type if rank1 > rank2 else pa.float64()
                else:
                    field_type = candidate if rank2 > rank1 else pa.float64()
            else:
                field_type = field_type if rank1 > rank2 else candidate
        elif isinstance(field_type, pa.TimestampType):
            rank1 = _TIMESTAMP_UNIT_ORDER[field_type.unit]
            rank2 = (
                _TIMESTAMP_UNIT_ORDER[candidate.unit]
                if isinstance(candidate, pa.TimestampType)
                else 0
            )
            field_type = field_type if rank1 > rank2 else candidate
    return field_type


def _pydala_compatible_schema(
    unified: pa.Schema, schemas: list[pa.Schema]
) -> pa.Schema:
    """Restore pydala's public schema-unification edge behavior."""
    fields = []
    for field in unified:
        source_types = [
            schema.field(field.name).type
            for schema in schemas
            if field.name in schema.names
        ]
        field_type = _pydala_legacy_type(source_types) if source_types else field.type
        fields.append(pa.field(field.name, field_type))
    return pa.schema(fields)


def unify_schemas(
    schemas: list[pa.Schema],
) -> tuple[pa.Schema, bool]:
    """Get the unified pyarrow schema for a list of schemas.

    Delegates generic unification to fsspeckit while retaining pydala's
    public conflict-resolution and metadata behavior.

    Args:
        schemas: Pyarrow schemas.

    Returns:
        Tuple of unified schema and whether all inputs were equal.
    """
    if not schemas:
        raise ValueError("At least one schema must be provided")
    if len(schemas) == 1:
        return schemas[0], True
    unified = _fsspeckit_unify_schemas(schemas, use_large_dtypes=True)
    unified = _pydala_compatible_schema(unified, schemas)
    schemas_equal = all(s == unified for s in schemas)
    return unified, schemas_equal


def _unify_repair_schemas(schemas: list[pa.Schema]) -> pa.Schema:
    """Build a repair target with pydala's established promotion policy."""
    if not schemas:
        raise ValueError("At least one schema must be provided")
    try:
        return pa.unify_schemas(schemas, promote_options="permissive")
    except pa.ArrowTypeError:
        try:
            unified, _ = unify_schemas(schemas)
        except (pa.ArrowException, ValueError):
            return unify_schemas_pl(schemas)
        return unified


def _plan_schema_repair(
    file_schemas: dict[str, pa.Schema],
    schema: pa.Schema | None = None,
    ts_unit: str | None = None,
    tz: str | None = None,
    file_versions: dict[str, str] | None = None,
    version: str | None = None,
) -> dict[str, object]:
    """Plan schema repair from one immutable snapshot of physical files.

    The input mapping is copied and each physical schema is normalized once for
    this operation.  No state is retained between calls, so callers can take a
    fresh snapshot after reconciling physical additions and removals.
    """
    snapshot = {
        path: convert_large_types_to_normal(file_schema)
        for path, file_schema in file_schemas.items()
    }

    if schema is None:
        target_schema = _unify_repair_schemas(list(snapshot.values()))
    else:
        target_schema = schema

    if ts_unit is not None or tz is not None:
        target_schema = convert_timestamp(target_schema, unit=ts_unit, tz=tz)
    target_schema = convert_large_types_to_normal(target_schema)

    files_to_repair = [
        path for path, file_schema in snapshot.items() if file_schema != target_schema
    ]
    if version is not None and file_versions is not None:
        files_to_repair.extend(
            path for path in snapshot if file_versions.get(path) != version
        )

    return {
        "files": sorted(set(files_to_repair)),
        "schema": target_schema,
    }


def _execute_schema_repair(
    files: list[str],
    schema: pa.Schema,
    filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
    n_jobs: int = -1,
    backend: str = "threading",
    verbose: bool = True,
    ts_unit: str | None = None,
    tz: str | None = None,
    alter_schema: bool = True,
    version: str | None = None,
    **kwargs,
) -> None:
    """Execute a prepared schema-repair plan without replanning it."""

    def _repair_schema(f, schema, filesystem):
        # pydala-specific coercions (missing fields, int->timestamp, str->bool)
        table = replace_schema(
            read_table(f, filesystem=filesystem, partitioning=None),
            schema=schema,
            ts_unit=ts_unit,
            tz=tz,
            alter_schema=alter_schema,
        )
        # Final fsspeckit cast to ensure that the table matches the target schema.
        write_kwargs = kwargs.copy()
        if version is not None:
            write_kwargs["version"] = version
        if "compression" not in write_kwargs:
            metadata = pq.read_metadata(f, filesystem=filesystem)
            if metadata.num_row_groups and metadata.row_group(0).num_columns:
                write_kwargs["compression"] = (
                    metadata.row_group(0).column(0).compression.lower()
                )
        table = _fsspeckit_cast_schema(table, schema)
        pq.write_table(
            table,
            f,
            filesystem=filesystem,
            coerce_timestamps=ts_unit,
            allow_truncated_timestamps=True,
            **write_kwargs,
        )

    if files:
        # Bind non-iterable arguments so fsspeckit's parallel runner only
        # iterates over the file list (pa.Schema is iterable and would
        # otherwise be mis-detected as a per-task iterable).
        _repair_bound = functools.partial(
            _repair_schema, schema=schema, filesystem=filesystem
        )
        _ = _fsspeckit_run_parallel(
            _repair_bound,
            files,
            backend=backend,
            n_jobs=n_jobs,
            verbose=verbose,
        )


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
    dry_run: bool = False,
    version: str | None = None,
    **kwargs,
):
    """Repairs the pyarrow schema of a parquet or arrow dataset.

    This adapter delegates schema unification and parallel execution to public
    ``fsspeckit`` primitives while preserving pydala-specific coercion behavior
    (string-to-bool, integer-to-timestamp, missing-field handling). When
    ``dry_run`` is ``True`` no data or metadata is mutated; a plan describing the
    files that would be rewritten is returned instead.

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
        alter_schema (bool): Whether to add missing fields while repairing files.
        dry_run (bool): If True, return a plan without reading or writing parquet data.
        version (str | None): Parquet format version passed to
            :func:`pyarrow.parquet.write_table` for every repair write. If
            ``None``, PyArrow's default is used. Existing callers may continue
            passing this option by keyword.
        **kwargs: Additional keyword arguments for pyarrow.parquet.write_table.

    Returns:
        dict | None: When ``dry_run`` is ``True``, returns ``{"files": [...], "schema": schema}``
        describing the repair plan; otherwise ``None``.
    """
    base_path = strip_protocol(base_path) if base_path is not None else None
    if files is None:
        if file_schemas is not None:
            files = list(file_schemas.keys())
        elif base_path is not None:
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

        def qualify(path: str) -> str:
            path = strip_protocol(path)
            return (
                path
                if path == base_path or path.startswith(base_path + "/")
                else posixpath.join(base_path, path)
            )

        files = [qualify(path) for path in files]
        file_schemas = {
            qualify(path): file_schema for path, file_schema in file_schemas.items()
        }

    # Ignore any schema entries outside the requested physical-file snapshot.
    file_schemas = {path: file_schemas[path] for path in files}

    file_versions = None
    if version is not None:
        file_versions = {
            path: pq.read_metadata(path, filesystem=filesystem).format_version
            for path in file_schemas
        }

    plan = _plan_schema_repair(
        file_schemas=file_schemas,
        schema=schema,
        ts_unit=ts_unit,
        tz=tz,
        file_versions=file_versions,
        version=version,
    )
    schema = plan["schema"]
    files_to_repair = plan["files"]

    if dry_run:
        return plan

    _execute_schema_repair(
        files=files_to_repair,
        schema=schema,
        filesystem=filesystem,
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
        ts_unit=ts_unit,
        tz=tz,
        alter_schema=alter_schema,
        version=version,
        **kwargs,
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
    base_path = strip_protocol(base_path) if base_path is not None else None
    files = [files] if isinstance(files, str) else list(files)
    if base_path is not None:
        files = [
            path
            if strip_protocol(path) == base_path
            or strip_protocol(path).startswith(base_path + "/")
            else posixpath.join(base_path, strip_protocol(path))
            for path in files
        ]

    def get_schema(f, filesystem):
        return {f: pq.read_schema(f, filesystem=filesystem)}

    schemas = _fsspeckit_run_parallel(
        get_schema,
        files,
        filesystem=filesystem,
        backend=backend,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return {key: value for d in schemas for key, value in d.items()}

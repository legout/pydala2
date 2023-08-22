import pyarrow as pa
import pyarrow.fs as pfs
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem

from .helpers import collect_file_schemas, run_parallel

# from .io import read_table, write_table


def sort_schema(schema: pa.Schema) -> pa.Schema:
    """Sort fields of a pyarrow schema in alphabetical order.

    Args:
        schema (pa.Schema): Pyarrow schema.

    Returns:
        pa.Schema: Sorted pyarrow schema.
    """
    return pa.schema(
        [
            pa.field(name, type_)
            for name, type_ in sorted(zip(schema.names, schema.types))
        ]
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


def _unify_schemas(
    schema1: pa.Schema,
    schema2: pa.Schema,
    ts_unit: str | None = "us",
    tz: str | None = "UTC",
    use_large_string: bool = False,
) -> tuple[dict, bool]:
    """Returns a unified pyarrow schema.

    Args:
        schema1 (pa.Schema): Pyarrow schema 1.
        schema2 (pa.Schema): Pyarrow schema 2.
        ts_unit (str|None): timestamp unit.
        tz (str|None): timezone for timestamp fields. Defaults to "UTC".
        use_large_string (bool): Convert pyarrow.large_string() to pyarrow.string(). Defaults to False.

    Returns:
        Tuple[dict, bool]: Unified pyarrow schema, bool value if schemas were equal.
    """

    numeric_rank = [
        pa.null(),
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
        pa.float16(),
        pa.float32(),
        pa.float64(),
        pa.string(),
    ]
    timestamp_rank = ["ns", "us", "ms", "s"]
    string_rank = [pa.string(), pa.utf8(), pa.large_string(), pa.large_utf8()]

    # check for equal columns and column order
    if schema1.names == schema2.names:
        # if schema1.types == schema2.types:
        #    return schema1, True

        all_names = schema1.names
        file_schemas_equal = True

    elif sorted(schema1.names) == sorted(schema2.names):
        all_names = sorted(schema1.names)
        file_schemas_equal = False

    else:
        all_names = sorted(set(schema1.names + schema2.names))
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
            if type1 in numeric_rank:
                rank1 = numeric_rank.index(type1) if type1 in numeric_rank else 0
                rank2 = numeric_rank.index(type2) if type2 in numeric_rank else 0

            elif type1 in string_rank:
                rank1 = string_rank.index(type1) if type1 in string_rank else 0
                rank2 = string_rank.index(type2) if type2 in string_rank else 0

            elif isinstance(type1, pa.TimestampType):
                rank1 = string_rank.index(type1.unit) if type1 in string_rank else 0
                rank2 = string_rank.index(type2.unit) if type2 in string_rank else 0
            else:
                rank1 = 1
                rank2 = 0
            schema.append(pa.field(name, type1 if rank1 > rank2 else type2))

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
    tz: str | None = "UTC",
    use_large_string: bool = False,
) -> tuple[pa.Schema, bool]:
    """Get the unified pyarrow schema for a list of schemas.

    Args:
        schemas (list[pa.Schema]): Pyarrow schemas.
        ts_unit (str|None): timestamp unit.
        tz (str|None): timezone for timestamp fields. Defaults to "UTC".
        use_large_string (bool): Convert pyarrow.large_string() to pyarrow.string(). Defaults to False.

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
        )

        schemas_equal *= schemas_equal_

    return unified_schema, schemas_equal


def repair_schema(
    files: list[str],
    schema: pa.Schema | None = None,
    filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
    n_jobs: int = -1,
    backend: str = "threading",
    verbose: bool = True,
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
    """
    if schema is None:
        schemas = collect_file_schemas(
            files=files,
            filesystem=filesystem,
            n_jobs=n_jobs,
            backend=backend,
            verbose=verbose,
        )
        schema, schemas_equal = unify_schemas(schemas=list(schemas.values()))

        files = [f for f in files if schemas[f] != schema]

    def _repair_schema(f, schema, filesystem):
        filesystem.invalidate_cache()
        # file_schema = pq.read_metadata(f, filesystem=filesystem)
        # if file_schema!=schema:
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
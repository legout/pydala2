import posixpath
import re
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tqdm
from fsspec import AbstractFileSystem
from fsspec import filesystem as fsspec_filesystem
from joblib import Parallel, delayed

# from ..schema import convert_large_types_to_normal
from .datetime import timestamp_from_string
from .polars import pl

# Compile regex patterns once for efficiency
SPLIT_PATTERN = re.compile(
    r"<=|<|>=|>|=|!=|\s+[n,N][o,O][t,T]\s+[i,I][n,N]\s+|\s+[i,I][n,N]\s+|\s+[i,I][s,S]\s+[n,N][o,O][t,T]\s+"
    r"[n,N][u,U][l,L]{2}\s+|\s+[i,I][s,S]\s+[n,N][u,U][l,L]{2}\s+"
)
LOGICAL_OPERATORS_PATTERN = re.compile(
    r"\s+[a,A][n,N][d,D] [n,N][o,O][t,T]\s+|\s+[a,A][n,N][d,D]\s+|\s+[o,O][r,R] [n,N][o,O][t,T]\s+|\s+[o,O][r,R]\s+"
)


# def sql2pyarrow_filter(string: str, schema: pa.Schema):
#     """
#     Generates a filter expression for PyArrow based on a given string and schema.

#     Parameters:
#         string (str): The string containing the filter expression.
#         schema (pa.Schema): The PyArrow schema used to validate the filter expression.

#     Returns:
#         PyArrow.Expression: The generated filter expression.

#     """

#     def _parse_part(part):
#         split_pattern = (
#             r"<=|"
#             r"<|"
#             r">=|"
#             r">|"
#             r"=|"
#             r"!=|"
#             r"\s+[n,N][o,O][t,T]\s+[i,I][n,N]\s+|"
#             r"\s+[i,I][n,N]\s+|"
#             r"\s+[i,I][s,S]\s+[n,N][o,O][t,T]\s+[n,N][u,U][l,L]{2}\s+|"
#             r"\s+[i,I][s,S]\s+[n,N][u,U][l,L]{2}\s+"
#         )
#         sign = re.findall(split_pattern, part)[0]
#         # print(sign)
#         # if "<" in sign or ">" in sign or "=" in sign or "!" in sign:
#         field, val = [p.strip() for p in re.split(f"\\s*{sign}\\s*", part)]
#         # else:
#         #    field, val = [p.strip() for p in re.split(f"\s+{sign}\s+", part)]

#         # print(field, val)
#         type_ = schema.field(field).type
#         if "(" in val:
#             val = eval(val)

#         if pa.types.is_time(type_):
#             val = timestamp_from_string(val, exact=True)
#             if isinstance(val, pdl.DateTime):
#                 val = val.time()

#         elif pa.types.is_date(type_):
#             val = timestamp_from_string(val, exact=True)
#             if isinstance(val, pdl.DateTime):
#                 val = val.date()

#         elif pa.types.is_timestamp(type_):
#             val = timestamp_from_string(val, exact=True)

#         elif pa.types.is_integer_value(type_) or pa.types.is_integer(type_):
#             if isinstance(val, str):
#                 val = int(float(val.strip("'").replace(",", ".")))
#             elif isinstance(val, tuple):
#                 val = tuple(
#                     [
#                         (
#                             int(float(val_.strip(",").replace(",", ".")))
#                             if isinstance(val_, str)
#                             else val_
#                         )
#                         for val_ in val
#                     ]
#                 )
#             elif isinstance(val, list):
#                 val = [
#                     (
#                         int(float(val_.strip("'").replace(",", ".")))
#                         if isinstance(val_, str)
#                         else val_
#                     )
#                     for val_ in val
#                 ]

#         elif pa.types.is_float_value(type_) or pa.types.is_floating(type_):
#             if isinstance(val, str):
#                 val = float(val.strip("'").replace(",", "."))
#             elif isinstance(val, tuple):
#                 val = tuple(
#                     [
#                         (
#                             float(val_.strip(",").replace(",", "."))
#                             if isinstance(val_, str)
#                             else val_
#                         )
#                         for val_ in val
#                     ]
#                 )
#             elif isinstance(val, list):
#                 val = [
#                     (
#                         float(val_.strip("'").replace(",", "."))
#                         if isinstance(val_, str)
#                         else val_
#                     )
#                     for val_ in val
#                 ]

#         elif pa.types.is_boolean(type_):
#             val = bool(val.strip("'").replace(",", "."))

#         if isinstance(val, str):
#             val = val.strip("'")

#         if sign == ">=":
#             return pc.field(field) >= val
#         elif sign == ">":
#             return pc.field(field) > val
#         elif sign == "<=":
#             return pc.field(field) <= val
#         elif sign == "<":
#             return pc.field(field) < val
#         elif sign == "=":
#             return pc.field(field) == val
#         elif sign == "!=":
#             return pc.field(field) != val
#         elif sign.lower() == "in":
#             return pc.field(field).isin(val)
#         elif sign.lower() == "not in":
#             return ~pc.field(field).isin(val)
#         elif sign.lower() == "is null":
#             return pc.field(field).is_null(nan_is_null=True)
#         elif sign.lower() == "is not null":
#             return ~pc.field(field).is_null(nan_is_null=True)

#     parts = re.split(
#         (
#             r"\s+[a,A][n,N][d,D] [n,N][o,O][t,T]\s+|"
#             r"\s+[a,A][n,N][d,D]\s+|"
#             r"\s+[o,O][r,R] [n,N][o,O][t,T]\s+|"
#             r"\s+[o,O][r,R]\s+"
#         ),
#         string,
#     )
#     operators = [
#         op.strip()
#         for op in re.findall(
#             (
#                 r"\s+[a,A][n,N][d,D]\s+[n,N][o,O][t,T]\s+|"
#                 r"\s+[a,A][n,N][d,D]\s+|"
#                 r"[o,O][r,R]\s+[n,N][o,O][t,T]\s+|"
#                 r"\s+[o,O][r,R]\s+"
#             ),
#             string,
#         )
#     ]

#     # print(parts, operators)

#     if len(parts) == 1:
#         return _parse_part(parts[0])

#     expr = _parse_part(parts[0])

#     for part, operator in zip(parts[1:], operators):
#         if operator.lower() == "and":
#             expr = expr & _parse_part(part)
#         elif operator.lower() == "and not":
#             expr = expr & ~_parse_part(part)
#         elif operator.lower() == "or":
#             expr = expr | _parse_part(part)
#         elif operator.lower() == "or not":
#             expr = expr | ~_parse_part(part)

#     return expr


def sql2pyarrow_filter(string: str, schema: pa.Schema) -> pc.Expression:
    """
    Generates a filter expression for PyArrow based on a given string and schema.

    Parameters:
        string (str): The string containing the filter expression.
        schema (pa.Schema): The PyArrow schema used to validate the filter expression.

    Returns:
        pc.Expression: The generated filter expression.

    Raises:
        ValueError: If the input string is invalid or contains unsupported operations.
    """

    def parse_value(val: str, type_: pa.DataType) -> Any:
        """Parse and convert value based on the field type."""
        if isinstance(val, (tuple, list)):
            return type(val)(parse_value(v, type_) for v in val)

        if pa.types.is_timestamp(type_):
            return timestamp_from_string(val, exact=False, tz=type_.tz)
        elif pa.types.is_date(type_):
            return timestamp_from_string(val, exact=True).date()
        elif pa.types.is_time(type_):
            return timestamp_from_string(val, exact=True).time()

        elif pa.types.is_integer(type_):
            return int(float(val.strip("'").replace(",", ".")))
        elif pa.types.is_floating(type_):
            return float(val.strip("'").replace(",", "."))
        elif pa.types.is_boolean(type_):
            return val.lower().strip("'") in ("true", "1", "yes")
        else:
            return val.strip("'")

    def _parse_part(part: str) -> pc.Expression:
        match = SPLIT_PATTERN.search(part)
        if not match:
            raise ValueError(f"Invalid condition: {part}")

        sign = match.group().lower().strip()
        field, val = [p.strip() for p in SPLIT_PATTERN.split(part)]

        if field not in schema.names:
            raise ValueError(f"Unknown field: {field}")

        type_ = schema.field(field).type
        val = parse_value(val, type_)

        operations = {
            ">=": lambda f, v: pc.field(f) >= v,
            ">": lambda f, v: pc.field(f) > v,
            "<=": lambda f, v: pc.field(f) <= v,
            "<": lambda f, v: pc.field(f) < v,
            "=": lambda f, v: pc.field(f) == v,
            "!=": lambda f, v: pc.field(f) != v,
            "in": lambda f, v: pc.field(f).isin(v),
            "not in": lambda f, v: ~pc.field(f).isin(v),
            "is null": lambda f, v: pc.field(f).is_null(nan_is_null=True),
            "is not null": lambda f, v: ~pc.field(f).is_null(nan_is_null=True),
        }

        if sign not in operations:
            raise ValueError(f"Unsupported operation: {sign}")

        return operations[sign](field, val)

    parts = LOGICAL_OPERATORS_PATTERN.split(string)
    operators = [op.lower().strip() for op in LOGICAL_OPERATORS_PATTERN.findall(string)]

    if len(parts) == 1:
        return _parse_part(parts[0])

    expr = _parse_part(parts[0])
    for part, operator in zip(parts[1:], operators):
        if operator == "and":
            expr = expr & _parse_part(part)
        elif operator == "and not":
            expr = expr & ~_parse_part(part)
        elif operator == "or":
            expr = expr | _parse_part(part)
        elif operator == "or not":
            expr = expr | ~_parse_part(part)
        else:
            raise ValueError(f"Unsupported logical operator: {operator}")

    return expr


def sql2polars_filter(string: str, schema: pl.Schema) -> pl.Expr:
    """
    Generates a filter expression for Polars based on a given string and schema.

    Parameters:
        string (str): The string containing the filter expression.
        schema (pl.Schema): The Polars schema used to validate the filter expression.

    Returns:
        pl.Expr: The generated filter expression.

    Raises:
        ValueError: If the input string is invalid or contains unsupported operations.
    """

    def parse_value(val: str, dtype: pl.DataType) -> Any:
        """Parse and convert value based on the field type."""
        if isinstance(val, (tuple, list)):
            return type(val)(parse_value(v, dtype) for v in val)

        if dtype == pl.Datetime:
            return timestamp_from_string(val, exact=False, tz=dtype.time_zone)
        elif dtype == pl.Date:
            return timestamp_from_string(val, exact=True).date()
        elif dtype == pl.Time:
            return timestamp_from_string(val, exact=True).time()
        elif dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
            return int(float(val.strip("'").replace(",", ".")))
        elif dtype in (pl.Float32, pl.Float64):
            return float(val.strip("'").replace(",", "."))
        elif dtype == pl.Boolean:
            return val.lower().strip("'") in ("true", "1", "yes")
        else:
            return val.strip("'")

    def _parse_part(part: str) -> pl.Expr:
        match = SPLIT_PATTERN.search(part)
        if not match:
            raise ValueError(f"Invalid condition: {part}")

        sign = match.group().lower().strip()
        field, val = [p.strip() for p in SPLIT_PATTERN.split(part)]

        if field not in schema.names():
            raise ValueError(f"Unknown field: {field}")

        dtype = schema[field]
        val = parse_value(val, dtype)

        operations = {
            ">=": lambda f, v: pl.col(f) >= v,
            ">": lambda f, v: pl.col(f) > v,
            "<=": lambda f, v: pl.col(f) <= v,
            "<": lambda f, v: pl.col(f) < v,
            "=": lambda f, v: pl.col(f) == v,
            "!=": lambda f, v: pl.col(f) != v,
            "in": lambda f, v: pl.col(f).is_in(v),
            "not in": lambda f, v: ~pl.col(f).is_in(v),
            "is null": lambda f, v: pl.col(f).is_null(),
            "is not null": lambda f, v: pl.col(f).is_not_null(),
        }

        if sign not in operations:
            raise ValueError(f"Unsupported operation: {sign}")

        return operations[sign](field, val)

    parts = LOGICAL_OPERATORS_PATTERN.split(string)
    operators = [op.lower().strip() for op in LOGICAL_OPERATORS_PATTERN.findall(string)]

    if len(parts) == 1:
        return _parse_part(parts[0])

    expr = _parse_part(parts[0])
    for part, operator in zip(parts[1:], operators):
        if operator == "and":
            expr = expr & _parse_part(part)
        elif operator == "and not":
            expr = expr & ~_parse_part(part)
        elif operator == "or":
            expr = expr | _parse_part(part)
        elif operator == "or not":
            expr = expr | ~_parse_part(part)
        else:
            raise ValueError(f"Unsupported logical operator: {operator}")

    return expr


def read_table(
    path: str,
    schema: pa.Schema | None = None,
    filesystem: AbstractFileSystem | None = None,
    partitioning: str | list[str] | None = None,
) -> pa.Table:
    """Loads the data for the given file path into a pyarrow table.

    Args:
        path (str): File path.
        schema (pa.Schema | None, optional): Pyarrow schema. Defaults to None.
        filesystem (AbstractFileSystem | None, optional): Filesystem. Defaults to None.
        partitioning (str | list[str] | None, optional): Partitioning of the file. Could be 'hive' for
            hive style partitioning or a list of column names. Defaults to None.

    Returns:
        pa.Table: Pyarrow table.
    """
    # sourcery skip: avoid-builtin-shadow
    if filesystem is None:
        filesystem = fsspec_filesystem("file")

    filesystem.invalidate_cache()

    table = pq.read_table(pa.BufferReader(filesystem.read_bytes(path)), schema=schema)

    if partitioning is not None:
        partitions = get_partitions_from_path(path, partitioning=partitioning)

        for key, values in partitions:
            if key in table.column_names:
                table = table.drop(key)
            table = table.append_column(
                field_=key, column=pa.array([values] * len(table))
            )
    if schema is not None:
        return table.cast(schema)

    return table


def run_parallel(
    func: callable,
    func_params: list[any],
    *args,
    n_jobs: int = -1,
    backend: str = "threading",
    verbose: bool = True,
    **kwargs,
) -> list[any]:
    """Runs a function for a list of parameters in parallel.

    Args:
        func (Callable): function to run in Parallelallel.
        func_params (list[any]): parameters for the function
        n_jobs (int, optional): Number of joblib workers. Defaults to -1.
        backend (str, optional): joblib backend. Valid options are
        `loky`,`threading`, `mutliprocessing` or `sequential`.  Defaults to "threading".

    Returns:
        list[any]: Function output.
    """
    if verbose:
        return Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(func)(fp, *args, **kwargs) for fp in tqdm.tqdm(func_params)
        )

    else:
        return Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(func)(fp, *args, **kwargs) for fp in func_params
        )


def get_partitions_from_path(
    path: str, partitioning: str | list[str] | None = None
) -> list[tuple]:
    """Get the dataset partitions from the file path.

    Args:
        path (str): File path.
        partitioning (str | list[str] | None, optional): Partitioning type. Defaults to None.

    Returns:
        list[tuple]: Partitions.
    """
    if "." in path:
        path = posixpath.dirname(path)

    parts = path.split("/")

    if isinstance(partitioning, str):
        if partitioning == "hive":
            return [tuple(p.split("=")) for p in parts if "=" in p]

        else:
            return [
                (partitioning, parts[0]),
            ]
    else:
        return list(zip(partitioning, parts[-len(partitioning) :]))


def humanize_size(size: int, unit="MB") -> float:
    "Human-readable size"
    if unit.lower() == "b":
        return round(size, 1)
    elif unit.lower() == "kb":
        return round(size / 1024, 1)
    elif unit.lower() == "mb":
        return round(size / 1024**2, 1)
    elif unit.lower() == "gb":
        return round(size / 1024**3, 1)
    elif unit.lower() == "tb":
        return round(size / 1024**4, 1)
    elif unit.lower() == "pb":
        return round(size / 1024**5, 1)


def humanized_size_to_bytes(size: str) -> float:
    "Human-readable size t bytes"
    unit = re.sub("[0-9 ]", "", size).lower()
    size = float(re.sub("[a-zA-Z ]", "", size))
    if unit == "b":
        return int(size)
    elif unit == "kb":
        return int(size * 1024)
    elif unit == "mb":
        return int(size * 1024**2)
    elif unit == "gb":
        return int(size * 1024**3)
    elif unit == "tb":
        return int(size * 1024**4)
    elif unit == "pb":
        return int(size * 1024**5)


def getattr_rec(obj, attr_str):
    """
    Recursively retrieves an attribute from an object based on a dot-separated string.

    Args:
        obj: The object from which to retrieve the attribute.
        attr_str: A dot-separated string representing the attribute to retrieve.

    Returns:
        The value of the attribute.

    Raises:
        AttributeError: If the attribute does not exist.
    """
    attrs = attr_str.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def setattr_rec(obj, attr_str, value):
    """
    Recursively sets the value of an attribute in an object.

    Args:
        obj (object): The object to set the attribute in.
        attr_str (str): The attribute string in dot notation (e.g., "attr1.attr2.attr3").
        value: The value to set the attribute to.

    Returns:
        None
    """
    attrs = attr_str.split(".")
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)


def delattr_rec(obj, attr_str):
    """
    Recursively deletes an attribute from an object.

    Args:
        obj: The object from which to delete the attribute.
        attr_str: A string representing the attribute to be deleted.
                  The attribute can be nested using dot notation (e.g., 'attr1.attr2.attr3').

    Raises:
        AttributeError: If the attribute does not exist.

    Example:
        obj = SomeObject()
        attr_str = 'attr1.attr2.attr3'
        delattr_rec(obj, attr_str)
    """
    attrs = attr_str.split(".")
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    delattr(obj, attrs[-1])


def get_nested_keys(d, parent_key=""):
    """
    Recursively retrieves all the nested keys from a dictionary.

    Args:
        d (dict): The dictionary to retrieve the keys from.
        parent_key (str, optional): The parent key to prepend to the nested keys. Defaults to "".

    Returns:
        list: A list of all the nested keys in the dictionary.
    """
    keys = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        keys.append(new_key)
        if isinstance(v, dict):
            keys.extend(get_nested_keys(v, new_key))
    return keys


def unify_schemas_pl(schemas: list[pa.Schema]) -> pl.Schema:
    """
    Unifies a list of Pyarrow schemas into a single schema.

    Args:
        schemas (list[pl.Schema]): List of Polars schemas.

    Returns:
        pa.Schema: Unified schema.
    """
    schema = (
        pl.concat(
            [
                # pl.from_arrow(pa.Table.from_pylist([], schema=schema))
                pl.from_arrow(schema.empty_table())
                for schema in schemas
            ],
            how="diagonal_relaxed",
        )
        .to_arrow()
        .schema
    )

    return schema

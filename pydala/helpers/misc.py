from typing import Any

from fsspec_utils.utils.misc import run_parallel
import posixpath
import re

import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
from fsspec import filesystem as fsspec_filesystem, AbstractFileSystem
from joblib import Parallel, delayed
from .polars import pl


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
    func_params: list[Any],
    *args,
    n_jobs: int = -1,
    backend: str = "threading",
    verbose: bool = True,
    **kwargs,
) -> list[Any]:
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
) -> list[tuple[str, str]]:
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


def humanize_size(size: int, unit: str = "MB") -> float:
    """Convert bytes to human-readable size in specified unit.

    Args:
        size: Size in bytes.
        unit: Target unit for conversion ('B', 'KB', 'MB', 'GB', 'TB', 'PB').

    Returns:
        Size converted to the specified unit, rounded to 1 decimal place.

    Raises:
        ValueError: If size is negative or unit is invalid.
    """
    if size < 0:
        raise ValueError("size must be a non-negative integer")

    unit = unit.lower()
    valid_units = ["b", "kb", "mb", "gb", "tb", "pb"]
    if unit not in valid_units:
        raise ValueError(f"unit must be one of {valid_units}")

    if unit == "b":
        return round(size, 1)
    elif unit == "kb":
        return round(size / 1024, 1)
    elif unit == "mb":
        return round(size / 1024**2, 1)
    elif unit == "gb":
        return round(size / 1024**3, 1)
    elif unit == "tb":
        return round(size / 1024**4, 1)
    elif unit == "pb":
        return round(size / 1024**5, 1)


def humanized_size_to_bytes(size: str) -> int:
    """Convert human-readable size string to bytes.

    Args:
        size: Human-readable size string (e.g., '100 MB', '1.5 GB').

    Returns:
        Size in bytes.

    Raises:
        ValueError: If size format is invalid or contains negative values.
    """
    if not isinstance(size, str):
        raise ValueError("size must be a string")

    unit = re.sub("[0-9 ]", "", size).lower()
    size_str = re.sub("[a-zA-Z ]", "", size)

    if not size_str:
        raise ValueError("size must contain a numeric value")

    try:
        size_val = float(size_str)
    except ValueError:
        raise ValueError("size must contain a valid numeric value")

    if size_val < 0:
        raise ValueError("size must be a non-negative number")

    valid_units = ["b", "kb", "mb", "gb", "tb", "pb"]
    if unit not in valid_units:
        raise ValueError(f"unit must be one of {valid_units}")

    if unit == "b":
        return int(size_val)
    elif unit == "kb":
        return int(size_val * 1024)
    elif unit == "mb":
        return int(size_val * 1024**2)
    elif unit == "gb":
        return int(size_val * 1024**3)
    elif unit == "tb":
        return int(size_val * 1024**4)
    elif unit == "pb":
        return int(size_val * 1024**5)


def getattr_rec(obj: Any, attr_str: str) -> Any:
    """Recursively get a nested attribute from an object.

    Args:
        obj: Object to get attribute from.
        attr_str: Dot-separated attribute path (e.g., 'a.b.c').

    Returns:
        The value of the nested attribute.
    """
    """
    Recursively retrieves an attribute from an object based on a dot-separated string.

    Args:
        obj: The object from which to retrieve the attribute.
        attr_str: A dot-separated string representing the attribute to retrieve.

    Returns:
        The value of the attribute.

    Raises:
        AttributeError: If the attribute does not exist.
        ValueError: If attr_str is not a string or is empty.
    """
    if not isinstance(attr_str, str):
        raise ValueError("attr_str must be a string")

    if not attr_str.strip():
        raise ValueError("attr_str cannot be empty")

    attrs = attr_str.split(".")
    for attr in attrs:
        if not attr.strip():
            raise ValueError("attribute name cannot be empty")
        obj = getattr(obj, attr)
    return obj


def setattr_rec(obj: Any, attr_str: str, value: Any) -> None:
    """Recursively set a nested attribute on an object.

    Args:
        obj: Object to set attribute on.
        attr_str: Dot-separated attribute path (e.g., 'a.b.c').
        value: Value to set the attribute to.

    Returns:
        None
    """
    """
    Recursively sets the value of an attribute in an object.

    Args:
        obj (object): The object to set the attribute in.
        attr_str (str): The attribute string in dot notation (e.g., "attr1.attr2.attr3").
        value: The value to set the attribute to.

    Returns:
        None

    Raises:
        ValueError: If attr_str is not a string or is empty.
    """
    if not isinstance(attr_str, str):
        raise ValueError("attr_str must be a string")

    if not attr_str.strip():
        raise ValueError("attr_str cannot be empty")

    attrs = attr_str.split(".")
    for attr in attrs[:-1]:
        if not attr.strip():
            raise ValueError("attribute name cannot be empty")
        obj = getattr(obj, attr)

    if not attrs[-1].strip():
        raise ValueError("attribute name cannot be empty")
    setattr(obj, attrs[-1], value)


def delattr_rec(obj: Any, attr_str: str) -> None:
    """Recursively delete a nested attribute from an object.

    Args:
        obj: Object to delete attribute from.
        attr_str: Dot-separated attribute path (e.g., 'a.b.c').

    Returns:
        None
    """
    """
    Recursively deletes an attribute from an object.

    Args:
        obj: The object from which to delete the attribute.
        attr_str: A string representing the attribute to be deleted.
                  The attribute can be nested using dot notation (e.g., 'attr1.attr2.attr3').

    Raises:
        AttributeError: If the attribute does not exist.
        ValueError: If attr_str is not a string or is empty.

    Example:
        obj = SomeObject()
        attr_str = 'attr1.attr2.attr3'
        delattr_rec(obj, attr_str)
    """
    if not isinstance(attr_str, str):
        raise ValueError("attr_str must be a string")

    if not attr_str.strip():
        raise ValueError("attr_str cannot be empty")

    attrs = attr_str.split(".")
    for attr in attrs[:-1]:
        if not attr.strip():
            raise ValueError("attribute name cannot be empty")
        obj = getattr(obj, attr)

    if not attrs[-1].strip():
        raise ValueError("attribute name cannot be empty")
    delattr(obj, attrs[-1])


def get_nested_keys(d: dict, parent_key: str = "") -> list[str]:
    """Get all nested keys from a dictionary as dot-separated paths.

    Args:
        d: Dictionary to extract keys from.
        parent_key: Parent key prefix for nested keys.

    Returns:
        List of all keys as dot-separated paths.
    """
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
    """Unify multiple PyArrow schemas into a single Polars schema.

    This function takes multiple schemas and creates a unified schema
    that can accommodate all fields from all input schemas.

    Args:
        schemas: List of PyArrow schemas to unify.

    Returns:
        Unified Polars schema.
    """
    """
    Unifies a list of Pyarrow schemas into a single schema.

    Args:
        schemas (list[pa.Schema]): List of PyArrow schemas.

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

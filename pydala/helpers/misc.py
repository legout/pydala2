import os
import re

import pendulum as pdl
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import tqdm
from joblib import Parallel, delayed

from .datetime import timestamp_from_string

# from .schema import shrink_large_string, convert_timestamp
# from .polars_ext import pl


def str2pyarrow_filter(string: str, schema: pa.Schema):
    """
    Generates a filter expression for PyArrow based on a given string and schema.

    Parameters:
        string (str): The string containing the filter expression.
        schema (pa.Schema): The PyArrow schema used to validate the filter expression.

    Returns:
        PyArrow.Expression: The generated filter expression.

    """

    def _parse_part(part):
        split_pattern = (
            "<=|"
            "<|"
            ">=|"
            ">|"
            "=|"
            "!=|"
            "\s+[n,N][o,O][t,T]\s+[i,I][n,N]\s+|"
            "\s+[i,I][n,N]\s+|"
            "\s+[i,I][s,S]\s+[n,N][o,O][t,T]\s+[n,N][u,U][l,L]{2}\s+|"
            "\s+[i,I][s,S]\s+[n,N][u,U][l,L]{2}\s+"
        )
        sign = re.findall(split_pattern, part)[0]
        # print(sign)
        # if "<" in sign or ">" in sign or "=" in sign or "!" in sign:
        field, val = [p.strip() for p in re.split(f"\s*{sign}\s*", part)]
        # else:
        #    field, val = [p.strip() for p in re.split(f"\s+{sign}\s+", part)]

        # print(field, val)
        type_ = schema.field(field).type
        if "(" in val:
            val = eval(val)

        if pa.types.is_time(type_):
            val = timestamp_from_string(val, exact=True)
            if isinstance(val, pdl.DateTime):
                val = val.time()

        elif pa.types.is_date(type_):
            val = timestamp_from_string(val, exact=True)
            if isinstance(val, pdl.DateTime):
                val = val.date()

        elif pa.types.is_timestamp(type_):
            val = timestamp_from_string(val, exact=True)

        elif pa.types.is_integer_value(type_) or pa.types.is_integer(type_):
            if isinstance(val, str):
                val = int(float(val.strip("'").replace(",", ".")))
            elif isinstance(val, tuple):
                val = tuple(
                    [
                        (
                            int(float(val_.strip(",").replace(",", ".")))
                            if isinstance(val_, str)
                            else val_
                        )
                        for val_ in val
                    ]
                )
            elif isinstance(val, list):
                val = [
                    (
                        int(float(val_.strip("'").replace(",", ".")))
                        if isinstance(val_, str)
                        else val_
                    )
                    for val_ in val
                ]

        elif pa.types.is_float_value(type_) or pa.types.is_floating(type_):
            if isinstance(val, str):
                val = float(val.strip("'").replace(",", "."))
            elif isinstance(val, tuple):
                val = tuple(
                    [
                        (
                            float(val_.strip(",").replace(",", "."))
                            if isinstance(val_, str)
                            else val_
                        )
                        for val_ in val
                    ]
                )
            elif isinstance(val, list):
                val = [
                    (
                        float(val_.strip("'").replace(",", "."))
                        if isinstance(val_, str)
                        else val_
                    )
                    for val_ in val
                ]

        elif pa.types.is_boolean(type_):
            val = bool(val.strip("'").replace(",", "."))

        if isinstance(val, str):
            val = val.strip("'")

        if sign == ">=":
            return pc.field(field) >= val
        elif sign == ">":
            return pc.field(field) > val
        elif sign == "<=":
            return pc.field(field) <= val
        elif sign == "<":
            return pc.field(field) < val
        elif sign == "=":
            return pc.field(field) == val
        elif sign == "!=":
            return pc.field(field) != val
        elif sign.lower() == "in":
            return pc.field(field).isin(val)
        elif sign.lower() == "not in":
            return ~pc.field(field).isin(val)
        elif sign.lower() == "is null":
            return pc.field(field).is_null(nan_is_null=True)
        elif sign.lower() == "is not null":
            return ~pc.field(field).is_null(nan_is_null=True)

    parts = re.split(
        (
            "\s+[a,A][n,N][d,D] [n,N][o,O][t,T]\s+|"
            "\s+[a,A][n,N][d,D]\s+|"
            "\s+[o,O][r,R] [n,N][o,O][t,T]\s+|"
            "\s+[o,O][r,R]\s+"
        ),
        string,
    )
    operators = [
        op.strip()
        for op in re.findall(
            (
                "\s+[a,A][n,N][d,D]\s+[n,N][o,O][t,T]\s+|"
                "\s+[a,A][n,N][d,D]\s+|"
                "[o,O][r,R]\s+[n,N][o,O][t,T]\s+|"
                "\s+[o,O][r,R]\s+"
            ),
            string,
        )
    ]

    # print(parts, operators)

    if len(parts) == 1:
        return _parse_part(parts[0])

    expr = _parse_part(parts[0])

    for part, operator in zip(parts[1:], operators):
        if operator.lower() == "and":
            expr = expr & _parse_part(part)
        elif operator.lower() == "and not":
            expr = expr & ~_parse_part(part)
        elif operator.lower() == "or":
            expr = expr | _parse_part(part)
        elif operator.lower() == "or not":
            expr = expr | ~_parse_part(part)

    return expr


def get_timestamp_column(df: pl.DataFrame | pl.LazyFrame | pa.Table) -> str | list[str]:
    if isinstance(df, pa.Table):
        df = pl.from_arrow(df)
    return [
        name
        for name, type_ in df.schema.items()
        if type_ in [pl.Time(), pl.Datetime(), pl.Date()]
    ]


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
        path = os.path.dirname(path)

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

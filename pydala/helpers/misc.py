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
    def _parse_part(part):
        split_pattern = (
            "<=|<|>=|>|=|!=|\s+[i,I][n,N]\s+|\s+[i,I][s,S] [n,N][u,U][l,L]{2}\s+"
        )
        sign = re.findall(split_pattern, part)[0]
        # print(sign)
        field, val = [p.strip() for p in re.split(f"\s*{sign}\s*", part)]
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

        elif pa.types.is_integer_value(type_):
            val = int(val.strip("'").replace(",", "."))

        elif pa.types.is_float_value(type_) or pa.types.is_floating(type_):
            val = float(val.strip("'").replace(",", "."))

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
        elif sign.lower() == "is null":
            return pc.field(field).is_null(nan_is_null=True)

    parts = re.split(
        "\s+[a,A][n,N][d,D] [n,N][o,O][t,T]\s+|\s+[a,A][n,N][d,D]\s+|\s+[o,O][r,R] [n,N][o,O][t,T]\s+|\s+[o,O][r,R]\s+",
        string,
    )
    operators = re.findall(
        "[a,A][n,N][d,D] [n,N][o,O][t,T]|[a,A][n,N][d,D]|[o,O][r,R] [n,N][o,O][t,T]|[o,O][r,R]",
        string,
    )

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
        df = pl.from_arrow(df[0])
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


def partition_by(
    df: pl.DataFrame | pl.LazyFrame,
    timestamp_column: str | None = None,
    columns: str | list[str] | None = None,
    strftime: str | list[str] | None = None,
    timedelta: str | list[str] | None = None,
    num_rows: int | None = None,
) -> list[tuple[dict, pl.DataFrame | pl.LazyFrame]]:
    if timestamp_column is None:
        timestamp_column = get_timestamp_column(df)
        if len(timestamp_column):
            timestamp_column = timestamp_column[0]

    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        columns_ = columns.copy()
    else:
        columns_ = []

    drop_columns = columns_.copy()

    if strftime is not None:
        if isinstance(strftime, str):
            strftime = [strftime]

        df = df.with_striftime_columns(
            timestamp_column=timestamp_column, strftime=strftime
        )
        strftime_columns = [
            f"_strftime_{strftime_.replaace('%', '')}_" for strftime_ in strftime
        ]
        columns_ += strftime_columns
        drop_columns += strftime_columns

    if timedelta:
        if isinstance(timedelta, str):
            timedelta = [timedelta]

        df = df.with_duration_columns(
            timestamp_column=timestamp_column, timedelta=timedelta
        )
        timedelta_columns = [f"_timedelta_{timedelta_}_" for timedelta_ in timedelta]
        columns_ += timedelta_columns
        drop_columns += timedelta_columns

    if num_rows:
        df = df.with_row_count_ext(over=columns).with_columns(
            (pl.col("row_nr") - 1) // num_rows
        )
        columns_ += ["row_nr"]
        drop_columns += ["row_nr"]

    if columns_:
        datetime_columns = {
            col: col in [col.lower() for col in columns_]
            for col in [
                "year",
                "month",
                "week",
                "yearday",
                "monthday",
                "weekday",
                "strftime",
            ]
            if col not in [table_col.lower() for table_col in df.columns]
        }
        if len(datetime_columns) and timestamp_column:
            df = df.with_datepart_columns(
                timestamp_column=timestamp_column, **datetime_columns
            )

        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        columns_ = [col for col in columns_ if col in df.columns]

        partitions = [
            (p.select(columns_).unique().to_dicts()[0], p.drop(drop_columns))
            for p in df.partition_by(
                by=columns_,
                as_dict=False,
                maintain_order=True,
            )
        ]

        return partitions

    return [({}, df)]


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

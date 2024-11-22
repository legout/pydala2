import datetime as dt
import re
from functools import lru_cache

import pendulum as pdl
import polars as pl
import polars.selectors as cs
import pyarrow as pa


def get_timestamp_column(df: pl.DataFrame | pl.LazyFrame | pa.Table) -> str | list[str]:
    if isinstance(df, pa.Table):
        df = pl.from_arrow(df).lazy()

    # if isinstance(df, pl.LazyFrame):
    #    return df.collect_schema().names()

    return df.select(cs.datetime() | cs.date()).collect_schema().names()


def get_timedelta_str(timedelta_string: str, to: str = "polars") -> str:
    polars_timedelta_units = [
        "ns",
        "us",
        "ms",
        "s",
        "m",
        "h",
        "d",
        "w",
        "mo",
        "y",
    ]
    duckdb_timedelta_units = [
        "nanosecond",
        "microsecond",
        "millisecond",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
    ]

    unit = re.sub("[0-9]", "", timedelta_string).strip()
    val = timedelta_string.replace(unit, "").strip()
    if to == "polars":
        return (
            timedelta_string
            if unit in polars_timedelta_units
            else val
            + dict(zip(duckdb_timedelta_units, polars_timedelta_units))[
                re.sub("s$", "", unit)
            ]
        )

    if unit in polars_timedelta_units:
        return (
            f"{val} " + dict(zip(polars_timedelta_units, duckdb_timedelta_units))[unit]
        )

    return f"{val} " + re.sub("s$", "", unit)


@lru_cache(maxsize=128)
def timestamp_from_string(
    timestamp: str,
    tz: str | None = None,
    exact: bool = True,
    strict: bool = False,
    naive: bool = False,
) -> pdl.DateTime | pdl.Date | pdl.Time | dt.datetime | dt.date | dt.time:
    """
    Converts a string like "2023-01-01 10:00:00" into a datetime.datetime object.

    Args:
        string (str): The string representation of the timestamp, e.g. "2023-01-01 10:00:00".
        tz (str, optional): The timezone to use for the timestamp. Defaults to None.
        exact (bool, optional): Whether to use exact parsing. Defaults to True.
        strict (bool, optional): Whether to use strict parsing. Defaults to False.
        naive (bool, optional): Whether to return a naive datetime without a timezone. Defaults to False.

    Returns:
        datetime.datetime: The datetime object.
    """
    # Extract the timezone from the string if not provided
    # tz = extract_timezone(timestamp) if tz is None else tz
    # timestamp = timestamp.replace(tz, "").strip() if tz else timestamp

    pdl_timestamp = pdl.parse(timestamp, exact=exact, strict=strict)

    if isinstance(pdl_timestamp, pdl.DateTime):
        if tz is not None:
            pdl_timestamp = pdl_timestamp.naive().set(tz=tz)
        if naive or tz is None:
            pdl_timestamp = pdl_timestamp.naive()

    return pdl_timestamp


def timedelta_from_string(
    timedelta_string: str, as_timedelta
) -> pdl.Duration | dt.timedelta:
    """
    Converts a string like "2d10s" into a datetime.timedelta object.

    Args:
        string (str): The string representation of the timedelta, e.g. "2d10s".

    Returns:
        datetime.timedelta: The timedelta object.
    """
    # Extract the numeric value and the unit from the string
    matches = re.findall(r"(\d+)([a-zA-Z]+)", timedelta_string)
    if not matches:
        raise ValueError("Invalid timedelta string")

    # Initialize the timedelta object
    delta = pdl.duration()

    # Iterate over each match and accumulate the timedelta values
    for value, unit in matches:
        # Map the unit to the corresponding timedelta attribute
        unit_mapping = {
            "us": "microseconds",
            "ms": "milliseconds",
            "s": "seconds",
            "m": "minutes",
            "h": "hours",
            "d": "days",
            "w": "weeks",
            "mo": "months",
            "y": "years",
        }
        if unit not in unit_mapping:
            raise ValueError("Invalid timedelta unit")

        # Update the timedelta object
        kwargs = {unit_mapping[unit]: int(value)}
        delta += pdl.duration(**kwargs)

    return delta.as_timedelta if as_timedelta else delta


# def extract_timezone(timestamp_string):
#     """
#     Extracts the timezone from a timestamp string.

#     Args:
#         timestamp_string (str): The input timestamp string.

#     Returns:
#         str: The extracted timezone.
#     """
#     pattern = r"\b([a-zA-Z]+/{0,1}[a-zA-Z_ ]*)\b"  # Matches the timezone portion
#     match = re.search(pattern, timestamp_string)
#     if match:
#         timezone = match.group(0)
#         return timezone
#     else:
#         return None

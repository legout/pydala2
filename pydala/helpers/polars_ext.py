import pandas as pd
import polars as pl

from .datetime import get_timedelta_str
from .misc import get_timestamp_column
from functools import partial

# import string


def unnest_all(df: pl.DataFrame, seperator="_", fields: list[str] | None = None):
    def _unnest_all(struct_columns):
        if fields is not None:
            return (
                df.with_columns(
                    [
                        pl.col(col).struct.rename_fields(
                            [
                                f"{col}{seperator}{field_name}"
                                for field_name in df[col].struct.fields
                            ]
                        )
                        for col in struct_columns
                    ]
                )
                .unnest(struct_columns)
                .select(
                    list(set(df.columns) - set(struct_columns))
                    + sorted(
                        [
                            f"{col}{seperator}{field_name}"
                            for field_name in fields
                            for col in struct_columns
                        ]
                    )
                )
            )

        return df.with_columns(
            [
                pl.col(col).struct.rename_fields(
                    [
                        f"{col}{seperator}{field_name}"
                        for field_name in df[col].struct.fields
                    ]
                )
                for col in struct_columns
            ]
        ).unnest(struct_columns)

    struct_columns = [
        col for col in df.columns if df[col].dtype == pl.Struct()
    ]  # noqa: F821
    while len(struct_columns):
        df = _unnest_all(struct_columns=struct_columns)
        struct_columns = [col for col in df.columns if df[col].dtype == pl.Struct()]
    return df


def _opt_dtype_(s: pl.Series, strict: bool = True) -> pl.Series:
    try:
        if (
            s.str.contains("^[0-9,\.-]{1,}$") | s.is_null() | s.str.contains("^$")
        ).all():
            s = (
                s.str.replace_all(",", ".")
                .str.replace_all("^0$", "+0")
                .str.strip_chars_start("0")
                .str.replace_all("\.0*$", "")
            )
            if s.dtype == pl.Utf8():
                s = s.set(s == "-", None)
                s = s.set(s == "", None)
            if (
                s.str.contains("\.").any()
                # | s.is_null().any() # null / None is valid in Int
                # | s.str.contains("^$").any()
                | s.str.contains("NaN").any()
            ):
                s = (
                    # s.str.replace("^$", pl.lit("NaN"))
                    s.cast(pl.Float64(), strict=True).shrink_dtype()
                )
            else:
                if (s.str.lengths() > 0).all():
                    s = s.cast(pl.Int64(), strict=True).shrink_dtype()
        # cast str to datetime
        elif (
            s.str.contains("^\d{4}-\d{2}-\d{2}$")
            | s.str.contains("^\d{1,2}\/\d{1,2}\/\d{4}$")
            | s.str.contains("^\d{4}-\d{2}-\d{2}T{0,1}\s{0,1}\d{2}:\d{2}:\d{0,2}$")
            | s.str.contains(
                "^\d{4}-\d{2}-\d{2}T{0,1}\s{0,1}\d{2}:\d{2}:\d{2}\.\d{1,}$"
            )
            | s.str.contains(
                "^\d{4}-\d{2}-\d{2}T{0,1}\s{0,1}\d{2}:\d{2}:\d{2}\.\d{1,}\w{0,1}\+\d{0,2}:\d{0,2}:\d{0,2}$"
            )
            | s.is_null()
            | s.str.contains("^$")
        ).all() and s.dtype == pl.Utf8():
            s = pl.Series(name=s.name, values=pd.to_datetime(s)).cast(pl.Datetime("us"))

        elif s.str.contains("^[T,t]rue|[F,f]alse$").all():
            s = s.str.contains("^[T,t]rue$", strict=True)

    except Exception as e:
        if strict:
            e.add_note(
                "if you were trying to cast Utf8 to temporal dtypes, consider using `strptime` or setting `strict=False`"
            )
            raise e

    return s


def opt_dtype(
    df: pl.DataFrame, exclude: str | list[str] | None = None, strict: bool = True
) -> pl.DataFrame:
    _opt_dtype_strict = partial(_opt_dtype_, strict=strict)
    _opt_dtype_not_strict = partial(_opt_dtype_, strict=False)
    return (
        df.with_columns(
            pl.all()
            .exclude(exclude)
            .map(_opt_dtype_strict if strict else _opt_dtype_not_strict)
        )
        if exclude is not None
        else df.with_columns(
            pl.all().map(_opt_dtype_strict if strict else _opt_dtype_not_strict)
        )
    )


def explode_all(df: pl.DataFrame | pl.LazyFrame):
    list_columns = [col for col in df.columns if df[col].dtype == pl.List()]
    for col in list_columns:
        df = df.explode(col)
    return df


def with_strftime_columns(
    df: pl.DataFrame | pl.LazyFrame,
    timestamp_column: str,
    strftime: str | list[str],
    column_names: str | list[str] | None = None,
):
    if isinstance(strftime, str):
        strftime = [strftime]
    if isinstance(column_names, str):
        column_names = [column_names]

    if column_names is None:
        column_names = [
            f"_strftime_{strftime_.replace('%', '').replace('-', '_')}_"
            for strftime_ in strftime
        ]
    return df.with_columns(
        [
            pl.col(timestamp_column).dt.strftime(strftime_).alias(column_name)
            for strftime_, column_name in zip(strftime, column_names)
        ]
    )


def with_truncated_columns(
    df: pl.DataFrame | pl.LazyFrame,
    timestamp_column: str,
    truncate: str | list[str],
    column_names: str | list[str] | None = None,
):
    if isinstance(truncate, str):
        truncate = [truncate]

    if isinstance(column_names, str):
        column_names = [column_names]

    if column_names is None:
        column_names = [
            f"_truncated_{truncate_.replace(' ', '_')}_" for truncate_ in truncate
        ]

    truncate = [get_timedelta_str(truncate_, to="polars") for truncate_ in truncate]
    return df.with_columns(
        [
            pl.col(timestamp_column).dt.truncate(truncate_).alias(column_name)
            for truncate_, column_name in zip(truncate, column_names)
        ]
    )


def with_datepart_columns(
    df: pl.DataFrame | pl.LazyFrame,
    timestamp_column: str | None = None,
    year: bool = False,
    month: bool = False,
    week: bool = False,
    yearday: bool = False,
    monthday: bool = False,
    weekday: bool = False,
    strftime: str | None = None,
):
    if not timestamp_column:
        timestamp_column = get_timestamp_column(df)

    if strftime:
        if isinstance(strftime, str):
            strftime = [strftime]
        column_names = [
            f"_strftime_{strftime_.replace('%', '').replace('-', '_')}_"
            for strftime_ in strftime
        ]
    else:
        strftime = []
        column_names = []

    if year:
        strftime.append("%Y")
        column_names.append("year")
    if month:
        strftime.append("%m")
        column_names.append("month")
    if week:
        strftime.append("%W")
        column_names.append("week")
    if yearday:
        strftime.append("%j")
        column_names.append("year_day")
    if monthday:
        strftime.append("%d")
        column_names.append("month_day")
    if weekday:
        strftime.append("%a")
        column_names.append("week_day")

    return with_strftime_columns(
        df=df,
        timestamp_column=timestamp_column,
        strftime=strftime,
        column_names=column_names,
    )


def with_row_count(
    df: pl.DataFrame | pl.LazyFrame,
    over: str | list[str] | None = None,
):
    if over:
        if len(over) == 0:
            over = None

    if isinstance(over, str):
        over = [over]

    if over:
        return df.with_columns(pl.lit(1).alias("row_nr")).with_columns(
            pl.col("row_nr").cumsum().over(over)
        )
    else:
        return df.with_columns(pl.lit(1).alias("row_nr")).with_columns(
            pl.col("row_nr").cumsum()
        )


def delta(
    df1: pl.DataFrame | pl.LazyFrame,
    df2: pl.DataFrame | pl.LazyFrame,
    subset: str | list[str] | None = None,
    eager: bool = False,
) -> pl.LazyFrame:
    if subset is None:
        subset = df1.columns
    if isinstance(subset, str):
        subset = [subset]

    if isinstance(df1, pl.LazyFrame) and isinstance(df2, pl.DataFrame):
        df2 = df2.lazy()

    elif isinstance(df1, pl.DataFrame) and isinstance(df2, pl.LazyFrame):
        df1 = df1.lazy()

    df = (
        pl.concat(
            [
                df1.with_columns(pl.lit(1).alias("df")).with_row_count(),
                df2.select(df1.columns)
                .with_columns(pl.lit(2).alias("df"))
                .with_row_count(),
            ],
            how="vertical_relaxed",
        )
        .filter((pl.count().over(subset) == 1) & (pl.col("df") == 1))
        .select(pl.exclude(["df", "row_nr"]))
    )

    if eager and isinstance(df, pl.LazyFrame):
        return df.collect()
    return df


pl.DataFrame.unnest_all = unnest_all
pl.DataFrame.explode_all = explode_all
pl.DataFrame.opt_dtype = opt_dtype
pl.DataFrame.with_row_count_ext = with_row_count
pl.DataFrame.with_datepart_columns = with_datepart_columns
pl.DataFrame.with_duration_columns = with_truncated_columns
pl.DataFrame.with_striftime_columns = with_strftime_columns
pl.DataFrame.delta = delta

pl.LazyFrame.unnest_all = unnest_all
pl.LazyFrame.explode_all = explode_all
pl.LazyFrame.opt_dtype = opt_dtype
pl.LazyFrame.with_row_count_ext = with_row_count
pl.LazyFrame.with_datepart_columns = with_datepart_columns
pl.LazyFrame.with_duration_columns = with_truncated_columns
pl.LazyFrame.with_striftime_columns = with_strftime_columns
pl.LazyFrame.delta = delta

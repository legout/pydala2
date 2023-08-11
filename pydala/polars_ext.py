import polars as pl

from .helpers import get_timedelta_str, get_timestamp_column


def unnest_all(df: pl.DataFrame, seperator="_", fields: list[str] | None = None):
    def _unnest_all(struct_columns):
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

    struct_columns = [
        col for col in df.columns if df[col].dtype == pl.Struct()
    ]  # noqa: F821
    while len(struct_columns):
        df = _unnest_all(struct_columns=struct_columns)
        struct_columns = [col for col in df.columns if df[col].dtype == pl.Struct()]
    return df


def _opt_dtype(s: pl.Series) -> pl.Series:
    if (s.str.contains("^[0-9,\.-]{1,}$") | s.is_null()).all():
        s = s.str.replace_all(",", ".").str.replace_all(".0$", "").str.replace_all("^-$", "NaN")
        if s.str.contains("\.").any() | s.is_null().any() | s.str.contains("^$").any() | s.str.contains("NaN").any():
            s = (
                s.str.replace("^$", pl.lit("NaN"))
                .cast(pl.Float64(), strict=True)
                .shrink_dtype()
            )
        else:
            if (s.str.lengths() > 0).all():
                s = s.cast(pl.Int64(), strict=True).shrink_dtype()
    elif s.str.contains("^[T,t]rue|[F,f]alse$").all():
        s = s.str.contains("^[T,t]rue$", strict=True)

    return s


def opt_dtype(df: pl.DataFrame, exclude=None) -> pl.DataFrame:
    return df.with_columns(pl.all().map(_opt_dtype))


def explode_all(df: pl.DataFrame | pl.LazyFrame):
    list_columns = [col for col in df.columns if df[col].dtype == pl.List()]
    for col in list_columns:
        df = df.explode(col)
    return df


pl.DataFrame.unnest_all = unnest_all
pl.DataFrame.explode_all = explode_all


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


def with_duration_columns(
    df: pl.DataFrame | pl.LazyFrame,
    timestamp_column: str,
    timedelta: str | list[str],
    column_names: str | list[str] | None = None,
):
    if isinstance(timedelta, str):
        timedelta = [timedelta]

    if isinstance(column_names, str):
        column_names = [column_names]

    if column_names is None:
        column_names = [
            f"_timebucket_{timedelta_.replace(' ', '_')}_" for timedelta_ in timedelta
        ]

    timedelta = [get_timedelta_str(timedelta_, to="polars") for timedelta_ in timedelta]
    return df.with_columns(
        [
            pl.col(timestamp_column).dt.truncate(timedelta_).alias(column_name)
            for timedelta_, column_name in zip(timedelta, column_names)
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


pl.DataFrame.unnest_all = unnest_all
pl.DataFrame.explode_all = explode_all
pl.DataFrame.opt_dtype = opt_dtype
pl.DataFrame.with_row_count_ext = with_row_count
pl.DataFrame.with_datepart_columns = with_datepart_columns
pl.DataFrame.with_duration_columns = with_duration_columns
pl.DataFrame.with_striftime_columns = with_strftime_columns

pl.LazyFrame.unnest_all = unnest_all
pl.LazyFrame.explode_all = explode_all
pl.LazyFrame.opt_dtype = opt_dtype
pl.LazyFrame.with_row_count_ext = with_row_count
pl.LazyFrame.with_datepart_columns = with_datepart_columns
pl.LazyFrame.with_duration_columns = with_duration_columns
pl.LazyFrame.with_striftime_columns = with_strftime_columns

from __future__ import annotations

import datetime as dt
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc
import polars as pl
from sqlglot import exp, parse_one

from .datetime import timestamp_from_string


def _schema_names(schema: Any) -> list[str]:
    names = schema.names
    return cast(list[str], names() if callable(names) else list(names))


def sql2pyarrow_filter(string: str, schema: pa.Schema) -> Any:
    """Convert a SQL boolean expression into a PyArrow compute expression."""

    def parse_value(val: Any, type_: Any) -> Any:
        if isinstance(val, (tuple, list)):
            return type(val)(parse_value(v, type_) for v in val)

        val = str(val).strip().strip("'\"")

        if pa.types.is_timestamp(type_):
            return timestamp_from_string(val, tz=type_.tz)
        if pa.types.is_date(type_):
            parsed = timestamp_from_string(val)
            return parsed.date() if isinstance(parsed, dt.datetime) else parsed
        if pa.types.is_time(type_):
            parsed = timestamp_from_string(val)
            return parsed.time() if isinstance(parsed, dt.datetime) else parsed
        if pa.types.is_integer(type_):
            try:
                return int(float(val.replace(",", ".")))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid integer literal: {val}") from exc
        if pa.types.is_floating(type_):
            try:
                return float(val.replace(",", "."))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid floating literal: {val}") from exc
        if pa.types.is_boolean(type_):
            return val.lower() in ("true", "1", "yes")
        return val

    def _get_field_type_from_context(expr: Any) -> Any:
        names = _schema_names(schema)
        if isinstance(expr.this, exp.Column):
            field_name = expr.this.name
            if field_name in names:
                return schema.field(field_name).type
        if isinstance(expr.expression, exp.Column):
            field_name = expr.expression.name
            if field_name in names:
                return schema.field(field_name).type
        return None

    def _in_values(expr: exp.In, field_type: Any) -> Any:
        expressions = expr.args.get("expressions")
        if expressions is None and getattr(expr, "expression", None) is not None:
            expressions = getattr(expr.expression, "expressions", None)
        if expressions is None:
            return _convert_expression(expr.expression, field_type)
        return [_convert_expression(e, field_type) for e in expressions]

    def _convert_expression(expr: Any, field_type: Any = None) -> Any:
        if isinstance(expr, exp.Column):
            field_name = expr.name
            if field_name not in _schema_names(schema):
                raise ValueError(f"Unknown field: {field_name}")
            return pc.field(field_name)

        if isinstance(expr, exp.Literal):
            if field_type:
                return parse_value(expr.this, field_type)
            return expr.this

        if isinstance(expr, exp.Boolean):
            return bool(expr.this)

        if isinstance(expr, exp.Null):
            return None

        if isinstance(expr, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
            context_type = _get_field_type_from_context(expr)
            left = _convert_expression(expr.this, context_type)
            right = _convert_expression(expr.expression, context_type)
            if isinstance(expr, exp.EQ):
                return left == right
            if isinstance(expr, exp.NEQ):
                return left != right
            if isinstance(expr, exp.GT):
                return left > right
            if isinstance(expr, exp.GTE):
                return left >= right
            if isinstance(expr, exp.LT):
                return left < right
            return left <= right

        if isinstance(expr, exp.In):
            context_type = _get_field_type_from_context(expr)
            left = _convert_expression(expr.this, context_type)
            return left.isin(_in_values(expr, context_type))

        if isinstance(expr, exp.Not):
            inner = expr.this
            if isinstance(inner, exp.In):
                context_type = _get_field_type_from_context(inner)
                left = _convert_expression(inner.this, context_type)
                return ~left.isin(_in_values(inner, context_type))
            if isinstance(inner, exp.Is):
                return ~_convert_expression(inner.this).is_null(nan_is_null=True)
            return ~_convert_expression(inner)

        if isinstance(expr, exp.Is):
            return _convert_expression(expr.this).is_null(nan_is_null=True)

        if isinstance(expr, exp.And):
            return _convert_expression(expr.this) & _convert_expression(expr.expression)

        if isinstance(expr, exp.Or):
            return _convert_expression(expr.this) | _convert_expression(expr.expression)

        if isinstance(expr, exp.Paren):
            return _convert_expression(expr.this)

        raise ValueError(f"Unsupported expression type: {type(expr)}")

    try:
        return _convert_expression(parse_one(string))
    except Exception as exc:
        raise ValueError(f"Failed to parse SQL expression: {exc}") from exc


def sql2polars_filter(string: str, schema: pl.Schema) -> pl.Expr:
    """Convert a SQL boolean expression into a Polars expression."""

    def parse_value(val: Any, dtype: Any) -> Any:
        if isinstance(val, (tuple, list)):
            return type(val)(parse_value(v, dtype) for v in val)

        val = str(val).strip().strip("'\"")

        if dtype == pl.Datetime:
            return timestamp_from_string(val, tz=dtype.time_zone)
        if dtype == pl.Date:
            parsed = timestamp_from_string(val)
            return parsed.date() if isinstance(parsed, dt.datetime) else parsed
        if dtype == pl.Time:
            parsed = timestamp_from_string(val)
            return parsed.time() if isinstance(parsed, dt.datetime) else parsed
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            try:
                return int(float(val.replace(",", ".")))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid integer literal: {val}") from exc
        if dtype in (pl.Float32, pl.Float64):
            try:
                return float(val.replace(",", "."))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid floating literal: {val}") from exc
        if dtype == pl.Boolean:
            return val.lower() in ("true", "1", "yes")
        return val

    def _get_field_type_from_context(expr: Any) -> Any:
        names = _schema_names(schema)
        if isinstance(expr.this, exp.Column):
            field_name = expr.this.name
            if field_name in names:
                return schema[field_name]
        if isinstance(expr.expression, exp.Column):
            field_name = expr.expression.name
            if field_name in names:
                return schema[field_name]
        return None

    def _in_values(expr: exp.In, field_type: Any) -> Any:
        expressions = expr.args.get("expressions")
        if expressions is None and getattr(expr, "expression", None) is not None:
            expressions = getattr(expr.expression, "expressions", None)
        if expressions is None:
            return _convert_expression(expr.expression, field_type)
        return [_convert_expression(e, field_type) for e in expressions]

    def _convert_expression(expr: Any, field_type: Any = None) -> Any:
        if isinstance(expr, exp.Column):
            field_name = expr.name
            if field_name not in _schema_names(schema):
                raise ValueError(f"Unknown field: {field_name}")
            return pl.col(field_name)

        if isinstance(expr, exp.Literal):
            if field_type:
                return parse_value(expr.this, field_type)
            return expr.this

        if isinstance(expr, exp.Boolean):
            return bool(expr.this)

        if isinstance(expr, exp.Null):
            return None

        if isinstance(expr, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
            context_type = _get_field_type_from_context(expr)
            left = _convert_expression(expr.this, context_type)
            right = _convert_expression(expr.expression, context_type)
            if isinstance(expr, exp.EQ):
                return left == right
            if isinstance(expr, exp.NEQ):
                return left != right
            if isinstance(expr, exp.GT):
                return left > right
            if isinstance(expr, exp.GTE):
                return left >= right
            if isinstance(expr, exp.LT):
                return left < right
            return left <= right

        if isinstance(expr, exp.In):
            context_type = _get_field_type_from_context(expr)
            left = _convert_expression(expr.this, context_type)
            return left.is_in(_in_values(expr, context_type))

        if isinstance(expr, exp.Not):
            inner = expr.this
            if isinstance(inner, exp.In):
                context_type = _get_field_type_from_context(inner)
                left = _convert_expression(inner.this, context_type)
                return ~left.is_in(_in_values(inner, context_type))
            if isinstance(inner, exp.Is):
                return _convert_expression(inner.this).is_not_null()
            return ~_convert_expression(inner)

        if isinstance(expr, exp.Is):
            return _convert_expression(expr.this).is_null()

        if isinstance(expr, exp.And):
            return _convert_expression(expr.this) & _convert_expression(expr.expression)

        if isinstance(expr, exp.Or):
            return _convert_expression(expr.this) | _convert_expression(expr.expression)

        if isinstance(expr, exp.Paren):
            return _convert_expression(expr.this)

        raise ValueError(f"Unsupported expression type: {type(expr)}")

    try:
        return _convert_expression(parse_one(string))
    except Exception as exc:
        raise ValueError(f"Failed to parse SQL expression: {exc}") from exc


def get_table_names(sql_query: str) -> list[str]:
    """Extract table names from a SQL query."""
    return [table.name for table in parse_one(sql_query).find_all(exp.Table)]


__all__ = ["get_table_names", "sql2polars_filter", "sql2pyarrow_filter"]

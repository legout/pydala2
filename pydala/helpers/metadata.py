"""Helpers for conservative pruning from Parquet row-group metadata."""

from __future__ import annotations

import duckdb as _duckdb
import re
from typing import Any


_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_ATOM_RE = re.compile(
    r"^\s*(?P<column>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?P<operator>>=|<=|=|>|<)\s*(?P<literal>.+?)\s*$"
)
_NUMERIC_LITERAL_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
_DATE_LITERAL_RE = re.compile(
    r"^'(?P<date>\d{4}-\d{1,2}-\d{1,2})"
    r"(?P<time>[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?)?'$"
)


def _relation_columns(metadata_table: Any) -> set[str]:
    """Return relation column names, or an empty set when unavailable."""
    try:
        return set(metadata_table.columns)
    except (AttributeError, TypeError):
        return set()


def _struct_columns(metadata_table: Any, columns: set[str]) -> set[str]:
    """Identify metadata columns containing min/max statistics structs."""
    result: set[str] = set()
    for column in columns:
        try:
            if metadata_table.select(column).types[0].id == "struct":
                result.add(column)
        except (AttributeError, IndexError, TypeError, _duckdb.Error):
            # A malformed or unavailable metadata column must not be pruned.
            continue
    return result


def _unquoted_identifiers(expression: str) -> set[str]:
    """Extract identifiers while ignoring identifiers embedded in literals."""
    without_literals = re.sub(r"'(?:''|[^'])*'", " ", expression)
    return set(_IDENTIFIER_RE.findall(without_literals))


def _strip_outer_parentheses(expression: str) -> str:
    """Strip only balanced pairs wrapping an entire expression."""
    expression = expression.strip()
    while expression.startswith("(") and expression.endswith(")"):
        depth = 0
        quote = False
        wraps_expression = True
        index = 0
        while index < len(expression):
            char = expression[index]
            if char == "'":
                if quote and index + 1 < len(expression) and expression[index + 1] == "'":
                    index += 2
                    continue
                quote = not quote
            elif not quote:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0 and index != len(expression) - 1:
                        wraps_expression = False
                        break
            index += 1
        if quote or depth != 0 or not wraps_expression:
            break
        expression = expression[1:-1].strip()
    return expression


def _split_top_level_and(expression: str) -> list[str] | None:
    """Split a SQL expression on top-level AND, rejecting malformed syntax."""
    parts: list[str] = []
    start = 0
    depth = 0
    quote = False
    index = 0
    while index < len(expression):
        char = expression[index]
        if char == "'":
            if quote and index + 1 < len(expression) and expression[index + 1] == "'":
                index += 2
                continue
            quote = not quote
            index += 1
            continue
        if quote:
            index += 1
            continue
        if char == "(":
            depth += 1
            index += 1
            continue
        if char == ")":
            depth -= 1
            if depth < 0:
                return None
            index += 1
            continue
        if depth == 0 and expression[index : index + 3].lower() == "and":
            before = expression[index - 1] if index else " "
            after = expression[index + 3] if index + 3 < len(expression) else " "
            if not (before.isalnum() or before == "_") and not (
                after.isalnum() or after == "_"
            ):
                parts.append(expression[start:index].strip())
                start = index + 3
                index += 3
                continue
        index += 1
    if quote or depth != 0:
        return None
    parts.append(expression[start:].strip())
    return [part for part in parts if part]


def _parse_supported_atom(expression: str) -> tuple[str, str, str, str] | None:
    """Parse a simple comparison and classify its literal as numeric/date/timestamp."""
    expression = _strip_outer_parentheses(expression)
    match = _ATOM_RE.fullmatch(expression)
    if match is None:
        return None
    column = match.group("column")
    operator = match.group("operator")
    literal = match.group("literal").strip()
    if _NUMERIC_LITERAL_RE.fullmatch(literal):
        return column, operator, literal, "numeric"
    date_match = _DATE_LITERAL_RE.fullmatch(literal)
    if date_match is not None:
        kind = "timestamp" if date_match.group("time") is not None else "date"
        return column, operator, literal, kind
    return None


def _translate_statistics_atom(
    column: str, operator: str, literal: str, literal_kind: str
) -> str:
    """Translate a supported row predicate into a conservative range predicate."""
    cast = ""
    if literal_kind in {"date", "timestamp"}:
        cast_name = "DATE" if literal_kind == "date" else "TIMESTAMP"
        cast = f"::{cast_name}"
    value = f"{literal}{cast}"
    minimum = f"{column}.min"
    maximum = f"{column}.max"
    if operator in {">", ">="}:
        return f"({maximum}{cast}{operator}{value} OR {maximum} IS NULL)"
    if operator in {"<", "<="}:
        return f"({minimum}{cast}{operator}{value} OR {minimum} IS NULL)"
    return (
        f"({minimum}{cast}<={value} OR {minimum} IS NULL) AND "
        f"({maximum}{cast}>={value} OR {maximum} IS NULL)"
    )


def _plan_metadata_predicate(metadata_table: Any, filter_expr: str | None) -> str | None:
    """Plan a metadata predicate, returning ``None`` when safe translation is unknown.

    The returned expression is evaluated against a DuckDB relation whose struct
    columns contain ``min`` and ``max`` statistics.  ``None`` means callers must
    retain every physical file candidate; it is intentionally also used for an
    absent filter.
    """
    if filter_expr is None or not isinstance(filter_expr, str) or not filter_expr.strip():
        return None

    columns = _relation_columns(metadata_table)
    struct_columns = _struct_columns(metadata_table, columns)
    if not struct_columns:
        # No statistics columns are referenced by this relation; preserve the
        # existing direct metadata filtering behavior for partition/file fields.
        return filter_expr

    identifiers = _unquoted_identifiers(filter_expr)
    if not identifiers.intersection(struct_columns):
        return filter_expr

    predicates = _split_top_level_and(filter_expr.rstrip().rstrip(";"))
    if predicates is None:
        return None
    translated: list[str] = []
    translated_statistics = False
    for predicate in predicates:
        parsed = _parse_supported_atom(predicate)
        if parsed is None:
            # Non-statistics metadata predicates remain unchanged. Any
            # expression touching a statistics struct is unsafe to rewrite.
            if _unquoted_identifiers(predicate).intersection(struct_columns):
                return None
            translated.append(predicate.strip())
            continue
        column, operator, literal, literal_kind = parsed
        if column in struct_columns:
            translated.append(
                _translate_statistics_atom(column, operator, literal, literal_kind)
            )
            translated_statistics = True
        else:
            translated.append(predicate.strip())

    if not translated_statistics:
        return filter_expr
    return " AND ".join(translated)


def _plan_non_statistics_predicate(
    metadata_table: Any, filter_expr: str | None
) -> str | None:
    """Keep only metadata predicates independent of statistics structs."""
    if filter_expr is None or not isinstance(filter_expr, str) or not filter_expr.strip():
        return None

    columns = _relation_columns(metadata_table)
    struct_columns = _struct_columns(metadata_table, columns)
    if not struct_columns:
        return filter_expr
    if not _unquoted_identifiers(filter_expr).intersection(struct_columns):
        return filter_expr

    predicates = _split_top_level_and(filter_expr.rstrip().rstrip(";"))
    if predicates is None:
        return None

    non_statistics: list[str] = []
    for predicate in predicates:
        parsed = _parse_supported_atom(predicate)
        if parsed is not None and parsed[0] in struct_columns:
            continue
        if _unquoted_identifiers(predicate).intersection(struct_columns):
            continue
        non_statistics.append(predicate.strip())
    return " AND ".join(non_statistics) if non_statistics else None


def _prune_metadata_files(
    metadata_table: Any, filter_expr: str | None, files: list[str]
) -> list[str]:
    """Select physical candidate files using conservative metadata statistics."""
    physical_files = list(dict.fromkeys(files))
    planned = _plan_metadata_predicate(metadata_table, filter_expr)
    if planned is None:
        # An unsafe statistics expression may still carry an independent,
        # safely evaluable metadata predicate.
        planned = _plan_non_statistics_predicate(metadata_table, filter_expr)
        if planned is None:
            return physical_files

    try:
        scanned = metadata_table.filter(planned)
        selected = {row[0] for row in scanned.select("file_path").fetchall()}
    except (AttributeError, IndexError, TypeError, _duckdb.Error):
        # Any translation/type/schema failure must retain all physical files.
        return physical_files

    # Physical discovery is authoritative; stale sidecar rows cannot reappear.
    return [file_path for file_path in physical_files if file_path in selected]

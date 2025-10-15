"""Security utilities for PyDala2."""

import os
import posixpath
import re
import string
from typing import Any
from fsspec.core import split_protocol


def strip_protocol(path: str) -> str:
    """Strips the protocol from a given path.

    Args:
        path (str): The input path which may contain a protocol.
    Returns:
        str: The path without the protocol.
    """
    protocol, path = split_protocol(path)
    return path


def escape_sql_identifier(identifier: str) -> str:
    """
    Escape SQL identifiers to prevent SQL injection.

    Args:
        identifier: The SQL identifier to escape

    Returns:
        Escaped identifier safe for use in SQL queries

    Raises:
        ValueError: If identifier contains invalid characters
    """
    if not identifier:
        raise ValueError("Identifier cannot be empty")

    # Check for invalid characters
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
        raise ValueError(f"Invalid identifier: {identifier}")

    # Double quote identifiers for SQL standard compliance
    return f'"{identifier.replace('"', '""')}"'


def escape_sql_literal(value: Any) -> str:
    """
    Escape SQL literal values to prevent SQL injection.

    Args:
        value: The value to escape

    Returns:
        Escaped string literal safe for use in SQL queries
    """
    if value is None:
        return "NULL"

    if isinstance(value, str):
        # Escape single quotes by doubling them
        escaped = value.replace("'", "''")
        return f"'{escaped}'"

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"

    # For other types, convert to string and escape
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def validate_partition_name(name: str) -> bool:
    """
    Validate partition names for security.

    Args:
        name: Partition name to validate

    Returns:
        True if valid, False otherwise
    """
    if not name or len(name) > 255:
        return False

    # Only allow alphanumeric, underscore, and hyphen
    return re.match(r"^[a-zA-Z0-9_-]+$", name) is not None


def validate_partition_value(value: Any) -> bool:
    """
    Validate partition values for security.

    Args:
        value: Partition value to validate

    Returns:
        True if valid, False otherwise
    """
    if value is None:
        return True

    if isinstance(value, str):
        # Prevent path traversal and special characters
        if any(char in value for char in ["../", "..\\", "\0", "\n", "\r"]):
            return False
        return len(value) <= 1024

    if isinstance(value, (int, float, bool)):
        return True

    return False


def sanitize_filter_expression(expr: str) -> str:
    """
    Sanitize filter expressions to prevent SQL injection.

    Args:
        expr: Filter expression to sanitize

    Returns:
        Sanitized expression
    """
    # Remove potentially dangerous characters
    expr = expr.replace("\0", "").replace("\n", " ").replace("\r", " ")

    # Prevent comment injection
    expr = re.sub(r"--.*$", "", expr)
    expr = re.sub(r"/\*.*?\*/", "", expr, flags=re.DOTALL)

    # Balance quotes
    single_quotes = expr.count("'")
    if single_quotes % 2 != 0:
        raise ValueError("Unbalanced quotes in filter expression")

    return expr.strip()


def validate_path(path: str, base_path: str | None = None) -> str:
    """
    Validate and normalize a path to prevent directory traversal.

    Args:
        path: The path to validate
        base_path: Optional base path to validate against

    Returns:
        Normalized, safe path

    Raises:
        ValueError: If path contains traversal attempts or is invalid
    """
    path = strip_protocol(path) if path else path
    if not path:
        raise ValueError("Path cannot be empty")

    # Remove null bytes and other dangerous characters
    path = path.replace("\0", "").strip()

    # Normalize the path
    normalized = os.path.normpath(path)

    # Check for traversal attempts
    if ".." in normalized or normalized.startswith(("/", "\\")):
        raise ValueError(f"Path traversal attempt detected: {path}")

    # If base_path is provided, ensure the path stays within it
    if base_path:
        base_path = strip_protocol(base_path)
        base_normalized = os.path.normpath(base_path)
        full_path = os.path.join(base_normalized, normalized)
        common_path = os.path.commonpath([base_normalized, full_path])

        if common_path != base_normalized:
            raise ValueError(f"Path escapes base directory: {path}")

    return normalized


def safe_join(base_path: str, *paths: str) -> str:
    """
    Safely join paths, preventing directory traversal.

    Args:
        base_path: The base directory
        *paths: Path components to join

    Returns:
        Safe joined path

    Raises:
        ValueError: If resulting path would escape base directory
    """
    base_path = strip_protocol(base_path) if base_path else base_path
    if not base_path:
        raise ValueError("Base path cannot be empty")

    # Join all paths
    joined = os.path.join(base_path, *paths)

    # Normalize and validate
    normalized = os.path.normpath(joined)
    base_normalized = os.path.normpath(base_path)

    # Ensure the result stays within base path
    if not normalized.startswith(base_normalized):
        raise ValueError(f"Path traversal attempt in: {os.path.join(*paths)}")

    return normalized


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent directory traversal and invalid characters.

    Args:
        filename: The filename to sanitize

    Returns:
        Sanitized filename
    """
    if not filename:
        raise ValueError("Filename cannot be empty")

    # Remove path separators and dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", filename)

    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(". ")

    # Ensure it's not empty after sanitization
    if not sanitized:
        sanitized = "unnamed"

    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[: 255 - len(ext)] + ext

    return sanitized

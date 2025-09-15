"""
Sorting logic handler for Pydala table operations.
"""

from typing import Any, Dict, List, Tuple, Union


class SortHandler:
    """Handles sorting configuration for different data storage types."""

    @staticmethod
    def normalize_sort_input(
        sort_by: Union[str, List[str], List[Tuple[str, str]]]
    ) -> List[Tuple[str, str]]:
        """
        Normalize various sort input formats to a consistent format.

        Args:
            sort_by: Sorting specification in various formats

        Returns:
            List of tuples with (column_name, direction)
        """
        if isinstance(sort_by, str):
            # Handle comma-separated string with optional directions
            sort_by = [s.split(" ") for s in sort_by.split(",")]
            sort_by = [[s[0], "ascending"] if len(s) == 1 else s for s in sort_by]

        elif isinstance(sort_by, (list, tuple)):
            if not sort_by:
                return []

            if isinstance(sort_by[0], str):
                # List of strings (column names only)
                sort_by = [[s, "ascending"] if isinstance(s, str) else s for s in sort_by]

            if isinstance(sort_by[0], list):
                # List of lists, ensure proper format
                sort_by = [[s[0], s[1]] for s in sort_by]

        # Normalize direction strings
        normalized = []
        for col, direction in sort_by:
            direction_lower = direction.lower()
            if direction_lower in ["asc", "ascending"]:
                normalized.append((col, "ascending"))
            elif direction_lower in ["desc", "descending"]:
                normalized.append((col, "descending"))
            else:
                raise ValueError(f"Invalid sort direction: {direction}")

        return normalized

    @staticmethod
    def get_sort_by(
        sort_by: Union[str, List[str], List[Tuple[str, str]]],
        target_type: str = "pyarrow"
    ) -> Union[Dict, str]:
        """
        Get sorting configuration formatted for the target data storage type.

        Args:
            sort_by: Sorting specification
            target_type: Target data storage type ("pyarrow", "duckdb", "polars")

        Returns:
            Formatted sorting configuration
        """
        if sort_by is None:
            return None

        normalized = SortHandler.normalize_sort_input(sort_by)

        if target_type == "pyarrow":
            return {"sorting": normalized}

        elif target_type == "duckdb":
            # Convert to DuckDB ORDER BY syntax
            sort_parts = []
            for col, direction in normalized:
                dir_sql = "ASC" if direction == "ascending" else "DESC"
                sort_parts.append(f"{col} {dir_sql}")
            return ", ".join(sort_parts)

        elif target_type == "polars":
            # Split into columns and descending flags
            by = [col for col, _ in normalized]
            descending = [direction == "descending" for _, direction in normalized]
            return {"by": by, "descending": descending}

        else:
            raise ValueError(f"Unsupported target type: {target_type}")
# Utilities

Import helpers from their defining modules. Only `sort_schema` and `unify_schemas` are also exported from the top-level package.

| Module | Helpers |
| --- | --- |
| `pydala.schema` | `sort_schema`, `unify_schemas`, `convert_timestamp`, `replace_schema` |
| `pydala.helpers.sql` | `sql2pyarrow_filter`, `sql2polars_filter` (both take an expression and a schema) |
| `pydala.helpers.datetime` | `get_timestamp_column`, `timestamp_from_string`, `get_timedelta_str` |
| `pydala.helpers.misc` | `read_table`, `run_parallel`, `humanize_size`, `humanized_size_to_bytes` |
| `pydala.helpers.security` | `validate_path`, `safe_join`, `validate_partition_name`, `escape_sql_identifier`, `escape_sql_literal` |

For dataset and filesystem options, see [Core Classes](core.md) and [Filesystem](filesystem.md).

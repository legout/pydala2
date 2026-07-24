"""Characterization contract tests for pydala.schema transformations.

These tests pin the *current* behavior of the schema-tooling seam in
``pydala.schema`` so it can be verified unchanged when the implementation is
delegated to ``fsspeckit`` (see issue #4). They intentionally capture
pydala-specific adapter behaviors -- string-to-bool coercion, int-to-timestamp
casting, and the ``(schema, bool)`` tuple returned by ``unify_schemas`` -- that
a downstream replacement must preserve.
"""

from __future__ import annotations

import datetime
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from pydala.metadata import collect_parquet_metadata
from pydala.schema import (
    _plan_schema_repair,
    collect_file_schemas,
    convert_large_types_to_normal,
    convert_timestamp,
    replace_schema,
    repair_schema,
    unify_schemas,
)


class TestConvertLargeTypesToNormal(unittest.TestCase):
    """``convert_large_types_to_normal`` maps large Arrow types to standard ones."""

    def test_large_string_and_large_utf8_become_string(self):
        schema = pa.schema([("a", pa.large_string()), ("b", pa.large_utf8())])
        result = convert_large_types_to_normal(schema)

        self.assertEqual(result.field("a").type, pa.string())
        self.assertEqual(result.field("b").type, pa.string())
        # pa.string() and pa.utf8() are the same Arrow type; both large variants
        # collapse to it.
        self.assertEqual(result.field("a").type, pa.utf8())

    def test_large_binary_becomes_binary(self):
        schema = pa.schema([("blob", pa.large_binary())])
        result = convert_large_types_to_normal(schema)

        self.assertEqual(result.field("blob").type, pa.binary())

    def test_large_list_value_type_is_unwrapped(self):
        schema = pa.schema([("lst", pa.large_list(pa.int32()))])
        result = convert_large_types_to_normal(schema)

        self.assertTrue(pa.types.is_list(result.field("lst").type))
        self.assertEqual(result.field("lst").type.value_type, pa.int32())

    def test_normal_types_are_unchanged(self):
        schema = pa.schema(
            [
                ("i", pa.int64()),
                ("s", pa.string()),
                ("bin", pa.binary()),
                ("lst", pa.list_(pa.int32())),
            ]
        )
        result = convert_large_types_to_normal(schema)

        self.assertEqual(result, schema)

    def test_empty_schema_is_unchanged(self):
        result = convert_large_types_to_normal(pa.schema([]))
        self.assertEqual(result, pa.schema([]))
        self.assertEqual(len(result), 0)

    def test_field_metadata_and_nullability_are_preserved(self):
        schema = pa.schema(
            [pa.field("a", pa.large_string(), nullable=False, metadata={"k": b"v"})]
        )
        result = convert_large_types_to_normal(schema)

        self.assertEqual(result.field("a").type, pa.string())
        self.assertFalse(result.field("a").nullable)
        self.assertEqual(result.field("a").metadata, {b"k": b"v"})


class TestReplaceSchema(unittest.TestCase):
    """``replace_schema`` casts, adds, drops, and applies pydala-specific coercions."""

    def test_cast_table_to_a_schema_with_different_column_types(self):
        table = pa.table({"a": [1, 2, 3]}, schema=pa.schema([("a", pa.int32())]))
        new_schema = pa.schema([("a", pa.int64())])

        result = replace_schema(table, schema=new_schema)

        self.assertEqual(result.schema, new_schema)
        self.assertEqual(result.schema.field("a").type, pa.int64())

    def test_add_missing_columns_when_alter_schema_true(self):
        table = pa.table({"a": [1, 2]})
        schema = pa.schema([("a", pa.int32()), ("b", pa.string())])

        result = replace_schema(table, schema=schema, alter_schema=True)

        self.assertEqual(result.schema.names, ["a", "b"])
        # The missing field is null-filled with the target type.
        self.assertEqual(result.column("b").to_pylist(), [None, None])

    def test_drop_extra_columns_when_alter_schema_false(self):
        table = pa.table({"a": [1, 2], "extra": [9, 9]})
        schema = pa.schema([("a", pa.int32())])

        result = replace_schema(table, schema=schema, alter_schema=False)

        self.assertEqual(result.schema.names, ["a"])
        self.assertNotIn("extra", result.schema.names)

    def test_integer_column_is_cast_to_timestamp(self):
        table = pa.table(
            {"ts": [1700000000, 1700000001]},
            schema=pa.schema([("ts", pa.int64())]),
        )
        schema = pa.schema([("ts", pa.timestamp("us"))])

        result = replace_schema(table, schema=schema)

        self.assertTrue(pa.types.is_timestamp(result.schema.field("ts").type))
        # The integer microsecond epoch is interpreted as a timestamp.
        self.assertEqual(
            result.column("ts").to_pylist(),
            [
                datetime.datetime(1970, 1, 1, 0, 28, 20),
                datetime.datetime(1970, 1, 1, 0, 28, 20, 1),
            ],
        )

    def test_string_column_is_cast_to_bool_using_true_values(self):
        # "true", "wahr", "1", "yes" are recognized truthy; "no" is not.
        table = pa.table({"flag": ["true", "wahr", "1", "no", "yes"]})
        schema = pa.schema([("flag", pa.bool_())])

        result = replace_schema(table, schema=schema)

        self.assertTrue(pa.types.is_boolean(result.schema.field("flag").type))
        self.assertEqual(
            result.column("flag").to_pylist(),
            [True, True, True, False, True],
        )

    def test_unchanged_schema_returns_table_as_is(self):
        table = pa.table({"a": [1, 2, 3]}, schema=pa.schema([("a", pa.int32())]))

        result = replace_schema(table, schema=table.schema)

        self.assertEqual(result.schema, table.schema)


class TestUnifySchemas(unittest.TestCase):
    """``unify_schemas`` returns ``(schema, schemas_were_equal)``."""

    def test_two_identical_schemas_are_equal(self):
        schema = pa.schema([("a", pa.int32()), ("b", pa.string())])

        result, equal = unify_schemas([schema, schema])

        self.assertTrue(equal)
        self.assertEqual(result, schema)

    def test_different_column_order_is_not_equal(self):
        s1 = pa.schema([("a", pa.int32()), ("b", pa.string())])
        s2 = pa.schema([("b", pa.string()), ("a", pa.int32())])

        result, equal = unify_schemas([s1, s2])

        self.assertFalse(equal)
        # The unified result follows the first schema's field order.
        self.assertEqual(result.names, ["a", "b"])
        self.assertEqual(result, s1)

    def test_numeric_types_are_promoted_to_wider(self):
        s1 = pa.schema([("x", pa.int32())])
        s2 = pa.schema([("x", pa.int64())])

        result, equal = unify_schemas([s1, s2])

        self.assertFalse(equal)
        self.assertEqual(result.field("x").type, pa.int64())

    def test_disjoint_columns_form_a_union(self):
        s1 = pa.schema([("a", pa.int32())])
        s2 = pa.schema([("b", pa.string())])

        result, equal = unify_schemas([s1, s2])

        self.assertFalse(equal)
        self.assertEqual(result.names, ["a", "b"])
        self.assertEqual(result.field("a").type, pa.int32())
        self.assertEqual(result.field("b").type, pa.string())

    def test_single_schema_returns_equal(self):
        schema = pa.schema([("a", pa.int32())])

        result, equal = unify_schemas([schema])

        self.assertTrue(equal)
        self.assertEqual(result, schema)

    def test_preserves_large_types_when_schemas_differ(self):
        s1 = pa.schema([("value", pa.large_string())])
        s2 = pa.schema([("value", pa.string())])

        result, equal = unify_schemas([s1, s2])

        self.assertFalse(equal)
        self.assertEqual(result.field("value").type, pa.large_string())

    def test_signed_and_unsigned_conflicts_keep_legacy_float_promotion(self):
        s1 = pa.schema([("value", pa.int32())])
        s2 = pa.schema([("value", pa.uint32())])

        result, equal = unify_schemas([s1, s2])

        self.assertFalse(equal)
        self.assertEqual(result.field("value").type, pa.float64())

    def test_non_promoted_binary_conflicts_keep_first_type(self):
        s1 = pa.schema([("value", pa.binary())])
        s2 = pa.schema([("value", pa.large_binary())])

        result, equal = unify_schemas([s1, s2])

        self.assertFalse(equal)
        self.assertEqual(result.field("value").type, pa.binary())

    def test_timestamp_conflicts_follow_legacy_unit_and_schema_order(self):
        s1 = pa.schema([("ts", pa.timestamp("s", tz="UTC"))])
        s2 = pa.schema([("ts", pa.timestamp("ms", tz="America/New_York"))])

        result, equal = unify_schemas([s1, s2])

        self.assertFalse(equal)
        self.assertEqual(
            result.field("ts").type,
            pa.timestamp("s", tz="UTC"),
        )

    def test_timestamp_timezone_ties_follow_later_schema(self):
        s1 = pa.schema([("ts", pa.timestamp("us", tz="UTC"))])
        s2 = pa.schema([("ts", pa.timestamp("us", tz="America/New_York"))])

        result, equal = unify_schemas([s1, s2])

        self.assertFalse(equal)
        self.assertEqual(
            result.field("ts").type,
            pa.timestamp("us", tz="America/New_York"),
        )

    def test_legacy_unsupported_type_and_field_metadata_behavior(self):
        s1 = pa.schema(
            [
                pa.field(
                    "value",
                    pa.decimal128(10, 2),
                    nullable=False,
                    metadata={b"source": b"first"},
                )
            ],
            metadata={b"schema": b"first"},
        )
        s2 = pa.schema([("value", pa.decimal128(20, 4))])

        result, equal = unify_schemas([s1, s2])

        self.assertFalse(equal)
        self.assertEqual(result.field("value").type, pa.decimal128(10, 2))
        self.assertTrue(result.field("value").nullable)
        self.assertIsNone(result.field("value").metadata)
        self.assertIsNone(result.metadata)


class TestRepairSchema(unittest.TestCase):
    """``repair_schema`` execution and dry-run characterization."""

    def test_promotes_signed_and_unsigned_integers_without_float_narrowing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            int_path = f"{tmpdir}/int.parquet"
            uint_path = f"{tmpdir}/uint.parquet"
            pq.write_table(
                pa.table({"id": pa.array([16_777_217], type=pa.int32())}), int_path
            )
            pq.write_table(
                pa.table({"id": pa.array([16_777_218], type=pa.uint32())}), uint_path
            )

            repair_schema(
                files=[int_path, uint_path],
                n_jobs=1,
                backend="sequential",
                verbose=False,
            )

            self.assertEqual(pq.read_schema(int_path).field("id").type, pa.int64())
            self.assertEqual(pq.read_schema(uint_path).field("id").type, pa.int64())

    def test_explicit_parquet_version_is_applied_to_repair_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/values.parquet"
            pq.write_table(
                pa.table({"id": pa.array([1], type=pa.int64())}),
                path,
                version="2.6",
            )

            repair_schema(
                files=[path],
                schema=pa.schema([("id", pa.int64())]),
                version="1.0",
                n_jobs=1,
                backend="sequential",
                verbose=False,
            )

            self.assertEqual(pq.read_metadata(path).format_version, "1.0")
            self.assertEqual(pq.read_schema(path), pa.schema([("id", pa.int64())]))


class TestSchemaRepairPlanner(unittest.TestCase):
    """Shared planning keeps physical schema and repair constraints aligned."""

    def test_planner_uses_one_snapshot_for_schema_and_version_candidates(self):
        schema = pa.schema([("id", pa.int64())])
        snapshots = {"current.parquet": schema, "legacy.parquet": schema}

        plan = _plan_schema_repair(
            snapshots,
            file_versions={"current.parquet": "1.0", "legacy.parquet": "2.6"},
            version="1.0",
        )

        self.assertEqual(plan["schema"], schema)
        self.assertEqual(plan["files"], ["legacy.parquet"])
        self.assertEqual(
            snapshots, {"current.parquet": schema, "legacy.parquet": schema}
        )


class TestRepairSchemaDryRun(unittest.TestCase):
    """``repair_schema(dry_run=True)`` describes the planned mutation safely.

    These tests pin the public dry-run contract: it returns a dictionary
    describing which files would be rewritten and the target schema, without
    mutating any data. See issue #21.
    """

    _RUN_KW = dict(n_jobs=1, backend="sequential", verbose=False)

    def test_dry_run_returns_dictionary_with_files_and_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            int_path = f"{tmpdir}/int.parquet"
            uint_path = f"{tmpdir}/uint.parquet"
            pq.write_table(pa.table({"id": pa.array([1], type=pa.int32())}), int_path)
            pq.write_table(pa.table({"id": pa.array([2], type=pa.uint32())}), uint_path)

            plan = repair_schema(
                files=[int_path, uint_path], dry_run=True, **self._RUN_KW
            )

            self.assertIsInstance(plan, dict)
            self.assertEqual(set(plan.keys()), {"files", "schema"})
            self.assertEqual(set(plan["files"]), {int_path, uint_path})
            self.assertEqual(plan["schema"], pa.schema([("id", pa.int64())]))

    def test_dry_run_does_not_mutate_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            int_path = f"{tmpdir}/int.parquet"
            uint_path = f"{tmpdir}/uint.parquet"
            pq.write_table(pa.table({"id": pa.array([1], type=pa.int32())}), int_path)
            pq.write_table(pa.table({"id": pa.array([2], type=pa.uint32())}), uint_path)

            _ = repair_schema(files=[int_path, uint_path], dry_run=True, **self._RUN_KW)

            # Files remain untouched.
            self.assertEqual(pq.read_schema(int_path).field("id").type, pa.int32())
            self.assertEqual(pq.read_schema(uint_path).field("id").type, pa.uint32())

    def test_dry_run_lists_only_files_that_differ_from_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            small = f"{tmpdir}/small.parquet"
            wide = f"{tmpdir}/wide.parquet"
            pq.write_table(pa.table({"id": pa.array([1], type=pa.int32())}), small)
            pq.write_table(pa.table({"id": pa.array([2], type=pa.int64())}), wide)

            plan = repair_schema(files=[small, wide], dry_run=True, **self._RUN_KW)

            # The unified schema is int64; only the int32 file needs rewriting.
            self.assertEqual(plan["files"], [small])
            self.assertEqual(plan["schema"].field("id").type, pa.int64())

    def test_dry_run_returns_empty_files_when_already_unified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            a = f"{tmpdir}/a.parquet"
            b = f"{tmpdir}/b.parquet"
            for path in (a, b):
                pq.write_table(pa.table({"id": [1, 2]}), path)

            plan = repair_schema(files=[a, b], dry_run=True, **self._RUN_KW)

            self.assertEqual(plan["files"], [])

    def test_dry_run_target_schema_reflects_timestamp_unit_and_timezone(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s_us = pa.schema([("ts", pa.timestamp("us")), ("id", pa.int32())])
            s_ms = pa.schema([("ts", pa.timestamp("ms")), ("id", pa.int32())])
            p_us = f"{tmpdir}/us.parquet"
            p_ms = f"{tmpdir}/ms.parquet"
            pq.write_table(
                pa.table(
                    {
                        "ts": pa.array(
                            [datetime.datetime(2024, 1, 1)], pa.timestamp("us")
                        ),
                        "id": [1],
                    },
                    schema=s_us,
                ),
                p_us,
            )
            pq.write_table(
                pa.table(
                    {
                        "ts": pa.array(
                            [datetime.datetime(2024, 1, 2)], pa.timestamp("ms")
                        ),
                        "id": [2],
                    },
                    schema=s_ms,
                ),
                p_ms,
            )

            plan = repair_schema(
                files=[p_us, p_ms],
                dry_run=True,
                ts_unit="us",
                tz="UTC",
                **self._RUN_KW,
            )

            ts_type = plan["schema"].field("ts").type
            self.assertEqual(ts_type, pa.timestamp("us", tz="UTC"))
            self.assertEqual(set(plan["files"]), {p_us, p_ms})


class TestRepairSchemaTimestampConversion(unittest.TestCase):
    """Direct ``repair_schema`` execution applies timestamp unit/timezone conversion.

    See issue #21 — compatibility for timestamp/timezone repair.
    """

    _RUN_KW = dict(n_jobs=1, backend="sequential", verbose=False)

    def test_repair_unifies_timestamp_units(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s_us = pa.schema([("ts", pa.timestamp("us")), ("id", pa.int32())])
            s_ms = pa.schema([("ts", pa.timestamp("ms")), ("id", pa.int32())])
            p_us = f"{tmpdir}/us.parquet"
            p_ms = f"{tmpdir}/ms.parquet"
            pq.write_table(
                pa.table(
                    {
                        "ts": pa.array(
                            [datetime.datetime(2024, 1, 1)], pa.timestamp("us")
                        ),
                        "id": [1],
                    },
                    schema=s_us,
                ),
                p_us,
            )
            pq.write_table(
                pa.table(
                    {
                        "ts": pa.array(
                            [datetime.datetime(2024, 1, 2)], pa.timestamp("ms")
                        ),
                        "id": [2],
                    },
                    schema=s_ms,
                ),
                p_ms,
            )

            repair_schema(files=[p_us, p_ms], ts_unit="us", **self._RUN_KW)

            self.assertEqual(pq.read_schema(p_us).field("ts").type, pa.timestamp("us"))
            self.assertEqual(pq.read_schema(p_ms).field("ts").type, pa.timestamp("us"))

    def test_repair_applies_timezone_to_timestamp_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            schema = pa.schema([("ts", pa.timestamp("us")), ("id", pa.int32())])
            a = f"{tmpdir}/a.parquet"
            b = f"{tmpdir}/b.parquet"
            for i, path in enumerate((a, b)):
                pq.write_table(
                    pa.table(
                        {
                            "ts": pa.array(
                                [datetime.datetime(2024, 1, 1 + i)],
                                pa.timestamp("us"),
                            ),
                            "id": [i],
                        },
                        schema=schema,
                    ),
                    path,
                )

            repair_schema(files=[a, b], tz="UTC", **self._RUN_KW)

            self.assertEqual(
                pq.read_schema(a).field("ts").type,
                pa.timestamp("us", tz="UTC"),
            )
            self.assertEqual(
                pq.read_schema(b).field("ts").type,
                pa.timestamp("us", tz="UTC"),
            )

    def test_repair_promotes_disjoint_columns_via_unification(self):
        """Repair adds missing columns so every file converges to the union schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a = f"{tmpdir}/a.parquet"
            b = f"{tmpdir}/b.parquet"
            pq.write_table(pa.table({"id": [1, 2], "name": ["x", "y"]}), a)
            pq.write_table(pa.table({"id": [3, 4], "value": [1.0, 2.0]}), b)

            repair_schema(files=[a, b], alter_schema=True, **self._RUN_KW)

            schema_a = pq.read_schema(a)
            schema_b = pq.read_schema(b)
            self.assertEqual(set(schema_a.names), {"id", "name", "value"})
            self.assertEqual(set(schema_b.names), {"id", "name", "value"})


class TestCollectFileSchemas(unittest.TestCase):
    """Schema collection exercises the public schema batch flow."""

    def test_collects_multiple_file_schemas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for index, dtype in enumerate((pa.int32(), pa.int64())):
                path = f"{tmpdir}/part-{index}.parquet"
                table = pa.table(
                    {"value": pa.array([index], type=dtype)},
                    schema=pa.schema([("value", dtype)]),
                )
                pq.write_table(table, path)
                paths.append(path)

            schema_result = collect_file_schemas(
                paths, n_jobs=2, backend="threading", verbose=False
            )
            metadata_result = collect_parquet_metadata(
                paths, n_jobs=2, backend="threading", verbose=False
            )

        self.assertEqual(set(schema_result), set(paths))
        self.assertEqual(schema_result[paths[0]], pa.schema([("value", pa.int32())]))
        self.assertEqual(schema_result[paths[1]], pa.schema([("value", pa.int64())]))
        self.assertEqual(set(metadata_result), set(paths))
        self.assertTrue(
            all(metadata.num_rows == 1 for metadata in metadata_result.values())
        )


class TestConvertTimestamp(unittest.TestCase):
    """``convert_timestamp`` rewrites timestamp fields' unit and timezone."""

    def test_change_timezone_on_timestamp_field(self):
        schema = pa.schema([("ts", pa.timestamp("us")), ("id", pa.int32())])

        result = convert_timestamp(schema, tz="UTC")

        self.assertEqual(result.field("ts").type, pa.timestamp("us", tz="UTC"))
        # Non-timestamp fields are left untouched.
        self.assertEqual(result.field("id").type, pa.int32())

    def test_change_timestamp_unit(self):
        schema = pa.schema([("ts", pa.timestamp("us"))])

        result = convert_timestamp(schema, unit="ms")

        self.assertEqual(result.field("ts").type, pa.timestamp("ms"))

    def test_remove_timezone(self):
        schema = pa.schema([("ts", pa.timestamp("us", tz="UTC"))])

        result = convert_timestamp(schema, remove_tz=True)

        self.assertEqual(result.field("ts").type, pa.timestamp("us"))
        self.assertIsNone(result.field("ts").type.tz)

    def test_timestamp_field_position_is_preserved(self):
        schema = pa.schema(
            [("before", pa.int32()), ("ts", pa.timestamp("us")), ("after", pa.string())]
        )

        result = convert_timestamp(schema, tz="UTC")

        self.assertEqual(result.names, ["before", "ts", "after"])
        self.assertEqual(result.field("ts").type, pa.timestamp("us", tz="UTC"))

    def test_preserves_timestamp_field_attributes(self):
        schema = pa.schema(
            [
                pa.field(
                    "ts",
                    pa.timestamp("us"),
                    nullable=False,
                    metadata={b"source": b"ingest"},
                )
            ],
            metadata={b"schema": b"metadata"},
        )

        result = convert_timestamp(schema, unit="ms", tz="UTC")

        self.assertEqual(result.field("ts").type, pa.timestamp("ms", tz="UTC"))
        self.assertFalse(result.field("ts").nullable)
        self.assertEqual(result.field("ts").metadata, {b"source": b"ingest"})
        self.assertEqual(result.metadata, {b"schema": b"metadata"})


class TestRepairSchemaLargeTypeDetection(unittest.TestCase):
    """Raw physical schema differences must be detected for repair (issue #36).

    The repair planner previously normalized every physical schema before both
    target selection and candidate detection, hiding real differences such as
    ``large_string`` versus ``string``. Candidate detection must now compare the
    raw physical schema against the canonical (normalized) target.
    """

    _RUN_KW = dict(n_jobs=1, backend="sequential", verbose=False)

    def test_dry_run_selects_raw_large_string_file_when_target_is_string(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            normal = f"{tmpdir}/normal.parquet"
            large = f"{tmpdir}/large.parquet"
            pq.write_table(pa.table({"x": pa.array(["a"], type=pa.string())}), normal)
            pq.write_table(
                pa.table({"x": pa.array(["b"], type=pa.large_string())}), large
            )

            plan = repair_schema(files=[normal, large], dry_run=True, **self._RUN_KW)

            self.assertEqual(plan["files"], [large])
            self.assertEqual(plan["schema"], pa.schema([("x", pa.string())]))

    def test_dry_run_leaves_already_canonical_string_file_unselected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            a = f"{tmpdir}/a.parquet"
            b = f"{tmpdir}/b.parquet"
            pq.write_table(pa.table({"x": pa.array(["a"], type=pa.string())}), a)
            pq.write_table(pa.table({"x": pa.array(["b"], type=pa.string())}), b)

            plan = repair_schema(files=[a, b], dry_run=True, **self._RUN_KW)

            self.assertEqual(plan["files"], [])

    def test_planner_compares_raw_schema_to_canonical_target_directly(self):
        normal_schema = pa.schema([("x", pa.string())])
        large_schema = pa.schema([("x", pa.large_string())])

        plan = _plan_schema_repair(
            {"normal.parquet": normal_schema, "large.parquet": large_schema}
        )

        self.assertEqual(plan["schema"], pa.schema([("x", pa.string())]))
        self.assertEqual(plan["files"], ["large.parquet"])

    def test_real_repair_rewrites_large_string_to_string_preserving_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            normal = f"{tmpdir}/normal.parquet"
            large = f"{tmpdir}/large.parquet"
            pq.write_table(pa.table({"x": pa.array(["a"], type=pa.string())}), normal)
            pq.write_table(
                pa.table({"x": pa.array(["hello"], type=pa.large_string())}), large
            )

            repair_schema(files=[normal, large], **self._RUN_KW)

            self.assertEqual(pq.read_schema(normal).field("x").type, pa.string())
            self.assertEqual(pq.read_schema(large).field("x").type, pa.string())
            self.assertEqual(pq.read_table(large).column("x").to_pylist(), ["hello"])

    def test_failed_cast_leaves_original_file_intact_and_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/overflow.parquet"
            # int64 value far outside int32 range; casting to int32 must fail.
            pq.write_table(pa.table({"id": pa.array([2**40], type=pa.int64())}), path)
            original_schema = pq.read_schema(path)
            original_bytes = pq.read_table(path).to_pydict()

            with self.assertRaises(Exception):
                repair_schema(
                    files=[path],
                    schema=pa.schema([("id", pa.int32())]),
                    **self._RUN_KW,
                )

            # The original file is untouched: same schema, same values.
            self.assertEqual(pq.read_schema(path), original_schema)
            self.assertEqual(pq.read_table(path).to_pydict(), original_bytes)


if __name__ == "__main__":
    unittest.main()

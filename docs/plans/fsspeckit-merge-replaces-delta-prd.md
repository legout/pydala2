# Replace `mode="delta"` with fsspeckit Merge — PRD

> **Status:** Proposed
>
> **Repository:** `legout/pydala2`
>
> **Dependency:** `fsspeckit>=0.27.1`

## Summary

Replace pydala2's custom `mode="delta"` write path with an explicit
`ParquetDataset.merge(...)` interface backed by fsspeckit's typed merge
implementation.

The existing `mode="delta"` behavior is **insert-if-key-absent**, not an
update: it anti-joins incoming rows against existing rows on `delta_subset`
and appends only unseen rows. Its semantic replacement is therefore
fsspeckit `strategy="insert"`, not `upsert`.

Add first-class `insert`, `update`, and `upsert` strategies through
`ParquetDataset.merge(...)`. Keep `mode="delta"` temporarily as a deprecated
compatibility alias for `merge(strategy="insert")`, then remove the custom
delta implementation in a future major release.

## Evidence

Current pydala2 execution path:

1. `BaseDataset.write_to_dataset(mode="delta")` builds a `Writer`.
2. `Writer.execute` sorts, optionally deduplicates, casts schema, and derives
   partition columns.
3. `_get_delta_other_df` reads potentially overlapping target rows.
4. `Writer.delta` calls `fsspeckit.datasets.polars.delta`, an anti-join of
   incoming rows against target rows.
5. Only unmatched incoming rows are appended as new files. No target file is
   rewritten.

Verified behavior:

```text
target:   {id: 1, value: old}
incoming: {id: 1, value: new}, {id: 2, value: two}
mode="delta", delta_subset=["id"]
result:   {id: 1, value: old}, {id: 2, value: two}
```

The same test through the existing `FsspeckitParquetAdapter` and
`PyarrowDatasetIO.merge(strategy="insert", key_columns=["id"])` produced:

```text
target_count_before=1, inserted=1, updated=0, target_count_after=2
result: {id: 1, value: old}, {id: 2, value: two}
```

This proves semantic compatibility for the documented
`delta_subset=[key_columns]` use case and confirms the existing adapter owns
the correct filesystem/path seam.

## Problem

The custom delta path duplicates behavior now owned by fsspeckit and has
several correctness and maintainability problems:

- `mode="delta"` is misnamed and documented as an update even though it never
  updates matching rows.
- Delta behavior depends on `self.is_loaded`; an unloaded dataset silently
  degrades to append behavior.
- Target filtering, anti-join semantics, and key handling are maintained
  separately from fsspeckit's tested merge implementation.
- There is no first-class update/upsert operation despite docs attempting to
  construct one manually.
- The write mode string is not validated; unknown modes effectively append.
- Typed merge outcomes (inserted/updated counts and affected files) are lost.
- Delta handling is mixed into `Writer.execute`, making the write module
  shallower and harder to test.

## Decision

### Public interface

Add a first-class merge method to `ParquetDataset`:

```python
from typing import Literal
from fsspeckit.core.incremental import MergeResult

result: MergeResult = dataset.merge(
    data,
    strategy="insert",              # "insert" | "update" | "upsert"
    key_columns=["id"],
    partition_by=["year", "month"],
    backend="pyarrow",              # "pyarrow" | "duckdb"
    compression="zstd",
    max_rows_per_file=10_000_000,
    row_group_size=256_000,
)
```

Interface requirements:

- `key_columns` is optional for new `merge()` calls. When omitted, infer
  whole-row identity from source columns that are present in every prepared
  source table. Key equality is null-safe: null matches null and differs from
  every non-null value. Callers should pass explicit business keys when a
  changed value must update an existing row.
- `partition_by` defaults to the dataset's existing partition columns.
- `backend="pyarrow"` is the default because it supports arbitrary fsspec
  filesystems; DuckDB remains an explicit alternative.
- The method accepts the same input types as `write_to_dataset` (Polars,
  PyArrow, pandas, DuckDB relations, lists), normalized internally to
  PyArrow tables.
- The method returns fsspeckit's typed `MergeResult`.
- `append` and `overwrite` remain write operations; they do not route through
  merge.

Do **not** add `mode="merge"` or `mode="upsert"` to `write_to_dataset`.
Merge is a distinct operation with stronger invariants, different return
semantics, and selective file rewrites. A separate interface keeps this
complexity behind a deep module.

### Adapter seam

Extend `FsspeckitParquetAdapter` with:

```python
def merge(
    self,
    data: pa.Table | list[pa.Table],
    path: str,
    strategy: Literal["insert", "update", "upsert"],
    key_columns: str | list[str],
    *,
    partition_columns: str | list[str] | None = None,
    backend: Literal["pyarrow", "duckdb"] = "pyarrow",
    **merge_options,
) -> MergeResult:
    ...
```

The adapter owns:

- pydala-relative path to backend-filesystem path conversion;
- filesystem and credential reuse;
- PyArrow/DuckDB handler construction and lifetime;
- translation from pydala's `partition_by` name to fsspeckit's
  `partition_columns` name;
- forwarding the typed result unchanged.

Callers must not instantiate fsspeckit dataset handlers directly.

### Input preparation seam

Extract the transformation half of `Writer.execute` into a reusable method,
for example:

```python
def prepare(
    self,
    *,
    sort_by=None,
    unique=False,
    ts_unit="us",
    tz=None,
    remove_tz=False,
    alter_schema=False,
    partition_by=None,
    timestamp_column=None,
) -> pa.Table:
    ...
```

`prepare` owns the current ordering:

1. source input normalization;
2. sorting;
3. optional uniqueness;
4. target-schema casting/evolution;
5. derived date-part partition columns;
6. Arrow conversion.

`Writer.execute` calls `prepare` and then writes. `ParquetDataset.merge` calls
`prepare` and delegates to `FsspeckitParquetAdapter.merge`. This avoids
copying transformation logic into the dataset class.

For list inputs, prepare every item and pass all prepared tables to one
fsspeckit merge call so duplicate-key resolution is deterministic across the
whole source batch.

## Strategy Semantics

| Strategy | Matching source key | New source key | Target files |
| --- | --- | --- | --- |
| `insert` | ignored; target row preserved | appended | never rewrites existing files |
| `update` | target row replaced | rejected/ignored by fsspeckit contract | rewrites only affected files |
| `upsert` | target row replaced | appended | rewrites only affected files |

Source rows with duplicate merge keys follow fsspeckit's established rule:
**last source row wins**.

Partition columns for existing keys are immutable. Attempting to move an
existing key to another partition raises before publication.

## `mode="delta"` Compatibility

During the compatibility release:

```python
dataset.write_to_dataset(
    data,
    mode="delta",
    delta_subset=["id"],
)
```

must:

1. emit `DeprecationWarning` pointing to `dataset.merge`;
2. delegate to `merge(strategy="insert", key_columns=delta_subset)`;
3. work whether or not the dataset was previously loaded;
4. return `MergeResult` (the docs currently claim `None`; actual metadata-list
   behavior is not a stable documented contract);
5. refresh pydala file, metadata, table, and cache state after completion.

If `delta_subset` is omitted, preserve the old exact-row behavior for the
compatibility release by deriving keys from common source/target
columns. Emit a stronger warning that explicit `key_columns` will be required
when `mode="delta"` is removed.

A future major release removes:

- `mode="delta"`;
- `delta_subset`;
- `Writer.delta`;
- `_get_delta_other_df`;
- the Polars anti-join write path.

## Intentional Behavior Changes

| Area | Existing delta | fsspeckit-backed merge |
| --- | --- | --- |
| Dataset load state | delta only if `self.is_loaded`; otherwise append | always inspects target files |
| Null keys | Polars anti-join treats nulls as equal | null-safe equality; null matches null only |
| Duplicate source keys | retained unless `unique=` | last source row wins |
| Missing key specification | all common columns | all common prepared source columns |
| Outcome | metadata list / docs claim `None` | typed `MergeResult` |
| Existing row changes | impossible | explicit `update` / `upsert` |

These differences are safety improvements and must be called out in the
migration guide.

## Metadata, Cache, and Error Handling

After a merge, call `ParquetDataset._refresh_after_rewrite()` so pydala's file
list, aggregate metadata, table state, and caches reflect inserted and/or
rewritten files.

Do not translate a failed merge into `PartialWriteError`: fsspeckit merge owns
its staging/replacement lifecycle and raises/returns its own typed failure
contract. If pydala metadata refresh fails *after* a successful merge, raise a
new completion error that retains the `MergeResult` for recovery and auditing.

## Non-Goals

- Reimplement merge logic in pydala.
- Preserve undocumented source-duplicate behavior.
- Add delete semantics; fsspeckit merge currently reports `deleted=0`.
- Replace `append` or `overwrite`.
- Change compaction, deduplication, or repartition maintenance interfaces.
- Remove `mode="delta"` in the first release; it receives a deprecation cycle.

## Implementation Plan

### Phase 1 — Freeze current behavior with contracts

- Add tests proving `mode="delta"` is insert-if-absent, not update.
- Cover keyed delta, exact-row delta (`delta_subset=None`), unloaded datasets,
  duplicate source keys, null keys, partitioned datasets, and list inputs.
- Correct the current misleading upsert example in the test expectations, but
  leave docs migration to the final phase.

### Phase 2 — Extract reusable source preparation

- Add `Writer.prepare(...) -> pa.Table`.
- Refactor `Writer.execute` to consume it without behavior changes for append
  and overwrite.
- Unit-test Polars/LazyFrame/PyArrow/pandas/DuckDB inputs, schema casting,
  date-part derivation, sorting, uniqueness, and list preparation.

### Phase 3 — Add the fsspeckit merge adapter

- Add `FsspeckitParquetAdapter.merge` with PyArrow and DuckDB backends.
- Reuse the existing adapter path/filesystem seam.
- Contract-test local and memory filesystems, composite keys, and partitioned
  insert/update/upsert behavior.

### Phase 4 — Add `ParquetDataset.merge`

- Normalize input through `Writer.prepare`.
- Delegate through `FsspeckitParquetAdapter.merge`.
- Refresh metadata/cache/table state.
- Return typed `MergeResult`.
- Test empty targets, missing targets, no-op insert/update, selective rewrite,
  partition immutability, schema handling, and failure propagation.

### Phase 5 — Deprecate and redirect delta mode

- Validate write modes explicitly.
- Redirect `mode="delta"` to `merge(strategy="insert")` with warnings.
- Implement compatibility key inference when `delta_subset=None`.
- Remove the old load-state-dependent anti-join path from active execution.
- Add parity tests between legacy delta fixtures and merge insert results.

### Phase 6 — Documentation and examples

- Add a merge guide with insert/update/upsert semantics.
- Add a delta migration guide.
- Replace docs that call delta an update or show it as an upsert.
- Document null-safe key equality, last-source-row-wins, return-type change, and
  explicit key requirements.

## Acceptance Criteria

- `ParquetDataset.merge` supports `insert`, `update`, and `upsert` through
  fsspeckit on local and memory filesystems.
- The verified keyed-delta example is behaviorally identical under
  `strategy="insert"`.
- `mode="delta"` delegates to merge and warns; it no longer depends on
  `self.is_loaded`.
- Existing append/overwrite tests remain unchanged and green.
- Metadata and table state are correct immediately after merge.
- Partitioned merges preserve hive layout and partition immutability.
- Explicit merge keys may contain null values and use null-safe equality.
- Documentation no longer claims delta updates existing rows.
- The old anti-join implementation is unreachable from active write paths.
- Layering, typing, and test suites pass.

## Risks

### Source duplicate behavior changes

fsspeckit deduplicates incoming merge keys with last-row-wins semantics.
Mitigation: document and test it; callers needing another winner must sort the
source before merge.

### Nullable-key behavior diverges across layers

pydala must not filter or reject nullable key columns before delegating to
fsspeckit. Mitigation: require fsspeckit 0.27.1, keep pydala's adapter as a
pass-through, and cover explicit, inferred, and delta-compatible nullable keys
at the public dataset seam.

### Mixed-schema inserted files

The current Writer performs schema casting before delta. The merge path must
retain that preparation step; bypassing it can create incompatible parquet
file schemas.

### Metadata refresh failure after successful merge

The data operation can succeed while pydala metadata refresh fails. Preserve
the `MergeResult` on the completion error and provide a documented
`dataset.load(reload_metadata=True)` recovery path.

### Existing absolute-path edge cases

The adapter is proven for pydala's supported `FileSystem(bucket=...), path`
shape. Add regression coverage for absolute local paths before declaring them
supported through merge.

## Derived Tickets

This PRD is implemented through separate, independently reviewable issues:

1. [#28 — Freeze current delta semantics with contract tests](https://github.com/legout/pydala2/issues/28).
2. [#25 — Extract reusable `Writer.prepare` source normalization](https://github.com/legout/pydala2/issues/25).
3. [#29 — Add `FsspeckitParquetAdapter.merge`](https://github.com/legout/pydala2/issues/29).
4. [#30 — Add public `ParquetDataset.merge` with metadata refresh](https://github.com/legout/pydala2/issues/30).
5. [#27 — Deprecate/redirect `mode="delta"` and validate write modes](https://github.com/legout/pydala2/issues/27).
6. [#26 — Update merge/delta documentation and examples](https://github.com/legout/pydala2/issues/26).

Tracking issue: [#24](https://github.com/legout/pydala2/issues/24).

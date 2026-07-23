# Merge and Delta Migration

`ParquetDataset.merge` is the public interface for key-based insert, update, and upsert operations. It prepares supported source data, delegates the rewrite to fsspeckit, refreshes dataset state, and returns a typed `MergeResult`.

## Set up a dataset

A merge can create a missing target. Sources may be PyArrow tables or record batches, Polars data frames or lazy frames, pandas data frames, DuckDB relations, or lists of those types.

```python
import pyarrow as pa

from pydala import ParquetDataset
from pydala.filesystem import FileSystem

filesystem = FileSystem(bucket=".", cached=False)
dataset = ParquetDataset(path="customers", filesystem=filesystem)

initial_result = dataset.merge(
    pa.table({"customer_id": [1], "name": ["Ada"]}),
    strategy="upsert",
    key_columns=["customer_id"],
)
```

Pass explicit `key_columns` for stable business-key identity. Composite keys use
a list such as `key_columns=["tenant_id", "customer_id"]`. Key equality is
null-safe: null matches null and differs from every non-null value.

When `key_columns` is omitted or set to `None`, pydala uses every prepared
source column that is present in every source table. This is whole-row identity,
not a business-key upsert: changing any value, including changing between null
and non-null, makes a new key, so an `upsert` inserts that changed row rather
than replacing the old one. Use explicit keys whenever an incoming row should
update an existing one.

## Insert rows whose keys are absent

Use `insert` when existing target rows must remain unchanged. A source row with an existing key is ignored; only absent keys are inserted.

```python
insert_result = dataset.merge(
    pa.table(
        {
            "customer_id": [1, 2],
            "name": ["Ignored replacement", "Grace"],
        }
    ),
    strategy="insert",
    key_columns=["customer_id"],
    backend="pyarrow",
)

print(insert_result.inserted)  # 1
print(insert_result.updated)   # 0
```

PyArrow is the default backend and works with generic fsspec filesystems. Passing `backend="pyarrow"` is optional.

## Update rows whose keys exist

Use `update` when only matching target rows should change. Source rows with absent keys are ignored.

```python
update_result = dataset.merge(
    pa.table(
        {
            "customer_id": [1, 99],
            "name": ["Ada Lovelace", "Ignored new customer"],
        }
    ),
    strategy="update",
    key_columns=["customer_id"],
)

print(update_result.updated)   # 1
print(update_result.inserted)  # 0
```

## Insert and update in one operation

Use `upsert` to update matching keys and insert absent keys.

```python
upsert_result = dataset.merge(
    pa.table(
        {
            "customer_id": [2, 3],
            "name": ["Grace Hopper", "Katherine"],
        }
    ),
    strategy="upsert",
    key_columns=["customer_id"],
)

print(upsert_result.updated)   # 1
print(upsert_result.inserted)  # 1
```

A list of sources is merged as one logical batch. If a key occurs more than once in the source batch, the last source row wins. This includes keys with null components: two nulls in the same key position match, while null never matches a non-null value.

## Partition and backend rules

Pass pydala-style partition columns with `partition_by`. Existing keys cannot move between partitions during `update` or `upsert`; partition values for matching keys are immutable. Use a separate data-replacement workflow when rows must move to another partition.

The available backends are:

- `backend="pyarrow"`: the default; supports local and generic fsspec filesystems.
- `backend="duckdb"`: explicit opt-in for filesystems DuckDB can write to natively, such as local filesystems. Unsupported protocols raise `ValueError` instead of silently changing backends.

Compression, file sizing, source sorting, schema preparation, timestamp conversion, and fsspeckit merge tuning can be passed through `merge`, for example `compression="zstd"`, `max_rows_per_file=1_000_000`, or `merge_max_memory_mb=512`.

## Read the result

`merge` returns fsspeckit's typed `MergeResult`. Useful fields include:

- `source_count`
- `target_count_before` and `target_count_after`
- `inserted`, `updated`, and `deleted`
- `rewritten_files`, `inserted_files`, and `preserved_files`
- `metrics`

After a successful merge, the same `ParquetDataset` instance is refreshed and immediately readable. Merge behavior does not depend on whether the dataset was loaded before the call.

## Recover after refresh failure

A physical merge can succeed even if the following metadata refresh fails. In that case pydala raises `PartialMergeError` and preserves the successful result on both `error.merge_result` and `dataset.last_merge_result`.

Do not repeat the merge: its file changes already succeeded. Record the retained result, then retry the dataset refresh with `dataset.load(update_metadata=True, reload_metadata=True)` or reconstruct the dataset instance from the same path and filesystem.

## Migrate from `mode="delta"`

`mode="delta"` has always meant insert-if-absent: matching target rows remain unchanged. It is not an update or an upsert.

| Deprecated call | Replacement | Meaning |
| --- | --- | --- |
| `write_to_dataset(data, mode="delta", delta_subset=keys)` | `merge(data, strategy="insert", key_columns=keys)` | Preserve matching rows and insert absent keys. |
| `write_to_dataset(data, mode="delta", delta_subset=None)` | Choose explicit keys and call `merge(data, strategy="insert", key_columns=keys)` | Replace temporary common source/target-column inference. |

The compatibility form still delegates to `merge(strategy="insert")` and returns `MergeResult`:

```python
legacy_result = dataset.write_to_dataset(
    pa.table({"customer_id": [4], "name": ["Dorothy"]}),
    mode="delta",
    delta_subset=["customer_id"],
)
```

New code should call `merge` directly:

```python
replacement_result = dataset.merge(
    pa.table({"customer_id": [5], "name": ["Mary"]}),
    strategy="insert",
    key_columns=["customer_id"],
)
```

During the compatibility release, delta emits `DeprecationWarning`, uses null-safe key equality, resolves duplicate source keys with the last row, works independently of load state, and returns a typed result. When `delta_subset=None`, pydala temporarily infers common source and target columns and warns that explicit keys will be required.

`mode="delta"`, `delta_subset`, `Writer.delta`, and `_get_delta_other_df` are scheduled for removal in the next major release. Migrate callers before that release.

### Design references

The migration was designed in the [fsspeckit merge PRD](../plans/fsspeckit-merge-replaces-delta-prd.md) and tracked by [epic #24](https://github.com/legout/pydala2/issues/24) with child issues [#25](https://github.com/legout/pydala2/issues/25), [#26](https://github.com/legout/pydala2/issues/26), [#27](https://github.com/legout/pydala2/issues/27), [#28](https://github.com/legout/pydala2/issues/28), [#29](https://github.com/legout/pydala2/issues/29), and [#30](https://github.com/legout/pydala2/issues/30).

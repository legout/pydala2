---
status: accepted
---

# Consolidate dataset lifecycle and retire optimization monkey-patching

`ParquetDataset` will preserve its current public API while replacing its duplicated metadata/runtime initialization and `Optimize` monkey-patching with one explicit lifecycle. This choice makes physical Parquet files authoritative, makes metadata an explicit reconciled sidecar, and removes hidden initialization order from the class hierarchy.

## Decision

- Introduce one private storage base for path, filesystem, cache, and directory lifecycle state.
- Keep `BaseDataset` responsible for generic runtime state and keep metadata classes responsible for Parquet sidecars and repair behavior.
- Have concrete dataset classes load eagerly exactly once after their complete state is initialized; `BaseDataset.__init__` must not dynamically dispatch to an overridable `load()` method.
- Make the runtime layer own `name` and the DuckDB connection. Metadata uses that connection during `ParquetDataset` construction; directly constructed metadata objects may create one only when none is supplied.
- Treat empty datasets as valid, but surface contextual errors for corrupt or unreadable metadata sidecars instead of swallowing them.
- Make physical filesystem discovery the canonical implementation of `dataset.files`; retain `files_in_metadata` and `files_in_file_metadata` as explicit sidecar views.
- Move optimization methods directly onto `ParquetDataset`, remove the method-assignment monkey-patch block, and retain `Optimize` as a compatibility-only subclass.

## Considered options

We rejected a metadata-store composition rewrite because forwarding the existing inherited metadata API would create a larger, riskier compatibility migration. We also rejected retaining the current diamond initialization and monkey-patching because both conceal lifecycle ownership and permit stale metadata to masquerade as data files.

## Consequences

Dataset construction remains eager and public imports remain stable, but internal initialization becomes explicit and testable. Metadata repair, update, and vacuum use the same invariant: physical files determine existence; metadata describes and is reconciled against those files.

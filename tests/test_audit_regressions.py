"""Regressions for the Ponytail audit: each test reproduces a verified defect."""

import pickle

import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
import yaml

from pydala import Catalog, ParquetDataset
from pydala.helpers.sql import sql2polars_filter
from pydala.table import PydalaTable


def _ids(path):
    return sorted(
        i for f in path.rglob("*.parquet") for i in pq.ParquetFile(f).read().column("id").to_pylist()
    )


def test_overwrite_into_empty_and_reopen(tmp_path):
    path = tmp_path / "ds"
    ds = ParquetDataset(str(path))
    ds.write_to_dataset(pa.table({"id": [1]}), mode="overwrite")
    assert _ids(path) == [1]
    ds.write_to_dataset(pa.table({"id": [9]}), mode="overwrite")
    assert _ids(path) == [9]
    assert ParquetDataset(str(path)).table.arrow.to_pylist() == [{"id": 9}]


def test_overwrite_with_stale_instance(tmp_path):
    path = tmp_path / "ds"
    a = ParquetDataset(str(path))
    a.write_to_dataset(pa.table({"id": [1]}))
    ParquetDataset(str(path)).write_to_dataset(pa.table({"id": [2]}))
    a.write_to_dataset(pa.table({"id": [3]}), mode="overwrite")
    assert _ids(path) == [3]
    assert ParquetDataset(str(path)).table.arrow.to_pylist() == [{"id": 3}]


def test_metadata_scan_matches_file_statistics(tmp_path):
    path = tmp_path / "ds"
    path.mkdir()
    for i in range(60):
        pq.write_table(pa.table({"id": [i] * 2}), path / f"f{i:03d}.parquet")
    ds = ParquetDataset(str(path))
    for file, lo in ds.metadata_table.select("file_path, id.min").fetchall():
        assert int(file[1:4]) == lo
    assert ds.scan("id = 14").to_arrow()["id"].to_pylist() == [14, 14]


def test_legacy_pickle_sidecar_is_never_loaded(tmp_path):
    class Boom:
        def __reduce__(self):
            return (RuntimeError, ("executed",))

    path = tmp_path / "ds"
    ds = ParquetDataset(str(path))
    ds.write_to_dataset(pa.table({"id": [1]}))
    (path / "_file_metadata").write_bytes(pickle.dumps(Boom()))
    fresh = ParquetDataset(str(path))
    assert fresh._read_file_metadata() is None
    fresh.load(reload_metadata=True)
    assert fresh.table.arrow.to_pylist() == [{"id": 1}]


def test_append_with_same_basename_keeps_both(tmp_path):
    path = tmp_path / "ds"
    ds = ParquetDataset(str(path))
    ds.write_to_dataset(pa.table({"id": [1]}), basename="batch")
    ds.write_to_dataset(pa.table({"id": [2]}), basename="batch")
    assert _ids(path) == [1, 2]


def test_schema_changing_append_is_visible(tmp_path):
    path = tmp_path / "ds"
    ds = ParquetDataset(str(path))
    ds.write_to_dataset(pa.table({"id": [1]}))
    ds.write_to_dataset(pa.table({"id": [2], "extra": ["x"]}), alter_schema=True)
    ds.load()
    assert sorted(r["id"] for r in ds.table.arrow.to_pylist()) == [1, 2]


def test_partition_value_equal_to_dataset_name(tmp_path):
    ds = ParquetDataset(str(tmp_path / "sales"))
    ds.write_to_dataset(pa.table({"id": [1], "region": ["sales"]}), partition_by=["region"])
    ds.load()
    assert ds.table.arrow.num_rows == 1


def test_scoped_catalog_edit_keeps_other_namespaces(tmp_path):
    cfg = tmp_path / "catalog.yml"
    tb = {"filesystem": "local", "path": "sales", "format": "parquet"}
    cfg.write_text(
        yaml.safe_dump(
            {
                "filesystem": {"local": {"protocol": "file", "bucket": "."}},
                "tables": {"default": {"sales": tb}, "other": {"copy": tb}},
            }
        )
    )
    catalog = Catalog(str(cfg), namespace="default")
    catalog.create_table(table_name="alias", **tb)
    assert set(yaml.safe_load(cfg.read_text())["tables"]) == {"default", "other"}
    catalog.delete_table("alias")
    saved = yaml.safe_load(cfg.read_text())["tables"]
    assert "alias" not in saved["default"] and "copy" in saved["other"]


def test_filter_is_honored_by_batch_reader_and_duckdb_export():
    t = PydalaTable(pa.table({"id": [1, 2, 3]}))
    f = pds.field("id") > 1
    assert t.to_batch_reader(filter=f, sort_by="id").read_all()["id"].to_pylist() == [2, 3]
    rel = t.filter("id > 0", use="duckdb")
    assert rel.to_arrow_table(filter=f)["id"].to_pylist() == [2, 3]


def test_polars_filter_keeps_large_integer_exact():
    import polars as pl

    df = pl.DataFrame({"id": [2**53, 2**53 + 1]}, schema={"id": pl.Int64})
    out = df.filter(sql2polars_filter(f"id IN ({2**53 + 1})", df.schema))
    assert out["id"].to_list() == [2**53 + 1]


def test_failed_later_batch_reports_partial_write(tmp_path):
    import pytest

    from pydala.io import PartialWriteError

    ds = ParquetDataset(str(tmp_path / "ds"))
    with pytest.raises(PartialWriteError):
        ds.write_to_dataset([pa.table({"id": [1]}), pa.table({"typo": [2]})], sort_by="id")


def test_supplied_duckdb_threads_are_kept(tmp_path):
    import duckdb

    con = duckdb.connect(config={"threads": 1})
    ParquetDataset(str(tmp_path / "ds"), ddb_con=con)
    assert con.execute("select current_setting('threads')").fetchone() == (1,)


def test_compact_by_timeperiod_dry_run_without_sort(tmp_path):
    import datetime as dt

    ds = ParquetDataset(str(tmp_path / "ds"))
    ds.write_to_dataset(pa.table({"id": [1, 2], "ts": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)]}))
    ds.load()
    assert ds.compact_by_timeperiod("1d", dry_run=True) is not None


def test_scoped_catalog_update_keeps_structure(tmp_path):
    cfg = tmp_path / "catalog.yml"
    tb = {"filesystem": "local", "path": "sales", "format": "parquet"}
    cfg.write_text(
        yaml.safe_dump(
            {
                "filesystem": {"local": {"protocol": "file", "bucket": "."}},
                "tables": {"default": {"sales": tb}, "other": {"copy": tb}},
            }
        )
    )
    Catalog(str(cfg), namespace="default").update("sales", timestamp_column="ts")
    saved = yaml.safe_load(cfg.read_text())["tables"]
    assert saved["default"]["sales"] == {**tb, "timestamp_column": "ts"}
    assert saved["other"] == {"copy": tb}


def test_failed_vacuum_keeps_catalog_entry(tmp_path):
    import pytest

    cfg = tmp_path / "catalog.yml"
    tb = {"filesystem": "local", "path": "sales", "format": "csv"}
    cfg.write_text(
        yaml.safe_dump(
            {
                "filesystem": {"local": {"protocol": "file", "bucket": "."}},
                "tables": {"default": {"sales": tb}},
            }
        )
    )
    (tmp_path / "sales").mkdir()
    (tmp_path / "sales" / "a.csv").write_text("id\n1\n")
    catalog = Catalog(str(cfg), namespace="default")
    with pytest.raises(TypeError):
        catalog.delete_table("sales", vacuum=True)
    assert "sales" in yaml.safe_load(cfg.read_text())["tables"]["default"]
    assert "sales" not in catalog.table


def test_overwrite_to_zero_rows_refreshes_live_files_and_scan_after_schema_append(tmp_path):
    ds = ParquetDataset(str(tmp_path / "ds"))
    ds.write_to_dataset(pa.table({"id": [1]}))
    ds.load()
    ds.write_to_dataset(pa.table({"id": [2], "extra": ["x"]}), alter_schema=True)
    ds.load()
    assert sorted(r["id"] for r in ds.scan("id = 2").arrow.to_pylist()) == [2]
    ds.write_to_dataset(pa.table({"id": pa.array([], pa.int64())}), mode="overwrite")
    assert ds.files == [] and ds.count_rows() == 0


def test_filtered_duckdb_batch_reader_streams(tmp_path):
    t = PydalaTable(pa.table({"id": [1, 2, 3]})).filter("id > 0", use="duckdb")
    reader = t.to_batch_reader(filter=pds.field("id") > 1, columns="id", batch_size=1)
    assert sum(b.num_rows for b in reader) == 2
    empty = PydalaTable(pa.table({"id": pa.array([], pa.int64())}))
    empty = empty.filter("id > 0", use="duckdb")
    assert empty.to_batch_reader(filter=pds.field("id") > 1).read_all().num_rows == 0


def test_timeperiod_with_empty_window_does_not_touch_other_files(tmp_path):
    import datetime as dt

    ds = ParquetDataset(str(tmp_path / "ds"))
    for day in (1, 20):
        ds.write_to_dataset(pa.table({"id": [day], "ts": [dt.datetime(2024, 1, day)]}))
    ds.load()
    before = sorted(ds.files)
    ds.compact_by_timeperiod("5d", timestamp_column="ts", max_rows_per_file=10)
    assert sorted(ds.files) == before


def test_overwrite_dropping_timestamp_column_still_loads(tmp_path):
    import datetime as dt

    ds = ParquetDataset(str(tmp_path / "ds"))
    ds.write_to_dataset(pa.table({"id": [1], "ts": [dt.datetime(2024, 1, 1)]}))
    ds.load()
    ds.write_to_dataset(pa.table({"id": [2]}), mode="overwrite", alter_schema=True)
    assert ds.table.arrow.to_pylist() == [{"id": 2}]


def test_emptied_overwrite_keeps_partition_layout_and_refills_without_ts(tmp_path):
    import datetime as dt

    ds = ParquetDataset(str(tmp_path / "ds"), partitioning="hive")
    ds.write_to_dataset(
        pa.table({"id": [1], "region": ["a"], "ts": [dt.datetime(2024, 1, 1)]}),
        partition_by=["region"],
    )
    ds.load()
    ds.write_to_dataset(
        pa.table({"id": pa.array([], pa.int64()), "region": pa.array([], pa.string())}),
        mode="overwrite",
    )
    assert ds.files == []
    ds.write_to_dataset(pa.table({"id": [2], "region": ["b"]}))
    ds.load()
    assert any(f.startswith("region=b/") for f in ds.files)
    assert ds.table.arrow.num_rows == 1

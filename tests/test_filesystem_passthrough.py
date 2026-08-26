import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.implementations.memory import MemoryFileSystem

from pydala import ParquetDataset
from pydala.filesystem import FileSystem


def test_supplied_unrooted_filesystem_is_preserved() -> None:
    fs = MemoryFileSystem()
    assert FileSystem(fs=fs) is fs


def test_absolute_local_dataset_path_is_not_rebased(tmp_path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    pq.write_table(pa.table({"id": [1]}), dataset_path / "part.parquet")

    dataset = ParquetDataset(str(dataset_path))

    assert len(dataset.files) == 1

from fsspec.implementations.memory import MemoryFileSystem

from pydala.filesystem import FileSystem


def test_supplied_unrooted_filesystem_is_preserved() -> None:
    fs = MemoryFileSystem()
    assert FileSystem(fs=fs) is fs

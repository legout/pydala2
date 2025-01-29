import posixpath

import pyarrow.fs as pfs
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem

from .misc import run_parallel


def collect_parquet_metadata(
    files: list[str] | str,
    base_path: str | None = None,
    filesystem: AbstractFileSystem | pfs.FileSystem | None = None,
    n_jobs: int = -1,
    backend: str = "threading",
    verbose: bool = True,
) -> dict[str, pq.FileMetaData]:
    """Collect all metadata information of the given parqoet files.

    Args:
        files (list[str] | str): Parquet files.
        filesystem (AbstractFileSystem | pfs.FileSystem | None, optional): Filesystem. Defaults to None.
        n_jobs (int, optional): n_jobs parameter of joblib.Parallel. Defaults to -1.
        backend (str, optional): backend parameter of joblib.Parallel. Defaults to "threading".
        verbose (bool, optional): Wheter to show the task progress using tqdm or not. Defaults to True.

    Returns:
        dict[str, pq.FileMetaData]: Parquet metadata of the given files.
    """

    def get_metadata(f, base_path, filesystem):
        if base_path is not None:
            path = posixpath.join(base_path, f)
        else:
            path = f
        return {f: pq.read_metadata(path, filesystem=filesystem)}

    metadata = run_parallel(
        get_metadata,
        files,
        base_path=base_path,
        filesystem=filesystem,
        backend=backend,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return {key: value for d in metadata for key, value in d.items()}


def remove_from_metadata(
    metadata: pq.FileMetaData,
    rm_files: list[str] | None = None,
    keep_files: list[str] | None = None,
    base_path: str = None,
) -> pq.FileMetaData:
    """
    Removes row groups from the metadata of the dataset.
    This method removes row groups from the given metadata based on the given files.
    Files in `rm_files` will be removed from the metadata. Files in `keep_files` will be kept in the metadata.


    Args:
        metadata (pq.FileMetaData): The metadata of the dataset.
        rm_files (list[str]): The files to delete from the metadata.
        keep_files (list[str]): The files to keep in the metadata.
    Returns:
        pq.FileMetaData: The updated metadata of the dataset.
    """
    row_groups = []
    if rm_files is not None:
        if base_path is not None:
            rm_files = [f.replace(base_path, "").lstrip("/") for f in rm_files]

        # row_groups to keep
        row_groups += [
            metadata.row_group(i)
            for i in range(metadata.num_row_groups)
            if metadata.row_group(i).column(0).file_path not in rm_files
        ]
    if keep_files is not None:
        if base_path is not None:
            keep_files = [f.replace(base_path, "").lstrip("/") for f in keep_files]

        # row_groups to keep
        row_groups += [
            metadata.row_group(i)
            for i in range(metadata.num_row_groups)
            if metadata.row_group(i).column(0).file_path in keep_files
        ]

    if len(row_groups):
        new_metadata = row_groups[0]
        for rg in row_groups[1:]:
            new_metadata.append_row_groups(rg)

        return new_metadata

    return metadata

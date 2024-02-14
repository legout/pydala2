from fsspec import AbstractFileSystem
import pyarrow.filesystem as pfs
import pyarrow.parquet as pq
from .misc import run_parallel


def collect_parquet_metadata(
    files: list[str] | str,
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

    def get_metadata(f, filesystem):
        return {f: pq.read_metadata(f, filesystem=filesystem)}

    metadata = run_parallel(
        get_metadata,
        files,
        filesystem=filesystem,
        backend=backend,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return {key: value for d in metadata for key, value in d.items()}


def remove_from_metadata(
    metadata: pq.FileMetaData,
    files: list[str],
) -> pq.FileMetaData:
    """
    Removes files from the metadata of the dataset.
    This method removes files from the metadata of the dataset and writes the updated metadata to the metadata file.
    Args:
        metadata (pq.FileMetaData): The metadata of the dataset.
        files (list[str]): The files to hold in the metadata.
    Returns:
        pq.FileMetaData: The updated metadata of the dataset.
    """
    row_groups = [
        metadata.row_group(i)
        for i in range(metadata.num_row_groups)
        if metadata.row_group(i).column(0).file_path in files
    ]
    new_metadata = row_groups[0]
    for rg in row_groups[1:]:
        new_metadata.append_row_groups(rg)

    return new_metadata

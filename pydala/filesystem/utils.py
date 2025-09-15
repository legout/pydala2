"""
Filesystem utility functions.
"""
import posixpath
from typing import List, Union

from fsspec import AbstractFileSystem
from ..helpers.misc import run_parallel


def get_new_file_names(src: List[str], dst: List[str]) -> List[str]:
    """Returns a list of new file names that are not in the destination list.

    Args:
        src: List of source file paths
        dst: List of destination file paths

    Returns:
        List of new file names that are not in the destination list
    """
    if not dst:
        return src

    src_file_names = [posixpath.basename(f).split(".")[0] for f in src]
    src_ext = posixpath.basename(src[0]).split(".")[1]
    dst_file_names = [posixpath.basename(f).split(".")[0] for f in dst]

    return [
        posixpath.join(posixpath.dirname(src[0]), f"{f}.{src_ext}")
        for f in src_file_names
        if f not in dst_file_names
    ]


def list_files(
    fs: AbstractFileSystem,
    path: Union[str, List[str]],
    recursive: bool = False,
    maxdepth: int = None,
    files_only: bool = False,
    dirs_only: bool = False,
) -> List[str]:
    """List files with filtering options.

    Args:
        fs: Filesystem instance
        path: Path(s) to list
        recursive: Whether to list recursively
        maxdepth: Maximum depth for recursive listing
        files_only: Whether to return only files
        dirs_only: Whether to return only directories

    Returns:
        List of file/directory paths
    """
    if isinstance(path, str):
        if fs.isfile(path):
            path = [path]
        else:
            path = path.rstrip("*").rstrip("/")
            if fs.exists(path):
                if not recursive:
                    path = fs.glob(posixpath.join(path, "*"))
                else:
                    if maxdepth is not None:
                        path = fs.glob(posixpath.join(path, *(["*"] * maxdepth)))
                    else:
                        path = fs.glob(posixpath.join(path, "**"))
            else:
                path = fs.glob(path)

    if files_only:
        files = [f for f in path if "." in posixpath.basename(f)]
        files += [f for f in sorted(set(path) - set(files)) if fs.isfile(f)]
        return files

    if dirs_only:
        dirs = [f for f in path if "." not in posixpath.basename(f)]
        dirs += [f for f in sorted(set(path) - set(dirs)) if fs.isdir(f)]
        return dirs

    return path


def sync_folder(
    fs: AbstractFileSystem,
    src: str,
    dst: str,
    recursive: bool = False,
    maxdepth: int = None,
) -> None:
    """Sync files from source to destination.

    Args:
        fs: Filesystem instance
        src: Source directory
        dst: Destination directory
        recursive: Whether to sync recursively
        maxdepth: Maximum depth for recursive sync
    """
    src_files = list_files(fs, src, recursive=recursive, maxdepth=maxdepth, files_only=True)
    dst_files = list_files(fs, dst, recursive=recursive, maxdepth=maxdepth, files_only=True)

    if not src_files:
        return

    src_names = [posixpath.basename(f) for f in src_files]
    dst_names = [posixpath.basename(f) for f in dst_files]

    new_src = [
        posixpath.join(posixpath.dirname(src_files[0]), f)
        for f in sorted(set(src_names) - set(dst_names))
    ]

    if new_src:
        fs.cp(new_src, dst)


def json_to_parquet_single(
    fs: AbstractFileSystem,
    src: str,
    dst: str,
    flatten: bool = True,
    distinct: bool = True,
    sort_by: Union[str, List[str]] = None,
    auto_optimize_dtypes: bool = True,
    **kwargs,
) -> None:
    """Convert a single JSON file to Parquet.

    Args:
        fs: Filesystem instance
        src: Source JSON file path
        dst: Destination Parquet file path
        flatten: Whether to flatten nested data
        distinct: Whether to remove duplicates
        sort_by: Column(s) to sort by
        auto_optimize_dtypes: Whether to optimize data types
        **kwargs: Additional arguments
    """
    from .converters import JsonConverter, ConversionService

    conversion_service = ConversionService()
    json_converter = conversion_service.get_converter("json")
    parquet_converter = conversion_service.get_converter("parquet")

    data = json_converter.read(fs, src, as_dataframe=True, flatten=flatten)

    if auto_optimize_dtypes:
        data = data.opt_dtype()

    if distinct:
        data = data.unique(maintain_order=True)

    if sort_by is not None:
        data = data.sort(sort_by)

    if not dst.endswith(".parquet"):
        dst = posixpath.join(
            dst, f"{posixpath.basename(src).replace('.json', '.parquet')}"
        )

    parquet_converter.write(fs, data, dst, **kwargs)


def csv_to_parquet_single(
    fs: AbstractFileSystem,
    src: str,
    dst: str,
    distinct: bool = True,
    sort_by: Union[str, List[str]] = None,
    auto_optimize_dtypes: bool = True,
    **kwargs,
) -> None:
    """Convert a single CSV file to Parquet.

    Args:
        fs: Filesystem instance
        src: Source CSV file path
        dst: Destination Parquet file path
        distinct: Whether to remove duplicates
        sort_by: Column(s) to sort by
        auto_optimize_dtypes: Whether to optimize data types
        **kwargs: Additional arguments
    """
    from .converters import CsvConverter, ConversionService

    conversion_service = ConversionService()
    csv_converter = conversion_service.get_converter("csv")
    parquet_converter = conversion_service.get_converter("parquet")

    data = csv_converter.read(fs, src, **kwargs)

    if auto_optimize_dtypes:
        data = data.opt_dtype()

    if distinct:
        data = data.unique(maintain_order=True)

    if sort_by is not None:
        data = data.sort(sort_by)

    if not dst.endswith(".parquet"):
        dst = posixpath.join(
            dst, f"{posixpath.basename(src).replace('.csv', '.parquet')}"
        )

    parquet_converter.write(fs, data, dst, **kwargs)


def batch_json_to_parquet(
    fs: AbstractFileSystem,
    src: str,
    dst: str,
    flatten: bool = True,
    sync: bool = True,
    parallel: bool = True,
    distinct: bool = True,
    sort_by: Union[str, List[str]] = None,
    auto_optimize_dtypes: bool = True,
    **kwargs,
) -> None:
    """Convert JSON files to Parquet in batch.

    Args:
        fs: Filesystem instance
        src: Source directory
        dst: Destination directory
        flatten: Whether to flatten nested data
        sync: Whether to sync only new files
        parallel: Whether to process in parallel
        distinct: Whether to remove duplicates
        sort_by: Column(s) to sort by
        auto_optimize_dtypes: Whether to optimize data types
        **kwargs: Additional arguments
    """
    if sync:
        src_files = fs.glob(posixpath.join(src, "*.json"))
        dst_files = fs.glob(posixpath.join(dst, "*.parquet"))
        new_src_files = get_new_file_names(src_files, dst_files)
    else:
        new_src_files = fs.glob(posixpath.join(src, "*.json"))

    kwargs.pop("backend", None)
    if len(new_src_files) == 1:
        parallel = False

    if new_src_files:
        run_parallel(
            json_to_parquet_single,
            new_src_files,
            filesystem=fs,
            dst=dst,
            flatten=flatten,
            backend="threading" if parallel else "sequential",
            distinct=distinct,
            sort_by=sort_by,
            auto_optimize_dtypes=auto_optimize_dtypes,
            **kwargs,
        )


def batch_csv_to_parquet(
    fs: AbstractFileSystem,
    src: str,
    dst: str,
    sync: bool = True,
    parallel: bool = True,
    distinct: bool = True,
    sort_by: Union[str, List[str]] = None,
    auto_optimize_dtypes: bool = True,
    **kwargs,
) -> None:
    """Convert CSV files to Parquet in batch.

    Args:
        fs: Filesystem instance
        src: Source directory
        dst: Destination directory
        sync: Whether to sync only new files
        parallel: Whether to process in parallel
        distinct: Whether to remove duplicates
        sort_by: Column(s) to sort by
        auto_optimize_dtypes: Whether to optimize data types
        **kwargs: Additional arguments
    """
    if sync:
        src_files = fs.glob(posixpath.join(src, "*.csv"))
        dst_files = fs.glob(posixpath.join(dst, "*.parquet"))
        new_src_files = get_new_file_names(src_files, dst_files)
    else:
        new_src_files = fs.glob(posixpath.join(src, "*.csv"))

    kwargs.pop("backend", None)
    if len(new_src_files) == 1:
        parallel = False

    if new_src_files:
        run_parallel(
            csv_to_parquet_single,
            new_src_files,
            filesystem=fs,
            dst=dst,
            backend="threading" if parallel else "sequential",
            distinct=distinct,
            sort_by=sort_by,
            auto_optimize_dtypes=auto_optimize_dtypes,
            **kwargs,
        )
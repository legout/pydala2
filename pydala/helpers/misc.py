import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, TypeVar
from functools import wraps
import time

T = TypeVar('T')


def read_table(path: str, filesystem=None, **kwargs) -> "pyarrow.Table":
    """
    Read a parquet file using pyarrow.

    Args:
        path: Path to the parquet file
        filesystem: Optional fsspec filesystem instance
        **kwargs: Additional arguments for pq.read_table

    Returns:
        pyarrow.Table
    """
    if filesystem is not None:
        return pq.read_table(path, filesystem=filesystem, **kwargs)
    else:
        return pq.read_table(path, **kwargs)


def run_parallel(
    func: Callable[..., T],
    items: List[Any],
    *args,
    max_workers: int = None,
    use_threads: bool = True,
    **kwargs
) -> List[T]:
    """
    Run a function in parallel over a list of items.

    Args:
        func: Function to run
        items: List of items to process
        *args: Additional positional arguments for the function
        max_workers: Maximum number of worker threads
        use_threads: Whether to use threads (if False, processes would be used)
        **kwargs: Additional keyword arguments for the function

    Returns:
        List of results in the same order as items
    """
    if not items:
        return []

    if len(items) == 1:
        # No need for parallelization with a single item
        return [func(items[0], *args, **kwargs)]

    # Determine number of workers
    if max_workers is None:
        max_workers = min(len(items), 10)  # Cap at 10 workers by default

    results = []

    if use_threads:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(func, item, *args, **kwargs): item
                for item in items
            }

            # Collect results in order
            for future in as_completed(future_to_item):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error processing {future_to_item[future]}: {e}")
                    raise
    else:
        # Sequential execution
        for item in items:
            results.append(func(item, *args, **kwargs))

    return results




# Import additional utilities from fsspec-utils if available
try:
    from fsspec_utils.utils.misc import *
except ImportError:
    pass
import threading
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

import psutil
from fsspeckit import AbstractFileSystem, filesystem


def get_credentials_from_fsspec(
    fs: AbstractFileSystem, redact_secrets: bool = True
) -> dict[str, str]:
    """
    Safely extract credentials from fsspec filesystem.

    Args:
        fs: The fsspec filesystem object
        redact_secrets: If True, redact sensitive values from returned dict

    Returns:
        Dictionary with credential information (secrets redacted if redact_secrets=True)
    """
    if "s3" in fs.protocol:
        credentials = fs.s3._get_credentials()

        if redact_secrets:
            # For logging and display purposes, redact actual secrets
            return {
                "access_key": f"REDACTED({len(credentials.access_key) if credentials.access_key else 0} chars)",
                "secret_key": f"REDACTED({len(credentials.secret_key) if credentials.secret_key else 0} chars)",
                "session_token": f"REDACTED({len(credentials.token) if credentials.token else 0} chars)"
                if credentials.token
                else None,
                "endpoint_override": fs.s3._endpoint.host if fs.s3._endpoint else None,
            }
        else:
            # Only return actual credentials when explicitly requested
            return {
                "access_key": credentials.access_key,
                "secret_key": credentials.secret_key,
                "session_token": credentials.token,
                "endpoint_override": fs.s3._endpoint.host if fs.s3._endpoint else None,
            }

    return {}


# Back-compat alias for the original misspelled public name.
get_credentials_from_fssspec = get_credentials_from_fsspec


def get_total_directory_size(directory: str) -> int:
    """Calculate the total size of all files in a directory.

    Args:
        directory: Path to the directory to analyze.

    Returns:
        Total size in bytes of all files in the directory and subdirectories.
    """
    return sum(f.stat().st_size for f in Path(directory).glob("**/*") if f.is_file())


class throttle(object):
    """
    Decorator that prevents a function from being called more than once every
    time period.
    To create a function that cannot be called more than once a minute:
        @throttle(minutes=1)
        def my_fun():
            pass
    """

    def __init__(self, seconds=0, minutes=0, hours=0):
        self.throttle_period = timedelta(seconds=seconds, minutes=minutes, hours=hours)
        self.time_of_last_call = datetime.min

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            now = datetime.now()
            time_since_last_call = now - self.time_of_last_call

            if time_since_last_call > self.throttle_period:
                self.time_of_last_call = now
                return fn(*args, **kwargs)

        return wrapper


def sizeof_fmt(num: float, suffix="B") -> str:
    """Format a number of bytes as a human-readable string.

    Args:
        num: Number of bytes to format.
        suffix: Suffix to append to the formatted string.

    Returns:
        Human-readable string representation (e.g., '1.5 MiB').
    """
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


class DiskUsageTracker:
    """Thread-safe disk usage tracking with delta calculations.

    This class tracks disk usage over time, calculating deltas between
    measurements to show how much disk space has been consumed.
    """

    def __init__(self) -> None:
        """Initialize the DiskUsageTracker."""
        self._last_free = None
        self._first_free = None
        self._lock = threading.Lock()

    def get_usage_message(self, storage: str) -> str:
        """Get friendly disk usage message with deltas.

        Args:
            storage: Path to the storage location to check.

        Returns:
            Formatted string with usage information including deltas.
        """
        with self._lock:
            usage = psutil.disk_usage(storage)
            if self._first_free is None:
                self._first_free = usage.free
            current_usage = get_total_directory_size(storage)
            message = (
                f"{sizeof_fmt(current_usage)} used {sizeof_fmt(usage.free)} available"
            )
            if self._last_free is not None:
                downloaded_recently = self._last_free - usage.free
                if downloaded_recently > 10_000_000:
                    downloaded_since_start = self._first_free - usage.free
                    if downloaded_recently != downloaded_since_start:
                        message += f" delta: {sizeof_fmt(downloaded_recently)}"
                    message += (
                        f" delta since start: {sizeof_fmt(downloaded_since_start)}"
                    )

            self._last_free = usage.free
            return message

    def reset(self) -> None:
        """Reset tracking state.

        Clears all stored usage information and starts fresh tracking.
        """
        with self._lock:
            self._first_free = None
            self._last_free = None


# Global instance for backward compatibility
_disk_tracker = DiskUsageTracker()


def get_friendly_disk_usage(storage: str) -> str:
    """Get friendly disk usage message (legacy function).

    This is a convenience function that uses the global DiskUsageTracker instance.

    Args:
        storage: Path to the storage location to check.

    Returns:
        Formatted string with disk usage information.
    """
    return _disk_tracker.get_usage_message(storage)


def FileSystem(
    bucket: str | None = None,
    fs: AbstractFileSystem | None = None,
    profile: str | None = None,
    key: str | None = None,
    endpoint_url: str | None = None,
    secret: str | None = None,
    token: str | None = None,
    protocol: str | None = None,
    cached: bool = False,
    cache_storage="~/.tmp",
    check_files: bool = False,
    # cache_check: int = 120,
    # expire_time: int = 24 * 60 * 60,
    # same_names: bool = False,
    **kwargs,
) -> AbstractFileSystem:
    protocol_or_path = protocol
    if protocol is None:
        protocol_or_path = bucket or "file"
    else:
        if bucket is None:
            protocol_or_path = protocol
        else:
            protocol_or_path = f"{protocol}://{bucket}"

    return filesystem(
        protocol_or_path=protocol_or_path,
        base_fs=fs,
        profile=profile,
        key=key,
        endpoint_url=endpoint_url,
        secret=secret,
        token=token,
        cached=cached,
        cache_storage=cache_storage,
        **kwargs,
    )


def clear_cache(fs: AbstractFileSystem | None) -> None:
    """Invalidate caches on a filesystem and any nested base filesystem."""
    if fs is None:
        return
    fs.invalidate_cache()
    fs.clear_instance_cache()
    if hasattr(fs, "fs"):
        fs.fs.invalidate_cache()
        fs.fs.clear_instance_cache()

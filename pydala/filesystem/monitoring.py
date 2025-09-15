"""
Monitoring module for filesystem operations.
"""
import threading
from pathlib import Path
from typing import Optional

import psutil


def sizeof_fmt(num: float, suffix: str = "B") -> str:
    """Format file size in human readable format.

    Args:
        num: Size in bytes
        suffix: Unit suffix

    Returns:
        Formatted size string
    """
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def get_total_directory_size(directory: str) -> int:
    """Calculate total size of directory.

    Args:
        directory: Directory path

    Returns:
        Total size in bytes
    """
    return sum(f.stat().st_size for f in Path(directory).glob("**/*") if f.is_file())


class DiskUsageTracker:
    """Thread-safe disk usage tracking."""

    def __init__(self):
        self._last_free: Optional[int] = None
        self._first_free: Optional[int] = None
        self._lock = threading.Lock()

    def get_usage_message(self, storage: str) -> str:
        """Get friendly disk usage message with deltas.

        Args:
            storage: Storage path to check

        Returns:
            Formatted usage message
        """
        with self._lock:
            usage = psutil.disk_usage(storage)
            if self._first_free is None:
                self._first_free = usage.free

            current_usage = get_total_directory_size(storage)
            message = f"{sizeof_fmt(current_usage)} used {sizeof_fmt(usage.free)} available"

            if self._last_free is not None:
                downloaded_recently = self._last_free - usage.free
                if downloaded_recently > 10_000_000:
                    downloaded_since_start = self._first_free - usage.free
                    if downloaded_recently != downloaded_since_start:
                        message += f" delta: {sizeof_fmt(downloaded_recently)}"
                    message += f" delta since start: {sizeof_fmt(downloaded_since_start)}"

            self._last_free = usage.free
            return message

    def reset(self) -> None:
        """Reset tracking state."""
        with self._lock:
            self._first_free = None
            self._last_free = None


class MonitoringService:
    """Service for monitoring filesystem operations and system resources."""

    def __init__(self):
        self._disk_tracker = DiskUsageTracker()

    def get_disk_usage(self, storage: str) -> str:
        """Get disk usage information.

        Args:
            storage: Storage path

        Returns:
            Formatted disk usage message
        """
        return self._disk_tracker.get_usage_message(storage)

    def reset_disk_tracking(self) -> None:
        """Reset disk usage tracking."""
        self._disk_tracker.reset()

    def get_system_memory_usage(self) -> dict:
        """Get current system memory usage.

        Returns:
            Dictionary with memory information
        """
        memory = psutil.virtual_memory()
        return {
            "total": sizeof_fmt(memory.total),
            "available": sizeof_fmt(memory.available),
            "used": sizeof_fmt(memory.used),
            "percent": f"{memory.percent:.1f}%",
        }

    def get_system_cpu_usage(self) -> dict:
        """Get current system CPU usage.

        Returns:
            Dictionary with CPU information
        """
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        return {
            "usage_percent": f"{cpu_percent:.1f}%",
            "logical_cores": cpu_count,
            "physical_cores": psutil.cpu_count(logical=False),
        }
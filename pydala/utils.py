import datetime as dt
import os
import posixpath
import threading
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
from fsspec import AbstractFileSystem
from fsspec.implementations.cache_mapper import AbstractCacheMapper
from loguru import logger


def get_credentials_from_fssspec(fs: AbstractFileSystem, redact_secrets: bool = True) -> Dict[str, str]:
    \"\"\"
    Safely extract credentials from fsspec filesystem.

    Args:
        fs: The fsspec filesystem object
        redact_secrets: If True, redact sensitive values from returned dict

    Returns:
        Dictionary with credential information (secrets redacted if redact_secrets=True)
    \"\"\"
    if \"s3\" in fs.protocol:
        credentials = fs.s3._get_credentials()

        if redact_secrets:
            return {
                \"access_key\": f\"REDACTED({len(credentials.access_key) if credentials.access_key else 0} chars)\",
                \"secret_key\": f\"REDACTED({len(credentials.secret_key) if credentials.secret_key else 0} chars)\",
                \"session_token\": f\"REDACTED({len(credentials.token) if credentials.token else 0} chars)\" if credentials.token else None,
                \"endpoint_override\": fs.s3._endpoint.host if fs.s3._endpoint else None,
            }
        else:
            return {
                \"access_key\": credentials.access_key,
                \"secret_key\": credentials.secret_key,
                \"session_token\": credentials.token,
                \"endpoint_override\": fs.s3._endpoint.host if fs.s3._endpoint else None,
            }

    return {}


def get_total_directory_size(directory: str) -> int:
    return sum(f.stat().st_size for f in Path(directory).glob(\"**/*\") if f.is_file())


class FileNameCacheMapper(AbstractCacheMapper):
    def __init__(self, directory: str):
        from .helpers.security import validate_path
        self.directory = validate_path(directory)

    def __call__(self, path: str) -> str:
        from .helpers.security import validate_path, safe_join
        validated_path = validate_path(path)
        full_path = safe_join(self.directory, validated_path)
        parent_dir = posixpath.dirname(full_path)
        os.makedirs(parent_dir, exist_ok=True)
        return validated_path


class throttle:
    \"\"\"
    Decorator that prevents a function from being called more than once every
    time period.
    To create a function that cannot be called more than once a minute:
        @throttle(minutes=1)
        def my_fun():
            pass
    \"\"\"

    def __init__(self, seconds: int = 0, minutes: int = 0, hours: int = 0):
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


def sizeof_fmt(num: float, suffix: str = \"B\") -> str:
    for unit in (\"\", \"Ki\", \"Mi\", \"Gi\", \"Ti\", \"Pi\", \"Ei\", \"Zi\"):
        if abs(num) < 1024.0:
            return f\"{num:3.1f}{unit}{suffix}\"
        num /= 1024.0
    return f\"{num:.1f}Yi{suffix}\"


class DiskUsageTracker:
    \"\"\"Thread-safe disk usage tracking.\"\"\"

    def __init__(self):
        self._last_free: Optional[int] = None
        self._first_free: Optional[int] = None
        self._lock = threading.Lock()

    def get_usage_message(self, storage: str) -> str:
        \"\"\"Get friendly disk usage message with deltas.\"\"\"
        with self._lock:
            usage = psutil.disk_usage(storage)
            if self._first_free is None:
                self._first_free = usage.free
            current_usage = get_total_directory_size(storage)
            message = f\"{sizeof_fmt(current_usage)} used {sizeof_fmt(usage.free)} available\"
            if self._last_free is not None:
                downloaded_recently = self._last_free - usage.free
                if downloaded_recently > 10_000_000:
                    downloaded_since_start = self._first_free - usage.free
                    if downloaded_recently != downloaded_since_start:
                        message += f\" delta: {sizeof_fmt(downloaded_recently)}\"
                    message += f\" delta since start: {sizeof_fmt(downloaded_since_start)}\"

            self._last_free = usage.free
            return message

    def reset(self):
        \"\"\"Reset tracking state.\"\"\"
        with self._lock:
            self._first_free = None
            self._last_free = None


# Global instance for backward compatibility
_disk_tracker = DiskUsageTracker()


def get_friendly_disk_usage(storage: str) -> str:
    \"\"\"Get friendly disk usage message (legacy function).\"\"\"
    return _disk_tracker.get_usage_message(storage)


def get_new_file_names(src: List[str], dst: List[str]) -> List[str]:
    \"\"\"
    Returns a list of new file names that are not in the destination list

    Parameters
    ----------
    src : list[str]
        List of source file paths
    dst : list[str]
        List of destination file paths

    Returns
    -------
    list[str]
        List of new file names that are not in the destination list
    \"\"\"
    if len(dst) == 0:
        return src
    src_file_names = [posixpath.basename(f).split(\".\")[0] for f in src]
    src_ext = posixpath.basename(src[0]).split(\".\")[1]
    dst_file_names = [posixpath.basename(f).split(\".\")[0] for f in dst]

    return [
        posixpath.join(posixpath.dirname(src[0]), f\"{f}.{src_ext}\")
        for f in src_file_names
        if f not in dst_file_names
    ]
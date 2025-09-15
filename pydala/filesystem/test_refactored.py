"""
Test script to verify the refactored filesystem module works correctly.
"""
import tempfile
import os
from pathlib import Path

import polars as pl

# Test the new refactored module
def test_refactored_filesystem():
    print("Testing refactored filesystem module...")

    # Test 1: Create local filesystem using new service
    from pydala.filesystem import FileSystemService, CacheManager, MonitoringService

    # Create services
    cache_manager = CacheManager(cache_storage=tempfile.mkdtemp())
    monitoring_service = MonitoringService()
    fs_service = FileSystemService(
        cache_manager=cache_manager,
        monitoring_service=monitoring_service
    )

    # Create local filesystem
    fs = fs_service.create_filesystem(protocol="file", cached=True)
    print(f"✓ Created filesystem: {type(fs).__name__}")

    # Test 2: Create test data
    test_data = pl.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'value': [10.5, 20.3, 30.7]
    })

    # Test 3: Write and read Parquet using service
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "test.parquet")

        # Write using service
        fs_service.write_data(fs, test_data, parquet_path, format="parquet")
        print("✓ Wrote Parquet data using service")

        # Read using service
        read_data = fs_service.read_data(fs, parquet_path, format="parquet")
        print("✓ Read Parquet data using service")
        assert test_data.equals(read_data), "Data mismatch!"

    # Test 4: Test backward compatibility
    from pydala.filesystem import FileSystem

    fs_legacy = FileSystem(protocol="file")

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "test.parquet")

        # Write using legacy API
        fs_legacy.write_parquet(test_data, parquet_path)
        print("✓ Wrote Parquet data using legacy API")

        # Read using legacy API
        read_data = fs_legacy.read_parquet(parquet_path)
        print("✓ Read Parquet data using legacy API")
        assert test_data.equals(read_data), "Data mismatch!"

    # Test 5: Test monitoring
    usage = monitoring_service.get_disk_usage(tmpdir)
    print(f"✓ Disk monitoring works: {usage}")

    # Test 6: Test conversion service
    from pydala.filesystem import ConversionService

    conversion_service = ConversionService()
    parquet_converter = conversion_service.get_converter("parquet")
    print("✓ Conversion service works")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    test_refactored_filesystem()
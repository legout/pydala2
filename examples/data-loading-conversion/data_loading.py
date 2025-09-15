"""
Data Loading and Conversion Example

This example demonstrates how to load data from various sources and convert
between different formats using PyDala2.
"""

import os
import tempfile
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pac
import pyarrow.json as paj

# Import PyDala2 components
from pydala.dataset import ParquetDataset, CsvDataset, JsonDataset
from pydala.table import PydalaTable
from pydala.catalog import Catalog

# Optional imports
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


def create_sample_csv_data(path: Path, num_records: int = 1000):
    """Create sample CSV data for demonstration."""
    csv_file = path / "sample_data.csv"

    # Generate sample data
    records = []
    start_date = datetime(2024, 1, 1)

    for i in range(num_records):
        record = {
            'id': i + 1,
            'name': f'Product_{i % 100}',
            'category': ['Electronics', 'Clothing', 'Food', 'Books'][i % 4],
            'price': round((i % 100) * 1.5 + 10, 2),
            'quantity': i % 50 + 1,
            'date': (start_date + timedelta(days=i % 365)).strftime('%Y-%m-%d'),
            'is_active': i % 2 == 0
        }
        records.append(record)

    # Write to CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    return csv_file


def create_sample_json_data(path: Path, num_records: int = 500):
    """Create sample JSON data for demonstration."""
    json_file = path / "sample_data.json"

    # Generate sample nested data
    records = []

    for i in range(num_records):
        record = {
            'user_id': f'user_{i:04d}',
            'profile': {
                'name': f'User {i}',
                'age': 20 + (i % 60),
                'city': ['New York', 'London', 'Tokyo', 'Paris'][i % 4]
            },
            'orders': [
                {
                    'order_id': f'order_{i}_{j}',
                    'amount': round((j + 1) * 25.5, 2),
                    'items': j + 1
                }
                for j in range(i % 3 + 1)
            ],
            'registration_date': (datetime(2023, 1, 1) + timedelta(days=i)).isoformat(),
            'active': True
        }
        records.append(record)

    # Write to JSON
    with open(json_file, 'w') as f:
        json.dump(records, f, indent=2)

    return json_file


def example_1_loading_from_csv():
    """Example 1: Loading data from CSV files."""
    print("\n=== Example 1: Loading from CSV ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create sample CSV data
        csv_file = create_sample_csv_data(temp_path, 1000)

        # Method 1: Direct CSV loading
        print("\n1.1 Direct CSV loading...")
        csv_dataset = CsvDataset(csv_file)
        print(f"Loaded {len(csv_dataset)} records from CSV")
        print(f"Schema: {csv_dataset.schema}")

        # Method 2: Convert CSV to Parquet
        print("\n1.2 Converting CSV to Parquet...")
        parquet_path = temp_path / "converted_data"
        parquet_dataset = ParquetDataset.from_csv(
            csv_file,
            path=parquet_path,
            partition_cols=['category']
        )
        print(f"Converted to Parquet with {len(parquet_dataset.files)} files")

        # Method 3: Using PyDalaTable
        print("\n1.3 Using PyDalaTable...")
        table = csv_dataset.to_table()
        filtered = table.filter(table.price > 50)
        print(f"Filtered to {len(filtered)} records with price > 50")

        return csv_dataset, parquet_dataset, table


def example_2_loading_from_json():
    """Example 2: Loading data from JSON files."""
    print("\n=== Example 2: Loading from JSON ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create sample JSON data
        json_file = create_sample_json_data(temp_path, 500)

        # Load JSON data
        print("\n2.1 Loading JSON data...")
        json_dataset = JsonDataset(json_file)
        print(f"Loaded {len(json_dataset)} records from JSON")

        # Explore nested structure
        print("\n2.2 Exploring nested structure...")
        table = json_dataset.to_table()
        print(f"Columns: {table.columns}")

        # Flatten nested data
        print("\n2.3 Flattening nested data...")
        # This would typically require more complex transformation
        # For now, we'll show basic structure
        if 'profile' in table.columns:
            print("Found nested 'profile' field")

        return json_dataset


def example_3_format_conversion():
    """Example 3: Converting between formats."""
    print("\n=== Example 3: Format Conversion ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Start with CSV data
        csv_file = create_sample_csv_data(temp_path, 500)

        # CSV to Parquet
        print("\n3.1 CSV to Parquet...")
        csv_ds = CsvDataset(csv_file)
        parquet_path = temp_path / "parquet_output"
        parquet_ds = ParquetDataset.from_dataset(
            csv_ds,
            path=parquet_path,
            compression='zstd'
        )
        print(f"Converted to Parquet with compression")

        # Parquet to different formats
        print("\n3.2 Parquet to other formats...")
        table = parquet_ds.to_table()

        # To Arrow
        arrow_table = table.to_arrow()
        print(f"Converted to Arrow: {type(arrow_table)}")

        # To Pandas
        pandas_df = table.to_pandas()
        print(f"Converted to Pandas: {type(pandas_df)}")

        # To CSV (individual file)
        csv_output = temp_path / "output.csv"
        pandas_df.to_csv(csv_output, index=False)
        print(f"Saved to CSV: {csv_output}")

        # To JSON
        json_output = temp_path / "output.json"
        pandas_df.to_json(json_output, orient='records', date_format='iso')
        print(f"Saved to JSON: {json_output}")

        return parquet_ds, table


def example_4_working_with_partitioned_data():
    """Example 4: Working with partitioned datasets."""
    print("\n=== Example 4: Partitioned Datasets ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create partitioned dataset
        csv_file = create_sample_csv_data(temp_path, 2000)

        # Partition by multiple columns
        print("\n4.1 Creating partitioned dataset...")
        partitioned_path = temp_path / "partitioned_data"
        partitioned_ds = ParquetDataset.from_csv(
            csv_file,
            path=partitioned_path,
            partition_cols=['category', 'is_active']
        )

        print(f"Created partitioned dataset:")
        print(f" - Partition columns: category, is_active")
        print(f" - Number of files: {len(partitioned_ds.files)}")

        # Explore partitions
        print("\n4.2 Exploring partitions...")
        for file_path in partitioned_ds.files[:5]:  # Show first 5
            print(f"   {file_path}")

        # Query specific partition
        print("\n4.3 Querying specific partition...")
        table = partitioned_ds.to_table()
        electronics_data = table.filter(table.category == 'Electronics')
        print(f"Electronics category: {len(electronics_data)} records")

        return partitioned_ds


def example_5_integration_with_libraries():
    """Example 5: Integration with other data libraries."""
    print("\n=== Example 5: Library Integration ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create base dataset
        csv_file = create_sample_csv_data(temp_path, 1000)
        ds = ParquetDataset.from_csv(csv_file, temp_path / "base_data")
        table = ds.to_table()

        # Pandas integration
        print("\n5.1 Pandas integration...")
        pdf = table.to_pandas()
        pandas_result = pdf.groupby('category')['price'].agg(['mean', 'count'])
        print("Pandas aggregation result:")
        print(pandas_result)

        # Polars integration (if available)
        if POLARS_AVAILABLE:
            print("\n5.2 Polars integration...")
            polars_df = table.to_polars()
            polars_result = polars_df.groupby('category').agg([
                pl.col('price').mean(),
                pl.col('quantity').sum()
            ])
            print("Polars aggregation result:")
            print(polars_result)
        else:
            print("\n5.2 Polars not available")

        # DuckDB integration (if available)
        if DUCKDB_AVAILABLE:
            print("\n5.3 DuckDB integration...")
            # Register table with DuckDB
            con = duckdb.connect()
            con.register('pydala_table', table.to_arrow())

            # Run SQL query
            sql_result = con.execute("""
                SELECT
                    category,
                    AVG(price) as avg_price,
                    SUM(quantity) as total_quantity,
                    COUNT(*) as count
                FROM pydala_table
                GROUP BY category
                ORDER BY avg_price DESC
            """).fetchdf()

            print("DuckDB SQL result:")
            print(sql_result)
            con.close()
        else:
            print("\n5.3 DuckDB not available")

        return table


def example_6_handling_large_datasets():
    """Example 6: Efficient handling of large datasets."""
    print("\n=== Example 6: Large Dataset Handling ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create larger dataset
        print("\n6.1 Creating larger dataset...")
        csv_file = create_sample_csv_data(temp_path, 10000)  # 10K records

        # Load with batching
        print("\n6.2 Loading with batching...")
        ds = ParquetDataset.from_csv(csv_file, temp_path / "large_data")
        table = ds.to_table()

        # Use scanner for memory-efficient reading
        scanner = table.to_arrow_scanner(
            batch_size=1000,
            columns=['id', 'name', 'category', 'price']
        )

        # Process in batches
        batch_count = 0
        total_processed = 0

        print("Processing in batches...")
        for batch in scanner.to_batches():
            batch_count += 1
            total_processed += batch.num_rows

            # Example batch processing
            if batch_count <= 3:  # Show first 3 batches
                avg_price = batch.column('price').to_pandas().mean()
                print(f"  Batch {batch_count}: {batch.num_rows} rows, avg price: {avg_price:.2f}")

        print(f"\nProcessed {total_processed} total records in {batch_count} batches")

        # Memory-efficient filtering
        print("\n6.3 Memory-efficient filtering...")
        filtered_scanner = table.to_arrow_scanner(
            filter=pa.dataset.field('price') > 75
        )

        # Count without loading all data
        count = 0
        for batch in filtered_scanner.to_batches():
            count += batch.num_rows

        print(f"Records with price > 75: {count}")

        return ds


def main():
    """Run all examples."""
    print("PyDala2 Data Loading and Conversion Examples")
    print("=" * 50)

    print(f"\nLibrary Availability:")
    print(f"  - Polars: {'✓' if POLARS_AVAILABLE else '✗'}")
    print(f"  - DuckDB: {'✓' if DUCKDB_AVAILABLE else '✗'}")

    try:
        # Run all examples
        example_1_loading_from_csv()
        example_2_loading_from_json()
        example_3_format_conversion()
        example_4_working_with_partitioned_data()
        example_5_integration_with_libraries()
        example_6_handling_large_datasets()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
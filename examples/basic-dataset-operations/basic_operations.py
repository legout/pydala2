"""
Basic Dataset Operations Example

This example demonstrates fundamental PyDala2 dataset operations including:
- Creating datasets from various sources
- Basic data manipulation
- Working with metadata
- Dataset persistence
"""

import os
import tempfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Import PyDala2 components
from pydala.dataset import ParquetDataset, PyarrowDataset
from pydala.table import PydalaTable
from pydala.catalog import Catalog


def create_sample_data():
    """Create sample data for demonstration."""
    # Create a simple DataFrame
    data = {
        'id': range(1, 101),
        'name': [f'Item_{i}' for i in range(1, 101)],
        'category': ['A', 'B', 'C', 'D'] * 25,
        'value': [i * 1.5 for i in range(100)],
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='D')
    }
    return pd.DataFrame(data)


def example_1_creating_datasets():
    """Example 1: Creating datasets from different sources."""
    print("\n=== Example 1: Creating Datasets ===")

    # Create temporary directory for our data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create sample data
        df = create_sample_data()

        # Method 1: Create from pandas DataFrame
        print("\n1.1 Creating dataset from pandas DataFrame...")
        dataset_path = temp_path / "dataset_from_pandas"
        ds = ParquetDataset.from_pandas(
            df,
            path=dataset_path,
            partition_cols=['category']
        )
        print(f"Created dataset at: {dataset_path}")
        print(f"Dataset contains {len(ds)} records")

        # Method 2: Create from PyArrow Table
        print("\n1.2 Creating dataset from PyArrow Table...")
        arrow_table = pa.Table.from_pandas(df)
        dataset_path2 = temp_path / "dataset_from_arrow"
        ds2 = ParquetDataset.from_arrow(
            arrow_table,
            path=dataset_path2,
            row_group_size=25
        )
        print(f"Created dataset at: {dataset_path2}")

        # Method 3: Create from existing parquet files
        print("\n1.3 Creating dataset from existing parquet files...")
        # Save individual parquet files
        for category in ['A', 'B', 'C', 'D']:
            category_df = df[df['category'] == category]
            category_path = temp_path / f"raw_data/category={category}"
            category_path.mkdir(parents=True, exist_ok=True)
            category_df.to_parquet(category_path / f"data_{category}.parquet")

        ds3 = ParquetDataset(temp_path / "raw_data")
        print(f"Created dataset from {len(ds3.files)} files")

        return ds, ds2, ds3


def example_2_basic_operations():
    """Example 2: Basic dataset operations."""
    print("\n=== Example 2: Basic Operations ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a dataset
        df = create_sample_data()
        ds = ParquetDataset.from_pandas(df, path=temp_path / "operations_dataset")

        # Basic dataset info
        print(f"\n2.1 Dataset Information:")
        print(f" - Number of files: {len(ds.files)}")
        print(f" - Total rows: {len(ds)}")
        print(f" - Schema: {ds.schema}")

        # Convert to PydalaTable for advanced operations
        table = ds.to_table()

        # Select columns
        print(f"\n2.2 Column Selection:")
        selected = table.select(['name', 'value', 'category'])
        print(f"Selected columns: {selected.columns}")

        # Filter data
        print(f"\n2.3 Filtering Data:")
        filtered = table.filter(table.value > 50)
        print(f"Records with value > 50: {len(filtered)}")

        # Aggregate data
        print(f"\n2.4 Aggregation:")
        # Convert to pandas for aggregation (simple way)
        df_filtered = filtered.to_pandas()
        agg_result = df_filtered.groupby('category')['value'].agg(['mean', 'sum', 'count'])
        print(f"Aggregation by category:\n{agg_result}")

        return table


def example_3_metadata_handling():
    """Example 3: Working with metadata."""
    print("\n=== Example 3: Metadata Handling ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create dataset with custom metadata
        df = create_sample_data()
        ds = ParquetDataset.from_pandas(
            df,
            path=temp_path / "metadata_dataset",
            partition_cols=['category']
        )

        # Add custom metadata
        print("\n3.1 Adding Custom Metadata:")
        ds.metadata.update({
            'description': 'Sample dataset for demonstration',
            'created_by': 'PyDala2 examples',
            'version': '1.0',
            'tags': ['sample', 'demo', 'test']
        })
        ds.save_metadata()

        # Read metadata
        print(f"\n3.2 Reading Metadata:")
        print(f"Description: {ds.metadata.get('description')}")
        print(f"Created by: {ds.metadata.get('created_by')}")
        print(f"Version: {ds.metadata.get('version')}")

        # Work with schema metadata
        print(f"\n3.3 Schema Information:")
        for field in ds.schema:
            print(f" - {field.name}: {field.type}")

        return ds


def example_4_catalog_operations():
    """Example 4: Using the catalog system."""
    print("\n=== Example 4: Catalog Operations ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a catalog
        catalog_path = temp_path / "catalog.yaml"
        catalog = Catalog(catalog_path)

        # Create multiple datasets
        df = create_sample_data()

        # Dataset 1: Full data
        ds1 = ParquetDataset.from_pandas(
            df,
            path=temp_path / "full_dataset"
        )
        catalog.register_dataset("full_data", ds1)

        # Dataset 2: Filtered data
        df_filtered = df[df['category'].isin(['A', 'B'])]
        ds2 = ParquetDataset.from_pandas(
            df_filtered,
            path=temp_path / "filtered_dataset"
        )
        catalog.register_dataset("filtered_data", ds2)

        # List datasets
        print(f"\n4.1 Registered Datasets:")
        for name in catalog.list_datasets():
            print(f" - {name}")

        # Retrieve dataset
        print(f"\n4.2 Retrieving Dataset:")
        retrieved_ds = catalog.get_dataset("full_data")
        print(f"Retrieved dataset has {len(retrieved_ds)} records")

        # Save catalog
        catalog.save()
        print(f"\n4.3 Catalog saved to: {catalog_path}")

        return catalog


def example_5_table_operations():
    """Example 5: Advanced table operations."""
    print("\n=== Example 5: Table Operations ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create dataset
        df = create_sample_data()
        ds = ParquetDataset.from_pandas(df, path=temp_path / "table_dataset")
        table = ds.to_table()

        # Convert to different formats
        print("\n5.1 Format Conversions:")

        # To pandas
        pdf = table.to_pandas()
        print(f"Converted to pandas: {type(pdf)} with shape {pdf.shape}")

        # To PyArrow
        arrow_table = table.to_arrow()
        print(f"Converted to PyArrow: {type(arrow_table)} with {arrow_table.num_rows} rows")

        # Get scanner for efficient reading
        print(f"\n5.2 Using Scanner:")
        scanner = table.to_arrow_scanner(
            columns=['id', 'name', 'value'],
            filter=pa.dataset.field('value') > 75
        )

        # Read in batches
        batch_count = 0
        total_rows = 0
        for batch in scanner.to_batches():
            batch_count += 1
            total_rows += batch.num_rows

        print(f"Read {batch_count} batches with {total_rows} total rows")

        # Head operation
        print(f"\n5.3 Head Operation:")
        head_data = table.head(5)
        print(f"First 5 rows:\n{head_data}")

        return table


def main():
    """Run all examples."""
    print("PyDala2 Basic Dataset Operations Examples")
    print("=" * 50)

    try:
        # Run all examples
        example_1_creating_datasets()
        table = example_2_basic_operations()
        ds = example_3_metadata_handling()
        catalog = example_4_catalog_operations()
        table = example_5_table_operations()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
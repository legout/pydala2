"""
Metadata Management Example

This example demonstrates advanced metadata handling capabilities in PyDala2, including:
- Schema management and validation
- Metadata collection and synchronization
- Partition metadata handling
- Performance optimization through metadata
- Schema evolution and migration
"""

import os
import tempfile
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Import PyDala2 components
from pydala.dataset import ParquetDataset
from pydala.table import PydalaTable
from pydala.metadata import ParquetDatasetMetadata, PydalaDatasetMetadata
from pydala.schema import replace_schema, convert_large_types_to_normal
from pydala.catalog import Catalog


def create_complex_sample_data():
    """Create complex sample data with various data types for metadata demonstration."""
    np.random.seed(42)

    # Create data with various data types and structures
    data = {
        'id': range(1, 1001),
        'name': [f'Product_{i}' for i in range(1, 1001)],
        'category': ['Electronics', 'Clothing', 'Books', 'Home', 'Sports'] * 200,
        'price': np.random.uniform(10, 1000, 1000).round(2),
        'rating': np.random.uniform(1, 5, 1000).round(1),
        'in_stock': np.random.choice([True, False], 1000),
        'created_at': pd.date_range('2023-01-01', periods=1000, freq='6H'),
        'updated_at': pd.date_range('2023-01-01', periods=1000, freq='6H') + pd.Timedelta(days=np.random.randint(1, 365, 1000)),
        'tags': [[f'tag_{j}' for j in range(np.random.randint(1, 5))] for _ in range(1000)],
        'metadata': [{'source': 'web', 'priority': np.random.randint(1, 5)} for _ in range(1000)]
    }

    return pd.DataFrame(data)


def example_1_basic_metadata_operations():
    """Example 1: Basic metadata operations."""
    print("\n=== Example 1: Basic Metadata Operations ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create dataset
        df = create_complex_sample_data()
        ds = ParquetDataset.from_pandas(
            df,
            path=temp_path / "basic_metadata",
            partition_cols=['category']
        )

        # Access basic metadata
        print("\n1.1 Dataset Information:")
        print(f" - Number of files: {len(ds.files)}")
        print(f" - Total rows: {len(ds)}")
        print(f" - Schema: {ds.schema}")
        print(f" - Partitioning: {ds.partitioning}")

        # Add custom metadata
        print("\n1.2 Adding Custom Metadata:")
        custom_metadata = {
            'description': 'E-commerce product catalog dataset',
            'created_by': 'PyDala2 metadata example',
            'version': '1.0.0',
            'data_source': 'synthetic',
            'creation_date': datetime.now(timezone.utc).isoformat(),
            'tags': ['e-commerce', 'products', 'catalog'],
            'business_unit': 'retail',
            'data_classification': 'internal',
            'retention_policy': '5_years'
        }

        ds.metadata.update(custom_metadata)
        ds.save_metadata()

        # Read and verify metadata
        print("\n1.3 Reading Metadata:")
        for key, value in ds.metadata.items():
            print(f" - {key}: {value}")

        # Schema metadata
        print("\n1.4 Schema Metadata:")
        for field in ds.schema:
            field_metadata = field.metadata or {}
            print(f" - {field.name}: {field.type}")
            if field_metadata:
                print(f"   Metadata: {dict(field_metadata)}")

        return ds


def example_2_schema_management():
    """Example 2: Advanced schema management."""
    print("\n=== Example 2: Schema Management ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create initial dataset
        df = create_complex_sample_data()
        ds = ParquetDataset.from_pandas(
            df,
            path=temp_path / "schema_management"
        )

        print("\n2.1 Original Schema:")
        for field in ds.schema:
            print(f" - {field.name}: {field.type}")

        # Schema evolution - add new columns
        print("\n2.2 Schema Evolution - Adding Columns:")

        # Create new data with additional columns
        new_data = df.copy()
        new_data['discount_percent'] = np.random.uniform(0, 30, len(df)).round(1)
        new_data['is_featured'] = np.random.choice([True, False], len(df))

        # Create new dataset with evolved schema
        ds_evolved = ParquetDataset.from_pandas(
            new_data,
            path=temp_path / "schema_evolved"
        )

        print("\nEvolved Schema:")
        for field in ds_evolved.schema:
            print(f" - {field.name}: {field.type}")

        # Schema validation
        print("\n2.3 Schema Validation:")

        # Convert to table for schema operations
        table = ds.to_table()

        # Check for specific columns
        required_columns = ['id', 'name', 'category', 'price']
        missing_columns = [col for col in required_columns if col not in table.columns]

        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
        else:
            print("All required columns present")

        # Type checking
        print("\n2.4 Type Validation:")
        expected_types = {
            'id': pa.int64(),
            'name': pa.string(),
            'price': pa.float64(),
            'in_stock': pa.bool_()
        }

        for col_name, expected_type in expected_types.items():
            if col_name in table.schema.names:
                actual_type = table.schema.field(col_name).type
                if actual_type == expected_type:
                    print(f"✓ {col_name}: {actual_type}")
                else:
                    print(f"✗ {col_name}: expected {expected_type}, got {actual_type}")

        return ds, ds_evolved


def example_3_partition_metadata():
    """Example 3: Working with partition metadata."""
    print("\n=== Example 3: Partition Metadata ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create partitioned dataset
        df = create_complex_sample_data()
        ds = ParquetDataset.from_pandas(
            df,
            path=temp_path / "partitioned_data",
            partition_cols=['category', 'in_stock']
        )

        print("\n3.1 Partition Information:")
        print(f" - Partition columns: {ds.partitioning}")
        print(f" - Number of files: {len(ds.files)}")

        # Analyze partitions
        print("\n3.2 Partition Analysis:")

        # Get partition values
        partitions = {}
        for file_path in ds.files:
            # Extract partition values from path
            parts = file_path.split('/')
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    if key not in partitions:
                        partitions[key] = set()
                    partitions[key].add(value)

        for part_col, values in partitions.items():
            print(f" - {part_col}: {sorted(values)}")

        # Add partition-specific metadata
        print("\n3.3 Adding Partition Metadata:")

        partition_metadata = {
            'category': {
                'Electronics': {'description': 'Electronic devices and accessories', 'priority': 1},
                'Clothing': {'description': 'Apparel and fashion items', 'priority': 2},
                'Books': {'description': 'Books and publications', 'priority': 3},
                'Home': {'description': 'Home and garden products', 'priority': 4},
                'Sports': {'description': 'Sports and outdoor equipment', 'priority': 5}
            },
            'in_stock': {
                'True': {'description': 'Items currently in stock', 'availability': 'immediate'},
                'False': {'description': 'Items out of stock', 'availability': 'backorder'}
            }
        }

        ds.metadata.update({
            'partition_metadata': partition_metadata,
            'partition_strategy': 'hive',
            'partition_stats': {
                'category': len(partitions.get('category', set())),
                'in_stock': len(partitions.get('in_stock', set()))
            }
        })

        ds.save_metadata()

        # Partition pruning example
        print("\n3.4 Partition Pruning Example:")

        # Create filtered dataset
        table = ds.to_table()

        # Filter by partition column (efficient)
        electronics_table = table.filter(table.category == 'Electronics')
        print(f"Electronics products: {len(electronics_table)}")

        in_stock_table = table.filter(table.in_stock == True)
        print(f"In-stock products: {len(in_stock_table)}")

        return ds


def example_4_metadata_collection():
    """Example 4: Efficient metadata collection from large datasets."""
    print("\n=== Example 4: Metadata Collection ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create multiple datasets to simulate a large data environment
        datasets = {}

        for i in range(5):
            df = create_complex_sample_data()
            df['batch_id'] = f"batch_{i}"
            df['processing_date'] = datetime.now(timezone.utc)

            ds = ParquetDataset.from_pandas(
                df,
                path=temp_path / f"batch_{i}",
                partition_cols=['category']
            )

            # Add batch-specific metadata
            ds.metadata.update({
                'batch_id': f"batch_{i}",
                'processing_date': datetime.now(timezone.utc).isoformat(),
                'record_count': len(df),
                'data_quality_score': np.random.uniform(0.8, 0.99),
                'processing_stage': 'raw',
                'source_system': 'ecommerce_platform'
            })

            ds.save_metadata()
            datasets[f"batch_{i}"] = ds

        print("\n4.1 Collecting Metadata from Multiple Datasets:")

        # Collect metadata from all datasets
        collected_metadata = []

        for batch_name, ds in datasets.items():
            metadata_info = {
                'dataset_name': batch_name,
                'path': str(ds.path),
                'record_count': ds.metadata.get('record_count'),
                'processing_date': ds.metadata.get('processing_date'),
                'data_quality_score': ds.metadata.get('data_quality_score'),
                'file_count': len(ds.files),
                'schema_fields': len(ds.schema)
            }
            collected_metadata.append(metadata_info)

        # Create metadata summary
        metadata_df = pd.DataFrame(collected_metadata)
        print("\nMetadata Summary:")
        print(metadata_df.to_string(index=False))

        # Metadata analysis
        print("\n4.2 Metadata Analysis:")
        print(f"Total datasets: {len(datasets)}")
        print(f"Total records: {metadata_df['record_count'].sum()}")
        print(f"Average data quality: {metadata_df['data_quality_score'].mean():.3f}")
        print(f"Total files: {metadata_df['file_count'].sum()}")

        # Create catalog for metadata management
        print("\n4.3 Creating Metadata Catalog:")

        catalog_path = temp_path / "metadata_catalog.yaml"
        catalog = Catalog(catalog_path)

        # Register datasets with metadata
        for batch_name, ds in datasets.items():
            catalog.register_dataset(batch_name, ds)

        # Add catalog-level metadata
        catalog.metadata.update({
            'catalog_description': 'E-commerce batch processing catalog',
            'created_date': datetime.now(timezone.utc).isoformat(),
            'total_datasets': len(datasets),
            'total_records': metadata_df['record_count'].sum(),
            'refresh_frequency': 'daily'
        })

        catalog.save()
        print(f"Catalog saved with {len(catalog.list_datasets())} datasets")

        return catalog, datasets


def example_5_performance_optimization():
    """Example 5: Performance optimization through metadata."""
    print("\n=== Example 5: Performance Optimization ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create dataset with optimization metadata
        df = create_complex_sample_data()

        # Add optimization hints
        optimization_metadata = {
            'access_patterns': {
                'frequent_filters': ['category', 'in_stock', 'price_range'],
                'frequent_columns': ['id', 'name', 'price', 'category'],
                'sort_order': ['created_at', 'id'],
                'query_patterns': ['product_lookup', 'category_browse', 'price_filter']
            },
            'performance_hints': {
                'row_group_size': 100000,
                'compression': 'snappy',
                'encoding': 'dictionary',
                'statistics': True,
                'bloom_filters': ['id', 'name']
            },
            'cache_hints': {
                'cache_frequently_accessed': True,
                'cache_ttl_hours': 24,
                'prefetch_related': ['category', 'in_stock']
            }
        }

        ds = ParquetDataset.from_pandas(
            df,
            path=temp_path / "optimized_dataset",
            partition_cols=['category'],
            row_group_size=100000
        )

        # Add optimization metadata
        ds.metadata.update(optimization_metadata)
        ds.save_metadata()

        print("\n5.1 Optimization Metadata:")
        print(f" - Row group size: {optimization_metadata['performance_hints']['row_group_size']}")
        print(f" - Compression: {optimization_metadata['performance_hints']['compression']}")
        print(f" - Frequent filters: {optimization_metadata['access_patterns']['frequent_filters']}")

        # Demonstrate optimized queries
        print("\n5.2 Demonstrating Optimized Queries:")

        table = ds.to_table()

        # Query using frequent filter pattern
        print("Query 1: Category filter (optimized through partitioning)")
        electronics = table.filter(table.category == 'Electronics')
        print(f" - Result: {len(electronics)} records")

        # Query using frequent columns
        print("\nQuery 2: Selecting frequent columns")
        selected = table.select(['id', 'name', 'price', 'category'])
        print(f" - Selected {len(selected.columns)} columns")

        # Complex query combining multiple optimizations
        print("\nQuery 3: Complex optimized query")
        complex_query = table.filter(
            (table.category == 'Electronics') &
            (table.price > 100) &
            (table.in_stock == True)
        ).select(['id', 'name', 'price', 'rating'])

        print(f" - Complex query result: {len(complex_query)} records")

        # Statistics and indexing
        print("\n5.3 Dataset Statistics:")

        # Get basic statistics
        pdf = table.to_pandas()

        stats = {
            'total_records': len(pdf),
            'unique_categories': pdf['category'].nunique(),
            'price_range': [pdf['price'].min(), pdf['price'].max()],
            'avg_rating': pdf['rating'].mean(),
            'stock_percentage': (pdf['in_stock'].sum() / len(pdf)) * 100
        }

        for key, value in stats.items():
            print(f" - {key}: {value}")

        return ds


def example_6_metadata_driven_processing():
    """Example 6: Metadata-driven data processing."""
    print("\n=== Example 6: Metadata-Driven Processing ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create datasets with different processing requirements
        processing_configs = [
            {
                'name': 'high_priority',
                'priority': 1,
                'validation_level': 'strict',
                'quality_threshold': 0.95,
                'processing_speed': 'fast'
            },
            {
                'name': 'medium_priority',
                'priority': 2,
                'validation_level': 'normal',
                'quality_threshold': 0.85,
                'processing_speed': 'normal'
            },
            {
                'name': 'low_priority',
                'priority': 3,
                'validation_level': 'basic',
                'quality_threshold': 0.70,
                'processing_speed': 'slow'
            }
        ]

        datasets = {}

        for config in processing_configs:
            df = create_complex_sample_data()

            ds = ParquetDataset.from_pandas(
                df,
                path=temp_path / config['name']
            )

            # Add processing metadata
            ds.metadata.update({
                'processing_config': config,
                'data_classification': config['name'],
                'processing_requirements': {
                    'validation_needed': config['validation_level'] != 'basic',
                    'quality_check': True,
                    'compression_required': config['priority'] <= 2,
                    'indexing_required': config['priority'] == 1
                }
            })

            ds.save_metadata()
            datasets[config['name']] = ds

        print("\n6.1 Processing-Based Metadata:")

        # Sort by priority
        sorted_datasets = sorted(datasets.items(),
                               key=lambda x: x[1].metadata.get('processing_config', {}).get('priority', 999))

        for name, ds in sorted_datasets:
            config = ds.metadata.get('processing_config', {})
            print(f"\n{name}:")
            print(f" - Priority: {config.get('priority')}")
            print(f" - Validation: {config.get('validation_level')}")
            print(f" - Quality threshold: {config.get('quality_threshold')}")
            print(f" - Processing speed: {config.get('processing_speed')}")

        # Simulate processing pipeline
        print("\n6.2 Simulating Processing Pipeline:")

        for name, ds in sorted_datasets:
            config = ds.metadata.get('processing_config', {})
            requirements = ds.metadata.get('processing_requirements', {})

            print(f"\nProcessing {name}:")

            # Apply processing based on metadata
            if requirements.get('validation_needed'):
                print("  - Running validation...")

            if requirements.get('quality_check'):
                print("  - Performing quality check...")

            if requirements.get('compression_required'):
                print("  - Applying compression...")

            if requirements.get('indexing_required'):
                print("  - Creating indexes...")

            print(f"  - Processing complete (speed: {config.get('processing_speed')})")

        return datasets


def main():
    """Run all metadata management examples."""
    print("PyDala2 Metadata Management Examples")
    print("=" * 50)

    try:
        # Run all examples
        ds_basic = example_1_basic_metadata_operations()
        ds_schema, ds_evolved = example_2_schema_management()
        ds_partitioned = example_3_partition_metadata()
        catalog, datasets = example_4_metadata_collection()
        ds_optimized = example_5_performance_optimization()
        datasets_processing = example_6_metadata_driven_processing()

        print("\n" + "=" * 50)
        print("All metadata management examples completed successfully!")
        print("\nKey Learnings:")
        print("- Metadata management is crucial for data governance")
        print("- Schema evolution enables flexible data structures")
        print("- Partition metadata improves query performance")
        print("- Metadata collection enables data cataloging")
        print("- Performance optimization through metadata")
        print("- Metadata-driven processing pipelines")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
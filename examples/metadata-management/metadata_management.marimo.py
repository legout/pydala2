import marimo

__generated_with = "0.15.3"
app = marimo.App(width="full")


@app.cell
def __():
    import os
    import tempfile
    import json
    import pandas as pd
    import numpy as np
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
    return (
        Any,
        Catalog,
        Dict,
        List,
        Optional,
        ParquetDataset,
        ParquetDatasetMetadata,
        Path,
        PydalaDatasetMetadata,
        PydalaTable,
        catalog,
        convert_large_types_to_normal,
        datetime,
        ds,
        json,
        np,
        os,
        pa,
        pd,
        pq,
        replace_schema,
        tempfile,
        timezone,
        typing,
    )


@app.cell
def __(np, pd):
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
    return create_complex_sample_data,


@app.cell
def __(create_complex_sample_data):
    # Create sample data
    df = create_complex_sample_data()
    print(f"Created sample data with {len(df)} rows")
    df.head()
    return create_complex_sample_data, df


@app.cell
def __(tempfile, Path, df, ParquetDataset):
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    print(f"Using temporary directory: {temp_dir}")

    # Create dataset
    ds = ParquetDataset.from_pandas(
        df,
        path=temp_path / "basic_metadata",
        partition_cols=['category']
    )
    return ds, temp_dir, temp_path


@app.cell
def __(ds):
    # Access basic metadata
    print("1.1 Dataset Information:")
    print(f" - Number of files: {len(ds.files)}")
    print(f" - Total rows: {len(ds)}")
    print(f" - Schema: {ds.schema}")
    print(f" - Partitioning: {ds.partitioning}")


@app.cell
def __(ds, datetime, timezone):
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


@app.cell
def __(ds):
    # Read and verify metadata
    print("\n1.3 Reading Metadata:")
    for key, value in ds.metadata.items():
        print(f" - {key}: {value}")


@app.cell
def __(ds):
    # Schema metadata
    print("\n1.4 Schema Metadata:")
    for field in ds.schema:
        field_metadata = field.metadata or {}
        print(f" - {field.name}: {field.type}")
        if field_metadata:
            print(f"   Metadata: {dict(field_metadata)}")


@app.cell
def __(temp_path, df, ParquetDataset, np):
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
    return ds_evolved, new_data


@app.cell
def __(ds_evolved):
    print("\nEvolved Schema:")
    for field in ds_evolved.schema:
        print(f" - {field.name}: {field.type}")


@app.cell
def __(ds):
    # Convert to table for schema operations
    table = ds.to_table()

    # Check for specific columns
    required_columns = ['id', 'name', 'category', 'price']
    missing_columns = [col for col in required_columns if col not in table.columns]

    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
    else:
        print("All required columns present")
    return missing_columns, required_columns, table


@app.cell
def __(pa, table):
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
    return actual_type, col_name, expected_type, expected_types


@app.cell
def __(temp_path, df, ParquetDataset):
    # Create partitioned dataset
    print("3.1 Partition Information:")
    ds_partitioned = ParquetDataset.from_pandas(
        df,
        path=temp_path / "partitioned_data",
        partition_cols=['category', 'in_stock']
    )

    print(f" - Partition columns: {ds_partitioned.partitioning}")
    print(f" - Number of files: {len(ds_partitioned.files)}")
    return ds_partitioned,


@app.cell
def __(ds_partitioned):
    # Analyze partitions
    print("\n3.2 Partition Analysis:")

    # Get partition values
    partitions = {}
    for file_path in ds_partitioned.files:
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
    return partitions, values,


@app.cell
def __(ds_partitioned, partitions):
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

    ds_partitioned.metadata.update({
        'partition_metadata': partition_metadata,
        'partition_strategy': 'hive',
        'partition_stats': {
            'category': len(partitions.get('category', set())),
            'in_stock': len(partitions.get('in_stock', set()))
        }
    })

    ds_partitioned.save_metadata()
    return partition_metadata,


@app.cell
def __(ds_partitioned):
    # Partition pruning example
    print("\n3.4 Partition Pruning Example:")

    # Create filtered dataset
    table = ds_partitioned.to_table()

    # Filter by partition column (efficient)
    electronics_table = table.filter(table.category == 'Electronics')
    print(f"Electronics products: {len(electronics_table)}")

    in_stock_table = table.filter(table.in_stock == True)
    print(f"In-stock products: {len(in_stock_table)}")
    return electronics_table, in_stock_table, table


@app.cell
def __(temp_path, create_complex_sample_data, ParquetDataset, datetime, timezone, np):
    # Create multiple datasets to simulate a large data environment
    datasets = {}

    for i in range(3):
        df_batch = create_complex_sample_data()
        df_batch['batch_id'] = f"batch_{i}"
        df_batch['processing_date'] = datetime.now(timezone.utc)

        ds_batch = ParquetDataset.from_pandas(
            df_batch,
            path=temp_path / f"batch_{i}",
            partition_cols=['category']
        )

        # Add batch-specific metadata
        ds_batch.metadata.update({
            'batch_id': f"batch_{i}",
            'processing_date': datetime.now(timezone.utc).isoformat(),
            'record_count': len(df_batch),
            'data_quality_score': np.random.uniform(0.8, 0.99),
            'processing_stage': 'raw',
            'source_system': 'ecommerce_platform'
        })

        ds_batch.save_metadata()
        datasets[f"batch_{i}"] = ds_batch
    return batch_name, datasets, ds_batch, df_batch, i


@app.cell
def __(datasets, pd):
    # Collect metadata from all datasets
    print("4.1 Collecting Metadata from Multiple Datasets:")

    collected_metadata = []

    for batch_name, ds_batch in datasets.items():
        metadata_info = {
            'dataset_name': batch_name,
            'path': str(ds_batch.path),
            'record_count': ds_batch.metadata.get('record_count'),
            'processing_date': ds_batch.metadata.get('processing_date'),
            'data_quality_score': ds_batch.metadata.get('data_quality_score'),
            'file_count': len(ds_batch.files),
            'schema_fields': len(ds_batch.schema)
        }
        collected_metadata.append(metadata_info)

    # Create metadata summary
    metadata_df = pd.DataFrame(collected_metadata)
    print("\nMetadata Summary:")
    metadata_df
    return batch_name, collected_metadata, ds_batch, metadata_df, metadata_info


@app.cell
def __(datasets, metadata_df):
    # Metadata analysis
    print("4.2 Metadata Analysis:")
    print(f"Total datasets: {len(datasets)}")
    print(f"Total records: {metadata_df['record_count'].sum()}")
    print(f"Average data quality: {metadata_df['data_quality_score'].mean():.3f}")
    print(f"Total files: {metadata_df['file_count'].sum()}")


@app.cell
def __(temp_path, datasets, Catalog, datetime, timezone, metadata_df):
    # Create catalog for metadata management
    print("4.3 Creating Metadata Catalog:")

    catalog_path = temp_path / "metadata_catalog.yaml"
    catalog = Catalog(catalog_path)

    # Register datasets with metadata
    for batch_name, ds_batch in datasets.items():
        catalog.register_dataset(batch_name, ds_batch)

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
    return batch_name, catalog, catalog_path, ds_batch


@app.cell
def __(temp_path, df, ParquetDataset):
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

    ds_optimized = ParquetDataset.from_pandas(
        df,
        path=temp_path / "optimized_dataset",
        partition_cols=['category'],
        row_group_size=100000
    )

    # Add optimization metadata
    ds_optimized.metadata.update(optimization_metadata)
    ds_optimized.save_metadata()
    return ds_optimized, optimization_metadata


@app.cell
def __(optimization_metadata):
    print("5.1 Optimization Metadata:")
    print(f" - Row group size: {optimization_metadata['performance_hints']['row_group_size']}")
    print(f" - Compression: {optimization_metadata['performance_hints']['compression']}")
    print(f" - Frequent filters: {optimization_metadata['access_patterns']['frequent_filters']}")


@app.cell
def __(ds_optimized):
    # Demonstrate optimized queries
    print("5.2 Demonstrating Optimized Queries:")

    table = ds_optimized.to_table()

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
    return complex_query, electronics, selected, table


@app.cell
def __(ds_optimized, pd):
    # Statistics and indexing
    print("5.3 Dataset Statistics:")

    # Get basic statistics
    table = ds_optimized.to_table()
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
    return pdf, stats, table, value


@app.cell
def __(temp_path, create_complex_sample_data, ParquetDataset):
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

    datasets_processing = {}

    for config in processing_configs:
        df_proc = create_complex_sample_data()

        ds_proc = ParquetDataset.from_pandas(
            df_proc,
            path=temp_path / config['name']
        )

        # Add processing metadata
        ds_proc.metadata.update({
            'processing_config': config,
            'data_classification': config['name'],
            'processing_requirements': {
                'validation_needed': config['validation_level'] != 'basic',
                'quality_check': True,
                'compression_required': config['priority'] <= 2,
                'indexing_required': config['priority'] == 1
            }
        })

        ds_proc.save_metadata()
        datasets_processing[config['name']] = ds_proc
    return config, datasets_processing, df_proc, ds_proc, processing_configs


@app.cell
def __(datasets_processing):
    print("6.1 Processing-Based Metadata:")

    # Sort by priority
    sorted_datasets = sorted(datasets_processing.items(),
                           key=lambda x: x[1].metadata.get('processing_config', {}).get('priority', 999))

    for name, ds_proc in sorted_datasets:
        config = ds_proc.metadata.get('processing_config', {})
        print(f"\n{name}:")
        print(f" - Priority: {config.get('priority')}")
        print(f" - Validation: {config.get('validation_level')}")
        print(f" - Quality threshold: {config.get('quality_threshold')}")
        print(f" - Processing speed: {config.get('processing_speed')}")
    return config, ds_proc, name, sorted_datasets


@app.cell
def __(sorted_datasets):
    # Simulate processing pipeline
    print("6.2 Simulating Processing Pipeline:")

    for name, ds_proc in sorted_datasets:
        config = ds_proc.metadata.get('processing_config', {})
        requirements = ds_proc.metadata.get('processing_requirements', {})

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
    return config, ds_proc, name, requirements,


@app.cell
def __():
    print("Key Learnings:")
    print("- Metadata management is crucial for data governance")
    print("- Schema evolution enables flexible data structures")
    print("- Partition metadata improves query performance")
    print("- Metadata collection enables data cataloging")
    print("- Performance optimization through metadata")
    print("- Metadata-driven processing pipelines")

    print("\nBest Practices:")
    print("- Always add descriptive metadata to datasets")
    print("- Use partitioning for large datasets")
    print("- Implement schema validation")
    print("- Create metadata catalogs for organization")
    print("- Optimize based on access patterns")
    print("- Automate processing with metadata-driven workflows")


@app.cell
def __(temp_dir):
    import shutil
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    app.run()
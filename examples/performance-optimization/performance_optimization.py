"""
Performance Optimization Example

This example demonstrates performance optimization techniques in PyDala2, including:
- Memory management and efficient data handling
- Query optimization strategies
- Parallel processing and chunking
- Caching strategies
- Large dataset handling
"""

import os
import tempfile
import time
import psutil
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pds
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Generator
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import PyDala2 components
from pydala.dataset import ParquetDataset
from pydala.table import PydalaTable
from pydala.catalog import Catalog


def create_large_dataset(n_records: int = 100000) -> pd.DataFrame:
    """Create a large dataset for performance testing."""
    np.random.seed(42)

    start_date = datetime(2023, 1, 1)
    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Toys', 'Beauty']
    regions = ['North', 'South', 'East', 'West', 'Central']
    segments = ['Premium', 'Standard', 'Basic']

    data = {
        'id': range(1, n_records + 1),
        'timestamp': [start_date + timedelta(seconds=np.random.randint(0, 31536000)) for _ in range(n_records)],
        'category': np.random.choice(categories, n_records),
        'region': np.random.choice(regions, n_records),
        'customer_segment': np.random.choice(segments, n_records),
        'value': np.random.uniform(1, 1000, n_records).round(2),
        'quantity': np.random.randint(1, 100, n_records),
        'price': np.random.uniform(10, 500, n_records).round(2),
        'discount': np.random.uniform(0, 0.5, n_records).round(3),
        'is_active': np.random.choice([True, False], n_records, p=[0.8, 0.2]),
        'priority': np.random.randint(1, 6, n_records),
        'score': np.random.uniform(0, 1, n_records).round(3),
        'metadata_size': np.random.randint(100, 10000, n_records)
    }

    df = pd.DataFrame(data)
    df['total_value'] = df['value'] * df['quantity'] * (1 - df['discount'])
    df['efficiency'] = df['total_value'] / df['price']
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year

    return df


def measure_memory_usage() -> Dict[str, float]:
    """Measure current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }


def example_1_memory_optimization():
    """Example 1: Memory optimization techniques."""
    print("\n=== Example 1: Memory Optimization ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        print("\n1.1 Baseline Memory Usage:")
        baseline_memory = measure_memory_usage()
        print(f"Baseline RSS: {baseline_memory['rss_mb']:.2f} MB")

        # Create dataset
        df = create_large_dataset(50000)
        ds = ParquetDataset.from_pandas(
            df,
            path=temp_path / "large_dataset",
            partition_cols=['year', 'month'],
            row_group_size=10000
        )

        print(f"\n1.2 Dataset Created:")
        print(f" - Records: {len(df)}")
        print(f" - Columns: {len(df.columns)}")
        print(f" - File count: {len(ds.files)}")

        memory_after_create = measure_memory_usage()
        print(f"Memory after creation: {memory_after_create['rss_mb']:.2f} MB")

        print("\n1.3 Column Pruning:")

        # Select only needed columns
        start_time = time.time()
        table = ds.to_table()
        selected_columns = table.select(['id', 'category', 'value', 'total_value', 'region'])
        end_time = time.time()

        memory_pruned = measure_memory_usage()
        print(f"Column selection time: {end_time - start_time:.3f} seconds")
        print(f"Memory after pruning: {memory_pruned['rss_mb']:.2f} MB")
        print(f"Selected {len(selected_columns.columns)} columns from {len(table.columns)}")

        print("\n1.4 Efficient Data Types:")

        # Convert to more efficient data types
        df_optimized = df.copy()

        # Convert to categorical where appropriate
        df_optimized['category'] = df_optimized['category'].astype('category')
        df_optimized['region'] = df_optimized['region'].astype('category')
        df_optimized['customer_segment'] = df_optimized['customer_segment'].astype('category')

        # Downcast numeric types
        df_optimized['value'] = pd.to_numeric(df_optimized['value'], downcast='float')
        df_optimized['quantity'] = pd.to_numeric(df_optimized['quantity'], downcast='integer')
        df_optimized['priority'] = pd.to_numeric(df_optimized['priority'], downcast='integer')

        memory_optimized = measure_memory_usage()
        print(f"Memory after type optimization: {memory_optimized['rss_mb']:.2f} MB")

        print(f"\n1.5 Memory Savings:")
        print(f" - Original: {memory_after_create['rss_mb']:.2f} MB")
        print(f" - Optimized: {memory_optimized['rss_mb']:.2f} MB")
        print(f" - Savings: {((memory_after_create['rss_mb'] - memory_optimized['rss_mb']) / memory_after_create['rss_mb'] * 100):.1f}%")

        return ds, table


def example_2_query_optimization():
    """Example 2: Query optimization strategies."""
    print("\n=== Example 2: Query Optimization ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create optimized dataset
        df = create_large_dataset(30000)
        ds = ParquetDataset.from_pandas(
            df,
            path=temp_path / "optimized_dataset",
            partition_cols=['region', 'year'],
            row_group_size=5000
        )

        table = ds.to_table()

        print("\n2.1 Partition Pruning:")

        # Test partition-based filtering
        start_time = time.time()
        north_region = table.filter(table.region == 'North')
        end_time = time.time()
        partition_time = end_time - start_time
        print(f"Partition filter time: {partition_time:.3f} seconds")
        print(f"North region records: {len(north_region)}")

        # Test non-partition filter
        start_time = time.time()
        high_value = table.filter(table.value > 500)
        end_time = time.time()
        non_partition_time = end_time - start_time
        print(f"Non-partition filter time: {non_partition_time:.3f} seconds")
        print(f"High value records: {len(high_value)}")

        print(f"\nPartition pruning speedup: {non_partition_time / partition_time:.1f}x")

        print("\n2.2 Predicate Pushdown:")

        # Test combined filters
        start_time = time.time()
        complex_filter = table.filter(
            (table.region == 'North') &
            (table.customer_segment == 'Premium') &
            (table.value > 200) &
            (table.is_active == True)
        )
        end_time = time.time()
        complex_time = end_time - start_time
        print(f"Complex filter time: {complex_time:.3f} seconds")
        print(f"Complex filter results: {len(complex_filter)} records")

        print("\n2.3 Efficient Scanning:")

        # Use PyArrow scanner
        start_time = time.time()
        scanner = table.to_arrow_scanner(
            columns=['id', 'category', 'value', 'region', 'total_value'],
            filter=pa.dataset.field('value') > 100
        )

        # Process in batches
        total_rows = 0
        batch_count = 0
        for batch in scanner.to_batches():
            total_rows += batch.num_rows
            batch_count += 1
            if batch_count >= 10:  # Limit for demo
                break

        end_time = time.time()
        scanner_time = end_time - start_time
        print(f"Scanner processing time: {scanner_time:.3f} seconds")
        print(f"Processed {batch_count} batches with {total_rows} rows")

        print("\n2.4 Projection Pushdown:")

        # Test column selection performance
        start_time = time.time()
        few_columns = table.select(['id', 'value', 'region'])
        end_time = time.time()
        projection_time = end_time - start_time

        start_time = time.time()
        many_columns = table.select(table.columns)  # All columns
        end_time = time.time()
        all_columns_time = end_time - start_time

        print(f"Few columns time: {projection_time:.3f} seconds")
        print(f"All columns time: {all_columns_time:.3f} seconds")
        print(f"Projection speedup: {all_columns_time / projection_time:.1f}x")

        return ds, table


def process_chunk(chunk: pd.DataFrame, filter_conditions: Dict[str, Any]) -> pd.DataFrame:
    """Process a chunk of data with given filter conditions."""
    result = chunk.copy()

    for column, condition in filter_conditions.items():
        if isinstance(condition, dict):
            if 'min' in condition:
                result = result[result[column] >= condition['min']]
            if 'max' in condition:
                result = result[result[column] <= condition['max']]
            if 'equals' in condition:
                result = result[result[column] == condition['equals']]
        elif isinstance(condition, list):
            result = result[result[column].isin(condition)]

    return result


def example_3_parallel_processing():
    """Example 3: Parallel processing and chunking."""
    print("\n=== Example 3: Parallel Processing ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create dataset
        df = create_large_dataset(50000)
        ds = ParquetDataset.from_pandas(df, path=temp_path / "parallel_dataset")

        print(f"\n3.1 Dataset Info:")
        print(f" - Total records: {len(df)}")
        print(f" - CPU cores available: {mp.cpu_count()}")

        print("\n3.2 Sequential Processing:")

        # Sequential processing
        start_time = time.time()
        sequential_results = []
        chunk_size = 10000

        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            filtered_chunk = process_chunk(chunk, {'value': {'min': 100}, 'region': ['North', 'South']})
            sequential_results.append(filtered_chunk)

        sequential_df = pd.concat(sequential_results)
        sequential_time = time.time() - start_time
        print(f"Sequential processing time: {sequential_time:.3f} seconds")
        print(f"Sequential results: {len(sequential_df)} records")

        print("\n3.3 Parallel Processing (Thread Pool):")

        # Parallel processing with threads
        start_time = time.time()
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

        with ThreadPoolExecutor(max_workers=4) as executor:
            parallel_results = list(executor.map(
                lambda chunk: process_chunk(chunk, {'value': {'min': 100}, 'region': ['North', 'South']}),
                chunks
            ))

        parallel_df = pd.concat(parallel_results)
        parallel_time = time.time() - start_time
        print(f"Parallel processing time: {parallel_time:.3f} seconds")
        print(f"Parallel results: {len(parallel_df)} records")
        print(f"Speedup: {sequential_time / parallel_time:.1f}x")

        print("\n3.4 Batch Processing with PyArrow:")

        # Use PyArrow batch processing
        table = ds.to_table()
        start_time = time.time()

        scanner = table.to_arrow_scanner(
            filter=(pa.dataset.field('value') > 100) &
                  (pa.dataset.field('region').isin(['North', 'South']))
        )

        batch_results = []
        for batch in scanner.to_batches():
            # Process batch (convert to pandas for demonstration)
            batch_df = batch.to_pandas()
            batch_results.append(batch_df)

        if batch_results:
            batch_df = pd.concat(batch_results)
            batch_time = time.time() - start_time
            print(f"Batch processing time: {batch_time:.3f} seconds")
            print(f"Batch processing results: {len(batch_df)} records")

        return ds, sequential_df, parallel_df


def example_4_caching_strategies():
    """Example 4: Caching strategies."""
    print("\n=== Example 4: Caching Strategies ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create dataset
        df = create_large_dataset(20000)
        ds = ParquetDataset.from_pandas(df, path=temp_path / "cache_dataset")

        print(f"\n4.1 Dataset Info:")
        print(f" - Records: {len(df)}")
        print(f" - Memory usage: {measure_memory_usage()['rss_mb']:.2f} MB")

        print("\n4.2 Query Without Caching:")

        # Multiple queries without caching
        queries = [
            lambda t: t.filter(t.region == 'North'),
            lambda t: t.filter(t.value > 500),
            lambda t: t.filter(t.customer_segment == 'Premium'),
            lambda t: t.filter((t.region == 'North') & (t.value > 500)),
        ]

        times_no_cache = []
        for i, query in enumerate(queries):
            start_time = time.time()
            table = ds.to_table()  # Reload table each time
            result = query(table)
            end_time = time.time()
            times_no_cache.append(end_time - start_time)
            print(f"Query {i+1}: {end_time - start_time:.3f} seconds, {len(result)} records")

        print(f"\nAverage time without caching: {np.mean(times_no_cache):.3f} seconds")

        print("\n4.3 Query With Caching:")

        # Cache the table in memory
        cached_table = ds.to_table()
        times_with_cache = []

        for i, query in enumerate(queries):
            start_time = time.time()
            result = query(cached_table)
            end_time = time.time()
            times_with_cache.append(end_time - start_time)
            print(f"Query {i+1}: {end_time - start_time:.3f} seconds, {len(result)} records")

        print(f"\nAverage time with caching: {np.mean(times_with_cache):.3f} seconds")
        print(f"Cache speedup: {np.mean(times_no_cache) / np.mean(times_with_cache):.1f}x")

        print("\n4.4 Selective Caching:")

        # Cache frequently accessed subsets
        north_data = cached_table.filter(cached_table.region == 'North')
        premium_data = cached_table.filter(cached_table.customer_segment == 'Premium')

        print(f"Cached North data: {len(north_data)} records")
        print(f"Cached Premium data: {len(premium_data)} records")

        # Query cached subsets
        start_time = time.time()
        north_premium = north_data.filter(north_data.customer_segment == 'Premium')
        end_time = time.time()
        print(f"Cached subset query time: {end_time - start_time:.3f} seconds")

        # Compare with full table query
        start_time = time.time()
        full_table_result = cached_table.filter(
            (cached_table.region == 'North') &
            (cached_table.customer_segment == 'Premium')
        )
        end_time = time.time()
        print(f"Full table query time: {end_time - start_time:.3f} seconds")

        return ds, cached_table


def example_5_large_dataset_handling():
    """Example 5: Large dataset handling strategies."""
    print("\n=== Example 5: Large Dataset Handling ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create multiple smaller datasets to simulate large data
        datasets = []
        for i in range(5):
            df = create_large_dataset(10000)
            df['batch_id'] = f"batch_{i}"
            ds = ParquetDataset.from_pandas(
                df,
                path=temp_path / f"batch_{i}",
                partition_cols=['region']
            )
            datasets.append(ds)

        print(f"\n5.1 Created {len(datasets)} datasets")
        total_records = sum(len(ds) for ds in datasets)
        print(f"Total records across all datasets: {total_records}")

        print("\n5.2 Streaming Processing:")

        # Process datasets in streaming fashion
        start_time = time.time()
        all_results = []

        for i, ds in enumerate(datasets):
            table = ds.to_table()
            # Process only high-value items
            high_value = table.filter(table.value > 300)
            all_results.append(high_value)

            if i % 2 == 0:  # Progress indicator
                print(f"Processed batch {i+1}/{len(datasets)}")

        final_result = pa.concat_tables(all_results)
        streaming_time = time.time() - start_time
        print(f"Streaming processing time: {streaming_time:.3f} seconds")
        print(f"Final result: {len(final_result)} records")

        print("\n5.3 Memory-Efficient Aggregation:")

        # Aggregate results incrementally
        start_time = time.time()
        aggregated_results = []

        for ds in datasets:
            table = ds.to_table()
            df_chunk = table.to_pandas()

            # Aggregate this chunk
            chunk_agg = df_chunk.groupby(['region', 'customer_segment']).agg({
                'value': ['sum', 'mean', 'count'],
                'quantity': 'sum'
            }).round(2)

            aggregated_results.append(chunk_agg)

        # Combine results
        final_agg = pd.concat(aggregated_results).groupby(['region', 'customer_segment']).sum()
        aggregation_time = time.time() - start_time
        print(f"Memory-efficient aggregation time: {aggregation_time:.3f} seconds")

        print("\n5.4 Lazy Evaluation:")

        # Use PyArrow's lazy evaluation
        start_time = time.time()
        lazy_results = []

        for ds in datasets:
            # Create scanner without immediate execution
            scanner = ds.to_arrow_scanner(
                columns=['region', 'customer_segment', 'value', 'quantity'],
                filter=pa.dataset.field('value') > 200
            )

            # Execute lazily
            for batch in scanner.to_batches():
                # Process batch
                lazy_results.append(batch)

        if lazy_results:
            combined_result = pa.Table.from_batches(lazy_results)
            lazy_time = time.time() - start_time
            print(f"Lazy evaluation time: {lazy_time:.3f} seconds")
            print(f"Lazy evaluation results: {combined_result.num_rows} records")

        return datasets, final_result


def example_6_performance_monitoring():
    """Example 6: Performance monitoring and optimization."""
    print("\n=== Example 6: Performance Monitoring ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create dataset
        df = create_large_dataset(30000)
        ds = ParquetDataset.from_pandas(
            df,
            path=temp_path / "monitoring_dataset",
            partition_cols=['region', 'year'],
            row_group_size=5000
        )

        print(f"\n6.1 Performance Monitoring Setup:")
        print(f" - Dataset size: {len(df)} records")
        print(f" - Partitions: {len(ds.files)} files")

        print("\n6.2 Query Performance Analysis:")

        # Define test queries
        test_queries = [
            ("Simple Filter", lambda t: t.filter(t.value > 100)),
            ("Complex Filter", lambda t: t.filter((t.value > 200) & (t.region == 'North') & (t.customer_segment == 'Premium'))),
            ("Aggregation", lambda t: t.to_pandas().groupby('region')['value'].mean()),
            ("Column Selection", lambda t: t.select(['id', 'value', 'region'])),
            ("Join Simulation", lambda t: t.filter(t.value > 150)),  # Simulate join operation
        ]

        performance_results = []

        for query_name, query_func in test_queries:
            # Measure memory before
            mem_before = measure_memory_usage()

            # Execute query
            start_time = time.time()
            table = ds.to_table()
            result = query_func(table)
            end_time = time.time()

            # Measure memory after
            mem_after = measure_memory_usage()

            query_time = end_time - start_time
            memory_delta = mem_after['rss_mb'] - mem_before['rss_mb']

            performance_results.append({
                'query': query_name,
                'time_seconds': query_time,
                'memory_delta_mb': memory_delta,
                'result_size': len(result) if hasattr(result, '__len__') else 'N/A'
            })

            print(f"{query_name}:")
            print(f"  Time: {query_time:.3f} seconds")
            print(f"  Memory delta: {memory_delta:+.2f} MB")
            print(f"  Result size: {len(result) if hasattr(result, '__len__') else 'N/A'}")

        print("\n6.3 Performance Summary:")
        perf_df = pd.DataFrame(performance_results)
        print(perf_df[['query', 'time_seconds', 'memory_delta_mb']].to_string(index=False))

        print("\n6.4 Optimization Recommendations:")

        # Analyze performance and provide recommendations
        avg_time = perf_df['time_seconds'].mean()
        avg_memory = perf_df['memory_delta_mb'].abs().mean()

        recommendations = []

        if avg_time > 1.0:
            recommendations.append("- Consider partitioning for better query performance")
            recommendations.append("- Use column pruning to reduce data transfer")

        if avg_memory > 50:
            recommendations.append("- Implement memory-efficient data types")
            recommendations.append("- Use streaming processing for large datasets")

        if perf_df['time_seconds'].max() > 2.0:
            recommendations.append("- Add caching for frequently accessed data")
            recommendations.append("- Consider data pre-aggregation")

        recommendations.append("- Monitor query performance regularly")
        recommendations.append("- Use appropriate indexing strategies")

        for rec in recommendations:
            print(f"  {rec}")

        return ds, performance_results


def main():
    """Run all performance optimization examples."""
    print("PyDala2 Performance Optimization Examples")
    print("=" * 50)

    try:
        # Run all examples
        ds1, table1 = example_1_memory_optimization()
        ds2, table2 = example_2_query_optimization()
        ds3, seq_df, par_df = example_3_parallel_processing()
        ds4, cached_table = example_4_caching_strategies()
        datasets, result = example_5_large_dataset_handling()
        ds6, perf_results = example_6_performance_monitoring()

        print("\n" + "=" * 50)
        print("All performance optimization examples completed successfully!")
        print("\nKey Learnings:")
        print("- Memory optimization through column pruning and efficient data types")
        print("- Query optimization with partition pruning and predicate pushdown")
        print("- Parallel processing for improved throughput")
        print("- Caching strategies for frequently accessed data")
        print("- Large dataset handling with streaming and lazy evaluation")
        print("- Performance monitoring and optimization")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
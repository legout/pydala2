"""
Advanced Querying Example

This example demonstrates advanced querying capabilities in PyDala2, including:
- Complex filtering and predicates
- SQL integration with DuckDB
- Window functions and aggregations
- Multi-dataset queries
- Performance optimization techniques
"""

import os
import tempfile
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pds
import duckdb
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

# Import PyDala2 components
from pydala.dataset import ParquetDataset
from pydala.table import PydalaTable
from pydala.catalog import Catalog
from pydala.helpers.sql import sql2pyarrow_filter


def create_complex_sales_data():
    """Create complex sales data for advanced querying examples."""
    np.random.seed(42)

    # Generate realistic sales data
    n_records = 10000
    start_date = datetime(2023, 1, 1)

    data = {
        'sale_id': range(1, n_records + 1),
        'customer_id': np.random.randint(1, 1001, n_records),
        'product_id': np.random.randint(1, 501, n_records),
        'store_id': np.random.randint(1, 51, n_records),
        'sale_date': [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_records)],
        'sale_amount': np.random.uniform(10, 1000, n_records).round(2),
        'quantity': np.random.randint(1, 10, n_records),
        'discount_percent': np.random.uniform(0, 30, n_records).round(1),
        'payment_method': np.random.choice(['Credit Card', 'Cash', 'Debit Card', 'Digital Wallet'], n_records),
        'sales_rep_id': np.random.randint(1, 101, n_records),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_records),
        'is_online': np.random.choice([True, False], n_records, p=[0.3, 0.7]),
        'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], n_records, p=[0.2, 0.5, 0.3]),
        'return_flag': np.random.choice([0, 1], n_records, p=[0.95, 0.05])  # 5% return rate
    }

    df = pd.DataFrame(data)

    # Calculate derived fields
    df['net_amount'] = df['sale_amount'] * (1 - df['discount_percent'] / 100) * df['quantity']
    df['month'] = df['sale_date'].dt.month
    df['quarter'] = df['sale_date'].dt.quarter
    df['year'] = df['sale_date'].dt.year
    df['day_of_week'] = df['sale_date'].dt.dayofweek

    return df


def create_product_dimension_data():
    """Create product dimension data."""
    np.random.seed(42)

    categories = ['Electronics', 'Clothing', 'Home', 'Sports', 'Books', 'Toys', 'Beauty', 'Food']

    data = {
        'product_id': range(1, 501),
        'product_name': [f'Product_{i}' for i in range(1, 501)],
        'category': np.random.choice(categories, 500),
        'base_price': np.random.uniform(5, 500, 500).round(2),
        'cost': np.random.uniform(2, 250, 500).round(2),
        'supplier_id': np.random.randint(1, 51, 500),
        'launch_date': [datetime(2022, 1, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(500)],
        'is_active': np.random.choice([True, False], 500, p=[0.9, 0.1])
    }

    return pd.DataFrame(data)


def create_customer_dimension_data():
    """Create customer dimension data."""
    np.random.seed(42)

    data = {
        'customer_id': range(1, 1001),
        'customer_name': [f'Customer_{i}' for i in range(1, 1001)],
        'age': np.random.randint(18, 80, 1000),
        'gender': np.random.choice(['M', 'F'], 1000),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 1000),
        'state': np.random.choice(['NY', 'CA', 'IL', 'TX', 'AZ'], 1000),
        'registration_date': [datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1095)) for _ in range(1000)],
        'total_purchases': np.random.randint(0, 100, 1000),
        'avg_order_value': np.random.uniform(25, 500, 1000).round(2)
    }

    return pd.DataFrame(data)


def example_1_complex_filtering():
    """Example 1: Complex filtering and predicates."""
    print("\n=== Example 1: Complex Filtering ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create dataset
        df_sales = create_complex_sales_data()
        ds_sales = ParquetDataset.from_pandas(
            df_sales,
            path=temp_path / "sales_data",
            partition_cols=['region', 'year', 'quarter']
        )

        # Convert to table for querying
        table = ds_sales.to_table()

        print("\n1.1 Basic Filtering:")
        # Simple filters
        high_value = table.filter(table.sale_amount > 500)
        print(f"Sales over $500: {len(high_value)} records")

        online_sales = table.filter(table.is_online == True)
        print(f"Online sales: {len(online_sales)} records")

        print("\n1.2 Compound Conditions:")
        # Multiple conditions
        complex_filter = table.filter(
            (table.sale_amount > 200) &
            (table.region == 'North') &
            (table.customer_segment == 'Premium') &
            (table.return_flag == 0)
        )
        print(f"Complex filter result: {len(complex_filter)} records")

        print("\n1.3 Date Range Filtering:")
        # Date filtering
        date_filter = table.filter(
            (table.sale_date >= '2023-06-01') &
            (table.sale_date <= '2023-08-31')
        )
        print(f"Q3 2023 sales: {len(date_filter)} records")

        print("\n1.4 String Operations:")
        # String-based filtering
        credit_card_sales = table.filter(table.payment_method == 'Credit Card')
        print(f"Credit card sales: {len(credit_card_sales)} records")

        print("\n1.5 Numeric Range Filtering:")
        # Numeric ranges
        mid_range_sales = table.filter(
            (table.sale_amount >= 100) &
            (table.sale_amount <= 300)
        )
        print(f"Sales between $100-$300: {len(mid_range_sales)} records")

        return table, ds_sales


def example_2_sql_integration():
    """Example 2: SQL integration with DuckDB."""
    print("\n=== Example 2: SQL Integration ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create datasets
        df_sales = create_complex_sales_data()
        df_products = create_product_dimension_data()
        df_customers = create_customer_dimension_data()

        ds_sales = ParquetDataset.from_pandas(df_sales, path=temp_path / "sales")
        ds_products = ParquetDataset.from_pandas(df_products, path=temp_path / "products")
        ds_customers = ParquetDataset.from_pandas(df_customers, path=temp_path / "customers")

        print("\n2.1 Basic SQL Queries:")

        # Connect to DuckDB
        con = duckdb.connect()

        # Register datasets as DuckDB tables
        con.register('sales', df_sales)
        con.register('products', df_products)
        con.register('customers', df_customers)

        # Basic aggregation
        result = con.execute("""
            SELECT
                region,
                COUNT(*) as total_sales,
                SUM(sale_amount) as total_revenue,
                AVG(sale_amount) as avg_sale_amount
            FROM sales
            GROUP BY region
            ORDER BY total_revenue DESC
        """).fetchdf()

        print("Sales by region:")
        print(result)

        print("\n2.2 Complex SQL with Joins:")

        # Join with dimensions
        join_result = con.execute("""
            SELECT
                p.category,
                c.city,
                COUNT(*) as sales_count,
                SUM(s.sale_amount) as total_revenue,
                AVG(s.sale_amount) as avg_sale
            FROM sales s
            JOIN products p ON s.product_id = p.product_id
            JOIN customers c ON s.customer_id = c.customer_id
            WHERE s.sale_date >= '2023-07-01'
            GROUP BY p.category, c.city
            HAVING COUNT(*) > 10
            ORDER BY total_revenue DESC
            LIMIT 10
        """).fetchdf()

        print("Top category-city combinations (July+ 2023):")
        print(join_result)

        print("\n2.3 Window Functions:")

        # Window functions
        window_result = con.execute("""
            SELECT
                customer_id,
                COUNT(*) as purchase_count,
                SUM(sale_amount) as total_spent,
                AVG(sale_amount) as avg_purchase,
                RANK() OVER (ORDER BY SUM(sale_amount) DESC) as customer_rank
            FROM sales
            GROUP BY customer_id
            ORDER BY total_spent DESC
            LIMIT 10
        """).fetchdf()

        print("Top 10 customers by total spend:")
        print(window_result)

        print("\n2.4 Time Series Analysis:")

        # Time-based analysis
        time_result = con.execute("""
            SELECT
                DATE_TRUNC('month', sale_date) as month,
                COUNT(*) as sales_count,
                SUM(sale_amount) as monthly_revenue,
                AVG(sale_amount) as avg_sale_amount,
                COUNT(DISTINCT customer_id) as unique_customers
            FROM sales
            GROUP BY DATE_TRUNC('month', sale_date)
            ORDER BY month
        """).fetchdf()

        print("Monthly sales summary:")
        print(time_result)

        # Clean up
        con.close()

        return result, join_result, window_result, time_result


def example_3_advanced_aggregations():
    """Example 3: Advanced aggregations and analytics."""
    print("\n=== Example 3: Advanced Aggregations ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create dataset
        df_sales = create_complex_sales_data()
        ds_sales = ParquetDataset.from_pandas(df_sales, path=temp_path / "sales")
        table = ds_sales.to_table()

        print("\n3.1 Multi-level Grouping:")

        # Convert to pandas for complex aggregations
        df = table.to_pandas()

        # Multi-level grouping
        multi_level = df.groupby(['region', 'customer_segment', 'payment_method']).agg({
            'sale_amount': ['sum', 'mean', 'count'],
            'quantity': ['sum', 'mean'],
            'customer_id': 'nunique'
        }).round(2)

        print("Multi-level aggregation:")
        print(multi_level.head(10))

        print("\n3.2 Pivot Tables:")

        # Pivot table
        pivot_result = df.pivot_table(
            values='sale_amount',
            index='region',
            columns='customer_segment',
            aggfunc='sum',
            fill_value=0
        ).round(2)

        print("Regional sales by customer segment:")
        print(pivot_result)

        print("\n3.3 Rolling Aggregations:")

        # Time-based rolling calculations
        df_sorted = df.sort_values('sale_date')
        df_sorted['rolling_avg_7d'] = df_sorted['sale_amount'].rolling(window=7, min_periods=1).mean()
        df_sorted['rolling_sum_30d'] = df_sorted['sale_amount'].rolling(window=30, min_periods=1).sum()

        rolling_result = df_sorted.groupby(df_sorted['sale_date'].dt.to_period('M')).agg({
            'sale_amount': 'sum',
            'rolling_avg_7d': 'mean',
            'rolling_sum_30d': 'mean'
        }).round(2)

        print("Monthly rolling averages:")
        print(rolling_result.head())

        print("\n3.4 Percentile Analysis:")

        # Percentile calculations
        percentile_result = df.groupby('region')['sale_amount'].agg([
            'count', 'min', 'max', 'mean', 'median',
            ('p25', lambda x: x.quantile(0.25)),
            ('p75', lambda x: x.quantile(0.75)),
            ('p90', lambda x: x.quantile(0.90)),
            ('p95', lambda x: x.quantile(0.95))
        ]).round(2)

        print("Regional sales percentiles:")
        print(percentile_result)

        return multi_level, pivot_result, rolling_result, percentile_result


def example_4_performance_optimization():
    """Example 4: Performance optimization techniques."""
    print("\n=== Example 4: Performance Optimization ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create larger dataset for performance testing
        df_sales = create_complex_sales_data()

        # Create dataset with optimization hints
        ds_sales = ParquetDataset.from_pandas(
            df_sales,
            path=temp_path / "optimized_sales",
            partition_cols=['region', 'year', 'quarter'],
            row_group_size=10000
        )

        table = ds_sales.to_table()

        print("\n4.1 Column Pruning:")

        # Select only needed columns
        print("Selecting specific columns:")
        start_time = datetime.now()
        selected = table.select(['customer_id', 'sale_amount', 'region', 'customer_segment'])
        end_time = datetime.now()
        print(f"Column selection time: {(end_time - start_time).total_seconds():.3f} seconds")
        print(f"Selected {len(selected.columns)} columns from {len(table.columns)}")

        print("\n4.2 Partition Pruning:")

        # Filter on partition columns
        print("Partition-based filtering:")
        start_time = datetime.now()
        north_region = table.filter(table.region == 'North')
        end_time = datetime.now()
        print(f"Partition filter time: {(end_time - start_time).total_seconds():.3f} seconds")
        print(f"North region records: {len(north_region)}")

        print("\n4.3 Efficient Scanning:")

        # Use PyArrow scanner for efficient reading
        print("Using PyArrow scanner:")
        start_time = datetime.now()
        scanner = table.to_arrow_scanner(
            columns=['customer_id', 'sale_amount', 'region'],
            filter=pa.dataset.field('sale_amount') > 100
        )

        # Read first batch
        first_batch = next(scanner.to_batches())
        end_time = datetime.now()
        print(f"Scanner setup time: {(end_time - start_time).total_seconds():.3f} seconds")
        print(f"First batch size: {first_batch.num_rows}")

        print("\n4.4 Batch Processing:")

        # Process data in batches
        print("Batch processing:")
        batch_size = 1000
        total_rows = 0
        batch_count = 0

        start_time = datetime.now()
        for batch in scanner.to_batches():
            batch_count += 1
            total_rows += batch.num_rows
            if batch_count >= 5:  # Process first 5 batches
                break

        end_time = datetime.now()
        print(f"Processed {batch_count} batches with {total_rows} total rows")
        print(f"Batch processing time: {(end_time - start_time).total_seconds():.3f} seconds")

        print("\n4.5 Predicate Pushdown:")

        # Combine multiple filters efficiently
        print("Combined filtering:")
        start_time = datetime.now()
        complex_filter = table.filter(
            (table.region == 'North') &
            (table.sale_amount > 200) &
            (table.customer_segment == 'Premium') &
            (table.is_online == True)
        )
        end_time = datetime.now()
        print(f"Complex filter time: {(end_time - start_time).total_seconds():.3f} seconds")
        print(f"Complex filter result: {len(complex_filter)} records")

        return ds_sales, table


def example_5_multi_dataset_queries():
    """Example 5: Querying across multiple datasets."""
    print("\n=== Example 5: Multi-Dataset Queries ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create multiple related datasets
        df_sales = create_complex_sales_data()
        df_products = create_product_dimension_data()
        df_customers = create_customer_dimension_data()

        # Create separate datasets
        ds_sales = ParquetDataset.from_pandas(df_sales, path=temp_path / "sales")
        ds_products = ParquetDataset.from_pandas(df_products, path=temp_path / "products")
        ds_customers = ParquetDataset.from_pandas(df_customers, path=temp_path / "customers")

        # Create catalog
        catalog_path = temp_path / "catalog.yaml"
        catalog = Catalog(catalog_path)

        # Register datasets
        catalog.register_dataset("sales", ds_sales)
        catalog.register_dataset("products", ds_products)
        catalog.register_dataset("customers", ds_customers)

        print("\n5.1 Cross-Dataset Analysis:")

        # Load datasets for analysis
        sales_table = ds_sales.to_table()
        products_table = ds_products.to_table()
        customers_table = ds_customers.to_table()

        # Convert to pandas for joining
        sales_df = sales_table.to_pandas()
        products_df = products_table.to_pandas()
        customers_df = customers_table.to_pandas()

        print("\n5.2 Manual Joins:")

        # Join sales with products
        sales_products = sales_df.merge(
            products_df[['product_id', 'category', 'base_price']],
            on='product_id',
            how='left'
        )

        # Join with customers
        full_data = sales_products.merge(
            customers_df[['customer_id', 'age', 'city', 'customer_segment']],
            on='customer_id',
            how='left'
        )

        print(f"Joined dataset shape: {full_data.shape}")
        print("Sample of joined data:")
        print(full_data[['sale_amount', 'category', 'age', 'city']].head())

        print("\n5.3 Cross-Dataset Filtering:")

        # Example: High-value electronics purchases by premium customers
        high_value_electronics = full_data[
            (full_data['category'] == 'Electronics') &
            (full_data['customer_segment'] == 'Premium') &
            (full_data['sale_amount'] > 500)
        ]

        print(f"High-value electronics by premium customers: {len(high_value_electronics)} records")
        print(f"Average sale amount: ${high_value_electronics['sale_amount'].mean():.2f}")

        print("\n5.4 Aggregated Analysis:")

        # Multi-dataset aggregation
        category_segment_analysis = full_data.groupby(['category', 'customer_segment']).agg({
            'sale_amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique',
            'age': 'mean'
        }).round(2)

        print("Category-Customer Segment Analysis:")
        print(category_segment_analysis.head(10))

        print("\n5.5 Time-based Cross-Dataset Analysis:")

        # Add time-based analysis
        full_data['month'] = full_data['sale_date'].dt.to_period('M')
        monthly_trends = full_data.groupby(['month', 'category']).agg({
            'sale_amount': 'sum',
            'customer_id': 'nunique'
        }).round(2)

        print("Monthly category trends:")
        print(monthly_trends.head(10))

        return catalog, full_data


def example_6_advanced_analytics():
    """Example 6: Advanced analytics and insights."""
    print("\n=== Example 6: Advanced Analytics ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create dataset
        df_sales = create_complex_sales_data()
        ds_sales = ParquetDataset.from_pandas(df_sales, path=temp_path / "sales")
        table = ds_sales.to_table()
        df = table.to_pandas()

        print("\n6.1 Customer Analytics:")

        # Customer lifetime value calculation
        customer_ltv = df.groupby('customer_id').agg({
            'sale_amount': 'sum',
            'quantity': 'sum',
            'sale_date': ['min', 'max', 'count']
        }).round(2)

        # Flatten multi-level columns
        customer_ltv.columns = ['total_spent', 'total_quantity', 'first_purchase', 'last_purchase', 'purchase_count']
        customer_ltv['customer_lifetime_days'] = (customer_ltv['last_purchase'] - customer_ltv['first_purchase']).dt.days
        customer_ltv['avg_purchase_value'] = customer_ltv['total_spent'] / customer_ltv['purchase_count']

        print("Customer LTV summary:")
        print(customer_ltv.describe())

        print("\n6.2 Product Performance:")

        # Product performance analysis
        product_performance = df.groupby('product_id').agg({
            'sale_amount': 'sum',
            'quantity': 'sum',
            'customer_id': 'nunique',
            'sale_id': 'count'
        }).round(2)

        product_performance.columns = ['total_revenue', 'total_quantity', 'unique_customers', 'total_transactions']
        product_performance['avg_transaction_value'] = product_performance['total_revenue'] / product_performance['total_transactions']

        print("Top 10 products by revenue:")
        print(product_performance.sort_values('total_revenue', ascending=False).head(10))

        print("\n6.3 Market Basket Analysis:")

        # Simple market basket analysis (products purchased together)
        from collections import defaultdict

        # Create basket analysis
        baskets = defaultdict(lambda: defaultdict(int))

        for customer_id in df['customer_id'].unique():
            customer_products = set(df[df['customer_id'] == customer_id]['product_id'])
            for product1 in customer_products:
                for product2 in customer_products:
                    if product1 != product2:
                        baskets[product1][product2] += 1

        # Find most frequent product pairs
        product_pairs = []
        for product1, related in baskets.items():
            for product2, count in related.items():
                if count > 5:  # Threshold for significance
                    product_pairs.append((product1, product2, count))

        product_pairs.sort(key=lambda x: x[2], reverse=True)
        print("Top 10 product pairs purchased together:")
        for i, (p1, p2, count) in enumerate(product_pairs[:10]):
            print(f"{i+1}. Product {p1} + Product {p2}: {count} times")

        print("\n6.4 Seasonal Analysis:")

        # Seasonal trends
        df['month'] = df['sale_date'].dt.month
        df['quarter'] = df['sale_date'].dt.quarter
        df['day_of_week'] = df['sale_date'].dt.dayofweek

        seasonal_analysis = df.groupby('month').agg({
            'sale_amount': 'sum',
            'sale_id': 'count',
            'customer_id': 'nunique'
        }).round(2)

        seasonal_analysis.columns = ['monthly_revenue', 'transaction_count', 'unique_customers']
        seasonal_analysis['avg_transaction_value'] = seasonal_analysis['monthly_revenue'] / seasonal_analysis['transaction_count']

        print("Monthly seasonal patterns:")
        print(seasonal_analysis)

        print("\n6.5 Geographic Analysis:")

        # Regional performance
        geo_analysis = df.groupby('region').agg({
            'sale_amount': 'sum',
            'customer_id': 'nunique',
            'sale_id': 'count',
            'quantity': 'sum'
        }).round(2)

        geo_analysis.columns = ['total_revenue', 'unique_customers', 'transaction_count', 'total_quantity']
        geo_analysis['avg_order_value'] = geo_analysis['total_revenue'] / geo_analysis['transaction_count']
        geo_analysis['avg_quantity'] = geo_analysis['total_quantity'] / geo_analysis['transaction_count']

        print("Regional performance:")
        print(geo_analysis.sort_values('total_revenue', ascending=False))

        return customer_ltv, product_performance, seasonal_analysis, geo_analysis


def main():
    """Run all advanced querying examples."""
    print("PyDala2 Advanced Querying Examples")
    print("=" * 50)

    try:
        # Run all examples
        table, ds_sales = example_1_complex_filtering()
        sql_results = example_2_sql_integration()
        agg_results = example_3_advanced_aggregations()
        perf_results = example_4_performance_optimization()
        catalog, full_data = example_5_multi_dataset_queries()
        analytics_results = example_6_advanced_analytics()

        print("\n" + "=" * 50)
        print("All advanced querying examples completed successfully!")
        print("\nKey Learnings:")
        print("- Complex filtering with compound conditions")
        print("- SQL integration with DuckDB for powerful queries")
        print("- Advanced aggregations and analytics")
        print("- Performance optimization techniques")
        print("- Multi-dataset querying capabilities")
        print("- Advanced analytics and insights")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
"""
Example usage of the simplified dataset module.
"""

import polars as pl
from simplified.dataset import ParquetDataset, JsonDataset, CsvDataset
from simplified.filtering import DateRangeFilter


def basic_usage():
    """Basic dataset operations."""
    # Create dataset (simplified initialization)
    dataset = ParquetDataset(
        path="/data/sales_data",
        partitioning=["year", "month"],
        timestamp_column="transaction_date"
    )

    # Write data (simplified API)
    df = pl.read_csv("new_sales.csv")
    dataset.write(df, mode="append")

    # Query data (simplified filtering)
    filtered = dataset.filter("total_amount > 1000 AND region = 'West'")

    # Get results
    result_df = filtered.to_polars()


def advanced_filtering_example():
    """Advanced filtering with date ranges."""
    dataset = ParquetDataset(path="/data/events", timestamp_column="event_time")

    # Date-based filtering (simplified API)
    from simplified.filtering import DateRangeFilter
    date_filter = DateRangeFilter(dataset, "event_time")

    # Filter last 7 days
    recent_events = date_filter.filter_last("7d")

    # Filter specific date range
    march_events = date_filter.filter_range(
        start="2024-03-01",
        end="2024-04-01"
    )


def optimization_example():
    """Dataset optimization example."""
    dataset = ParquetDataset(path="/data/large_dataset", partitioning="hive")

    # Auto-optimization (simpler API)
    dataset.optimize(method="auto")

    # Specific optimizations
    dataset.optimize_by_rows(max_rows_per_file=1_000_000)
    dataset.optimize_by_time(interval="1d", timestamp_column="date")
    dataset.optimize_partitions()

    # Schema optimization
    dataset.optimize_schema(infer_schema_rows=50_000)


def partitioning_example():
    """Working with partitioned datasets."""
    dataset = ParquetDataset(
        path="/data/logs",
        partitioning=["year", "month", "day"],
        timestamp_column="timestamp"
    )

    # Access partition information (simplified)
    partition_names = dataset.partitioning_manager.get_partition_names(dataset)
    partition_values = dataset.partitioning_manager.get_partition_values(dataset)

    # Write to specific partition (simplified)
    new_data = pl.DataFrame({...})
    dataset.write(
        new_data,
        partition_by=["year", "month", "day"],
        mode="append"
    )

    # Optimize specific partition
    dataset.optimize_partitions(max_files_per_partition=2)


def workflow_example():
    """Complete workflow example."""
    # 1. Create dataset
    sales_data = ParquetDataset(
        path="sales_data",
        partitioning=["year", "month"],
        cached=True,
        timestamp_column="sale_date"
    )

    # 2. Load initial data
    initial_df = pl.read_csv("initial_sales_2023.csv")
    sales_data.write(initial_df, mode="overwrite")

    # 3. Append new data
    new_data = pl.read_csv("sales_jan_2024.csv")
    sales_data.write(
        new_data,
        partition_by=["year", "month"],
        mode="append"
    )

    # 4. Query recent sales
    from simplified.filtering import DateRangeFilter
    date_filter = DateRangeFilter(sales_data, "sale_date")
    this_month = date_filter.filter_last("1M")

    # 5. Filter high-value sales
    high_value = this_month.filter("amount > 10000")

    # 6. Analyze results
    summary = high_value.to_polars().group_by("product_category").agg(
        pl.sum("amount").alias("total_sales"),
        pl.count().alias("num_sales")
    )

    # 7. Optimize dataset periodically
    sales_data.optimize(method="auto")


def error_handling_example():
    """Better error handling in simplified API."""
    try:
        # Simplified error handling
        dataset = ParquetDataset("/invalid/path")
        dataset.write(
            data,
            mode="invalid_mode"  # Will raise ValueError with clear message
        )
    except (FileNotFoundError, ValueError, ConnectionError) as e:
        logger.error(f"Dataset operation failed: {e}")
        # Handle specific errors appropriately


if __name__ == "__main__":
    print("Simplified Dataset Examples:")
    print("1. Basic Usage")
    print("2. Filtering")
    print("3. Optimization")
    print("4. Partitioning")
    print("5. Complete Workflow")
    print("6. Error Handling")
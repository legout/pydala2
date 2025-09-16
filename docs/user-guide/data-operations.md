# Data Operations

This guide covers advanced data operations and transformations you can perform with PyDala2.

## Filtering Operations

### Basic Filtering with Automatic Backend Selection

```python
from pydala import ParquetDataset

dataset = ParquetDataset("data/sales")

# Simple filters use PyArrow automatically
simple_result = dataset.filter("region = 'North America'")

# Complex filters automatically use DuckDB
complex_result = dataset.filter("""
    (category = 'Electronics' AND price > 500) OR
    (category = 'Books' AND rating >= 4.5) OR
    (customer_id IN (SELECT DISTINCT customer_id FROM premium_customers))
""")

# String operations use DuckDB
string_result = dataset.filter("""
    product_name LIKE '%Laptop%'
    AND customer_email LIKE '%@company.com'
    AND NOT product_description LIKE '%refurbished%'
""")

# Get results as Polars DataFrame
df = complex_result.collect()
```

### Advanced Filter Patterns

```python
# Date range filtering
date_result = dataset.filter("""
    sale_date BETWEEN '2023-01-01' AND '2023-12-31'
    AND sale_date < CURRENT_DATE
""")

# Using subqueries
subquery_result = dataset.filter("""
    customer_id IN (
        SELECT customer_id
        FROM customers
        WHERE segment = 'Premium'
        AND total_lifetime_value > 10000
    )
""")

# Pattern matching with regex
regex_result = dataset.filter("""
    product_code ~ '^[A-Z]{2}-\\d{4}$'
    AND product_name !~ '.*(test|demo|sample).*'
""")

# NULL handling
null_result = dataset.filter("""
    (status IS NOT NULL AND status != 'cancelled')
    OR (refund_amount IS NULL AND order_date > '2023-06-01')
""")
```

### Explicit Backend Selection

```python
# Force using PyArrow for simple operations
arrow_result = dataset.filter("region = 'US'", use="pyarrow")

# Force using DuckDB for complex operations
duckdb_result = dataset.filter("""
    product_name LIKE '%test%'
    OR customer_id IN (SELECT id FROM high_value_customers)
""", use="duckdb")
```

## Aggregation Operations

### Using DuckDB for Aggregations

```python
# Basic aggregations
agg_result = dataset.ddb_con.sql("""
    SELECT
        region,
        category,
        COUNT(*) as order_count,
        SUM(amount) as total_revenue,
        AVG(amount) as avg_order_value,
        MIN(amount) as min_order,
        MAX(amount) as max_order,
        COUNT(DISTINCT customer_id) as unique_customers
    FROM dataset
    WHERE date >= '2023-01-01'
    GROUP BY region, category
    HAVING COUNT(*) > 100
    ORDER BY total_revenue DESC
""").pl()

# Window functions
window_result = dataset.ddb_con.sql("""
    SELECT
        order_id,
        customer_id,
        order_date,
        amount,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id
            ORDER BY order_date
        ) as order_sequence,
        RANK() OVER (
            PARTITION BY customer_id
            ORDER BY amount DESC
        ) as amount_rank,
        LAG(amount, 1) OVER (
            PARTITION BY customer_id
            ORDER BY order_date
        ) as prev_amount,
        SUM(amount) OVER (
            PARTITION BY customer_id
            ORDER BY order_date
            RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW
        ) as rolling_30d_sum
    FROM dataset
    ORDER BY customer_id, order_date
""").pl()
```

### Time-based Aggregations

```python
# Time series aggregations
time_agg = dataset.ddb_con.sql("""
    SELECT
        DATE_TRUNC('month', order_date) as month,
        DATE_TRUNC('week', order_date) as week,
        region,
        COUNT(*) as orders,
        SUM(amount) as revenue,
        AVG(amount) as avg_order,
        COUNT(DISTINCT customer_id) as customers
    FROM dataset
    WHERE order_date >= '2023-01-01'
    GROUP BY month, week, region
    ORDER BY month, week
""").pl()

# Growth calculations
growth_result = dataset.ddb_con.sql("""
    WITH monthly_data AS (
        SELECT
            DATE_TRUNC('month', order_date) as month,
            SUM(amount) as revenue
        FROM dataset
        GROUP BY month
    )
    SELECT
        month,
        revenue,
        LAG(revenue, 1) OVER (ORDER BY month) as prev_revenue,
        (revenue - LAG(revenue, 1) OVER (ORDER BY month)) /
            LAG(revenue, 1) OVER (ORDER BY month) * 100 as growth_rate
    FROM monthly_data
    ORDER BY month
""").pl()
```

## Data Transformations

### Using Polars for Complex Transformations

```python
import polars as pl

# Read data
df = dataset.table.pl.collect()

# Chain transformations
transformed = (
    df
    .filter(pl.col("status") == "completed")
    .with_columns([
        # Date transformations
        pl.col("order_date").dt.year().alias("year"),
        pl.col("order_date").dt.month().alias("month"),
        pl.col("order_date").dt.day().alias("day"),
        pl.col("order_date").dt.weekday().alias("day_of_week"),

        # Business calculations
        (pl.col("unit_price") * pl.col("quantity")).alias("total_amount"),
        (pl.col("unit_price") * pl.col("quantity") * 0.1).alias("tax_amount"),
        (pl.col("unit_price") * pl.col("quantity") * 1.1).alias("total_with_tax"),

        # Conditional columns
        pl.when(pl.col("total_amount") > 1000)
        .then("High Value")
        .when(pl.col("total_amount") > 500)
        .then("Medium Value")
        .otherwise("Low Value")
        .alias("order_value_category")
    ])
    .group_by(["year", "month", "customer_segment"])
    .agg([
        pl.count("order_id").alias("order_count"),
        pl.sum("total_amount").alias("total_revenue"),
        pl.mean("total_amount").alias("avg_order_value"),
        pl.col("customer_id").n_unique().alias("unique_customers"),
        pl.quantile("total_amount", 0.95).alias("p95_order_value")
    ])
)

# Write transformed data
output_dataset = ParquetDataset("data/transformed_sales")
output_dataset.write_to_dataset(
    transformed,
    partition_by=["year", "month"],
    sort_by="total_revenue DESC"
)
```

### Schema Evolution and Type Conversions

```python
# Add new columns with type conversions
enhanced_data = df.with_columns([
    # String to categorical
    pl.col("category").cast(pl.Categorical),

    # Numeric conversions
    pl.col("price").cast(pl.Float32),
    pl.col("quantity").cast(pl.Int32),

    # Date operations
    pl.col("order_date").cast(pl.Date),

    # New calculated columns
    (pl.col("price") * pl.col("quantity")).alias("total_price"),
    pl.col("customer_email").str.to_lowercase().alias("email_lower")
])

# Write with schema evolution
dataset.write_to_dataset(
    enhanced_data,
    mode="append",
    alter_schema=True
)
```

## Join Operations

### Using DuckDB for Complex Joins

```python
# Multiple dataset joins via catalog
from pydala import Catalog

catalog = Catalog("catalog.yaml")

# Get datasets
orders = catalog.get_table("orders")
customers = catalog.get_table("customers")
products = catalog.get_table("products")

# Complex join with aggregations
join_result = orders.ddb_con.sql("""
    SELECT
        o.order_id,
        o.order_date,
        c.customer_name,
        c.segment,
        p.product_name,
        p.category,
        o.quantity,
        o.unit_price,
        (o.quantity * o.unit_price) as total_amount,
        ROW_NUMBER() OVER (
            PARTITION BY c.customer_id
            ORDER BY o.order_date DESC
        ) as recent_order_rank
    FROM orders o
    INNER JOIN customers c ON o.customer_id = c.id
    INNER JOIN products p ON o.product_id = p.id
    WHERE o.order_date >= '2023-01-01'
      AND c.segment IN ('Premium', 'VIP')
      AND p.category IN ('Electronics', 'Appliances')
    QUALIFY recent_order_rank <= 5  -- Get only 5 most recent orders per customer
    ORDER BY c.segment, c.customer_name, o.order_date DESC
""").pl()
```

### Self-Joins and Recursive Patterns

```python
# Customer purchase patterns
self_join_result = dataset.ddb_con.sql("""
    WITH customer_orders AS (
        SELECT
            customer_id,
            order_date,
            total_amount,
            LAG(order_date, 1) OVER (
                PARTITION BY customer_id
                ORDER BY order_date
            ) as prev_order_date
        FROM dataset
    )
    SELECT
        customer_id,
        COUNT(*) as order_count,
        AVG(total_amount) as avg_order_value,
        AVG(DATEDIFF('day', prev_order_date, order_date)) as avg_days_between_orders,
        COUNT(CASE WHEN DATEDIFF('day', prev_order_date, order_date) <= 7
             THEN 1 END) as repeat_purchase_count
    FROM customer_orders
    WHERE prev_order_date IS NOT NULL
    GROUP BY customer_id
    HAVING COUNT(*) >= 2
""").pl()
```

## Advanced Write Patterns

### Conditional and Batched Writes

```python
# Write in batches with validation
def write_in_batches(dataset, data, batch_size=100000):
    """Write data in batches with validation"""
    total_rows = len(data)

    for i in range(0, total_rows, batch_size):
        batch = data.slice(i, batch_size)

        # Validate batch
        if batch["amount"].min() < 0:
            print(f"Warning: Negative amounts found in batch {i//batch_size}")

        # Write batch
        dataset.write_to_dataset(
            batch,
            mode="append",
            update_metadata=(i + batch_size >= total_rows)  # Update metadata on last batch
        )

        print(f"Written batch {i//batch_size + 1}/{(total_rows-1)//batch_size + 1}")

# Usage
write_in_batches(dataset, large_dataframe)
```

### Upsert Operations

```python
def upsert_data(dataset, new_data, key_columns):
    """Implement upsert functionality"""
    # Get existing keys
    existing_keys = dataset.ddb_con.sql(f"""
        SELECT {', '.join(key_columns)}
        FROM dataset
    """).pl()

    # Find new vs existing records
    new_keys = new_data.select(key_columns)

    # Insert new records
    new_records = new_data.join(
        existing_keys,
        on=key_columns,
        how="anti"
    )

    if len(new_records) > 0:
        dataset.write_to_dataset(
            new_records,
            mode="append"
        )
        print(f"Inserted {len(new_records)} new records")

    # Update existing records using delta mode
    if len(new_data) > len(new_records):
        dataset.write_to_dataset(
            new_data,
            mode="delta",
            delta_subset=key_columns
        )
        print(f"Updated existing records")

# Usage
upsert_data(dataset, updated_customers, ["customer_id"])
```

## Data Quality Operations

### Automated Data Validation

```python
def validate_dataset_quality(dataset):
    """Perform comprehensive data quality checks"""
    # Read sample for validation
    sample = dataset.table.pl.collect().head(10000)

    quality_report = {
        "row_count": dataset.count_rows(),
        "column_count": len(sample.columns),
        "null_counts": {},
        "data_types": {},
        "value_ranges": {},
        "duplicate_rows": 0,
        "issues": []
    }

    # Check each column
    for col in sample.columns:
        # Null counts
        null_count = sample[col].null_count()
        quality_report["null_counts"][col] = null_count

        # Data types
        quality_report["data_types"][col] = str(sample[col].dtype)

        # Value ranges for numeric columns
        if sample[col].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
            quality_report["value_ranges"][col] = {
                "min": sample[col].min(),
                "max": sample[col].max(),
                "mean": sample[col].mean()
            }

            # Check for outliers
            q1 = sample[col].quantile(0.25)
            q3 = sample[col].quantile(0.75)
            iqr = q3 - q1
            outliers = sample.filter(
                (sample[col] < q1 - 1.5 * iqr) |
                (sample[col] > q3 + 1.5 * iqr)
            )

            if len(outliers) > 0:
                quality_report["issues"].append(
                    f"Column {col} has {len(outliers)} potential outliers"
                )

    # Check for duplicates
    quality_report["duplicate_rows"] = sample.is_duplicated().sum()

    return quality_report

# Usage
quality = validate_dataset_quality(dataset)
print(f"Quality Report: {quality}")
```

### Automated Data Cleaning

```python
def clean_dataset(dataset, rules):
    """Clean dataset based on business rules"""
    # Read data
    df = dataset.table.pl.collect()

    # Apply cleaning rules
    for rule in rules:
        if rule["type"] == "remove_nulls":
            df = df.filter(pl.col(rule["column"]).is_not_null())

        elif rule["type"] == "remove_outliers":
            col = rule["column"]
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df = df.filter(
                (df[col] >= q1 - 1.5 * iqr) &
                (df[col] <= q3 + 1.5 * iqr)
            )

        elif rule["type"] == "standardize_format":
            if rule["format"] == "uppercase":
                df = df.with_columns(
                    pl.col(rule["column"]).str.to_uppercase()
                )
            elif rule["format"] == "lowercase":
                df = df.with_columns(
                    pl.col(rule["column"]).str.to_lowercase()
                )

        elif rule["type"] == "fill_missing":
            if rule["strategy"] == "mean":
                fill_value = df[rule["column"]].mean()
            elif rule["strategy"] == "median":
                fill_value = df[rule["column"]].median()
            elif rule["strategy"] == "mode":
                fill_value = df[rule["column"]].mode()[0]

            df = df.with_columns(
                pl.col(rule["column"]).fill_null(fill_value)
            )

    # Write cleaned data
    cleaned_dataset = ParquetDataset(f"{dataset.path}_cleaned")
    cleaned_dataset.write_to_dataset(df)

    return cleaned_dataset

# Usage
cleaning_rules = [
    {"type": "remove_nulls", "column": "customer_id"},
    {"type": "remove_outliers", "column": "order_amount"},
    {"type": "standardize_format", "column": "email", "format": "lowercase"},
    {"type": "fill_missing", "column": "age", "strategy": "median"}
]

cleaned = clean_dataset(dataset, cleaning_rules)
```

## Performance Optimization Patterns

### Query Optimization

```python
# Optimize frequent queries with materialized views
class MaterializedView:
    def __init__(self, base_dataset, view_name, query):
        self.base_dataset = base_dataset
        self.view_name = view_name
        self.query = query
        self.view_dataset = ParquetDataset(f"views/{view_name}")

    def refresh(self):
        """Refresh the materialized view"""
        result = self.base_dataset.ddb_con.sql(self.query)
        self.view_dataset.write_to_dataset(
            result.pl(),
            mode="overwrite",
            update_metadata=True
        )

    def query(self, additional_filters=""):
        """Query the materialized view"""
        full_query = f"SELECT * FROM {self.view_name}"
        if additional_filters:
            full_query += f" WHERE {additional_filters}"

        return self.view_dataset.ddb_con.sql(full_query)

# Usage
mv = MaterializedView(
    dataset,
    "daily_sales_summary",
    """
    SELECT
        DATE_TRUNC('day', order_date) as day,
        region,
        category,
        COUNT(*) as orders,
        SUM(amount) as revenue,
        COUNT(DISTINCT customer_id) as customers
    FROM dataset
    GROUP BY day, region, category
    """
)

# Refresh daily
mv.refresh()

# Query the view
result = mv.query("day >= '2023-12-01'")
```

### Partitioning Optimization

```python
def optimize_partitioning(dataset, target_rows_per_file=1000000):
    """Analyze and optimize partitioning strategy"""
    # Get current statistics
    total_rows = dataset.count_rows()
    current_files = len(dataset.files)

    # Analyze partition cardinality
    if dataset.partition_names:
        partition_stats = {}

        for part_col in dataset.partition_names:
            unique_vals = dataset.ddb_con.sql(f"""
                SELECT DISTINCT {part_col}
                FROM dataset
            """).pl()

            partition_stats[part_col] = len(unique_vals)

        # Recommend optimal partitioning
        avg_rows_per_partition = total_rows / max(1, current_files)

        recommendations = []

        if avg_rows_per_partition < target_rows_per_file / 10:
            recommendations.append("Consider reducing partition levels - too many small files")
        elif avg_rows_per_partition > target_rows_per_file * 10:
            recommendations.append("Consider adding more partition levels - files too large")

        return {
            "total_rows": total_rows,
            "current_files": current_files,
            "avg_rows_per_file": avg_rows_per_partition,
            "partition_stats": partition_stats,
            "recommendations": recommendations
        }

# Usage
stats = optimize_partitioning(dataset)
print(f"Partitioning analysis: {stats}")
```

These patterns demonstrate the powerful data operations available in PyDala2, leveraging its dual-engine architecture to provide optimal performance for both simple and complex data processing tasks.
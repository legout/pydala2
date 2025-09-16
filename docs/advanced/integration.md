# Integration Patterns

This guide covers common integration patterns and use cases for PyDala2 with other tools and frameworks.

## Data Pipeline Integration

### Apache Airflow

```python
# Airflow DAG with PyDala2
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pydala import ParquetDataset, Catalog

def process_daily_sales():
    """Process daily sales data."""
    # Create dataset
    dataset = ParquetDataset("s3://sales-bucket/daily")

    # Read yesterday's data
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    data = dataset.filter(f"date = '{yesterday}'").table.to_polars()

    # Process data
    processed = (
        data
        .filter(pl.col("status") == "completed")
        .with_columns([
            (pl.col("amount") * 0.1).alias("tax"),
            (pl.col("amount") * 1.1).alias("total")
        ])
    )

    # Write to processed bucket
    processed_dataset = ParquetDataset("s3://processed-bucket/sales")
    processed_dataset.write_to_dataset(processed)

def update_dashboard():
    """Update dashboard with latest metrics."""
    catalog = Catalog("s3://metadata-bucket/catalog.yaml")

    # Calculate daily metrics
    metrics = catalog.ddb_con.sql("""
        SELECT
            date,
            COUNT(*) as orders,
            SUM(amount) as revenue,
            AVG(amount) as avg_order
        FROM daily_sales
        WHERE date = CURRENT_DATE - INTERVAL '1 day'
        GROUP BY date
    """).to_arrow()

    # Update dashboard database
    dashboard_db = ParquetDataset("s3://dashboard-bucket/metrics")
    dashboard_db.write_to_dataset(metrics)

with DAG(
    'daily_sales_processing',
    start_date=datetime(2023, 1, 1),
    schedule_interval='0 6 * * *',  # Daily at 6 AM
    catchup=False
) as dag:
    process_task = PythonOperator(
        task_id='process_daily_sales',
        python_callable=process_daily_sales
    )

    update_task = PythonOperator(
        task_id='update_dashboard',
        python_callable=update_dashboard
    )

    process_task >> update_task
```

### Prefect

```python
# Prefect flow with PyDala2
from prefect import flow, task
from prefect.context import get_run_context
from pydala import ParquetDataset

@task
def extract_data(source_path: str, date_filter: str):
    """Extract data from source."""
    dataset = ParquetDataset(source_path)
    return dataset.filter(f"date >= '{date_filter}'").table.to_polars()

@task
def transform_data(raw_data):
    """Transform raw data."""
    return (
        raw_data
        .filter(pl.col("status").is_in(["completed", "shipped"]))
        .group_by(["customer_id", "product_category"])
        .agg([
            pl.count("order_id").alias("order_count"),
            pl.sum("amount").alias("total_spent"),
            pl.col("order_date").max().alias("last_order_date")
        ])
    )

@task
def load_data(transformed_data, target_path: str):
    """Load transformed data."""
    dataset = ParquetDataset(target_path)
    dataset.write(transformed_data, mode='append')

@flow(name="ETL Pipeline")
def etl_pipeline(source_path: str, target_path: str, start_date: str):
    """Main ETL pipeline."""
    # Get flow run context
    context = get_run_context()
    run_id = context.flow_run.id

    # Extract
    raw_data = extract_data(source_path, start_date)

    # Transform
    transformed_data = transform_data(raw_data)

    # Load
    load_data(transformed_data, target_path)

    return f"Processed {len(transformed_data)} customer records"
```

## Machine Learning Integration

### Feature Store Integration

```python
# Feature store with PyDala2
import pyarrow as pa
from pydala import ParquetDataset, Catalog

class FeatureStore:
    """Simple feature store implementation."""

    def __init__(self, catalog_path: str):
        self.catalog = Catalog(catalog_path)

    def create_feature_view(self, name: str, query: str):
        """Create a feature view."""
        # Register feature view in catalog
        self.catalog._set_table_params(
            f"feature_views.{name}",
            query=query,
            type="feature_view",
            created_at=datetime.now().isoformat()
        )

    def get_features(self, view_name: str, entity_ids: list, feature_date: str):
        """Get features for entities."""
        query = self.catalog.get(f"feature_views.{view_name}").query

        # Execute query with parameters
        return self.catalog.query(
            query,
            parameters={
                "entity_ids": entity_ids,
                "feature_date": feature_date
            }
        )

# Usage
fs = FeatureStore("s3://feature-store/catalog.yaml")

# Create feature view
fs.create_feature_view(
    "customer_features",
    """
    SELECT
        customer_id,
        COUNT(DISTINCT order_id) as order_count_30d,
        SUM(amount) as total_spent_30d,
        AVG(amount) as avg_order_value_30d,
        DATEDIFF('day', MAX(order_date), ?) as days_since_last_order
    FROM sales
    WHERE order_date >= DATE_SUB(?, INTERVAL 30 DAY)
      AND order_date <= ?
    GROUP BY customer_id
    """
)

# Get features for ML
features = fs.get_features(
    "customer_features",
    entity_ids=[1, 2, 3, 4, 5],
    feature_date="2023-12-01"
)
```

### Model Training Integration

```python
# PyDala2 with scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_churn_model(dataset_path: str):
    """Train churn prediction model."""
    # Load data
    dataset = ParquetDataset(dataset_path)
    data = dataset.read(backend="pandas")

    # Prepare features
    features = data.drop(['customer_id', 'churned'], axis=1)
    labels = data['churned']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy:.2f}")

    # Save model and metadata
    model_path = f"models/churn_model_v{datetime.now().strftime('%Y%m%d')}.pkl"
    joblib.dump(model, model_path)

    # Save training metadata
    metadata = {
        "model_path": model_path,
        "accuracy": accuracy,
        "training_date": datetime.now().isoformat(),
        "features": list(features.columns),
        "n_samples": len(data)
    }

    metadata_dataset = ParquetDataset("models/metadata")
    metadata_dataset.write(
        pd.DataFrame([metadata]),
        mode="append"
    )

    return model, accuracy
```

### Batch Prediction

```python
# Batch prediction with PyDala2
def batch_predict(model_path: str, input_path: str, output_path: str):
    """Run batch predictions."""
    # Load model
    model = joblib.load(model_path)

    # Load input data
    input_dataset = ParquetDataset(input_path)
    data = input_dataset.read(backend="pandas")

    # Prepare features
    features = data.drop(['customer_id'], axis=1)

    # Make predictions
    predictions = model.predict_proba(features)[:, 1]
    results = data[['customer_id']].copy()
    results['churn_probability'] = predictions

    # Save results
    output_dataset = ParquetDataset(output_path)
    output_dataset.write(results, mode='overwrite')

    # Generate summary statistics
    summary = {
        "prediction_date": datetime.now().isoformat(),
        "total_predictions": len(results),
        "high_risk_customers": int((predictions > 0.7).sum()),
        "avg_churn_probability": float(predictions.mean())
    }

    return summary
```

## Web Application Integration

### FastAPI Integration

```python
# FastAPI with PyDala2
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from pydala import Catalog, ParquetDataset
import polars as pl

app = FastAPI()

# Initialize catalog
catalog = Catalog("catalog.yaml")

class SalesSummary(BaseModel):
    date: str
    total_sales: float
    total_orders: int
    avg_order_value: float

@app.get("/api/sales/summary", response_model=List[SalesSummary])
async def get_sales_summary(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    category: Optional[str] = Query(None, description="Product category")
):
    """Get sales summary for date range."""
    try:
        # Build query
        base_query = """
            SELECT
                DATE_TRUNC('day', order_date) as date,
                COUNT(*) as total_orders,
                SUM(amount) as total_sales,
                SUM(amount) / COUNT(*) as avg_order_value
            FROM sales
            WHERE order_date BETWEEN ? AND ?
        """
        params = [start_date, end_date]

        if category:
            base_query += " AND category = ?"
            params.append(category)

        base_query += " GROUP BY date ORDER BY date"

        # Execute query
        results = catalog.query(base_query, parameters=params)

        # Convert to response model
        return [
            SalesSummary(
                date=str(row["date"]),
                total_sales=float(row["total_sales"]),
                total_orders=int(row["total_orders"]),
                avg_order_value=float(row["avg_order_value"])
            )
            for row in results
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/products/{product_id}/analytics")
async def get_product_analytics(product_id: int):
    """Get analytics for a specific product."""
    try:
        # Get product dataset
        dataset = catalog.get_dataset("product_analytics")

        # Calculate analytics
        analytics = dataset.read(
            backend="polars",
            filters=f"product_id = {product_id}"
        ).select([
            pl.col("product_id"),
            pl.col("product_name"),
            pl.col("category"),
            pl.count("order_id").alias("total_orders"),
            pl.sum("amount").alias("total_revenue"),
            pl.mean("rating").alias("avg_rating"),
            pl.col("order_date").max().alias("last_ordered")
        ])

        if len(analytics) == 0:
            raise HTTPException(status_code=404, detail="Product not found")

        return analytics.to_dict(as_series=False)[0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Streamlit Dashboard

```python
# Streamlit dashboard with PyDala2
import streamlit as st
import plotly.express as px
from pydala import Catalog, ParquetDataset
import pandas as pd

# Initialize catalog
@st.cache_resource
def load_catalog():
    return Catalog("catalog.yaml")

catalog = load_catalog()

# Page title
st.title("Sales Analytics Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

# Date range
date_col, _ = st.columns([2, 1])
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

# Category filter
categories = catalog.query("SELECT DISTINCT category FROM sales ORDER BY category")
selected_category = st.sidebar.selectbox(
    "Category",
    options=["All"] + [row["category"] for row in categories]
)

# Load data based on filters
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_sales_data(start_date, end_date, category):
    """Load sales data with filters."""
    base_query = """
        SELECT
            order_date,
            category,
            product_name,
            amount,
            customer_id
        FROM sales
        WHERE order_date BETWEEN ? AND ?
    """
    params = [start_date, end_date]

    if category != "All":
        base_query += " AND category = ?"
        params.append(category)

    return catalog.query(base_query, parameters=params)

# Load data
sales_data = load_sales_data(start_date, end_date, selected_category)

# Convert to DataFrame for plotting
df = pd.DataFrame(sales_data)

# Key metrics
st.header("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Revenue", f"${df['amount'].sum():,.0f}")

with col2:
    st.metric("Total Orders", f"{len(df):,}")

with col3:
    st.metric("Average Order", f"${df['amount'].mean():.2f}")

with col4:
    st.metric("Unique Customers", f"{df['customer_id'].nunique():,}")

# Sales over time
st.header("Sales Trend")
df['order_date'] = pd.to_datetime(df['order_date'])
daily_sales = df.groupby(df['order_date'].dt.date)['amount'].sum().reset_index()

fig = px.line(
    daily_sales,
    x="order_date",
    y="amount",
    title="Daily Sales"
)
st.plotly_chart(fig, use_container_width=True)

# Sales by category
st.header("Sales by Category")
category_sales = df.groupby('category')['amount'].sum().reset_index()

fig = px.bar(
    category_sales,
    x="category",
    y="amount",
    title="Sales by Category"
)
st.plotly_chart(fig, use_container_width=True)

# Top products
st.header("Top Products")
top_products = df.groupby('product_name')['amount'].sum().nlargest(10).reset_index()

fig = px.bar(
    top_products,
    x="amount",
    y="product_name",
    orientation="h",
    title="Top 10 Products by Revenue"
)
st.plotly_chart(fig, use_container_width=True)
```

## Database Integration

### PostgreSQL Integration

```python
# Sync with PostgreSQL
import psycopg2
from psycopg2.extras import execute_batch
from pydala import ParquetDataset

def sync_to_postgres(dataset_path: str, table_name: str, db_config: dict):
    """Sync PyDala2 dataset to PostgreSQL."""
    # Connect to PostgreSQL
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    # Load dataset
    dataset = ParquetDataset(dataset_path)
    data = dataset.read(backend="pandas")

    # Create table if not exists
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        {', '.join([f'{col} {get_pg_type(dtype)}' for col, dtype in data.dtypes.items()])}
    )
    """
    cursor.execute(create_table_sql)

    # Prepare insert statement
    columns = ', '.join(data.columns)
    placeholders = ', '.join(['%s'] * len(data.columns))
    insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

    # Insert data in batches
    batch_size = 1000
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]
        execute_batch(
            cursor,
            insert_sql,
            [tuple(row) for row in batch.itertuples(index=False)]
        )
        conn.commit()
        print(f"Inserted {min(i + batch_size, len(data))} rows")

    cursor.close()
    conn.close()

def get_pg_type(dtype):
    """Map pandas dtype to PostgreSQL type."""
    if dtype == 'int64':
        return 'BIGINT'
    elif dtype == 'float64':
        return 'DOUBLE PRECISION'
    elif dtype == 'bool':
        return 'BOOLEAN'
    elif dtype == 'datetime64[ns]':
        return 'TIMESTAMP'
    else:
        return 'TEXT'
```

### BigQuery Integration

```python
# BigQuery integration
from google.cloud import bigquery
from google.oauth2 import service_account

def export_to_bigquery(dataset_path: str, project_id: str, dataset_id: str, table_id: str):
    """Export dataset to BigQuery."""
    # Initialize BigQuery client
    credentials = service_account.Credentials.from_service_account_file(
        'service-account.json'
    )
    client = bigquery.Client(credentials=credentials, project=project_id)

    # Load dataset
    dataset = ParquetDataset(dataset_path)
    data = dataset.read(backend="pandas")

    # Set up job config
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        source_format=bigquery.SourceFormat.PARQUET
    )

    # Get table reference
    table_ref = client.dataset(dataset_id).table(table_id)

    # Load data
    job = client.load_table_from_dataframe(
        data,
        table_ref,
        job_config=job_config
    )

    # Wait for job to complete
    job.result()
    print(f"Loaded {job.output_rows} rows into {dataset_id}.{table_id}")
```

## Real-time Integration

### Kafka Integration

```python
# Kafka producer with PyDala2
from kafka import KafkaProducer
import json
from pydala import ParquetDataset

def stream_to_kafka(dataset_path: str, topic: str, bootstrap_servers: str):
    """Stream dataset to Kafka."""
    # Initialize producer
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    # Load dataset
    dataset = ParquetDataset(dataset_path)
    data = dataset.read(backend="pandas")

    # Stream records
    for _, row in data.iterrows():
        message = row.to_dict()
        producer.send(topic, value=message)

    # Flush producer
    producer.flush()
    print(f"Streamed {len(data)} records to topic {topic}")

# Kafka consumer
from kafka import KafkaConsumer

def consume_from_kafka_to_dataset(topic: str, bootstrap_servers: str, output_path: str):
    """Consume from Kafka and save to dataset."""
    # Initialize consumer
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='earliest',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    # Collect messages
    records = []
    for message in consumer:
        records.append(message.value)

        # Batch save every 1000 records
        if len(records) >= 1000:
            df = pd.DataFrame(records)
            dataset = ParquetDataset(output_path)
            dataset.write(df, mode='append')
            records = []

    # Save remaining records
    if records:
        df = pd.DataFrame(records)
        dataset = ParquetDataset(output_path)
        dataset.write(df, mode='append')
```

## Monitoring and Observability

### Prometheus Integration

```python
# Prometheus metrics for PyDala2
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
READ_OPERATIONS = Counter(
    'pydala2_read_operations_total',
    'Total read operations',
    ['dataset', 'backend']
)

READ_DURATION = Histogram(
    'pydala2_read_duration_seconds',
    'Read operation duration',
    ['dataset', 'backend']
)

DATASET_SIZE = Gauge(
    'pydala2_dataset_size_bytes',
    'Dataset size in bytes',
    ['dataset']
)

# Monitor dataset operations
class MonitoredDataset:
    def __init__(self, dataset_path: str):
        self.dataset = ParquetDataset(dataset_path)
        self.dataset_path = dataset_path

    def read(self, **kwargs):
        """Monitored read operation."""
        backend = kwargs.get('backend', 'default')

        # Record start time
        start_time = time.time()

        try:
            # Perform read
            result = self.dataset.read(**kwargs)

            # Record metrics
            READ_OPERATIONS.labels(
                dataset=self.dataset_path,
                backend=backend
            ).inc()

            READ_DURATION.labels(
                dataset=self.dataset_path,
                backend=backend
            ).observe(time.time() - start_time)

            return result

        except Exception as e:
            # Increment error counter (not shown)
            raise e

    def update_size_gauge(self):
        """Update dataset size gauge."""
        size = sum(
            os.path.getsize(f)
            for f in self.dataset.files
        )
        DATASET_SIZE.labels(dataset=self.dataset_path).set(size)

# Start metrics server
start_http_server(8000)
```

### Logging Integration

```python
# Structured logging with ELK stack
import logging
import json
from datetime import datetime
from pydala import ParquetDataset

class JSONFormatter(logging.Formatter):
    """JSON formatter for ELK stack."""
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add extra fields if present
        if hasattr(record, 'dataset'):
            log_entry['dataset'] = record.dataset
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        if hasattr(record, 'duration'):
            log_entry['duration_ms'] = record.duration * 1000

        return json.dumps(log_entry)

# Configure logger
logger = logging.getLogger('pydala2')
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage with extra context
def monitored_read(dataset_path: str, **kwargs):
    start_time = time.time()
    logger.info(
        "Starting read operation",
        extra={
            'dataset': dataset_path,
            'operation': 'read'
        }
    )

    try:
        dataset = ParquetDataset(dataset_path)
        result = dataset.read(**kwargs)

        logger.info(
            "Completed read operation",
            extra={
                'dataset': dataset_path,
                'operation': 'read',
                'duration': time.time() - start_time,
                'rows_returned': len(result)
            }
        )

        return result

    except Exception as e:
        logger.error(
            "Read operation failed",
            extra={
                'dataset': dataset_path,
                'operation': 'read',
                'duration': time.time() - start_time,
                'error': str(e)
            }
        )
        raise
```

## Best Practices

1. **Use connection pooling** for database operations
2. **Implement retry logic** for network operations
3. **Cache frequently accessed data** to reduce I/O
4. **Monitor performance** with appropriate metrics
5. **Handle errors gracefully** with proper logging
6. **Use async patterns** for I/O-bound operations
7. **Batch operations** for better performance
8. **Validate data** before processing
9. **Document integrations** clearly
10. **Test integrations** thoroughly
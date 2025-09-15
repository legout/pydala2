import marimo

__generated_with = "0.15.3"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Data Loading and Conversion

        This notebook demonstrates how to load data from various sources and convert between different formats using PyDala2.
        """
    )
    return


@app.cell
def _():
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

    print(f"Library Availability:")
    print(f"  - Polars: {'✓' if POLARS_AVAILABLE else '✗'}")
    print(f"  - DuckDB: {'✓' if DUCKDB_AVAILABLE else '✗'}")
    return (
        CsvDataset,
        DUCKDB_AVAILABLE,
        JsonDataset,
        POLARS_AVAILABLE,
        ParquetDataset,
        Path,
        csv,
        datetime,
        duckdb,
        json,
        pa,
        pl,
        tempfile,
        timedelta,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Helper Functions for Creating Sample Data
        """
    )
    return


@app.cell
def _(Path, csv, datetime, json, timedelta):
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
    return create_sample_csv_data, create_sample_json_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example 1: Loading from CSV
        """
    )
    return


@app.cell
def _(Path, create_sample_csv_data, tempfile):
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    # Create sample CSV data
    csv_file = create_sample_csv_data(temp_path, 1000)
    print(f"Created CSV file: {csv_file}")
    return csv_file, temp_dir, temp_path


@app.cell
def _(CsvDataset, csv_file):
    # Method 1: Direct CSV loading
    print("1.1 Direct CSV loading...")
    csv_dataset = CsvDataset(csv_file)
    print(f"Loaded {len(csv_dataset)} records from CSV")
    print(f"Schema: {csv_dataset.schema}")
    return (csv_dataset,)


@app.cell
def _(ParquetDataset, csv_file, temp_path):
    print('\n1.2 Converting CSV to Parquet...')
    _parquet_path = temp_path / 'converted_data'
    parquet_dataset = ParquetDataset.from_csv(csv_file, path=_parquet_path, partition_cols=['category'])
    print(f'Converted to Parquet with {len(parquet_dataset.files)} files')
    return


@app.cell
def _(csv_dataset):
    # Method 3: Using PyDalaTable
    print("\n1.3 Using PyDalaTable...")
    table = csv_dataset.to_table()
    filtered = table.filter(table.price > 50)
    print(f"Filtered to {len(filtered)} records with price > 50")
    filtered.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example 2: Loading from JSON
        """
    )
    return


@app.cell
def _(create_sample_json_data, temp_path):
    # Create sample JSON data
    json_file = create_sample_json_data(temp_path, 500)
    print(f"Created JSON file: {json_file}")
    return (json_file,)


@app.cell
def _(JsonDataset, json_file):
    # Load JSON data
    print("2.1 Loading JSON data...")
    json_dataset = JsonDataset(json_file)
    print(f"Loaded {len(json_dataset)} records from JSON")
    return (json_dataset,)


@app.cell
def _(json_dataset):
    print('\n2.2 Exploring nested structure...')
    table_1 = json_dataset.to_table()
    print(f'Columns: {table_1.columns}')
    return


@app.cell
def _(json, json_file):
    # Show sample of nested data
    with open(json_file, 'r') as f:
        sample_record = json.load(f)[0]
    print("Sample JSON structure:")
    print(json.dumps(sample_record, indent=2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example 3: Format Conversion
        """
    )
    return


@app.cell
def _(CsvDataset, ParquetDataset, csv_file, temp_path):
    print('3.1 CSV to Parquet...')
    csv_ds = CsvDataset(csv_file)
    _parquet_path = temp_path / 'parquet_output'
    parquet_ds = ParquetDataset.from_dataset(csv_ds, path=_parquet_path, compression='zstd')
    print(f'Converted to Parquet with compression')
    return (parquet_ds,)


@app.cell
def _(parquet_ds):
    print('\n3.2 Parquet to other formats...')
    table_2 = parquet_ds.to_table()
    arrow_table = table_2.to_arrow()
    print(f'Converted to Arrow: {type(arrow_table)}')
    pandas_df = table_2.to_pandas()
    print(f'Converted to Pandas: {type(pandas_df)}')
    pandas_df.head()
    return (pandas_df,)


@app.cell
def _(pandas_df, temp_path):
    # Save to different formats
    csv_output = temp_path / "output.csv"
    pandas_df.to_csv(csv_output, index=False)
    print(f"Saved to CSV: {csv_output}")

    json_output = temp_path / "output.json"
    pandas_df.to_json(json_output, orient='records', date_format='iso')
    print(f"Saved to JSON: {json_output}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example 4: Working with Partitioned Datasets
        """
    )
    return


@app.cell
def _(ParquetDataset, csv_file, temp_path):
    # Create partitioned dataset
    print("4.1 Creating partitioned dataset...")
    partitioned_path = temp_path / "partitioned_data"
    partitioned_ds = ParquetDataset.from_csv(
        csv_file,
        path=partitioned_path,
        partition_cols=['category', 'is_active']
    )

    print(f"Created partitioned dataset:")
    print(f" - Partition columns: category, is_active")
    print(f" - Number of files: {len(partitioned_ds.files)}")
    return (partitioned_ds,)


@app.cell
def _(partitioned_ds):
    # Explore partitions
    print("\n4.2 Exploring partitions...")
    for file_path in partitioned_ds.files[:5]:  # Show first 5
        print(f"   {file_path}")
    return


@app.cell
def _(partitioned_ds):
    print('\n4.3 Querying specific partition...')
    table_3 = partitioned_ds.to_table()
    electronics_data = table_3.filter(table_3.category == 'Electronics')
    print(f'Electronics category: {len(electronics_data)} records')
    electronics_data.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example 5: Integration with Other Libraries
        """
    )
    return


@app.cell
def _(ParquetDataset, csv_file, temp_path):
    ds = ParquetDataset.from_csv(csv_file, temp_path / 'base_data')
    table_4 = ds.to_table()
    return (table_4,)


@app.cell
def _(table_4):
    print('5.1 Pandas integration...')
    pdf = table_4.to_pandas()
    pandas_result = pdf.groupby('category')['price'].agg(['mean', 'count'])
    print('Pandas aggregation result:')
    pandas_result
    return


@app.cell
def _(POLARS_AVAILABLE, pl, table_4):
    if POLARS_AVAILABLE:
        print('\n5.2 Polars integration...')
        polars_df = table_4.to_polars()
        polars_result = polars_df.groupby('category').agg([pl.col('price').mean(), pl.col('quantity').sum()])
        print('Polars aggregation result:')
        polars_result
    else:
        print('\n5.2 Polars not available')
    return


@app.cell
def _(DUCKDB_AVAILABLE, display, duckdb, table_4):
    if DUCKDB_AVAILABLE:
        print('\n5.3 DuckDB integration...')
        con = duckdb.connect()
        con.register('pydala_table', table_4.to_arrow())
        sql_result = con.execute('\n        SELECT\n            category,\n            AVG(price) as avg_price,\n            SUM(quantity) as total_quantity,\n            COUNT(*) as count\n        FROM pydala_table\n        GROUP BY category\n        ORDER BY avg_price DESC\n    ').fetchdf()
        print('DuckDB SQL result:')
        display(sql_result)
        con.close()
    else:
        print('\n5.3 DuckDB not available')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example 6: Handling Large Datasets
        """
    )
    return


@app.cell
def _(ParquetDataset, create_sample_csv_data, temp_path):
    print('6.1 Creating larger dataset...')
    large_csv_file = create_sample_csv_data(temp_path, 10000)
    large_ds = ParquetDataset.from_csv(large_csv_file, temp_path / 'large_data')
    table_5 = large_ds.to_table()
    return (table_5,)


@app.cell
def _(table_5):
    scanner = table_5.to_arrow_scanner(batch_size=1000, columns=['id', 'name', 'category', 'price'])
    batch_count = 0
    total_processed = 0
    print('Processing in batches...')
    for _batch in scanner.to_batches():
        batch_count = batch_count + 1
        total_processed = total_processed + _batch.num_rows
        if batch_count <= 3:
            avg_price = _batch.column('price').to_pandas().mean()
            print(f'  Batch {batch_count}: {_batch.num_rows} rows, avg price: {avg_price:.2f}')
    print(f'\nProcessed {total_processed} total records in {batch_count} batches')
    return


@app.cell
def _(pa, table_5):
    print('\n6.3 Memory-efficient filtering...')
    filtered_scanner = table_5.to_arrow_scanner(filter=pa.dataset.field('price') > 75)
    count = 0
    for _batch in filtered_scanner.to_batches():
        count = count + _batch.num_rows
    print(f'Records with price > 75: {count}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Cleanup

        Remove temporary files
        """
    )
    return


@app.cell
def _(temp_dir):
    import shutil
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory: {temp_dir}")
    print("\nAll examples completed successfully!")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

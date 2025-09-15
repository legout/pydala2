import marimo

__generated_with = "0.15.3"
app = marimo.App(width="full")


@app.cell
def __():
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
    return Catalog, ParquetDataset, Path, PyarrowDataset, PydalaTable, catalog, os, pa, pd, pq, tempfile


@app.cell
def __(pd):
    def create_sample_data():
        """Create sample data for demonstration."""
        data = {
            'id': range(1, 101),
            'name': [f'Item_{i}' for i in range(1, 101)],
            'category': ['A', 'B', 'C', 'D'] * 25,
            'value': [i * 1.5 for i in range(100)],
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='D')
        }
        return pd.DataFrame(data)
    return create_sample_data,


@app.cell
def __(create_sample_data):
    # Create sample data
    df = create_sample_data()
    print(f"Created sample data with {len(df)} rows")
    df.head()
    return create_sample_data, df


@app.cell
def __(tempfile, Path, df, ParquetDataset):
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    print(f"Using temporary directory: {temp_dir}")

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
    return dataset_path, ds, temp_dir, temp_path


@app.cell
def __(pa, temp_path, df, ParquetDataset):
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
    return arrow_table, dataset_path2, ds2


@app.cell
def __(temp_path, df, ParquetDataset):
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
    return category_df, category_path, ds3


@app.cell
def __(ds):
    # Basic dataset info
    print("2.1 Dataset Information:")
    print(f" - Number of files: {len(ds.files)}")
    print(f" - Total rows: {len(ds)}")
    print(f" - Schema: {ds.schema}")


@app.cell
def __(ds):
    # Convert to PydalaTable for advanced operations
    table = ds.to_table()

    # Select columns
    print("\n2.2 Column Selection:")
    selected = table.select(['name', 'value', 'category'])
    print(f"Selected columns: {selected.columns}")
    selected.head()
    return selected, table


@app.cell
def __(table):
    # Filter data
    print("\n2.3 Filtering Data:")
    filtered = table.filter(table.value > 50)
    print(f"Records with value > 50: {len(filtered)}")
    filtered.head()
    return filtered,


@app.cell
def __(filtered, pd):
    # Aggregate data
    print("\n2.4 Aggregation:")
    df_filtered = filtered.to_pandas()
    agg_result = df_filtered.groupby('category')['value'].agg(['mean', 'sum', 'count'])
    print(f"Aggregation by category:")
    agg_result
    return agg_result, df_filtered


@app.cell
def __(ds):
    # Add custom metadata
    print("3.1 Adding Custom Metadata:")
    ds.metadata.update({
        'description': 'Sample dataset for demonstration',
        'created_by': 'PyDala2 examples',
        'version': '1.0',
        'tags': ['sample', 'demo', 'test']
    })
    ds.save_metadata()


@app.cell
def __(ds):
    # Read metadata
    print("\n3.2 Reading Metadata:")
    print(f"Description: {ds.metadata.get('description')}")
    print(f"Created by: {ds.metadata.get('created_by')}")
    print(f"Version: {ds.metadata.get('version')}")


@app.cell
def __(ds):
    # Work with schema metadata
    print("\n3.3 Schema Information:")
    for field in ds.schema:
        print(f" - {field.name}: {field.type}")


@app.cell
def __(temp_path, Catalog, df, ParquetDataset):
    # Create a catalog
    catalog_path = temp_path / "catalog.yaml"
    catalog = Catalog(catalog_path)

    # Create multiple datasets
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
    return catalog, catalog_path, df_filtered, ds1, ds2


@app.cell
def __(catalog):
    # List datasets
    print("4.1 Registered Datasets:")
    for name in catalog.list_datasets():
        print(f" - {name}")


@app.cell
def __(catalog):
    # Retrieve dataset
    print("\n4.2 Retrieving Dataset:")
    retrieved_ds = catalog.get_dataset("full_data")
    print(f"Retrieved dataset has {len(retrieved_ds)} records")
    return retrieved_ds,


@app.cell
def __(catalog):
    # Save catalog
    catalog.save()
    print(f"\n4.3 Catalog saved to: {catalog_path}")


@app.cell
def __(table):
    # Convert to different formats
    print("5.1 Format Conversions:")

    # To pandas
    pdf = table.to_pandas()
    print(f"Converted to pandas: {type(pdf)} with shape {pdf.shape}")

    # To PyArrow
    arrow_table = table.to_arrow()
    print(f"Converted to PyArrow: {type(arrow_table)} with {arrow_table.num_rows} rows")
    return arrow_table, pdf


@app.cell
def __(pa, table):
    # Get scanner for efficient reading
    print("\n5.2 Using Scanner:")
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
    return batch_count, batch, scanner, total_rows


@app.cell
def __(table):
    # Head operation
    print("\n5.3 Head Operation:")
    head_data = table.head(5)
    head_data
    return head_data,


@app.cell
def __(temp_dir):
    import shutil
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory: {temp_dir}")
    return shutil,


if __name__ == "__main__":
    app.run()
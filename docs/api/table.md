# Table Operations

This document describes the `PydalaTable` class which provides a unified interface for table operations across multiple backends.

## PydalaTable

```python
class PydalaTable
```

The `PydalaTable` class provides a consistent interface for working with tabular data across different backends (PyArrow and DuckDB). It supports conversion to various formats and provides efficient data access methods.

## Shortcut Notation

PyDala2 provides a convenient shortcut notation using the `t` property to access the table interface:

```python
# Instead of writing:
df = dataset.table.to_polars()
table = dataset.table.to_arrow()
rel = dataset.table.to_duckdb()

# You can use the shortcut:
df = dataset.t.to_polars()
table = dataset.t.to_arrow()
rel = dataset.t.to_duckdb()

# The shortcut also provides direct property access:
lazy_df = dataset.t.pl  # Polars LazyFrame
df_pandas = dataset.t.df  # Pandas DataFrame
arrow_table = dataset.t.arrow  # PyArrow Table
duckdb_rel = dataset.t.ddb  # DuckDB relation
```

The shortcut notation is particularly useful when chaining operations:

```python
# Filter and export using shortcut
result = dataset.filter("amount > 100").t.to_pandas()

# Complex operations with shortcut
result = (
    dataset.t.pl
    .filter(pl.col("date") > "2023-01-01")
    .group_by("category")
    .agg(pl.mean("amount"))
    .collect()
)
```

### Constructor

```python
PydalaTable(
    result: pds.Dataset | duckdb.DuckDBPyRelation,
    ddb_con: duckdb.DuckDBPyConnection | None = None
) -> None
```

**Parameters:**
- `result`: The data source - either a PyArrow dataset or DuckDB relation
- `ddb_con`: Existing DuckDB connection. If None, creates a new one

**Example:**
```python
from pydala import PydalaTable, ParquetDataset
import pyarrow.dataset as pds

# From PyArrow dataset
dataset = pds.dataset("data/sales")
table = PydalaTable(dataset)

# From ParquetDataset
ds = ParquetDataset("data/sales")
table = PydalaTable(ds._arrow_dataset)
```

## Conversion Methods

### to_arrow_table
```python
def to_arrow_table(
    self,
    columns: str | list[str] | None = None,
    filter: pds.Expression | None = None,
    batch_size: int = 131072,
    sort_by: str | list[str] | list[tuple[str, str]] | None = None,
    distinct: bool = False,
    batch_readahead: int = 16,
    fragment_readahead: int = 4,
    fragment_scan_options: pds.FragmentScanOptions | None = None,
    use_threads: bool = True,
    memory_pool: pa.MemoryPool | None = None
) -> pa.Table
```
Convert the table to a PyArrow Table.

**Parameters:**
- `columns` (str | list[str]): Columns to include
- `filter` (pds.Expression): Filter expression to apply
- `batch_size` (int): Batch size for reading
- `sort_by` (str | list[str]): Column(s) to sort by
- `distinct` (bool): Whether to return distinct rows
- `batch_readahead` (int): Number of batches to read ahead
- `fragment_readahead` (int): Number of fragments to read ahead
- `fragment_scan_options` (pds.FragmentScanOptions): Scan options
- `use_threads` (bool): Whether to use threading
- `memory_pool` (pa.MemoryPool): Memory pool for allocation

**Returns:**
- `pa.Table`: The table as a PyArrow Table

**Example:**
```python
# Convert to Arrow table
table = dataset.table.to_arrow_table()

# With filters
import pyarrow.dataset as pds
filter_expr = (pds.field('amount') > 100) & (pds.field('category') == 'premium')
table = dataset.table.to_arrow_table(filter=filter_expr)

# With sorting
table = dataset.table.to_arrow_table(
    sort_by=['date', 'amount DESC'],
    columns=['id', 'date', 'amount']
)
```

### to_polars
```python
def to_polars(
    self,
    lazy: bool = True,
    columns: str | list[str] | None = None,
    sort_by: str | list[str] | None = None,
    distinct: bool = False,
    batch_size: int = 131072,
    **kwargs
) -> pl.DataFrame | pl.LazyFrame
```
Convert the table to a Polars DataFrame or LazyFrame.

**Parameters:**
- `lazy` (bool): If True, returns a LazyFrame; if False, returns a DataFrame
- `columns` (str | list[str]): Columns to include
- `sort_by` (str | list[str]): Column(s) to sort by
- `distinct` (bool): Whether to return distinct rows
- `batch_size` (int): Batch size for scanning
- `**kwargs`: Additional arguments

**Returns:**
- `pl.DataFrame | pl.LazyFrame`: Polars DataFrame or LazyFrame

**Example:**
```python
import polars as pl

# Get LazyFrame for lazy evaluation
lazy_df = dataset.table.to_polars(lazy=True)

# Perform operations
result = (
    lazy_df
    .filter(pl.col("amount") > 100)
    .group_by("category")
    .agg([
        pl.count("id").alias("count"),
        pl.mean("amount").alias("avg_amount")
    ])
    .collect()
)

# Get DataFrame directly
df = dataset.table.to_polars(lazy=False)
```

### to_pandas
```python
def to_pandas(
    self,
    columns: str | list[str] | None = None,
    sort_by: str | list[str] | None = None,
    distinct: bool = False,
    **kwargs
) -> pd.DataFrame
```
Convert the table to a pandas DataFrame.

**Parameters:**
- `columns` (str | list[str]): Columns to include
- `sort_by` (str | list[str]): Column(s) to sort by
- `distinct` (bool): Whether to return distinct rows
- `**kwargs`: Additional arguments

**Returns:**
- `pd.DataFrame`: The table as a pandas DataFrame

**Example:**
```python
# Convert to pandas DataFrame
df = dataset.table.to_pandas()

# With column selection
df = dataset.table.to_pandas(columns=['id', 'name', 'amount'])

# With sorting
df = dataset.table.to_pandas(
    sort_by='date DESC',
    distinct=False
)
```

### to_duckdb
```python
def to_duckdb(
    self,
    lazy: bool = True,
    columns: str | list[str] | None = None,
    batch_size: int = 131072,
    sort_by: str | list[str] | list[tuple[str, str]] | None = None,
    distinct: bool = False,
    **kwargs
) -> duckdb.DuckDBPyRelation
```
Convert the table to a DuckDB relation.

**Parameters:**
- `lazy` (bool): If True, returns a lazy relation; if False, materializes
- `columns` (str | list[str]): Columns to include
- `batch_size` (int): Batch size for scanning
- `sort_by` (str | list[str]): Column(s) to sort by
- `distinct` (bool): Whether to return distinct rows
- `**kwargs`: Additional arguments

**Returns:**
- `duckdb.DuckDBPyRelation`: DuckDB relation

**Example:**
```python
# Get DuckDB relation
rel = dataset.table.to_duckdb()

# Use SQL
result = rel.execute("SELECT category, AVG(amount) FROM rel GROUP BY category").df()

# Use relation API
result = rel.filter("amount > 100").project("category, amount").df()
```

### to_arrow_scanner
```python
def to_arrow_scanner(
    self,
    columns: str | list[str] | None = None,
    filter: pds.Expression | None = None,
    batch_size: int | None = None,
    sort_by: str | list[str] | list[tuple[str, str]] | None = None,
    batch_readahead: int | None = None,
    fragment_readahead: int | None = None,
    fragment_scan_options: pds.FragmentScanOptions | None = None,
    use_threads: bool | None = None,
    memory_pool: pa.MemoryPool | None = None
) -> pds.Scanner
```
Convert the table to an Arrow scanner for efficient data scanning.

**Returns:**
- `pds.Scanner`: Arrow scanner object

**Example:**
```python
# Create scanner
scanner = dataset.table.to_arrow_scanner(
    columns=['id', 'name', 'amount'],
    filter=pds.field('amount') > 100
)

# Read in batches
for batch in scanner.to_batches():
    process_batch(batch)
```

### to_batch_reader
```python
def to_batch_reader(
    self,
    columns: str | list[str] | None = None,
    filter: pds.Expression | None = None,
    lazy: bool = True,
    batch_size: int = 131072,
    sort_by: str | list[str] | list[tuple[str, str]] | None = None,
    distinct: bool = False,
    batch_readahead: int = 16,
    fragment_readahead: int = 4,
    fragment_scan_options: pds.FragmentScanOptions | None = None,
    use_threads: bool = True,
    memory_pool: pa.MemoryPool | None = None
) -> pa.RecordBatchReader
```
Convert the table to a batch reader for streaming operations.

**Returns:**
- `pa.RecordBatchReader`: RecordBatch reader

**Example:**
```python
# Stream data in batches
reader = dataset.table.to_batch_reader(
    batch_size=10000,
    columns=['id', 'name', 'amount']
)

for batch in reader:
    process_batch(batch)
```

## Properties

### arrow
```python
@property
def arrow(self) -> pa.Table
```
Get the table as a PyArrow Table (shortcut property).

**Returns:**
- `pa.Table`: PyArrow Table

**Example:**
```python
# Direct access to Arrow table
table = dataset.table.arrow
```

### pl
```python
@property
def pl(self) -> pl.LazyFrame
```
Get the table as a Polars LazyFrame (shortcut property).

**Returns:**
- `pl.LazyFrame`: Polars LazyFrame

**Example:**
```python
# Direct access to Polars LazyFrame
lazy_df = dataset.table.pl
result = lazy_df.filter(pl.col("amount") > 100).collect()
```

### df
```python
@property
def df(self) -> pd.DataFrame
```
Get the table as a pandas DataFrame (shortcut property).

**Returns:**
- `pd.DataFrame`: Pandas DataFrame

**Example:**
```python
# Direct access to pandas DataFrame
df = dataset.table.df
```

### ddb
```python
@property
def ddb(self) -> duckdb.DuckDBPyRelation
```
Get the table as a DuckDB relation (shortcut property).

**Returns:**
- `duckdb.DuckDBPyRelation`: DuckDB relation

**Example:**
```python
# Direct access to DuckDB relation
rel = dataset.table.ddb
result = rel.sql("SELECT COUNT(*) FROM rel").df()
```

### arrow_dataset
```python
@property
def arrow_dataset(self) -> pds.Dataset
```
Get the underlying PyArrow dataset.

**Returns:**
- `pds.Dataset`: PyArrow dataset

### duckdb_relation
```python
@property
def duckdb_relation(self) -> duckdb.DuckDBPyRelation
```
Get the DuckDB relation.

**Returns:**
- `duckdb.DuckDBPyRelation`: DuckDB relation

## Backend Selection

### Choosing the Right Backend

```python
# Use PyArrow for columnar operations
table = dataset.table
arrow_table = table.to_arrow_table()

# Use Polars for high-performance transformations
lazy_df = table.to_polars(lazy=True)
result = lazy_df.filter(...).group_by(...).collect()

# Use DuckDB for SQL operations
rel = table.to_duckdb()
sql_result = rel.execute("SELECT ...").df()

# Use Pandas for compatibility
df = table.to_pandas()
```

### Performance Considerations

- **PyArrow**: Best for columnar operations and Arrow ecosystem integration
- **Polars**: Fastest for complex transformations and aggregations
- **DuckDB**: Ideal for SQL queries and joins
- **Pandas**: Best for compatibility with existing code

### Memory Management

```python
# Use batch readers for large datasets
reader = table.to_batch_reader(batch_size=10000)
for batch in reader:
    # Process each batch
    pass

# Use lazy evaluation with Polars
lazy_df = table.to_polars(lazy=True)
# Operations are optimized before execution
result = lazy_df.filter(...).collect()

# Use DuckDB for out-of-core operations
rel = table.to_duckdb()
# DuckDB handles data larger than memory
```

## Common Operations

### Filtering

```python
# Arrow filtering
import pyarrow.dataset as pds
filter_expr = (pds.field('amount') > 100) & (pds.field('category') == 'premium')
result = table.to_arrow_table(filter=filter_expr)

# Polars filtering
lazy_df = table.to_polars(lazy=True)
result = lazy_df.filter(
    (pl.col('amount') > 100) & (pl.col('category') == 'premium')
).collect()

# DuckDB filtering
rel = table.to_duckdb()
result = rel.filter("amount > 100 AND category = 'premium'").df()
```

### Aggregation

```python
# Polars aggregation
lazy_df = table.to_polars(lazy=True)
result = (
    lazy_df
    .group_by('category')
    .agg([
        pl.count('id').alias('count'),
        pl.sum('amount').alias('total'),
        pl.mean('amount').alias('average')
    ])
    .sort('total', descending=True)
    .collect()
)

# DuckDB aggregation
rel = table.to_duckdb()
result = rel.execute("""
    SELECT
        category,
        COUNT(*) as count,
        SUM(amount) as total,
        AVG(amount) as average
    FROM table
    GROUP BY category
    ORDER BY total DESC
""").df()
```

### Joins

```python
# DuckDB join (most efficient)
rel1 = table1.to_duckdb()
rel2 = table2.to_duckdb()
result = rel1.join(rel2, 'id').df()

# Polars join
df1 = table1.to_polars()
df2 = table2.to_polars()
result = df1.join(df2, on='id')
```

## Advanced Usage

### Custom Arrow Expressions

```python
# Complex Arrow expressions
import pyarrow.compute as pc

filter_expr = (
    (pc.field('date') >= pc.scalar('2023-01-01')) &
    (pc.field('amount') > pc.scalar(100)) &
    (pc.is_in(pc.field('category'), pc.scalar(['A', 'B', 'C'])))
)

result = table.to_arrow_table(filter=filter_expr)
```

### SQL with DuckDB

```python
# Complex SQL queries
rel = table.to_duckdb()
result = rel.execute("""
    WITH monthly_stats AS (
        SELECT
            DATE_TRUNC('month', date) as month,
            category,
            COUNT(*) as orders,
            SUM(amount) as revenue
        FROM table
        WHERE date >= '2023-01-01'
        GROUP BY month, category
    )
    SELECT
        month,
        category,
        orders,
        revenue,
        revenue / orders as avg_order_value,
        LAG(revenue) OVER (PARTITION BY category ORDER BY month) as prev_month_revenue
    FROM monthly_stats
    ORDER BY month, category
""").df()
```

### Lazy Evaluation with Polars

```python
# Build complex query plan
query = (
    table.to_polars(lazy=True)
    .filter(pl.col('date') >= '2023-01-01')
    .group_by(['category', pl.col('date').dt.month()])
    .agg([
        pl.count('id').alias('orders'),
        pl.sum('amount').alias('revenue')
    ])
    .filter(pl.col('orders') > 10)
    .sort('revenue', descending=True)
)

# Execute once
result = query.collect()
```
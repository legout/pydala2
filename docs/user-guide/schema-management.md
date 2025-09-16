# Schema Management

PyDala2 provides comprehensive schema management capabilities to ensure data consistency and enable efficient operations.

## Understanding Schemas in PyDala2

A schema defines the structure of your data including:
- Column names and data types
- Nullability constraints
- Column metadata and descriptions
- Validation rules

```python
from pydala2 import ParquetDataset

dataset = ParquetDataset("data/sales")

# View the schema
schema = dataset.schema
print(schema)
```

## Schema Basics

### Viewing Schema Information

```python
# Get basic schema info
schema = dataset.schema

# Print column details
for field in schema:
    print(f"Column: {field.name}")
    print(f"  Type: {field.type}")
    print(f"  Nullable: {field.nullable}")
    print(f"  Metadata: {field.metadata}")
    print()
```

### Schema Types

PyDala2 supports all Arrow data types:

```python
# Numeric types
'int32', 'int64', 'float32', 'float64', 'decimal128'

# String and binary
'string', 'binary', 'large_string', 'large_binary'

# Date and time
'date32', 'date64', 'timestamp', 'time32', 'time64'

# List and struct types
'list', 'struct', 'large_list'

# Other
'boolean', 'null'
```

### Schema Operations

```python
# Check if column exists
has_id = 'id' in schema.names

# Get column by name
id_field = schema.field('id')

# Get column type
amount_type = schema.field('amount').type

# Schema size
num_columns = len(schema)
```

## Schema Validation

### Basic Validation

```python
# Validate data against schema
validation_result = dataset.validate_schema()

if validation_result.valid:
    print("Schema is valid")
else:
    print("Schema validation failed:")
    for error in validation_result.errors:
        print(f"  - {error}")
```

### Custom Validation Rules

```python
# Define custom validation rules
rules = {
    'id': {'required': True, 'type': 'int64'},
    'email': {'required': True, 'pattern': r'^[^@]+@[^@]+\.[^@]+$'},
    'amount': {'required': True, 'min': 0, 'max': 1000000},
    'status': {'allowed_values': ['active', 'inactive', 'pending']}
}

# Validate with custom rules
result = dataset.validate_data(rules=rules)
```

### Data Quality Checks

```python
# Comprehensive data quality validation
quality_report = dataset.data_quality_check()

print(f"Completeness: {quality_report.completeness:.2%}")
print(f"Consistency: {quality_report.consistency:.2%}")
print(f"Validity: {quality_report.validity:.2%}")

# Issues found
for issue in quality_report.issues:
    print(f"Row {issue.row}: {issue.message}")
```

## Schema Evolution

### Adding Columns

```python
# Add new column to existing dataset
import pandas as pd

# Read existing data
data = dataset.table.to_pandas()

# Add new column
data['discount_percent'] = 0.0

# Update schema metadata
dataset.update_schema_metadata({
    'discount_percent': {
        'description': 'Discount percentage applied',
        'unit': 'percentage',
        'default_value': 0.0
    }
})

# Write back
dataset.write(data, mode='overwrite')
```

### Modifying Column Types

```python
# Safe type conversion
def convert_column_safely(df, column, new_type):
    try:
        df[column] = df[column].astype(new_type)
        return df
    except Exception as e:
        print(f"Cannot convert {column} to {new_type}: {e}")
        return None

# Apply conversion
data = dataset.table.to_pandas()
data = convert_column_safely(data, 'user_id', 'string')
if data is not None:
    dataset.write(data, mode='overwrite')
```

### Schema Versioning

```python
# Track schema versions
dataset.add_schema_version({
    'version': '1.1',
    'changes': ['Added discount_percent column'],
    'author': 'data_team',
    'date': '2023-12-01'
})

# List schema versions
versions = dataset.list_schema_versions()
for v in versions:
    print(f"Version {v.version}: {v.changes}")
```

## Schema Metadata

### Column Documentation

```python
# Add descriptive metadata
schema_metadata = {
    'id': {
        'description': 'Unique identifier for the record',
        'type': 'integer',
        'primary_key': True
    },
    'customer_id': {
        'description': 'Foreign key to customers table',
        'type': 'integer',
        'references': 'customers.id'
    },
    'order_date': {
        'description': 'Date when order was placed',
        'type': 'datetime',
        'format': 'YYYY-MM-DD'
    },
    'total_amount': {
        'description': 'Total order amount including tax',
        'type': 'decimal',
        'precision': 10,
        'scale': 2,
        'currency': 'USD'
    }
}

dataset.update_schema_metadata(schema_metadata)
```

### Business Rules

```python
# Define business rules in schema
business_rules = {
    'order_status': {
        'allowed_values': ['pending', 'confirmed', 'shipped', 'delivered', 'cancelled'],
        'default': 'pending'
    },
    'quantity': {
        'min': 1,
        'max': 1000,
        'unit': 'items'
    },
    'email': {
        'validation': 'email_format',
        'required': True
    }
}

dataset.set_business_rules(business_rules)
```

### Data Classification

```python
# Classify data sensitivity
classification = {
    'ssn': {'classification': 'PII', 'access_level': 'restricted'},
    'email': {'classification': 'PII', 'access_level': 'internal'},
    'name': {'classification': 'personal', 'access_level': 'internal'},
    'amount': {'classification': 'financial', 'access_level': 'confidential'},
    'ip_address': {'classification': 'technical', 'access_level': 'internal'}
}

dataset.set_data_classification(classification)
```

## Schema Inference

### Automatic Schema Detection

```python
# Infer schema from CSV file
from pydala2 import infer_schema

schema = infer_schema("data.csv")
print(schema)

# Infer from sample data
import pandas as pd
sample_data = pd.read_csv("sample.csv", nrows=1000)
schema = infer_schema(sample_data)
```

### Schema Recommendations

```python
# Get schema optimization recommendations
recommendations = dataset.get_schema_recommendations()

for rec in recommendations:
    print(f"Column {rec.column}:")
    print(f"  Current type: {rec.current_type}")
    print(f"  Recommended type: {rec.recommended_type}")
    print(f"  Reason: {rec.reason}")
    print(f"  Impact: {rec.impact}")
    print()
```

### Schema Drift Detection

```python
# Detect schema changes over time
drift_report = dataset.detect_schema_drift()

if drift_report.has_drift:
    print("Schema drift detected:")
    for change in drift_report.changes:
        print(f"  {change.column}: {change.old_type} -> {change.new_type}")
        print(f"    First seen: {change.first_seen}")
        print(f"    Frequency: {change.frequency}")
```

## Schema Migration

### Safe Migration Process

```python
def safe_schema_migration(dataset, migration_func):
    """Safely migrate schema with backup"""

    # 1. Backup current data
    backup_path = f"{dataset.path}_backup_{int(time.time())}"
    backup = ParquetDataset(backup_path)
    backup.write(dataset.table.to_pandas())

    try:
        # 2. Apply migration
        data = dataset.table.to_pandas()
        data = migration_func(data)

        # 3. Validate
        if validate_migrated_data(data):
            # 4. Write back
            dataset.write(data, mode='overwrite')
            print("Migration successful")
            return True
        else:
            raise ValueError("Validation failed")

    except Exception as e:
        # 5. Restore from backup
        print(f"Migration failed: {e}")
        print("Restoring from backup...")
        dataset.write(backup.table.to_pandas(), mode='overwrite')
        return False
```

### Common Migrations

```python
# Example migration: Normalize date formats
def normalize_dates(df):
    df['order_date'] = pd.to_datetime(df['order_date']).dt.normalize()
    return df

# Execute migration
success = safe_schema_migration(dataset, normalize_dates)
```

## Schema Patterns

### Star Schema

```python
# Fact table schema
fact_schema = {
    'order_id': 'int64',
    'customer_id': 'int64',
    'product_id': 'int64',
    'date_id': 'int32',
    'quantity': 'int32',
    'unit_price': 'decimal(10,2)',
    'total_amount': 'decimal(10,2)',
    'discount_amount': 'decimal(10,2)'
}

# Dimension table schemas
date_schema = {
    'date_id': 'int32',
    'date': 'date32',
    'day': 'int32',
    'month': 'int32',
    'year': 'int32',
    'quarter': 'int32',
    'day_of_week': 'int32'
}
```

### Slowly Changing Dimensions

```python
# Type 2 SCD
customer_schema = {
    'customer_id': 'int64',
    'version': 'int32',
    'name': 'string',
    'email': 'string',
    'address': 'string',
    'valid_from': 'timestamp',
    'valid_to': 'timestamp',
    'current_flag': 'boolean'
}
```

### Nested Data

```python
# Schema for nested JSON
nested_schema = {
    'order_id': 'int64',
    'customer': {
        'id': 'int64',
        'name': 'string',
        'address': {
            'street': 'string',
            'city': 'string',
            'country': 'string'
        }
    },
    'items': [
        {
            'product_id': 'int64',
            'name': 'string',
            'quantity': 'int32',
            'price': 'decimal(10,2)'
        }
    ]
}
```

## Schema Utilities

### Schema Comparison

```python
# Compare two schemas
def compare_schemas(schema1, schema2):
    added = []
    removed = []
    changed = []

    fields1 = {f.name: f for f in schema1}
    fields2 = {f.name: f for f in schema2}

    # Check for added fields
    for name in fields2:
        if name not in fields1:
            added.append(name)

    # Check for removed fields
    for name in fields1:
        if name not in fields2:
            removed.append(name)

    # Check for type changes
    for name in fields1:
        if name in fields2 and fields1[name].type != fields2[name].type:
            changed.append((name, str(fields1[name].type), str(fields2[name].type)))

    return {
        'added': added,
        'removed': removed,
        'changed': changed
    }

# Usage
comparison = compare_schemas(old_schema, new_schema)
```

### Schema Documentation

```python
# Generate schema documentation
def generate_schema_doc(schema):
    doc = "# Schema Documentation\n\n"
    doc += "| Column | Type | Nullable | Description |\n"
    doc += "|--------|------|----------|-------------|\n"

    for field in schema:
        doc += f"| {field.name} | {field.type} | {field.nullable} | "
        doc += f"{field.metadata.get('description', '')} |\n"

    return doc

# Save to file
with open("schema.md", "w") as f:
    f.write(generate_schema_doc(dataset.schema))
```

## Best Practices

1. **Use consistent naming conventions** across all datasets
2. **Document columns** with clear descriptions and business context
3. **Validate data** before writing to ensure schema compliance
4. **Version schemas** when making breaking changes
5. **Use appropriate data types** for your use case
6. **Plan for schema evolution** from the start
7. **Monitor schema drift** in automated pipelines
8. **Back up data** before schema migrations
9. **Test migrations** in staging environments first
10. **Communicate changes** to downstream consumers
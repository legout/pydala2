"""
Refactored dataset module - simplified version that imports from modular components.

This module has been refactored to follow SOLID principles:
- Single Responsibility: Each class has a focused responsibility
- Open/Closed: Easy to extend with new functionality
- Liskov Substitution: Subclasses can replace their base classes
- Interface Segregation: Clients depend only on what they need
- Dependency Inversion: High-level modules don't depend on low-level details
"""

# Import all components from the new modular structure
from .dataset import (
    BaseDataset,
    ParquetDataset,
    PyarrowDataset,
    CsvDataset,
    JsonDataset,
    DatasetOptimizer,
)

# Re-export for backward compatibility
__all__ = [
    "BaseDataset",
    "ParquetDataset",
    "PyarrowDataset",
    "CsvDataset",
    "JsonDataset",
    "DatasetOptimizer",
]

# Note: The original 1,614-line file has been refactored into:
# - file_manager.py: Handles file operations (FileManager)
# - data_processor.py: Handles data processing (DataProcessor)
# - dataset_filter.py: Handles filtering operations (DatasetFilter)
# - dataset_writer.py: Handles write operations (DatasetWriter)
# - base.py: Simplified BaseDataset using composition
# - datasets.py: Specialized dataset classes
# - optimizer.py: Optimization functionality
# - __init__.py: Module exports

# Key improvements:
# 1. Reduced complexity: Large methods broken into smaller, focused functions
# 2. Better separation of concerns: Each class has a single responsibility
# 3. Composition over inheritance: Uses composition to combine functionality
# 4. Improved testability: Each component can be tested independently
# 5. Reduced code duplication: Shared functionality extracted
# 6. Better maintainability: Changes are isolated to specific components
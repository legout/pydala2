"""
Partitioning management for simplified dataset module.
"""

import logging
from typing import Optional, Union, List, Dict, Any

import pyarrow as pa

logger = logging.getLogger(__name__)


class PartitioningManager:
    """Manages dataset partitioning detection and configuration."""

    def __init__(self, config, filesystem_manager):
        self.config = config
        self.filesystem_manager = filesystem_manager

    def infer_partitioning(self) -> Optional[Union[str, List[str]]]:
        """Infer partitioning scheme from existing data."""
        # Handle explicit partitioning configuration
        if self.config.partitioning is not None:
            return self._handle_explicit_partitioning()

        # Try to infer partitioning from file paths
        return self._infer_from_files()

    def _handle_explicit_partitioning(self) -> Optional[Union[str, List[str]]]:
        """Handle explicitly configured partitioning."""
        partitioning = self.config.partitioning

        # Handle "ignore" special case
        if partitioning == "ignore":
            return None

        # Already properly formatted
        return partitioning

    def _infer_from_files(self) -> Optional[str]:
        """Infer partitioning from existing file paths."""
        try:
            # List files in the dataset directory
            files = self.filesystem_manager.filesystem.lss(self.config.path)

            # Check for hive-style partitioning (key=value)
            if any("=" in obj for obj in files):
                logger.info("Detected hive-style partitioning")
                return "hive"

        except FileNotFoundError:
            # Dataset directory doesn't exist yet, which is fine
            logger.debug(f"Dataset path not found during partitioning inference: {self.config.path}")
        except Exception as e:
            logger.warning(f"Error inferring partitioning: {e}")

        return None

    def get_partition_schema(self, dataset) -> Optional[pa.Schema]:
        """Get partitioning schema from loaded dataset."""
        if not dataset.is_loaded or not dataset._partitioning:
            return None

        try:
            if hasattr(dataset._arrow_dataset, "partitioning"):
                return dataset._arrow_dataset.partitioning.schema
        except Exception as e:
            logger.warning(f"Failed to get partition schema: {e}")

        return None

    def get_partition_names(self, dataset) -> List[str]:
        """Get list of partition column names."""
        schema = self.get_partition_schema(dataset)
        return list(schema.names) if schema else []

    def get_partition_values(self, dataset) -> Dict[str, List[Any]]:
        """Get unique values for each partition column."""
        if not dataset.is_loaded or not self.get_partition_schema(dataset):
            return {}

        try:
            partitioning = dataset._arrow_dataset.partitioning
            values = {}
            for i, name in enumerate(partitioning.schema.names):
                if i < len(partitioning.dictionaries):
                    values[name] = partitioning.dictionaries[i].to_pylist()
            return values
        except Exception as e:
            logger.warning(f"Failed to get partition values: {e}")
            return {}

    def build_partition_filter(self, partition_spec: Dict[str, Any]) -> str:
        """Build SQL filter expression for partition."""
        filters = []

        for name, value in partition_spec.items():
            if value is None:
                filters.append(f"{name} IS NULL")
            elif isinstance(value, str):
                filters.append(f"{name} = '{value}'")
            else:
                filters.append(f"{name} = {value}")

        return " AND ".join(filters)

    def validate_partition_spec(self, partition_spec: Dict[str, Any]) -> bool:
        """Validate partition specification."""
        if not partition_spec:
            return True

        for name, value in partition_spec.items():
            # Validate partition name
            if not isinstance(name, str) or not name.replace('_', '').isalnum():
                logger.error(f"Invalid partition name: {name}")
                return False

            # Validate partition value
            if value is not None and not isinstance(value, (str, int, float)):
                logger.error(f"Invalid partition value type for {name}: {type(value)}")
                return False

        return True
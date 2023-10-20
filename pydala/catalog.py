import os
import re
from collections import defaultdict

import duckdb
import pyarrow.parquet as pq

from .helpers import get_partitions_from_path
from .polars_ext import pl


class ParquetDatasetCatalog:
    def __init__(
        self,
        metadata: pq.FileMetaData | list[pq.FileMetaData],
        partitioning: None | str | list[str] = None,
    ):
        self.gen_catalog(metadata=metadata, partitioning=partitioning)
        self.ddb_con = duckdb.connect()
        self._files = sorted(self.catalog["file_path"].to_list())

    @property
    def files(self):
        return self._files

    def reset_scan_files(self):
        """
        Resets the list of scanned files to the original list of files.
        """
        self._is_scanned = False
        self._scan_files = self._files.copy()

    def gen_catalog(
        self,
        metadata: pq.FileMetaData | list[pq.FileMetaData],
        partitioning: None | str | list[str] = None,
    ):
        """
        Generates a polars DataFrame with statistics for each row group in the dataset.

        """

        catalog = defaultdict(list)
        for rg_num in range(metadata.num_row_groups):
            row_group = metadata.row_group(rg_num)
            file_path = rgc.column(0).file_path
            catalog["file_path"].append(file_path)
            catalog["num_columns"].append(row_group.num_columns)
            catalog["num_rows"].append(row_group.num_rows)
            catalog["total_byte_size"].append(row_group.total_byte_size)
            catalog["compression"].append(row_group.column(0).compression)

            if "=" in file_path:
                partitioning = partitioning or "hive"

            if partitioning is not None:
                partitions = get_partitions_from_path(
                    file_path, partitioning=partitioning
                )
                catalog.update(dict(partitions))

            for col_num in range(row_group(rg_num).num_columns):
                rgc = metadata.row_group(rg_num).column(col_num)
                rgc = rgc.to_dict()
                col_name = rgc.pop("path_in_schema")
                rgc.pop("file_path")
                rgc.pop("compression")
                rgc.update(rgc.pop("statistics"))
                catalog[col_name].append(rgc)

        self._catalog = pl.DataFrame(catalog)

    @property
    def catalog(self):
        if not hasattr(self, "_catalog"):
            self.gen_catalog()
        return self._catalog

    @staticmethod
    def _gen_filter(filter_expr: str, exclude_columns: list[str] = []) -> list[str]:
        """
        Generate a modified filter expression for a given filter expression and list of excluded columns.

        Args:
            filter_expr (str): The filter expression to modify.
            exclude_columns (list[str], optional): A list of columns to exclude from the modification. Defaults to [].

        Returns:
            list: A list of modified filter expressions.
        """
        # chech if filter_expr is a date string
        filter_expr_mod = []
        is_date = False
        res = re.search("(\d{4}-\d{1,2}-\d{1,2})", filter_expr)
        if res:
            if res.end() + 1 == res.endpos:
                is_date = True
        # print(is_date)
        if ">" in filter_expr:
            if not filter_expr.split(">")[0].lstrip("(") in exclude_columns:
                filter_expr_mod.append(
                    f"({filter_expr.replace('>', '_max::DATE>')} OR {filter_expr.split('>')[0]}_max::DATE IS NULL)"
                ) if is_date else filter_expr_mod.append(
                    f"({filter_expr.replace('>', '_max>')} OR {filter_expr.split('>')[0]}_max IS NULL)"
                )
            else:
                filter_expr_mod.append(filter_expr)
        elif "<" in filter_expr:
            if not filter_expr.split("<")[0].lstrip("(") in exclude_columns:
                filter_expr_mod.append(
                    f"({filter_expr.replace('<', '_min::DATE<')} OR {filter_expr.split('<')[0]}_min::DATE IS NULL)"
                ) if is_date else filter_expr_mod.append(
                    f"({filter_expr.replace('<', '_min<')} OR {filter_expr.split('<')[0]}_min IS NULL)"
                )
            else:
                filter_expr_mod.append(filter_expr)
        elif "=" in filter_expr:
            if not filter_expr.split("=")[0].lstrip("(") in exclude_columns:
                filter_expr_mod.append(
                    f"({filter_expr.replace('=', '_min::DATE<=')} OR {filter_expr.split('=')[0]}_min::DATE IS NULL)"
                ) if is_date else filter_expr_mod.append(
                    f"({filter_expr.replace('=', '_min<=')} OR {filter_expr.split('=')[0]}_min IS NULL)"
                )
                filter_expr_mod.append(
                    f"({filter_expr.replace('=', '_max::DATE>=')} OR {filter_expr.split('=')[0]}_max::DATE IS NULL)"
                ) if is_date else filter_expr_mod.append(
                    f"({filter_expr.replace('=', '_max>=')} OR {filter_expr.split('=')[0]}_max IS NULL)"
                )
        else:
            filter_expr_mod.append(filter_expr)

        return filter_expr_mod

    def scan(self, filter_expr: str | None = None, lazy: bool = True):
        """
        Scans the dataset for files that match the given filter expression.

        Args:
            filter_expr (str | None): A filter expression to apply to the dataset. Defaults to None.
            lazy (bool): Whether to perform the scan lazily or eagerly. Defaults to True.

        Returns:
            None
        """
        self._filter_expr = filter_expr
        if filter_expr is not None:
            filter_expr = [fe.strip() for fe in filter_expr.split("AND")]

            filter_expr_mod = []
            for fe in filter_expr:
                filter_expr_mod += self._gen_filter(
                    fe, exclude_columns=self.file_catalog.columns
                )

            self._filter_expr_mod = " AND ".join(filter_expr_mod)

            self._catalog_scanned = (
                self.ddb_con.from_arrow(self.file_catalog.to_arrow())
                .filter(self._filter_expr_mod)
                .pl()
            )
            self._scan_files = sorted(
                [
                    os.path.join(self._path, sf)
                    for sf in self._scanned_file_catalog["file_path"].to_list()
                ]
            )
        else:
            self.reset_scan_files()
        # return self

    def filter(self, filter_expr: str | None = None, lazy: bool = True):
        """
        Filters the dataset for files that match the given filter expression.

        Args:
            filter_expr (str | None): A filter expression to apply to the dataset. Defaults to None.
            lazy (bool): Whether to perform the scan lazily or eagerly. Defaults to True.

        Returns:
            None
        """
        self.scan(filter_expr=filter_expr, lazy=lazy)

    @property
    def filter_expr(self):
        """
        Returns the filter expression and its module.

        If the filter expression has not been set yet, it will return None for both the expression and its module.
        """
        if not hasattr(self, "_filter_expr"):
            self._filter_expr = None
            self._filter_expr_mod = None
        return self._filter_expr, self._filter_expr_mod

    @property
    def scan_files(self):
        if not hasattr(self, "_scan_files"):
            self.scan()
        return self._scan_files

    @property
    def is_scanned(self):
        """
        Check if all files in the dataset have been scanned.

        Returns:
            bool: True if all files have been scanned, False otherwise.
        """
        return sorted(self._scan_files) == sorted(self._files)
        # @staticmethod

    # def _get_row_group_stats(
    #     row_group: pq.RowGroupMetaData,
    #     partitioning: None | str | list[str] = None,
    # ):
    #     def get_column_stats(row_group_column):
    #         name = row_group_column.path_in_schema
    #         column_stats = {}
    #         column_stats[
    #             name + "_total_compressed_size"
    #         ] = row_group_column.total_compressed_size
    #         column_stats[
    #             name + "_total_uncompressed_size"
    #         ] = row_group_column.total_uncompressed_size

    #         column_stats[name + "_physical_type"] = row_group_column.physical_type
    #         if row_group_column.is_stats_set:
    #             column_stats[name + "_min"] = row_group_column.statistics.min
    #             column_stats[name + "_max"] = row_group_column.statistics.max
    #             column_stats[
    #                 name + "_null_count"
    #             ] = row_group_column.statistics.null_count
    #             column_stats[
    #                 name + "_distinct_count"
    #             ]: row_group_column.statistics.distinct_count
    #         return column_stats

    #     stats = {}
    #     file_path = row_group.column(0).file_path
    #     stats["file_path"] = file_path
    #     if "=" in file_path:
    #         partitioning = partitioning or "hive"
    #     if partitioning is not None:
    #         partitions = get_partitions_from_path(file_path, partitioning=partitioning)
    #         stats.update(dict(partitions))

    #     stats["num_columns"] = row_group.num_columns
    #     stats["num_rows"] = row_group.num_rows
    #     stats["total_byte_size"] = row_group.total_byte_size
    #     stats["compression"] = row_group.column(0).compression

    #     column_stats = [
    #         get_column_stats(row_group.column(i)) for i in range(row_group.num_columns)
    #     ]
    #     # column_stats = [await task for task in tasks]

    #     [stats.update(cs) for cs in column_stats]
    #     return stats

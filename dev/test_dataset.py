import pyarrow.dataset as pds
import pickle
from pydala.helpers._metadata import collect_parquet_metadata
import os
import glob
import pyarrow.parquet as pq


class PydalaMetadata:
    def __init__(self, path: str):
        self._base_path = path
        self._metadata_path = os.path.join(path, "metadata/")
        self._data_path = os.path.join(path, "data")

        self._files = glob.glob(self._data_path + "/**/*.parquet", recursive=True)

        self.metadata = None
        self.file_metadata = None

        # def load_ds(self):
        # if os.path.exists(self._metadata_path + "/_metadata"):
        #    self.ds = pds.paqrquet_dataset(self._metadata_path + "/_metadata")

    def load_metadata(self):
        if os.path.exists(self._metadata_path + "/_metadata"):
            self.metadata = pq.read_metadata(self._metadata_path + "/_metadata")

        if os.path.exists(self._data_path + "/file_metadata.pkl"):
            with open(self._data_path + "/file_metadatapokl", "rb") as f:
                self.file_metadata = pickle.load(f)

    def _collect_file_metadata(self, files):
        file_metadata = collect_parquet_metadata(files)
        for fn in file_metadata:
            file_metadata[fn].set_file_path(
                "../data/" + fn.split(self._data_path)[-1].lstrip("/")
            )

        if self.file_metadata:
            self.file_metadata.update(file_metadata)
        else:
            self.file_metadata = file_metadata

    def update_file_metadata(self):
        if not self.file_metadata:
            self.load_metadata()

        if self.file_metadata:
            files = sorted(set(self._files) - set(self.file_metadata.keys()))
        else:
            files = self._files

        self._collect_file_metadata(files)

        with open(self._metadata_path + "/file_metadata.pkl", "wb") as f:
            pickle.dump(self.file_metadata, f)

    def update_metadata(self):
        self.update_file_metadata()

        if self.metadata:
            files_in_metadata = [
                self.metadata.row_group(i).column(0).file_path
                for i in range(self.metadata.num_row_groups)
            ]
        else:
            files_in_metadata = []
        #    new_files = sorted(set(self._files) - set(files_in_metadata))
        #    for fn in new_files:
        #        self.metadata.append_row_groups(self.file_metadata[fn])
        # else:
        self.metadata = self.file_metadata[self._files[0]]
        for fn in self._files[1:]:
            if fn not in files_in_metadata:
                self.metadata.append_row_groups(self.file_metadata[fn])

        with open(os.path.join(self._metadata_path, "_metadata"), "wb") as f:
            self.metadata.write_metadata_file(f)

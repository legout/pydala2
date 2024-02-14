# %%
import pyarrow.dataset as pds
import s3fs
from fsspec.implementations.dirfs import DirFileSystem
import duckdb

con = duckdb.connect()

path = "traces"
bucket = "myBucket"

fs = DirFileSystem(path=bucket, fs=s3fs.S3FileSystem())

# %%time
ds1 = pds.parquet_dataset(path + "/_metadata", filesystem=fs)
res = con.sql("FROM ds1").filter("time>'2022-07-01' AND time<'2022-07-02'").df()

# CPU times: user 9.09 s, sys: 900 ms, total: 9.99 s
# Wall time: 7.08 s


# %%time
ds2 = pds.dataset(path, filesystem=fs)
res = con.sql("FROM ds2").filter("time>'2022-07-01' AND time<'2022-07-02'").df()

# CPU times: user 3min 32s, sys: 13.7 s, total: 3min 46s
# Wall time: 10min 27s

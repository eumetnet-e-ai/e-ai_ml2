import os
from ecmwf.opendata import Client

os.environ["HTTP_PROXY"]  = "http://ofsquid.dwd.de:8080"
os.environ["HTTPS_PROXY"] = "http://ofsquid.dwd.de:8080"

client = Client(
    source="ecmwf",   # or "aws", "azure", "google"
    model="ifs",
)

client.retrieve(
    time=0,
    type="fc",
    step=24,
    param=["2t", "msl"],
    target="ifs_2t.grib2")

print("Downloaded ifs_2t.grib2")

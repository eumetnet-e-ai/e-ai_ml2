import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
base_url = (
  "https://opendata.dwd.de/weather/"
  "satellite/clouds/CTH/")
response = requests.get(base_url)
soup = BeautifulSoup(response.text, "html.parser")
cth_files = sorted([ link.get("href")
  for link in soup.find_all("a")
  if link.get("href", "").endswith(".nc.bz2") ])
my_f = cth_files[-1]
url = urljoin(base_url,my_f)
with open("cth.nc.bz2", "wb") as f:
    f.write(requests.get(url).content)

import bz2

# Decompress downloaded CTH file
with bz2.BZ2File("cth.nc.bz2", "rb") as f_in:
    with open("cth.nc", "wb") as f_out:
        f_out.write(f_in.read())

print("✅ Decompressed cth.nc.bz2 → cth.nc")

import netCDF4 as nc

ds = nc.Dataset("cth.nc")

print("📦 Variables in file:")
for var in ds.variables:
    print(f" - {var}: {ds.variables[var].shape}")

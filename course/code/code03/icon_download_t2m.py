#!/usr/bin/env python3
"""
Download the latest ICON-EU 2 m temperature GRIB file from DWD Open Data.

- Finds latest available file
- Downloads .grib2.bz2
- Decompresses to .grib2
"""

import requests
import re
import bz2
from urllib.parse import urljoin
from pathlib import Path

BASE_URL = "https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/t_2m/"

def find_latest_t2m(base_url):
    """Return latest ICON-EU T2M GRIB (.grib2.bz2)."""
    resp = requests.get(base_url)
    resp.raise_for_status()

    matches = re.findall(
        r"icon-eu_europe_regular-lat-lon_single-level_(\d{10})_000_T_2M\.grib2\.bz2",
        resp.text,
    )

    if not matches:
        raise RuntimeError("No ICON-EU T2M files found.")

    latest = sorted(matches)[-1]
    filename = (
        f"icon-eu_europe_regular-lat-lon_single-level_"
        f"{latest}_000_T_2M.grib2.bz2"
    )
    return filename

def download_and_decompress(url, target_grib):
    """Download .bz2 file and decompress to .grib2."""
    bz2_file = target_grib.with_suffix(target_grib.suffix + ".bz2")

    print(f"Downloading {url} …")
    r = requests.get(url)
    r.raise_for_status()

    bz2_file.write_bytes(r.content)
    print(f"Saved {bz2_file}")

    print(f"Decompressing to {target_grib} …")
    with bz2.open(bz2_file, "rb") as f_in, open(target_grib, "wb") as f_out:
        f_out.write(f_in.read())

    bz2_file.unlink()
    print("Done.")

if __name__ == "__main__":
    filename = find_latest_t2m(BASE_URL)
    full_url = urljoin(BASE_URL, filename)

    download_and_decompress(
        full_url,
        Path("icon_eu_t2m_latest.grib2")
    )

#!/usr/bin/env python
# coding: utf-8

# ## Look at CBCF Data

# ## Demo Folder

# In[1]:


ls 2024


# ## Check File Contents

# In[2]:


import json
from pathlib import Path

folder = Path("./2024")

# Pick one D0 and one D1 file to inspect
sample_files = [
    folder / "D0-2024-05-01-dwd_crossborder.final_internal.json",
    folder / "D1-2024-05-01-dwd_crossborder.final_internal.json"
]

for f in sample_files:
    print(f"\n=== {f.name} ===")
    with open(f, encoding="utf-8") as fh:
        data = json.load(fh)
    
    # Print only top-level keys and a snippet
    if isinstance(data, dict):
        print("Keys:", list(data.keys()))
        for k, v in list(data.items())[:3]:  # show first 3 entries
            print(f"  {k}: {str(v)[:200]} ...")
    elif isinstance(data, list):
        print("Type: list, length:", len(data))
        print("First entry:", str(data[0])[:200], "...")


# ## Install Geopandas

# In[3]:


#! pip install geopandas


# ## CheckPolygon File

# In[4]:


import geopandas as gpd

# Load one file directly
gdf = gpd.read_file("./2024/D0-2024-05-01-dwd_crossborder.final_internal.json")
print(gdf.head())
print(gdf.columns)


# ## Visualize with Cartopy

# In[5]:


import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd  # <-- needed

# Load one file (GeoJSON)
file = "2024/D0-2024-05-01-dwd_crossborder.final_internal.json"
gdf = gpd.read_file(file)

# Make sure timestamps are parsed
gdf["VALID_TIMESTAMP"] = pd.to_datetime(gdf["VALID_TIMESTAMP"])

# Select only one valid time (optional)
subset = gdf[gdf["VALID_TIMESTAMP"].dt.hour == 12]  # e.g. 12 UTC polygons

# --- Plot with Cartopy ---
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add background features
ax.set_extent([-15, 30, 35, 60], crs=ccrs.PlateCarree())  # Europe bbox
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.3, alpha=0.5)

# Plot polygons (colored by LIKELINESS)
subset.plot(ax=ax, column="LIKELINESS", legend=True, alpha=0.5, edgecolor="black")

plt.title("CBCF D0 – 2024-05-01 12 UTC")
fig.savefig("cbcf_d0_20240501_12utc.png", dpi=200, bbox_inches="tight")
plt.show()


# In[6]:


# One row
row = gdf.iloc[0]

print(type(row.geometry))      # shapely.geometry.polygon.Polygon
print(row.geometry.area)       # area (in degrees², unless reprojected)
print(row.geometry.bounds)     # bounding box (minx, miny, maxx, maxy)
print(list(row.geometry.exterior.coords)[:5])  # first 5 coordinate pairs


# In[7]:


coords = list(row.geometry.exterior.coords)
for lon, lat in coords[:10]:  # print first 10
    print(lon, lat)


# In[8]:


import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

# Load file
file = "2024/D0-2024-05-01-dwd_crossborder.final_internal.json"
gdf = gpd.read_file(file)

# Parse timestamps
gdf["VALID_TIMESTAMP"] = pd.to_datetime(gdf["VALID_TIMESTAMP"])

# Select one polygon (first row for demo)
poly = gdf.iloc[0:1]   # still a GeoDataFrame with 1 row

# --- Plot with Cartopy ---
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Map background
ax.set_extent([-15, 30, 35, 60], crs=ccrs.PlateCarree())  # Europe
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)

# Plot the polygon
poly.plot(ax=ax, facecolor="red", edgecolor="black", alpha=0.5)

plt.title(f"One CBCF polygon\n{poly.iloc[0]['VALID_TIMESTAMP']}")
plt.show()


# In[9]:


import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import time
from IPython.display import clear_output, display

# Load file
file = "./2024/D0-2024-05-01-dwd_crossborder.final_internal.json"
gdf = gpd.read_file(file)
gdf["VALID_TIMESTAMP"] = pd.to_datetime(gdf["VALID_TIMESTAMP"])

# Sort by time
gdf = gdf.sort_values("VALID_TIMESTAMP")

# Unique forecast hours
times = gdf["VALID_TIMESTAMP"].unique()

# Figure once
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})

for t in times:
    ax.clear()
    ax.set_extent([-15, 30, 35, 60], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    # Subset polygons for this valid time
    subset = gdf[gdf["VALID_TIMESTAMP"] == t]

    # Plot polygons
    subset.plot(ax=ax, facecolor="red", edgecolor="black", alpha=0.4)

    # Title
    ax.set_title(f"CBCF polygons valid at {t}")

    clear_output(wait=True)
    display(fig)
    time.sleep(2)  # 2 sec pause

plt.close(fig)
print("Animation finished.")


# ## Check all files

# In[10]:


import geopandas as gpd
import pandas as pd
import glob

files = sorted(glob.glob("./2024/*.final_internal.json"))

summary = []
for f in files:
    if ".info." in f:   # skip sidecar info files
        continue
    try:
        gdf = gpd.read_file(f)
        gdf["VALID_TIMESTAMP"] = pd.to_datetime(gdf["VALID_TIMESTAMP"])
        summary.append({
            "file": f.split("\\")[-1],
            "features": len(gdf),
            "min_time": gdf["VALID_TIMESTAMP"].min(),
            "max_time": gdf["VALID_TIMESTAMP"].max(),
        })
    except Exception as e:
        summary.append({"file": f.split("\\")[-1], "error": str(e)})

df_summary = pd.DataFrame(summary)
display(df_summary.head(20))


# In[11]:


import re

df_summary["date"] = df_summary["file"].str.extract(r"(\d{4}-\d{2}-\d{2})")
df_summary["run"] = df_summary["file"].str.extract(r"(D\d)")

pivot = df_summary.pivot_table(
    index="date", 
    columns="run", 
    values="features", 
    aggfunc="first"
).sort_index()

display(pivot.head(50))  # first 20 days
pivot.plot(kind="bar", figsize=(12,4))


# In[12]:


#!pip install folium mapclassify


# In[ ]:


import geopandas as gpd

gdf = gpd.read_file("./2024/D0-2024-05-01-dwd_crossborder.final_internal.json")

# pick the first polygon's valid time
t0 = gdf.iloc[0]["VALID_TIMESTAMP"]

# select all polygons with this valid time
subset = gdf[gdf["VALID_TIMESTAMP"] == t0]

print("Time:", t0, "Polygons:", len(subset))
subset.explore(
    column="LIKELINESS",
    categorical=True,
    categories=["likely", "possible", "unlikely"],  # force consistent order
    cmap=["red", "yellow", "blue"],               # same length as categories
    style_kwds={"fillOpacity": 0.1, "weight": 0.5},
    highlight_kwds={"fillOpacity": 0.0, "weight": 0.5}
)



# In[ ]:


sample_files = df_summary["file"].sample(5, random_state=0)

for f in sample_files:
    gdf = gpd.read_file("" + f)
    print(f, len(gdf), gdf["VALID_TIMESTAMP"].min(), gdf["VALID_TIMESTAMP"].max())
    display(gdf.iloc[0:1].explore())  # if you want interactive map


# In[ ]:





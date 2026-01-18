#!/usr/bin/env python3
import eccodes
import numpy as np
import matplotlib.pyplot as plt

filename = "ifs_2t.grib2"

with open(filename, "rb") as f:
    gid_2t = None

    while True:
        gid = eccodes.codes_grib_new_from_file(f)
        if gid is None:
            break

        short_name = eccodes.codes_get(gid, "shortName")

        if short_name == "2t":
            gid_2t = gid
            break
        else:
            eccodes.codes_release(gid)

    if gid_2t is None:
        raise RuntimeError("2t field not found in GRIB file")

    nx = eccodes.codes_get(gid_2t, "Ni")
    ny = eccodes.codes_get(gid_2t, "Nj")
    values = eccodes.codes_get_array(gid_2t, "values")

    eccodes.codes_release(gid_2t)

# --------------------------------------------------
# Reshape and plot
# --------------------------------------------------
field = values.reshape(ny, nx)

plt.figure(figsize=(7, 3.5))
plt.imshow(field, origin="lower")
plt.colorbar(label="K")
plt.title("IFS 2 m temperature")

plt.tight_layout()
plt.show()

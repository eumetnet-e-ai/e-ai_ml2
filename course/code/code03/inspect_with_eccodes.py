#!/usr/bin/env python3
import sys
import eccodes

# --------------------------------------------------
# Command-line argument
# --------------------------------------------------
if len(sys.argv) != 2:
    print("Usage: python inspect_with_eccodes.py <gribfile>")
    sys.exit(1)

filename = sys.argv[1]

print(f"Inspecting GRIB file: {filename}")
print("-" * 60)

with open(filename, "rb") as f:
    msg = 0
    while True:
        gid = eccodes.codes_grib_new_from_file(f)
        if gid is None:
            break

        msg += 1

        short_name = eccodes.codes_get(gid, "shortName")
        level_type = eccodes.codes_get(gid, "typeOfLevel")
        level      = eccodes.codes_get(gid, "level")
        nvalues    = eccodes.codes_get_size(gid, "values")

        print(
            f"Message {msg:2d}: "
            f"{short_name:6s}  "
            f"{level_type:12s}={level:<5d}  "
            f"nvalues={nvalues}"
        )

        eccodes.codes_release(gid)

print("-" * 60)
print(f"Total messages: {msg}")

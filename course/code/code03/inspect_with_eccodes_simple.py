import eccodes

f = open("icon_eu_t2m_latest.grib2", "rb")
while True:
    gid = eccodes.codes_grib_new_from_file(f)
    if gid is None:
        break

    short = eccodes.codes_get(gid, "shortName")
    level = eccodes.codes_get(gid, "level")
    size  = eccodes.codes_get_size(gid, "values")
    print(f"sN={short}, lev={level}, s={size}")
    eccodes.codes_release(gid)
f.close()

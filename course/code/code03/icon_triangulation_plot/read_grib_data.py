import eccodes as ecc
import datetime


def read_grib_data(infile, shortname, level):
    """Read a single field from a GRIB file.

    Parameters
    ----------
    infile : str
        full path of GRIB data file to read
    shortname : str
        GRIB short name of the variable to read
    level : int or float
        level to read (eccodes key 'level', eg. 1 for T or 0.01 for W_SO)
        level=0 for single level variable!!

    Returns
    -------
    values : ndarray

    """


    with open(infile, 'rb') as f:
        print('reading grib file: '+infile)
        while True:     # walk through grib message until found 
            gid = ecc.codes_grib_new_from_file(f)
            
            if gid is None:
                print("Error reading variable {0}, level {1}, not found in file".format(
                    shortname, str(level)))
                break

            if (ecc.codes_get(gid, 'shortName') == shortname and
                ecc.codes_get(gid, 'level', float) == float(level)):
                print("Found variable {0}, level {1}".format(
                    shortname, str(level)))
                values    = ecc.codes_get_values(gid)
                long_name = ecc.codes_get(gid, 'name')
                units     = ecc.codes_get(gid, 'parameterUnits')
                initime   = ecc.codes_get(gid, 'dateTime')
                endstep   = ecc.codes_get(gid, 'endStep')
                timeunits = ecc.codes_get(gid, 'stepUnits')       # 0: minutes, 1: hours
                break

            ecc.codes_release(gid)

#-- Time conversion
    inidate = datetime.datetime.strptime(initime, "%Y%m%d%H%M")
    if timeunits == 0:
      time_change = datetime.timedelta(minutes=endstep)
    else:
      time_change = datetime.timedelta(hours=endstep)
    time = inidate + time_change

    print('time:                     ',time)

    return values, long_name, units, time

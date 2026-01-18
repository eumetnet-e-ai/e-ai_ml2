from netCDF4 import Dataset, num2date


def read_nc_data(infile, shortname, level, ntime):
    """Read a single field from a GRIB file.

    Parameters
    ----------
    infile : str
        full path of GRIB data file to read
    shortname : str
        GRIB short name of the variable to read
    level : int or float
        level to read (eccodes key 'level', eg. 1 for T or 0.01 for W_SO)

    Returns
    -------
    values : ndarray

    """

    print('opening ICON netcdf file:   ', infile)
    ncf    = Dataset(infile)

#-- units

    if hasattr(ncf.variables[shortname], 'units'):
      units  = ncf.variables[shortname].units
    else:
      units  = ''

#-- Data and dimensions

    dims   = ncf.variables[shortname].dimensions
    values = ncf.variables[shortname][:]   
    print('variable, units, shape:   ', shortname, '[', units, ']  ', values.shape, dims)
    
    if len(dims) == 3:
      values = values[ntime-1,level-1,:]
    else:
      values = values[ntime-1,:]

    long_name = ncf.variables[shortname].long_name

#-- Time for plot and plot file name

    dates     = num2date(ncf.variables['time'] , ncf.variables['time'].units)
    seconds   = [(d-dates[0]).total_seconds() for d in dates]

    time_date = dates[ntime-1]
    time_hour = seconds[ntime-1] / 3600
    print('time:                     ', time_date, '  ', time_hour, '[h]')

    time = dates

    return values, long_name, units, time

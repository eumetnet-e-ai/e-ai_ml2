import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from read_nc_data           import read_nc_data
from read_grib_data         import read_grib_data
from read_grid              import read_grid
from fix_dateline_triangles import fix_dateline_triangles

def read_and_plot(infile, gridfile, shortname, level, ntime, experiment, colormap, varmin, varmax, ncolors):
    """Read data and grid and plot.

    Parameters
    ----------
    infile : str
        full path of GRIB data file to read
    gridfile : str
        full path of the corresponding grid file
    shortname : str
        GRIB short name of the variable to plot
    level : int or float
        level of the variable to plot (eccodes key 'level')

    Returns
    -------
    fig : Figure

    """

    print('')

    if   infile.endswith('.nc'):
      nc_grib = 1
    elif infile.endswith('.grb'):
      nc_grib = 2
    else:
      print('file type not supported: ', infile)

    if nc_grib == 1:
      values1, long_name, units, time = read_nc_data(infile, shortname, level, ntime)
    else:
      values1, long_name, units, time = read_grib_data(infile, shortname, level)

    triangulation, lat    = read_grid(gridfile)
    triangulation, values = fix_dateline_triangles(triangulation, values1, False)

    fig, ax = plt.subplots(figsize=(11,6), subplot_kw=dict(projection=ccrs.PlateCarree()))

    plt.subplots_adjust(bottom=0.12, right=1.0)


#-- color map

    cmap = plt.get_cmap(colormap, ncolors)

#-- map on triangles

    if varmin == varmax:
      map = ax.tripcolor(triangulation, values, cmap=cmap)
    else:
      map = ax.tripcolor(triangulation, values, cmap=cmap, vmin=varmin, vmax=varmax)

#-- plot detail

    if nc_grib == 1:
      time_text = time[ntime-1].strftime('%d%b%Y+%Hh')
    else:
      time_text = time.strftime('%d%b%Y %Hh')
    title = experiment + '   ' + long_name + '   ' + time_text
    plt.title(title)
    ax.coastlines()     # requires HTTPS_PROXY=http://ofsquid.dwd.de:8080 for internet - and PlateCarree
    gl = ax.gridlines(draw_labels=True)
    gl.right_labels = False
    gl.top_labels   = False


#-- area weighted mean and RMS - subtitle

    weights = np.cos(np.deg2rad(lat))
    weights.name = "weights"
    
    var_weighted_lat = np.average(values1, weights=weights, axis=0)
    area_mean = np.mean(var_weighted_lat)
    
    var_sqr_weighted_lat = np.average(values1 ** 2, weights=weights, axis=0)
    area_rms = np.mean(var_sqr_weighted_lat) ** 0.5
    
    plt.text(0.5, -0.13, 
      'Min:'     + "{0:.4g}".format(np.min(values1)) +
      '    Max:' + "{0:.4g}".format(np.max(values1)) +
      '    Mean:'+ "{0:.4g}".format(area_mean) +
      '    RMS:' + "{0:.4g}".format(area_rms),
      va='bottom', ha='center', transform=ax.transAxes, fontsize=11)

#-- color bar

    cbar = plt.colorbar(map, shrink=0.85)
    cbar.ax.set_ylabel('['+units+']')


    return fig, ax, time_text

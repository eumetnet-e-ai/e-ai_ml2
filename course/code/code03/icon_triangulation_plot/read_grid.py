import matplotlib.tri as mtri
import numpy as np
import netCDF4 as nc


def read_grid(gridfile):
    """Read coordinates of triangle vertices from NetCDF file.

    Parameters
    ----------
    gridfile : str
        full path of NetCDF file containing ICON grid information

    Returns
    -------
    triangulation : Triangulation
        a Triangulation object suitable for plotting with matplotlib

    """

    print('Reading grid file:        ', gridfile)

    with nc.Dataset(gridfile) as f:
        #vlon and vlat in ICON grid file are in radians
        vlon = f['vlon'][:] * 180/np.pi
        vlat = f['vlat'][:] * 180/np.pi
        vertex_of_cell = f['vertex_of_cell'][:]
        clat = f['clat'][:]
    # vertex indices in grid file count from 1, Python wants 0
    vertex_of_cell = vertex_of_cell - 1
    triangulation = mtri.Triangulation(vlon, vlat, vertex_of_cell.T)

    return triangulation, clat

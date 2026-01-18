import numpy as np
import matplotlib.tri as mtri


def fix_dateline_triangles(triangulation, values, mask_only=False):
    """Fix triangles crossing the date line.

    Triangles crossing the horizontal map boundary are plotted across
    the whole two-dimensional plane of the PlateCarree projection
    (eg. from -180 degrees to 180 degrees), while they should "wrap
    around the back". For the respective triangles on either side of
    the plot, the vertices beyond the date line - on thus on the
    opposite side of the plot - are re-set to a value on the same side
    and the triangle is duplicated on the other side.

    To visualize this effect, use mask_only=True. In this case, the
    triangles are not duplicated and the respective triangles are only
    masked and will not be plotted.

    (ideas taken from the ICON Model Tutorial 2019, section 9.3.3)

    Parameters
    ----------
    triangulation : Triangulation
        the triangulation to be fixed
    values : ndarray
        the values corresponding to the triangulation
    mask_only : bool, optional
        whether to mask the triangles without changing the vertices

    Returns
    -------
    triangulation_fixed : Triangulation
        the triangulation with modified triangles and vertices
    values_fixed : ndarray
        the values with duplicated values for duplicated triangles appended

    """

    to_fix = np.argwhere(triangulation.x[triangulation.triangles].max(axis=1)
                         - triangulation.x[triangulation.triangles].min(axis=1)
                         > 200
                         )[:, 0]

    # create a new Triangulation object to avoid overwriting the original data
    triangulation_fixed = mtri.Triangulation(triangulation.x, triangulation.y, triangulation.triangles)

    if mask_only:
        triangulation_fixed.mask = np.full(triangulation.triangles.shape[0], False)
        triangulation_fixed.mask[to_fix] = True
    else:
        values_fixed = values.copy()
        k = triangulation.x.shape[0]
        for i in to_fix:
            # append the mirrored triangle and its value to the existing triangles and values
            triangle = triangulation.triangles[i]
            triangulation_fixed.triangles = np.vstack([triangulation_fixed.triangles, triangle])
            values_fixed = np.append(values_fixed, values[i])

            # adjust the vertices of the appended triangle such that all lon values are > 0
            idx_vertex = np.argwhere(triangulation.x[triangle]<0)
            for j in idx_vertex:
                triangulation_fixed.x = np.append(triangulation_fixed.x,
                                                  triangulation.x[triangle[j]] + 360)
                triangulation_fixed.y = np.append(triangulation_fixed.y,
                                                  triangulation.y[triangle[j]])
                triangulation_fixed.triangles[-1, j] = k
                k = k+1

            # adjust the vertices of the original, copied triangle such that all lon values are < 0
            idx_vertex = np.argwhere(triangulation.x[triangle]>0)
            for j in idx_vertex:
                triangulation_fixed.x = np.append(triangulation_fixed.x,
                                                  triangulation.x[triangle[j]] - 360)
                triangulation_fixed.y = np.append(triangulation_fixed.y,
                                                  triangulation.y[triangle[j]])
                triangulation_fixed.triangles[i, j] = k
                k = k+1

    return triangulation_fixed, values_fixed

"""
Plot ICON data on triangular grids

module load unsupported 
module load python-extras/3.8.12
module load eccodes/2.26.0-x86-python

requires HTTPS_PROXY=http://ofsquid.dwd.de:8080 to get coastline from internet

CPU time: about 100s for a plot of ICON at 13km resolution

Gernot Geppert, Feb. 2021
Martin Koehler, May  2022
"""

import matplotlib.pyplot as plt
from read_and_plot import read_and_plot


#-- setup

exp      = '042'
varname  = 't_2m'   # clct, sob_t, ktype, qc
level    = 0        # level 0 for single level variable
ntime    = 1        # for grib: always first step taken

#-- data file and associcated grid

dir      = '/hpc/uwork/mkoehler/run-icon/experiments'
gridfile = '/hpc/rhome/routfox/routfox/icon/grids/public/edzw/icon_grid_0012_R02B04_G.nc'
infile   = dir+'/couple_'+exp+'/couple_'+exp+'_atm_nat_mn_ML_19790101T000000Z.nc'

#-- set min/max and experiment name for title and color table info

experiment = 'ICON exp'+exp
varmin     = 0         # data minimum   automatic range if varmin=varmax
varmax     = 0         # data maximum
ncolors    = 10
colormap   = 'RdBu_r'  #'RdBu_r'  # 'gray' for clouds

#-- Select region (-180,180 proper cut at dateline)

lon1 = -180   # -180  120   123  110
lon2 =  180   #  180  170   125  155
lat1 =  -90   #  -90  -70   -53  -65
lat2 =   90   #   90  -30   -55  -40


#------------------------------------------------------------------------

fig, ax, time = read_and_plot(infile, gridfile, varname, level, ntime, experiment, \
                              colormap, varmin, varmax, ncolors)

plt.xlim(lon1,lon2)
plt.ylim(lat1,lat2)

lonlattext = 'lon'+str(lon1)+'to'+str(lon2)+'lat'+str(lat1)+'to'+str(lat2)
fig.savefig(varname+'_ICON_'+exp+'_'+time+'_'+lonlattext+'.png', bbox_inches='tight')
plt.show()

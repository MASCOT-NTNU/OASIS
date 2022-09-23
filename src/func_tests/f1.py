""" Generate mask indices to remove unnecessary data points from Delft3D """

import numpy as np
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from datetime import datetime
from matplotlib.cm import get_cmap


plg_path = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Projects/OASIS/OperationArea.csv"

plg = pd.read_csv(plg_path).to_numpy()

# plt.plot(plg[:, 1], plg[:, 0], 'k.-')
# plt.show()

datapath = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Delft3D/raw/Nov2016_sal_1.mat"
df = h5py.File(datapath, 'r')
data = df.get('data')
lon = np.array(data.get("X")).squeeze()
lat = np.array(data.get("Y")).squeeze()
depth = np.array(data.get("Z"))
Time = np.array(data.get('Time'))
timestamp = (Time - 719529) * 24 * 3600  # 719529 is how many days have passed from Jan1 0,
# to Jan1 1970. Since 1970Jan1, is used as the starting index for datetime
sal_data = np.array(data["Val"])
# string_date = datetime.fromtimestamp(timestamp_data[0]).strftime("%Y_%m")

#%%
plt.scatter(lon[0, :, :], lat[0, :, :], c=sal_data[0, :, :, 1], cmap=get_cmap("BrBG", 10), vmin=10, vmax=36)
plt.plot(grid[:, 1], grid[:, 0], 'k-.', markersize=1, alpha=.1)
plt.plot(plg[:, 1], plg[:, 0], 'r-.')
plt.colorbar()
plt.show()

#%% Operate on small dataset

# s1: reduce lat, lon
lat_flatten = lat.flatten()
lon_flatten = lon.flatten()

ind_reduced = np.where((lat_flatten <= 41.14) *
                       (lon_flatten <= -8.675) *
                       (lon_flatten >= -8.75))[0]

depth_flatten = depth[:, :, :, 0].flatten()
sal_flatten = sal_data[:, :, :, 0].flatten()

lat_reduced = lat_flatten[ind_reduced]
lon_reduced = lon_flatten[ind_reduced]
depth_reduced = depth_flatten[ind_reduced]
sal_reduced = sal_flatten[ind_reduced]

# s2: reduce depth
ind_reduced2 = np.where(depth_reduced > -6.)[0]

lat_reduced2 = lat_reduced[ind_reduced2]
lon_reduced2 = lon_reduced[ind_reduced2]
depth_reduced2 = depth_reduced[ind_reduced2]
sal_reduced2 = sal_reduced[ind_reduced2]

#%%
# plt.plot(depth[0, 1, 1, :])
# plt.show()
plt.plot(id1)
plt.show()


#%%
import time
from scipy.spatial.distance import cdist
from usr_func.vectorize import vectorize
from WGS import WGS

grid = pd.read_csv("grid.csv").to_numpy()
xgrid, ygrid = WGS.latlon2xy(grid[:, 0], grid[:, 1])
zgrid = grid[:, 2]


# for t in range(sal_data.shape[3]):
t = 0
# s1: flatten
depth_unique = np.nanmean(np.nanmean(np.nanmean(depth[:, :, :, :], axis=1), axis=1), axis=1)

# ind_depth = np.argmin(np.abs(zgrid[i] - depth_unique))
ind_depth = np.zeros_like(zgrid)
for i in range(zgrid.shape[0]):
    print(i)
    ind_depth[i] = np.nanargmin((zgrid[i] - depth_unique)**2)

#%%

sal_aggr = np.zeros_like(xgrid)
t1 = time.time()
for i in range(xgrid.shape[0]):
    print(i)

    lat_reduced = lat_flatten[ind_reduced]
    lon_reduced = lon_flatten[ind_reduced]

    xdata, ydata = WGS.latlon2xy(lat_reduced, lon_reduced)
    sal_flatten = sal_data[ind_depth, :, :, t].flatten()

    # depth_reduced = depth_flatten[ind_reduced]
    sal_reduced = sal_flatten[ind_reduced]
    ind_aggr = np.nanargmin((xgrid[i] - xdata)**2 +
                            (ygrid[i] - ydata)**2)
    sal_aggr[i] = sal_reduced[int(ind_aggr)]
t2 = time.time()
print("Time consumed: ", t2 - t1)


# if t > 1:
#     break


#%%c
from matplotlib.cm import get_cmap
ind_surface = np.where(zgrid == -0.5)[0]

plt.scatter(ygrid[ind_surface], xgrid[ind_surface], c=sal_aggr[ind_surface],
            cmap=get_cmap("BrBG", 10), vmin=10, vmax=36)
plt.colorbar()
plt.show()

# %%
# ind_surface = np.where(depth_reduced3 >= -.5)[0]
# plt.scatter(lon_reduced3[ind_surface], lat_reduced3[ind_surface],
#             c=sal_reduced3[ind_surface], cmap="RdBu", vmin=10, vmax=36)
# plt.plot(grid[:, 1], grid[:, 0], 'k.', markersize=1,alpha=.1)
# plt.colorbar()
# plt.show()

#%%

# s1:
# lat_flatten <= 41.14) *
#                        (lon_flatten <= -8.675) *
#                        (lon_flatten >= -8.75))[0

ind1, ind2, ind3 = np.where((lat < 41.14) *
                            (lon <= -8.675) *
                            (lon >= -8.75))

lat_reduced = lat[:, ind2, ind3]

#%%
plt.plot(depth[2, 123, 240, :])
plt.show()

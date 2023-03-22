import h5py
import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import cuda
import pandas as pd
import plotly
import plotly.graph_objects as go
import time
from matplotlib.cm import get_cmap
from scipy.spatial.distance import cdist
from WGS import WGS

print("hello")
import concurrent.futures

datapath = "data/"

def vectorize(v):
    return np.array(v).reshape(-1, 1)



path = "raw/Nov2016_sal_1.mat"

t1 = time.time()
data = h5py.File(path, 'r')
t2 = time.time()
print("Time consumed: ", t2 - t1)


data = data.get('data')
lon = np.array(data.get("X")).squeeze()
lat = np.array(data.get("Y")).squeeze()
depth = np.array(data.get("Z"))
Time = np.array(data.get('Time'))
timestamp = (Time - 719529) * 24 * 3600  # 719529 is how many days have passed from Jan1 0,
# to Jan1 1970. Since 1970Jan1, is used as the starting index for datetime
sal_data = np.array(data["Val"])

grid = pd.read_csv("grid.csv").to_numpy()
xgrid, ygrid = WGS.latlon2xy(grid[:, 0], grid[:, 1])
xgrid = xgrid.reshape(-1, 1).astype(np.float32)
ygrid = ygrid.reshape(-1, 1).astype(np.float32)
zgrid = grid[:, 2].reshape(-1, 1)



ind_depth, ind_row, ind_col = np.where((lat <= 41.14) *
                                       (lon <= -8.675) *
                                       (lon >= -8.75))



def get_data(t=0):
    # s1: loop timestamp
    # t = 1

    # s2: sort depth
    depth_unique = np.nanmean(np.nanmean(depth[:, :, :, t], axis=1), axis=1).reshape(-1, 1)

    # s3: according to depth, select closest layer, then select lat lon
    depth_grid = grid[:, 2].reshape(-1, 1)
    dm_depth = cdist(depth_grid, depth_unique)
    ind_depth = np.argmin(dm_depth, axis=1)

    # s4: then merge them together using reduced dataset
    lat_r = lat[0, ind_row, ind_col]
    lon_r = lon[0, ind_row, ind_col]
    sal_r = sal_data[:, ind_row, ind_col, t]

    xdata, ydata = WGS.latlon2xy(lat_r, lon_r)
    xdata = xdata.reshape(-1, 1).astype(np.float32)
    ydata = ydata.reshape(-1, 1).astype(np.float32)

    sal_total = np.empty([0, 1])

    N = 5000
    shorten = False

    for i in range(0, len(xgrid), N):
        t1 = time.time()
        ind_start = i 
        if i + N > len(xgrid):
            ind_end = len(xgrid)
            shorten = True
        else: 
            ind_end = i + N
        print(ind_start, ind_end)

        xv = xgrid[ind_start: ind_end].reshape(-1, 1)
        yv = ygrid[ind_start: ind_end].reshape(-1, 1)

        dm_x = cdist(xv, xdata, 'sqeuclidean')
        print("dm_x: ", dm_x.shape)
        dm_y = cdist(yv, ydata, 'sqeuclidean')
        print("dm_y: ", dm_y.shape)
        dm = dm_x + dm_y

        ind_min = np.argmin(dm, axis=1)
        if shorten is True:
            n = len(xv)
        else: 
            n = N
        sal_temp = np.zeros([n, 1])
        for j in range(n):
            sal_temp[j] = sal_r[ind_depth[ind_start+j], ind_min[j]]

        sal_total = np.append(sal_total, sal_temp)

        t2 = time.time()
        print("Time consumed: ", t2 - t1)

    dataset = np.hstack((xgrid, ygrid, zgrid, sal_total.reshape(-1, 1)))
    df = pd.DataFrame(dataset, columns=['x', 'y', 'z', 'salinity'])
    df.to_csv(datapath + "D_{:03d}.csv".format(t))


get_data(0)


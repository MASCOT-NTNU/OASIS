import mat73
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import time
from datetime import datetime
import h5py

import WGS

datapath = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Delft3D/Nov2016_sal_1.mat"
t1 = time.time()
data = mat73.loadmat(datapath)
# data = h5py.File(datapath, 'r')
data = data['data']
t2 = time.time()

print("data loading: ", t2 - t1)

#%%
lon = data["X"]
lat = data["Y"]
depth =data["Z"]
Time = data['Time']
timestamp_data = (Time - 719529) * 24 * 3600  # 719529 is how many days have passed from Jan1 0,
# to Jan1 1970. Since 1970Jan1, is used as the starting index for datetime
sal_data = data["Val"]
string_date = datetime.fromtimestamp(timestamp_data[0]).strftime("%Y_%m")

#%%
grid_delft3d = []
# for t in range(sal_data.shape[0]):
for t in range(1):
    for i in range(sal_data.shape[1]):
        print(i)
        for j in range(sal_data.shape[2]):
            for k in range(sal_data.shape[3]):
                grid_delft3d.append([lat[i, j, k], lon[i, j, k], depth[t, i, j, k], sal_data[t, i, j, k]])

#%%
grid_delft3d = np.array(grid_delft3d)

#%%
from WGS import WGS

xd, yd = WGS.latlon2xy(grid_delft3d[:, 0], grid_delft3d[:, 1])

#%%
from scipy.spatial.distance import cdist

grid = pd.read_csv("grid.csv").to_numpy()
xg, yg = WGS.latlon2xy(grid[:, 0], grid[:, 1])

#%%
from usr_func.vectorize import vectorize
t1 = time.time()
dmx = cdist(vectorize(xg), vectorize(xd))
t2 = time.time()
print("Distance matrix for x: ", t2 - t1)

#%%
import numpy as np
import pandas as pd
from WGS import WGS

grid = pd.read_csv("grid.csv").to_numpy()

x, y = WGS.latlon2xy(grid[:, 0], grid[:, 1])

print(np.diff(x))
print(np.diff(y))






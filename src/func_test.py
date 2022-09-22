import mat73
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import time
from datetime import datetime

import WGS

datapath = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Delft3D/Nov2016_sal_1.mat"
t1 = time.time()
data = mat73.loadmat(datapath)
data = data["data"]
t2 = time.time()

print("data loading: ", t2 - t1)

#%%
lon = data["X"]
lat = data["Y"]
depth = data["Z"]
Time = data['Time']
timestamp_data = (Time - 719529) * 24 * 3600  # 719529 is how many days have passed from Jan1 0,
# to Jan1 1970. Since 1970Jan1, is used as the starting index for datetime
sal_data = data["Val"]
string_date = datetime.fromtimestamp(timestamp_data[0]).strftime("%Y_%m")
#%%
import pandas as pd
path = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Projects/OASIS/OperationArea.csv"
plg = pd.read_csv(path).to_numpy()

#%%
figpath = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Projects/OASIS/fig/Delft3D/"

d_ave = np.mean(depth, axis=0)
sal_ave = np.mean(sal_data, axis=0)

for i in range(sal_data.shape[0]):
    print(i)
    plt.figure(figsize=(10, 10))
    plt.scatter(lon[:, :, 0], lat[:, :, 0], c=sal_data[i, :, :, 0], cmap=get_cmap("BrBG", 10), vmin=10, vmax=35)
    plt.colorbar()
    plt.plot(plg[:, 1], plg[:, 0], 'r-.')
    plt.xlabel("Lon")
    plt.ylabel("Lat")
    plt.title("Timestamp: {:d}".format(i))
    plt.savefig(figpath+"P_{:03d}.png".format(i))
    plt.close("all")
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

b1 = [41.065, -8.74]
b2 = [41.053, -8.814]

dx1 = 8300
dy1 = 4800

dx2 = 5100
dy2 = 6200

bb1 = np.array([[0, 0],
                [dy1, 0],
                [dy1, dx1],
                [0, dx1]])

bb2 = np.array([[0, 0],
                [dy2, 0],
                [dy2, dx2],
                [0, dx2]])

from WGS import WGS

lat1, lon1 = WGS.xy2latlon_with_origin(bb1[:, 1], bb1[:, 0], b1[0], b1[1])
lat2, lon2 = WGS.xy2latlon_with_origin(bb2[:, 1], bb2[:, 0], b2[0], b2[1])

box1 = np.stack((lat1, lon1), axis=1)
box2 = np.stack((lat2, lon2), axis=1)

# box1 = np.array([[41.065, -8.74],
#                  [41.065, -8.682],
#                  [41.140, -8.682],
#                  [41.140, -8.74],
#                  [41.065, -8.74]])
#
# box2 = np.array([[41.053, -8.814],
#                  [41.053, -8.74],
#                  [41.099, -8.74],
#                  [41.099, -8.814],
#                  [41.053, -8.814]])

plt.figure(figsize=(10, 10))
# plt.scatter(lon[:, :, 0], lat[:, :, 0], c=sal_ave[:, :, 0], cmap=get_cmap("BrBG", 10), vmin=10, vmax=35)
# plt.colorbar()
plt.plot(plg[:, 1], plg[:, 0], 'r-.')
plt.plot(lon1, lat1, 'b-.')
plt.plot(lon2, lat2, 'y-.')
# plt.plot(box1[:, 1], box1[:, 0], 'b-.')
# plt.plot(box2[:, 1], box2[:, 0], 'y-.')
plt.xlabel("Lon")
plt.ylabel("Lat")
plt.show()

df_b1 = pd.DataFrame(box1, columns=['lat', 'lon'])
df_b2 = pd.DataFrame(box2, columns=['lat', 'lon'])
df_b1.to_csv("Region1.csv", index=False)
df_b2.to_csv("Region2.csv", index=False)

#%%
from WGS import WGS
x, y = WGS.latlon2xy(box2[:, 0], box2[:, 1])

grid = np.stack((x, y), axis=1)
from scipy.spatial.distance import cdist

dm = cdist(grid, grid)
plt.imshow(dm)
plt.colorbar()
plt.show()

#%%
print(np.diff(x))
print(np.diff(y))

#%%
id = 6
plt.scatter(lon[:, :, id], lat[:, :, id], c=d_ave[:, :, id], cmap=get_cmap("BrBG", 10), vmin=-8.0, vmax=.0)
plt.plot(box1[:, 1], box1[:, 0], 'b-.')
plt.plot(box2[:, 1], box2[:, 0], 'y-.')
plt.colorbar()
plt.show()


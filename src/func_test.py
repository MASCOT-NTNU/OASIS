import mat73
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import time
from datetime import datetime


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
box1 = np.array([[41.065, -8.74],
                 [41.065, -8.682],
                 [41.140, -8.682],
                 [41.140, -8.74],
                 [41.065, -8.74]])

box2 = np.array([[41.053, -8.814],
                 [41.053, -8.74],
                 [41.099, -8.74],
                 [41.099, -8.814],
                 [41.053, -8.814]])

plt.figure(figsize=(10, 10))
plt.scatter(lon[:, :, 0], lat[:, :, 0], c=sal_ave[:, :, 0], cmap=get_cmap("BrBG", 10), vmin=10, vmax=35)
plt.colorbar()
plt.plot(plg[:, 1], plg[:, 0], 'r-.')
plt.plot(box1[:, 1], box1[:, 0], 'b-.')
plt.plot(box2[:, 1], box2[:, 0], 'y-.')
plt.xlabel("Lon")
plt.ylabel("Lat")
plt.show()

df_b1 = pd.DataFrame(box1, columns=['lat', 'lon'])
df_b2 = pd.DataFrame(box2, columns=['lat', 'lon'])
df_b1.to_csv("Region1.csv", index=False)
df_b2.to_csv("Region2.csv", index=False)






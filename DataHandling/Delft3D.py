import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import time

t1 = time.time()
path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Delft3D/raw/Nov/Nov2016_sal_1.mat"
data = h5py.File(path, 'r')
print("s1")
data = data.get('data')
print("s2")
lon = np.array(data.get("X")).squeeze()
print("s3")
lat = np.array(data.get("Y")).squeeze()
print("s4")
depth = np.array(data.get("Z"))
print("s5")
Time = np.array(data.get('Time'))
print("s6")
timestamp = (Time - 719529) * 24 * 3600  # 719529 is how many days have passed from Jan1 0,
# to Jan1 1970. Since 1970Jan1, is used as the starting index for datetime
sal_data = np.array(data["Val"])
t2 = time.time()
print("Finished data loading, time takes: ", t2 - t1)

plt.scatter(lon[0, :, :], lat[0, :, :], c=sal_data[0, :, :, 0],
            cmap=get_cmap("BrBG", 10), vmin=10, vmax=36)
plt.colorbar()
plt.show()




""" Generate mask indices to remove unnecessary data points from Delft3D """

import numpy as np
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from datetime import datetime

plg_path = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Projects/OASIS/OperationArea.csv"

plg = pd.read_csv(plg_path).to_numpy()

# plt.plot(plg[:, 1], plg[:, 0], 'k.-')
# plt.show()

datapath = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Delft3D/Nov2016_sal_1.mat"
df = h5py.File(datapath, 'r')
data = df.get('data')
lon = np.array(data.get("X")).squeeze()
lat = np.array(data.get("Y")).squeeze()
# depth =data.get("Z")
# Time = data.get('Time')
# timestamp_data = (Time - 719529) * 24 * 3600  # 719529 is how many days have passed from Jan1 0,
# # to Jan1 1970. Since 1970Jan1, is used as the starting index for datetime
# sal_data = data["Val"]
# string_date = datetime.fromtimestamp(timestamp_data[0]).strftime("%Y_%m")

border = Polygon(plg)
ind_valid = []
nd = lat.shape[0]
ns1 = lat.shape[1]
ns2 = lat.shape[2]

for i in range(nd):
    for j in range(ns1):
        for k in range(ns2):
            point = Point(lat[i, j, k], lon[i, j, k])
            if border.contains(point):
                ind_valid.append(True)
            else:
                ind_valid.append(False)





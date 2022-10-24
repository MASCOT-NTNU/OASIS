import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly
import plotly.graph_objects as go
import time
from matplotlib.cm import get_cmap
from scipy.spatial.distance import cdist
from WGS import WGS
import os
from datetime import datetime
import concurrent.futures
from shapely.geometry import Polygon, Point
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec
import pickle
from scipy.interpolate import interp2d

def vectorize(v):
    return np.array(v).reshape(-1, 1)

# s1: load operation area
plg_op = pd.read_csv("OPA.csv").to_numpy()
plg_op_shapely = Polygon(plg_op)
def is_point_legal(lat, lon):
    point = Point(lat, lon)
    return plg_op_shapely.contains(point)

lat_origin =  41.11041842
lon_origin = -8.42446588
wg = WGS()
wg.set_origin(lat_origin, lon_origin)
x, y = wg.latlon2xy(plg_op[:, 0], plg_op[:, 1])

N = 10000000

# def get_random_location(N):
#     for i in range(N):
xmin, ymin = map(np.amin, [x, y])
xmax, ymax = map(np.amax, [x, y])

plg_xy = np.stack((x, y), axis=1)
plg_shapely = Polygon(plg_xy)

def get_random_locations(n):
    locations = []
    for i in range(n):
        xr = np.random.uniform(xmin, xmax)
        yr = np.random.uniform(ymin, ymax)
        point = Point(xr, yr)
        if plg_shapely.contains(point):
            locations.append([xr, yr])
    locs = np.array(locations)
    print("Acceptance rate: ", len(locs) / N)
    print("Total {:d} random locations are generated!".format(len(locs)))
    print("Can be used for {:d} iterations in RRT* planning".format(int(len(locs)/2000)))
    return locs

TT = 800000 * np.ones(12).astype(int)
res = Parallel(n_jobs=6)(delayed(get_random_locations)(tt) for tt in TT)

rest = np.empty([0, 2])
for i in range(len(res)):
    rest = np.append(rest, res[0], axis=0)

plt.figure(figsize=(100, 100))
plt.plot(plg_xy[:, 1], plg_xy[:, 0], 'r-.')
# plt.plot(locs[:, 1], locs[:, 0], 'k.', alpha=.1)
plt.plot(rest[:, 1], rest[:, 0], 'k.', alpha=.1)


indices = np.random.uniform(0, 1., len(rest))

np.save("RRT_Random_Locations.npy", rest)
np.save("Goal_indices.npy", indices)

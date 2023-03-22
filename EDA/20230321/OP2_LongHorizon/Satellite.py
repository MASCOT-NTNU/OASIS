

# %%
"""
Section for Satellite
"""
import matplotlib.pyplot as plt
import rasterio
import numpy as np
import pandas as pd
from rasterio.plot import show, adjust_band
from shapely.geometry import Polygon, Point
plg = pd.read_csv("csv/polygon.csv").to_numpy()
plg_sh = Polygon(plg)

@np.vectorize
def get_legal_indices(lat, lon) -> bool:
    point = Point(lat, lon)
    legal = False
    if plg_sh.contains(point):
        legal = True
    return legal


file = "/Users/yaolin/Library/CloudStorage/OneDrive-NTNU/MASCOT_PhD/Data/Porto/20221111/satellite/satellite_opa.tif"
img = rasterio.open(file)
# img = georaster.MultiBandRaster(file)
b1 = img.read(1)
b2 = img.read(2)
b3 = img.read(3)

print("hello")
height = b1.shape[0]
width = b1.shape[1]
cols, rows = np.meshgrid(np.arange(width), np.arange(height))
xs, ys = rasterio.transform.xy(img.transform, rows, cols)
lons = np.array(xs).flatten()
lats = np.array(ys).flatten()

b1 = b1.flatten()
b2 = b2.flatten()
b3 = b3.flatten()

ind_legal = get_legal_indices(lats, lons)

#%%
bimg = b1 + b2 + b3
plt.scatter(lons[ind_legal], lats[ind_legal], c=bimg[ind_legal])
plt.colorbar()
plt.show()
"""
EDA handles the explorative data analysis for the mascot mission field work.
"""

from Field import Field
from GRF.GRF import GRF
import numpy as np
import pandas as pd
import os
from datetime import datetime
from matplotlib.cm import get_cmap
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from shapely.geometry import Polygon, Point
from matplotlib import tri


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20

field = Field()
grf = GRF()

#%%

""" Step I """
folderpath = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Porto/20221111/backseat/src/GRF/data/"

# c1: load conditional mean
path_mu = folderpath + "mu/"
files_mu = os.listdir(path_mu)
files_mu.sort()

# c2: load raw ctd
path_raw = folderpath + "raw_ctd/1668163814/"
files_raw = os.listdir(path_raw)
files_raw.sort()

# c3: load conditional variance
path_cov = folderpath + "Sigma/"
files_cov = os.listdir(path_cov)
files_cov.sort()

# set threshold
threshold = 30


#%%


def get_ep(mu, sigma_diag, threshold) -> np.ndarray:
    p = norm.cdf(threshold, mu, sigma_diag)
    return p


def is_masked(x, y) -> bool:
    loc = np.array([x, y])
    masked = False
    if not field.border_contains(loc):
        masked = True
    return masked


def plotf_vector(xplot, yplot, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10),
                 cbar_title='test', colorbar=True, vmin=None, vmax=None, ticks=None,
                 stepsize=None, threshold=None, polygon_border=None,
                 polygon_obstacle=None, xlabel=None, ylabel=None):
    """ Note for triangulation:
    - Maybe sometimes it cannot triangulate based on one axis, but changing to another axis might work.
    - So then the final output needs to be carefully treated so that it has the correct visualisation.
    - Also note, the floating point number can cause issues as well.
    """
    """ To show threshold as a red line, then vmin, vmax, stepsize, threshold needs to have values. """
    triangulated = tri.Triangulation(yplot, xplot)
    x_triangulated = xplot[triangulated.triangles].mean(axis=1)
    y_triangulated = yplot[triangulated.triangles].mean(axis=1)

    ind_mask = []
    for i in range(len(x_triangulated)):
        ind_mask.append(is_masked(y_triangulated[i], x_triangulated[i]))
    triangulated.set_mask(ind_mask)
    refiner = tri.UniformTriRefiner(triangulated)
    triangulated_refined, value_refined = refiner.refine_field(values.flatten(), subdiv=3)

    """ extract new x and y, refined ones. """
    xre_plot = triangulated_refined.x
    yre_plot = triangulated_refined.y

    ax = plt.gca()
    # ax.triplot(triangulated, lw=0.5, color='white')
    if np.any([vmin, vmax]):
        levels = np.arange(vmin, vmax, stepsize)
    else:
        levels = None
    if np.any(levels):
        linewidths = np.ones_like(levels) * .3
        colors = len(levels) * ['black']
        if threshold:
            dist = np.abs(threshold - levels)
            ind = np.where(dist == np.amin(dist))[0]
            linewidths[ind] = 10
            colors[ind[0]] = 'red'
        contourplot = ax.tricontourf(yre_plot, xre_plot, value_refined, levels=levels, cmap=cmap, alpha=alpha)
        ax.tricontour(yre_plot, xre_plot, value_refined, levels=levels, linewidths=linewidths, colors=colors,
                      alpha=alpha)
    else:
        contourplot = ax.tricontourf(yre_plot, xre_plot, value_refined, cmap=cmap, alpha=alpha)
        ax.tricontour(yre_plot, xre_plot, value_refined, vmin=vmin, vmax=vmax, alpha=alpha)

    if colorbar:
        cbar = plt.colorbar(contourplot, ax=ax, ticks=ticks)
        cbar.ax.set_title(cbar_title)

    ax.set_title(title)

    return ax, yre_plot, xre_plot, value_refined

grid = field.get_grid()
plg = field.get_polygon_border()

figpath = os.getcwd() + "/../../fig/OP2_LongHorizon/Experiment/"

"""
Plot prior field
"""
mu = grf.get_mu()
cov = grf.get_Sigma()

fig = plt.figure(figsize=(35, 15))
gs = GridSpec(nrows=1, ncols=3)
ax = fig.add_subplot(gs[0])
plotf_vector(grid[:, 1], grid[:, 0], mu, title="Prior salinity field", cmap=get_cmap("BrBG", 10),
             vmin=10, vmax=36, cbar_title="Salinity", stepsize=1.5, threshold=threshold)
ax.plot(plg[:, 1], plg[:, 0], 'r-.')
ax.set_aspect("equal")

ax = fig.add_subplot(gs[1])
p = get_ep(mu.flatten(), np.diag(cov), threshold)
plotf_vector(grid[:, 1], grid[:, 0], p, title="Prior excursion probability field", cmap=get_cmap("YlGnBu", 10),
             vmin=-.1, vmax=1.3, cbar_title="Prob", stepsize=.45, threshold=.5)
ax.plot(plg[:, 1], plg[:, 0], 'r-.')
ax.set_aspect("equal")

ax = fig.add_subplot(gs[2])
plotf_vector(grid[:, 1], grid[:, 0], np.sqrt(np.diag(cov)), title="Prior uncertainty field",
             cmap=get_cmap("RdBu", 10), vmin=0, vmax=1, cbar_title="STD")
ax.plot(plg[:, 1], plg[:, 0], 'r-.')
ax.set_aspect("equal")

plt.savefig(figpath + "P_000.png")
plt.close("all")


df_raw = np.empty([0, 4])
# for i in range(len(files_mu)):
for i in [len(files_mu)-1]:
    print(i)
    file_mu = files_mu[i]
    file_raw = files_raw[i]
    file_cov = files_cov[i]
    mu = np.load(path_mu + file_mu)
    cov = np.load(path_cov + file_cov)
    raw = pd.read_csv(path_raw + file_raw).to_numpy()
    df_raw = np.append(df_raw, raw, axis=0)

    fig = plt.figure(figsize=(35, 15))
    gs = GridSpec(nrows=1, ncols=3)
    ax = fig.add_subplot(gs[0])
    axes, x, y, z_mu = plotf_vector(grid[:, 1], grid[:, 0], mu, title="Updated salinity field", cmap=get_cmap("BrBG", 10),
                 vmin=10, vmax=36, cbar_title="Salinity", stepsize=1.5, threshold=threshold)
    #     im = ax.scatter(grid[:, 1], grid[:, 0], c=mu, s=200, cmap=get_cmap("BrBG", 10), vmin=10, vmax=33)
    ax.plot(df_raw[:, 1], df_raw[:, 0], 'k.-', linewidth=10)
    ax.plot(plg[:, 1], plg[:, 0], 'r-.')
    ax.set_aspect("equal")
    #     plt.colorbar(im)

    ax = fig.add_subplot(gs[1])
    p = get_ep(mu.flatten(), np.diag(cov), threshold)
    axes, x, y, z_p = plotf_vector(grid[:, 1], grid[:, 0], p, title="Updated excursion probability field", cmap=get_cmap("YlGnBu", 10),
                 vmin=-.1, vmax=1.3, cbar_title="Prob", stepsize=.45, threshold=.5)
    ax.plot(df_raw[:, 1], df_raw[:, 0], 'k.-', linewidth=10)
    ax.plot(plg[:, 1], plg[:, 0], 'r-.')
    ax.set_aspect("equal")

    ax = fig.add_subplot(gs[2])
    axes, x, y, z_std = plotf_vector(grid[:, 1], grid[:, 0], np.sqrt(np.diag(cov)), title="Updated uncertainty field",
                 cmap=get_cmap("RdBu", 10), vmin=0, vmax=1, cbar_title="STD")
    ax.plot(df_raw[:, 1], df_raw[:, 0], 'k.-', linewidth=10)
    ax.plot(plg[:, 1], plg[:, 0], 'r-.')
    ax.set_aspect("equal")

    plt.show()
    # plt.savefig(figpath + "P_{:03d}.png".format(i+1))
    # plt.close("all")



    # if i > 5:
    #     break
print("Finished plotting")

#%%
from WGS import WGS
lat, lon = WGS.xy2latlon(y, x)
# grid_wgs = np.stack((lat, lon), axis=1)
dataset_mu = np.stack((lat, lon, z_mu), axis=1)
dataset_cov = np.stack((lat, lon, z_std), axis=1)
dataset_ep = np.stack((lat, lon, z_p), axis=1)

df_mu = pd.DataFrame(dataset_mu, columns=['lat', 'lon', 'salinity'])
df_cov = pd.DataFrame(dataset_cov, columns=['lat', 'lon', 'std'])
df_ep = pd.DataFrame(dataset_ep, columns=['lat', 'lon', 'probability'])

df_mu.to_csv("csv/df_mu.csv", index=False)
df_cov.to_csv("csv/df_std.csv", index=False)
df_ep.to_csv("csv/df_ep.csv", index=False)

#%%
import matplotlib
cmap = get_cmap('YlGnBu', 3)    # PiYG

for i in range(cmap.N):
    rgba = cmap(i)
    # rgb2hex accepts rgb or rgba
    print(matplotlib.colors.rgb2hex(rgba))


#%%
"""
Section for Satellite
"""

import rasterio
import georaster
file = "/Users/yaolin/Library/CloudStorage/OneDrive-NTNU/MASCOT_PhD/Data/Porto/20221111/satellite/20221110_qgis.tif"
# img = rasterio.open(file)
img = georaster.MultiBandRaster(file)
# b1 = img.read(1)
# b2 = img.read(2)
# b3 = img.read(3)

# plt.imshow(b3)
img.plot()
plt.show()



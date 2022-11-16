"""
EDA handles the explorative data analysis for the mascot mission field work.
"""

from Field import Field
from GRF.GRF import GRF
from WGS import WGS
from usr_func.normalize import normalize
import numpy as np
import pandas as pd
import os
import time
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

# %%

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


# %%

lat, lon = WGS.xy2latlon(plg[:, 0], plg[:, 1])
plg_wgs = np.stack((lat, lon), axis=1)
plg_wgs_sh = Polygon(plg_wgs)
#%%

def get_ep(mu, sigma_diag, threshold) -> np.ndarray:
    p = norm.cdf(threshold, mu, sigma_diag)
    return p

def is_masked(x, y) -> bool:
    point = Point(x, y)
    # loc = np.array([x, y])
    masked = False
    if not plg_wgs_sh.contains(point):
    # if not field.border_contains(loc):
        masked = True
    return masked

def get_ibv(mu, sigma_diag, threshold):
    p = norm.cdf(threshold, mu, sigma_diag)
    bv = p * (1 - p)
    ibv = np.sum(bv)
    return ibv

def get_ei_field(mu, Sigma, threshold) -> tuple:
    t1 = time.time()
    Ngrid = len(mu)
    nugget = .04
    eibv_field = np.zeros([Ngrid])
    ivr_field = np.zeros([Ngrid])
    for i in range(Ngrid):
        SF = Sigma[:, i].reshape(-1, 1)
        MD = 1 / (Sigma[i, i] + nugget)
        VR = SF @ SF.T * MD
        SP = Sigma - VR
        sigma_diag = np.diag(SP).reshape(-1, 1)
        eibv_field[i] = get_ibv(mu, sigma_diag, threshold)
        ivr_field[i] = np.sum(np.diag(VR))
    eibv_field = normalize(eibv_field)
    ivr_field = 1 - normalize(ivr_field)
    t2 = time.time()
    print("Total EI field takes: ", t2 - t1, " seconds.")
    return eibv_field, ivr_field


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

lat, lon = WGS.xy2latlon(grid[:, 0], grid[:, 1])
grid = np.stack((lat, lon), axis=1)

mu_prior = grf.get_mu()
# cov = grf.get_Sigma()

df_raw = np.empty([0, 4])
for i in range(len(files_mu)):
# for i in [len(files_mu) - 1]:
    print(i)
    file_mu = files_mu[i]
    file_raw = files_raw[i]
    file_cov = files_cov[i]
    mu = np.load(path_mu + file_mu)
    cov = np.load(path_cov + file_cov)
    raw = pd.read_csv(path_raw + file_raw).to_numpy()
    df_raw = np.append(df_raw, raw, axis=0)

    lat, lon = WGS.xy2latlon(df_raw[:, 0], df_raw[:, 1])
    df_app = np.stack((lat, lon), axis=1)

    fig = plt.figure(figsize=(40, 30))
    gs = GridSpec(nrows=2, ncols=3)


    ax = fig.add_subplot(gs[0, 0])
    axes, x, y, z_mu_prior = plotf_vector(grid[:, 1], grid[:, 0], mu_prior, title="Prior salinity field",
                                    cmap=get_cmap("BrBG", 10),
                                    vmin=10, vmax=36, cbar_title="Salinity", stepsize=1.5, threshold=threshold)
    #     im = ax.scatter(grid[:, 1], grid[:, 0], c=mu, s=200, cmap=get_cmap("BrBG", 10), vmin=10, vmax=33)
    ax.plot(df_app[:, 1], df_app[:, 0], 'k.-', linewidth=10)
    ax.plot(plg_wgs[:, 1], plg_wgs[:, 0], 'r-.')
    ax.set_aspect("equal")

    eibv, ivr = get_ei_field(mu, cov, threshold)

    ax = fig.add_subplot(gs[0, 1])
    axes, x, y, z_eibv = plotf_vector(grid[:, 1], grid[:, 0], eibv, title="EIBV cost field",
                                    cmap=get_cmap("RdYlGn_r", 10), vmin=0, vmax=1, cbar_title="Cost")
    #     im = ax.scatter(grid[:, 1], grid[:, 0], c=mu, s=200, cmap=get_cmap("BrBG", 10), vmin=10, vmax=33)
    ax.plot(df_app[:, 1], df_app[:, 0], 'k.-', linewidth=10)
    ax.plot(plg_wgs[:, 1], plg_wgs[:, 0], 'r-.')
    ax.set_aspect("equal")

    ax = fig.add_subplot(gs[0, 2])
    axes, x, y, z_ivr = plotf_vector(grid[:, 1], grid[:, 0], ivr, title="IVR cost field",
                                    cmap=get_cmap("RdYlGn_r", 10), vmin=0, vmax=1, cbar_title="Cost")
    #     im = ax.scatter(grid[:, 1], grid[:, 0], c=mu, s=200, cmap=get_cmap("BrBG", 10), vmin=10, vmax=33)
    ax.plot(df_app[:, 1], df_app[:, 0], 'k.-', linewidth=10)
    ax.plot(plg_wgs[:, 1], plg_wgs[:, 0], 'r-.')
    ax.set_aspect("equal")


    ax = fig.add_subplot(gs[1, 0])
    axes, x, y, z_mu = plotf_vector(grid[:, 1], grid[:, 0], mu, title="Updated salinity field",
                                    cmap=get_cmap("BrBG", 10),
                                    vmin=10, vmax=36, cbar_title="Salinity", stepsize=1.5, threshold=threshold)
    #     im = ax.scatter(grid[:, 1], grid[:, 0], c=mu, s=200, cmap=get_cmap("BrBG", 10), vmin=10, vmax=33)
    ax.plot(df_app[:, 1], df_app[:, 0], 'k.-', linewidth=10)
    ax.plot(plg_wgs[:, 1], plg_wgs[:, 0], 'r-.')
    ax.set_aspect("equal")
    #     plt.colorbar(im)

    ax = fig.add_subplot(gs[1, 1])
    p = get_ep(mu.flatten(), np.diag(cov), threshold)
    axes, x, y, z_p = plotf_vector(grid[:, 1], grid[:, 0], p, title="Updated excursion probability field",
                                   cmap=get_cmap("YlGnBu", 10),
                                   vmin=-.1, vmax=1.3, cbar_title="Prob", stepsize=.45, threshold=.5)
    ax.plot(df_app[:, 1], df_app[:, 0], 'k.-', linewidth=10)
    ax.plot(plg_wgs[:, 1], plg_wgs[:, 0], 'r-.')
    ax.set_aspect("equal")

    ax = fig.add_subplot(gs[1, 2])
    axes, x, y, z_std = plotf_vector(grid[:, 1], grid[:, 0], np.sqrt(np.diag(cov)), title="Updated uncertainty field",
                                     cmap=get_cmap("RdBu", 10), vmin=0, vmax=1, cbar_title="STD")
    ax.plot(df_app[:, 1], df_app[:, 0], 'k.-', linewidth=10)
    ax.plot(plg_wgs[:, 1], plg_wgs[:, 0], 'r-.')
    ax.set_aspect("equal")

    # plt.show()
    plt.savefig(figpath + "P_{:03d}.png".format(i+1))
    plt.close("all")

    # if i > 5:
    #     break
print("Finished plotting")

# %%
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

# %%
import matplotlib

cmap = get_cmap('YlGnBu', 3)  # PiYG

for i in range(cmap.N):
    rgba = cmap(i)
    # rgb2hex accepts rgb or rgba
    print(matplotlib.colors.rgb2hex(rgba))

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

#%%
"""
MOHID section
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_mohid = pd.read_csv("csv/df_mohid.csv").to_numpy()


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

    # ind_mask = []
    # for i in range(len(x_triangulated)):
    #     ind_mask.append(is_masked(y_triangulated[i], x_triangulated[i]))
    # triangulated.set_mask(ind_mask)
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


# plt.scatter(lon_mohid, lat_mohid, c=sal_mohid, cmap=get_cmap("BrBG", 10), vmin=10, vmax=36)
plotf_vector(data_mohid[:, 1], data_mohid[:, 0], data_mohid[:, 2], title="MOHID", cmap=get_cmap("BrBG", 10),
             vmin=10, vmax=36, cbar_title="Salinity", stepsize=1.5, threshold=30)
plt.plot(plg[:, 1], plg[:, 0], 'r-.')
plt.show()

#%%
"""
Combine three data sources (AUV, MOHID, Satellite)
"""

from WGS import WGS
folderpath = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Porto/20221111/backseat/src/GRF/data/"
# c1: load conditional mean
path_mu = folderpath + "mu/"
files_mu = os.listdir(path_mu)
files_mu.sort()
data_mu = np.load(path_mu + files_mu[-1])
grid = field.get_grid()
lat, lon = WGS.xy2latlon(grid[:, 0], grid[:, 1])

fig = plt.figure(figsize=(50, 20))
gs = GridSpec(nrows=1, ncols=3)

# p1: AUV
ax = fig.add_subplot(gs[0])
plotf_vector(lon, lat, data_mu, title="Updated field", cmap=get_cmap("BrBG", 10),
             vmin=10, vmax=36, cbar_title="Salinity", stepsize=1.5, threshold=30)
ax.plot(plg[:, 1], plg[:, 0], 'r-.')
ax.set_aspect("equal")
ax.set_xlim([-8.75, -8.68])
ax.set_ylim([41.06, 41.16])

# p2: MOHID
ax = fig.add_subplot(gs[1])
plotf_vector(data_mohid[:, 1], data_mohid[:, 0], data_mohid[:, 2], title="MOHID", cmap=get_cmap("BrBG", 10),
             vmin=10, vmax=36, cbar_title="Salinity", stepsize=1.5, threshold=30)
ax.plot(plg[:, 1], plg[:, 0], 'r-.')
ax.set_aspect("equal")
ax.set_xlim([-8.75, -8.68])
ax.set_ylim([41.06, 41.16])

# p3: satellite
ax = fig.add_subplot(gs[2])
im = ax.scatter(lons[ind_legal], lats[ind_legal], c=bimg[ind_legal], cmap=get_cmap("RdBu"))
plt.colorbar(im)
ax.plot(plg[:, 1], plg[:, 0], 'r-.')
ax.set_aspect("equal")
ax.set_title("Satellite on 2022-11-10")
ax.set_xlim([-8.75, -8.68])
ax.set_ylim([41.06, 41.16])

figpath = os.getcwd() + "/../../fig/OP2_LongHorizon/Experiment/"

plt.savefig(figpath + "Comparison.png")

plt.show()
print("hello")

"""
EDA handles the explorative data analysis for the mascot mission field work.
"""

from Field import Field
from Config import Config
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


class EDA:

    def __init__(self):
        self.config = Config()
        self.polygon_border = self.config.get_polygon_operational_area()
        self.polygon_border_shapely = Polygon(self.polygon_border)
        self.field = Field()
        self.grf = GRF()
        self.grid = self.field.get_grid()
        self.lat, self.lon = WGS.xy2latlon(self.grid[:, 0], self.grid[:, 1])
        self.grid_wgs = np.stack((self.lat, self.lon), axis=1)
        self.mu_prior = self.grf.get_mu()
        self.cov = self.grf.get_Sigma()
        self.folderpath = "./GRF/data/"
        self.figpath = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Projects/OASIS/fig/OP2_LongHorizon/20230321/"
        self.load_mission_folders()

        eibv, ivr = self.grf.get_ei_field_total()
        cv = .5 * eibv + .5 * ivr
        self.plot_figures(self.grf.get_mu(), self.grf.get_Sigma(), cv, eibv, ivr, -1, np.array([[None, None]]))

    def load_mission_folders(self) -> None:
        # c1: load conditional mean
        self.path_mu = self.folderpath + "mu/"
        self.files_mu = os.listdir(self.path_mu)
        self.files_mu.sort()

        # c2: load raw ctd
        self.path_raw = self.folderpath + "raw_ctd/"
        self.files_raw = []
        for item in os.listdir(self.path_raw):
            if item.endswith(".csv"):
                self.files_raw.append(item)
        self.files_raw.sort()

        # c3: load conditional variance
        self.path_cov = self.folderpath + "Sigma/"
        self.files_cov = os.listdir(self.path_cov)
        self.files_cov.sort()

    def run_mission_recap(self) -> None:
        """ Run mission recap. """

        df_raw = np.empty([0, 4])

        for i in range(len(self.files_mu)):
            print(i)
            file_mu = self.files_mu[i]
            file_raw = self.files_raw[i]
            file_cov = self.files_cov[i]
            mu = np.load(self.path_mu + file_mu)
            cov = np.load(self.path_cov + file_cov)
            raw = pd.read_csv(self.path_raw + file_raw).to_numpy()
            self.grf.assimilate_data(raw)
            eibv_field, ivr_field = self.grf.get_ei_field_total()
            cv = .5 * eibv_field + .5 * ivr_field
            df_raw = np.append(df_raw, raw, axis=0)

            lat, lon = WGS.xy2latlon(df_raw[:, 0], df_raw[:, 1])
            loc_auv = np.stack((lat, lon), axis=1)

            self.plot_figures(mu, cov, cv, eibv_field, ivr_field, i, loc_auv)

    def plot_figures(self, mu, cov, cv, eibv_field, ivr_field, i, loc_auv) -> None:
        fig = plt.figure(figsize=(50, 10))
        gs = GridSpec(nrows=1, ncols=5)
        ax = fig.add_subplot(gs[0])
        self.plotf_vector(self.grid_wgs[:, 1], self.grid_wgs[:, 0], mu,
                          title="Salinity field at step {:03d}".format(i + 1),
                          cmap=get_cmap("BrBG", 10), vmin=10, vmax=33, colorbar=True, cbar_title="Salinity (psu)",
                          polygon_border=self.polygon_border, polygon_obstacle=None, stepsize=2.,
                          threshold=self.grf.get_threshold(), xlabel="Longitude", ylabel="Latitude")
        plt.plot(loc_auv[:, 1], loc_auv[:, 0], "k.-", linewidth=2)

        ax = fig.add_subplot(gs[1])
        self.plotf_vector(self.grid_wgs[:, 1], self.grid_wgs[:, 0], np.diag(cov),
                          title="Uncertainty field at step {:03d}".format(i + 1),
                          cmap=get_cmap("RdBu", 10), vmin=0, vmax=2, colorbar=True, cbar_title="STD", stepsize=.2,
                          polygon_border=self.polygon_border, polygon_obstacle=None,
                          xlabel="Longitude", ylabel="Latitude")
        plt.plot(loc_auv[:, 1], loc_auv[:, 0], "k.-", linewidth=2)
        # plt.scatter(grid[:, 1], grid[:, 0], c=mu, s=100, cmap=get_cmap("BrBG", 10), vmin=0, vmax=30)
        plt.savefig(self.figpath + "P_{:03d}.png".format(i + 1))

        ax = fig.add_subplot(gs[2])
        self.plotf_vector(self.grid_wgs[:, 1], self.grid_wgs[:, 0], cv,
                          title="Cost valley at step {:03d}".format(i + 1),
                          cmap=get_cmap("GnBu", 10), vmin=0, vmax=1.1, colorbar=True, cbar_title="STD", stepsize=.1,
                          polygon_border=self.polygon_border, polygon_obstacle=None,
                          xlabel="Longitude", ylabel="Latitude")
        plt.plot(loc_auv[:, 1], loc_auv[:, 0], "k.-", linewidth=2)
        # plt.scatter(grid[:, 1], grid[:, 0], c=mu, s=100, cmap=get_cmap("BrBG", 10), vmin=0, vmax=30)
        plt.savefig(self.figpath + "P_{:03d}.png".format(i + 1))

        ax = fig.add_subplot(gs[3])
        self.plotf_vector(self.grid_wgs[:, 1], self.grid_wgs[:, 0], eibv_field,
                          title="EIBV cost field at step {:03d}".format(i + 1),
                          cmap=get_cmap("GnBu", 10), vmin=0, vmax=1.1, colorbar=True, cbar_title="STD", stepsize=.1,
                          polygon_border=self.polygon_border, polygon_obstacle=None,
                          xlabel="Longitude", ylabel="Latitude")
        plt.plot(loc_auv[:, 1], loc_auv[:, 0], "k.-", linewidth=2)
        # plt.scatter(grid[:, 1], grid[:, 0], c=mu, s=100, cmap=get_cmap("BrBG", 10), vmin=0, vmax=30)
        plt.savefig(self.figpath + "P_{:03d}.png".format(i + 1))

        ax = fig.add_subplot(gs[4])
        self.plotf_vector(self.grid_wgs[:, 1], self.grid_wgs[:, 0], ivr_field,
                          title="IVR cost field at step {:03d}".format(i + 1),
                          cmap=get_cmap("GnBu", 10), vmin=0, vmax=1.1, colorbar=True, cbar_title="STD", stepsize=.1,
                          polygon_border=self.polygon_border, polygon_obstacle=None,
                          xlabel="Longitude", ylabel="Latitude")
        plt.plot(loc_auv[:, 1], loc_auv[:, 0], "k.-", linewidth=2)
        # plt.scatter(grid[:, 1], grid[:, 0], c=mu, s=100, cmap=get_cmap("BrBG", 10), vmin=0, vmax=30)
        plt.savefig(self.figpath + "P_{:03d}.png".format(i + 1))

        plt.close("all")

    def get_ep(self, mu, sigma_diag, threshold) -> np.ndarray:
        p = norm.cdf(threshold, mu, sigma_diag)
        return p

    def is_masked(self, lat, lon) -> bool:
        point = Point(lat, lon)
        masked = False
        if not self.polygon_border_shapely.contains(point):
            masked = True
        return masked

    def plotf_vector(self, lat, lon, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10),
                     cbar_title='test', colorbar=True, vmin=None, vmax=None, ticks=None,
                     stepsize=None, threshold=None, polygon_border=None,
                     polygon_obstacle=None, xlabel=None, ylabel=None):
        """ Note for triangulation:
        - Maybe sometimes it cannot triangulate based on one axis, but changing to another axis might work.
        - So then the final output needs to be carefully treated so that it has the correct visualisation.
        - Also note, the floating point number can cause issues as well.
        - Triangulation uses a different axis than lat lon after its done.
        """
        """ To show threshold as a red line, then vmin, vmax, stepsize, threshold needs to have values. """
        triangulated = tri.Triangulation(lat, lon)
        lat_triangulated = lat[triangulated.triangles].mean(axis=1)
        lon_triangulated = lon[triangulated.triangles].mean(axis=1)

        ind_mask = []
        for i in range(len(lat_triangulated)):
            ind_mask.append(self.is_masked(lon_triangulated[i], lat_triangulated[i]))
        triangulated.set_mask(ind_mask)
        refiner = tri.UniformTriRefiner(triangulated)
        triangulated_refined, value_refined = refiner.refine_field(values.flatten(), subdiv=3)

        ax = plt.gca()
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
                linewidths[ind[0]] = 4
                colors[ind[0]] = 'red'
            contourplot = ax.tricontourf(triangulated_refined, value_refined, levels=levels, cmap=cmap, alpha=alpha)
            ax.tricontour(triangulated_refined, value_refined, levels=levels, linewidths=linewidths, colors=colors,
                          alpha=alpha)
        else:
            contourplot = ax.tricontourf(triangulated_refined, value_refined, cmap=cmap, alpha=alpha)
            ax.tricontour(triangulated_refined, value_refined, vmin=vmin, vmax=vmax, alpha=alpha)

        if colorbar:
            cbar = plt.colorbar(contourplot, ax=ax, ticks=ticks)
            cbar.ax.set_title(cbar_title)
        ax.set_title(title)

        if polygon_border is not None:
            ax.plot(polygon_border[:, 1], polygon_border[:, 0], 'r-.')
        if polygon_obstacle is not None:
            ax.plot(polygon_obstacle[:, 1], polygon_obstacle[:, 0], 'r-.')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return ax


if __name__ == "__main__":
    e = EDA()
    e.run_mission_recap()

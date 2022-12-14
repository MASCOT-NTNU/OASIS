""" Unit test for GMRF
This module tests the GMRF object.
"""

from unittest import TestCase
from WGS import WGS
from GMRF.GMRF import GMRF
import numpy as np
from numpy import testing
import time
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from usr_func.interpolate_3d import interpolate_3d
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from usr_func.vectorize import vectorize


class TestGMRF(TestCase):
    """
    Test class for spde helper class.
    """

    def setUp(self) -> None:
        self.gmrf = GMRF()
        self.grid = self.gmrf.get_gmrf_grid()

    # def test_get_ind_from_location(self) -> None:
    #     """
    #     Test if given a location, it will return the correct index in GMRF grid.
    #     """
    #     # c1: one location
    #     ide = 10
    #     loc = self.gmrf.get_location_from_ind(ide)
    #     id = self.gmrf.get_ind_from_location(loc)
    #     self.assertEqual(ide, id)
    #
    #     # c2: more locations
    #     ide = [10, 12]
    #     loc = self.gmrf.get_location_from_ind(ide)
    #     id = self.gmrf.get_ind_from_location(loc)
    #     self.assertIsNone(testing.assert_array_equal(ide, id))
    #
    # def test_get_ibv(self) -> None:
    #     """
    #     Test if GMRF is able to compute IBV
    #     """
    #     # c1: mean at threshold
    #     threshold = 0
    #     mu = np.array([0])
    #     sigma_diag = np.array([1])
    #     ibv = self.gmrf.get_ibv(threshold, mu, sigma_diag)
    #     self.assertEqual(ibv, .25)
    #
    #     # c2: mean further away from threshold
    #     threshold = 0
    #     mu = np.array([3])
    #     sigma_diag = np.array([1])
    #     ibv = self.gmrf.get_ibv(threshold, mu, sigma_diag)
    #     self.assertLess(ibv, .01)
    #
    # def test_get_eibv_at_locations(self) -> None:
    #     """
    #     Test if it can return eibv for the given locations.
    #     """
    #     # id = [1, 2, 3, 4, 5]
    #     id = np.random.randint(0, 1000, 5)
    #     loc = self.gmrf.get_location_from_ind(id)
    #     eibv = self.gmrf.get_eibv_at_locations(loc)
    #     self.assertIsNotNone(eibv)
    #
    # def test_check_assimilation(self):
    #     print("hello world")
    #     x_start = 1000
    #     y_start = 500
    #     z_start = -.5
    #     x_end = 2000
    #     y_end = 1500
    #     z_end = -4.5
    #     N = 20
    #     x = np.linspace(x_start, x_end, N)
    #     y = np.linspace(y_start, y_end, N)
    #     z = np.linspace(z_start, z_end, N)
    #     dataset = np.vstack((x, y, z, np.zeros_like(z))).T
    #     ind, value, idm = self.gmrf.assimilate_data(dataset)
    #
    #     fig = go.Figure(data=go.Scatter3d(
    #         x=self.grid[:, 1],
    #         y=self.grid[:, 0],
    #         z=self.grid[:, 2],
    #         mode='markers',
    #         marker=dict(color='black', size=1, opacity=.5)
    #     ))
    #     fig.add_trace(go.Scatter3d(
    #         x=y,
    #         y=x,
    #         z=z,
    #         mode='lines+markers',
    #         marker=dict(color='red', size=5, opacity=.5),
    #         line=dict(color='red', width=4)
    #     ))
    #     fig.add_trace(go.Scatter3d(
    #         x=self.grid[ind, 1],
    #         y=self.grid[ind, 0],
    #         z=self.grid[ind, 2],
    #         mode='markers',
    #         marker=dict(color='blue', size=10, opacity=.5, symbol="square")
    #     ))
    #     fig.update_layout(scene_aspectmode='manual',
    #                       scene_aspectratio=dict(x=1, y=1, z=.05))
    #
    #     plotly.offline.plot(fig, filename="/Users/yaolin/Downloads/check_assimilation.html", auto_open=True)
    #
    # def test_assimilate_data(self):
    #     """
    #     Test if it can assimilate data with given dataset.
    #     - 100 grid points within the grid.
    #     - 10 replicates with 10 grid points not within the grid.
    #     - no location.
    #     """
    #     # c1: grid points on grid
    #     ind = np.random.randint(0, self.grid.shape[0], 100)
    #     x = self.grid[ind, 0]
    #     y = self.grid[ind, 1]
    #     z = self.grid[ind, 2]
    #     v = np.zeros_like(z)
    #     dataset = np.stack((x, y, z, v), axis=1)
    #     ida, sal_a, ind_min = self.gmrf.assimilate_data(dataset)
    #     id = np.where(np.abs(z) >= .25)[0]
    #     dx = self.grid[ind_min, 0] - x[id]
    #     dy = self.grid[ind_min, 1] - y[id]
    #     dz = self.grid[ind_min, 2] - z[id]
    #     gap = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    #     self.assertLess(np.amax(gap), 1)
    #
    #     # c2: random locations
    #     for i in range(10):
    #         ind = np.random.randint(0, self.grid.shape[0], 10)
    #         x = self.grid[ind, 0] + np.random.randn(len(ind))
    #         y = self.grid[ind, 1] + np.random.randn(len(ind))
    #         z = self.grid[ind, 2] + np.random.randn(len(ind))
    #         v = np.zeros_like(z)
    #         dataset = np.stack((x, y, z, v), axis=1)
    #         ida, sal_a, ind_min = self.gmrf.assimilate_data(dataset)
    #         id = np.where(np.abs(z) >= .25)[0]
    #         dx = self.grid[ind_min, 0] - x[id]
    #         dy = self.grid[ind_min, 1] - y[id]
    #         dz = self.grid[ind_min, 2] - z[id]
    #         gap = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    #         self.assertLess(np.amax(gap), 10)
    #
    #     # c3: no location
    #     dataset = np.empty([0, 4])
    #     ida, sal_a, ind_min = self.gmrf.assimilate_data(dataset)
    #     self.assertTrue([True if len(ida) == 0 else False])

    def test_get_updated_mu_mvar(self):
        """
        Test if the assimilation works as desired.
        - c1: 1 value.
        - c2: many values.
        """
        file = "/Users/yaolin/Downloads/"

        """ Interpolate first """
        self.grid_plot, self.ind_plot = self.interpolate_grid()
        self.plot_mu(file + "mu_cond0")
        self.plot_var(file + "var_cond0")

        # c1: random location
        x = 6000
        y = 3000
        depth = -.5
        value = 25
        dataset = np.array([[x, y, depth, value]])
        self.gmrf.assimilate_data(dataset)
        self.plot_mu(file + "mu_cond1")
        self.plot_var(file + "var_cond1")

        # c2: desired location
        x = 6500
        y = 3500
        z = -.5
        value = 15
        dataset = np.array([[x, y, z, value]])
        self.gmrf.assimilate_data(dataset)
        self.plot_mu(file + "mu_cond2")
        self.plot_var(file + "var_cond2")

        # c3: mutiple desired location
        x = np.array([7000, 7500, 7800])
        y = np.array([3000, 4000, 4750])
        z = np.array([-1.5, -.5, -2.5])
        value = np.array([25, 15, 10])
        dataset = np.vstack((x, y, z, value)).T
        self.gmrf.assimilate_data(dataset)
        self.plot_mu(file + "mu_cond3")
        self.plot_var(file + "var_cond3")

    def interpolate_grid(self) -> tuple:
        """
        This function only works for this specific case since the grid from spde is not rectangular for plotting purposes.
        """
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        z = self.grid[:, 2]
        xmin, ymin, zmin = map(np.amin, [x, y, z])
        xmax, ymax, zmax = map(np.amax, [x, y, z])

        xn = np.linspace(xmin, xmax, 25)
        yn = np.linspace(ymin, ymax, 25)
        zn = np.linspace(zmin, zmax, 5)
        grid = []
        ind = []
        t1 = time.time()
        for i in range(xn.shape[0]):
            for j in range(yn.shape[0]):
                for k in range(zn.shape[0]):
                    loc = [xn[i], yn[j], zn[k]]
                    grid.append(loc)
                    ind.append(self.gmrf.get_ind_from_location(np.array(loc)))
        t2 = time.time()
        print("Time for interpolation: ", t2 - t1)
        return np.array(grid), np.array(ind)

    def plot_mu(self, filename=None):
        vmin = 10
        vmax = 36

        mu = self.gmrf.get_mu()
        grid = self.grid_plot
        ind = self.ind_plot

        xplot = grid[:, 1]
        yplot = grid[:, 0]
        zplot = grid[:, 2]
        sal_plot = mu[ind]

        # points_mean, values_mean = interpolate_3d(xplot, yplot, zplot, sal_plot)

        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])

        # fig.add_trace(go.Scatter3d(
        #     x=xplot,
        #     y=yplot,
        #     z=zplot,
        #     mode="markers",
        #     marker=dict(
        #         size=5,
        #         cmin=vmin,
        #         cmax=vmax,
        #         opacity=.3,
        #         color=sal_plot,
        #         colorscale="BrBG",
        #         showscale=True,
        #     )))

        fig.add_trace(go.Volume(
            x=xplot,
            y=yplot,
            z=zplot,
            value=sal_plot,
            isomin=vmin,
            isomax=vmax,
            opacity=.3,
            surface_count=15,
            colorscale="BrBG",
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))

        # fig.add_trace(go.Volume(
        #     x=points_mean[:, 0],
        #     y=points_mean[:, 1],
        #     z=points_mean[:, 2],
        #     value=values_mean,
        #     isomin=vmin,
        #     isomax=vmax,
        #     opacity=.3,
        #     surface_count=10,
        #     colorscale="YlGnBu",
        #     caps=dict(x_show=False, y_show=False, z_show=False),
        # ))
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.25, y=-1.25, z=1.25)
        )
        fig.update_layout(
            title={
                'text': "Mean field",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            scene=dict(
                zaxis=dict(nticks=4, range=[-4.5, 0], ),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="East", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="North", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Depth", font=dict(size=18, family="Times New Roman")),
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=.5),
            scene_camera=camera,
        )
        plotly.offline.plot(fig, filename=filename + ".html", auto_open=True)

    def plot_var(self, filename=None):
        mvar = self.gmrf.get_mvar()
        grid = self.grid_plot
        ind = self.ind_plot

        xplot = grid[:, 1]
        yplot = grid[:, 0]
        zplot = grid[:, 2]
        var_plot = mvar[ind]

        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])

        # fig.add_trace(go.Scatter3d(
        #     x=xplot,
        #     y=yplot,
        #     z=zplot,
        #     mode="markers",
        #     marker=dict(
        #         size=5,
        #         opacity=.3,
        #         color=var_plot,
        #         colorscale="RdBu",
        #         showscale=True,
        #     )))

        fig.add_trace(go.Volume(
            x=xplot,
            y=yplot,
            z=zplot,
            value=var_plot,
            opacity=.3,
            surface_count=15,
            colorscale="RdBu",
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.25, y=-1.25, z=1.25)
        )
        fig.update_layout(
            title={
                'text': "Marginal variance field",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            scene=dict(
                zaxis=dict(nticks=4, range=[-4.5, 0], ),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="East", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="North", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Depth", font=dict(size=18, family="Times New Roman")),
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=.5),
            scene_camera=camera,
        )
        plotly.offline.plot(fig, filename=filename + ".html", auto_open=True)


# if __name__ == "__main__":
#     t = TestGMRF()
#     t.setUp()
#     t.test_get_mariginal_variance()

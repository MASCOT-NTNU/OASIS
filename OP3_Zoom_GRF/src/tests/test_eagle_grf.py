""" Unit test for GRF
This module tests the GRF object.
"""


from unittest import TestCase
from Eagle.GRF2D import GRF
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
# from Visualiser.Visualiser import plotf_vector
from matplotlib.cm import get_cmap
from numpy import testing


def plotf_vector(x, y, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10),
                 cbar_title='test', colorbar=True, vmin=None, vmax=None, ticks=None,
                 stepsize=None, threshold=None, polygon_border=None,
                 polygon_obstacle=None, xlabel=None, ylabel=None):
    """
    Remember x, y is plotting x, y, thus x along horizonal and y along vertical.
    """
    plt.scatter(x, y, c=values, cmap=get_cmap("BrBG", 10), s=200, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlim([np.amin(x), np.amax(x)])
    plt.ylim([np.amin(y), np.amax(y)])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if np.any(polygon_border):
        plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'k-.', lw=2)
        if np.any(polygon_obstacle):
            for i in range(len(polygon_obstacle)):
                plt.plot(polygon_obstacle[i][:, 1], polygon_obstacle[i][:, 0], 'k-.', lw=2)
    return plt.gca()


def plotf(self, v1, v2, title1="mean", title2="cov", vmin1=None, vmax1=None, vmin2=None, vmax2=None):
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(nrows=1, ncols=2)
    ax = fig.add_subplot(gs[0])
    plotf_vector(self.grid[:, 1], self.grid[:, 0], v1, vmin=vmin1, vmax=vmax1)
    plt.title(title1)

    ax = fig.add_subplot(gs[1])
    plotf_vector(self.grid[:, 1], self.grid[:, 0], v2, vmin=vmin2, vmax=vmax2)
    plt.title(title2)
    plt.show()


class TestGRF(TestCase):

    def setUp(self) -> None:
        self.g = GRF()
        self.grid = self.g.get_grid()
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        self.cov = self.g.get_Sigma()
        self.mu = self.g.get_mu()

    def test_prior_matern_covariance(self):
        print("S1")
        plotf(self, v1=self.g.get_mu(), v2 = np.diag(self.g.get_Sigma()), vmin1=10, vmax1=36, vmin2=0, vmax2=1)
        print("END S1")

    def test_assimilate(self):
        # c2: one
        print("S2")
        dataset = np.array([[6000, 3000, 0, 30]])
        self.g.assimilate_data(dataset)
        plotf(self, v1=self.g.get_mu(), v2=np.diag(self.g.get_Sigma()), vmin1=10, vmax1=36, vmin2=0, vmax2=1)

        # c3: multiple
        dataset = np.array([[5500, 2000,  0, 35],
                            [6000, 3000, 0, 20],
                            [6200, 2500, 0, 15],
                            [6600, 4000, 0, 20]])
        self.g.assimilate_data(dataset)
        plotf(self, v1=self.g.get_mu(), v2=np.diag(self.g.get_Sigma()), vmin1=10, vmax1=36, vmin2=0, vmax2=1)
        print("End S2")


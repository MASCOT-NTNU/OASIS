""" Unit test for GRF
This module tests the GRF object.
"""


from unittest import TestCase
from GRF import GRF
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
# from Visualiser.Visualiser import plotf_vector
from matplotlib.cm import get_cmap


def plotf_vector(x, y, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10),
                 cbar_title='test', colorbar=True, vmin=None, vmax=None, ticks=None,
                 stepsize=None, threshold=None, polygon_border=None,
                 polygon_obstacle=None, xlabel=None, ylabel=None):
    """
    Remember x, y is plotting x, y, thus x along horizonal and y along vertical.
    """
    vmin = np.amin(values)
    vmax = np.amax(values)
    plt.scatter(x, y, c=values, cmap=get_cmap("BrBG", 10), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlim([np.amin(x), np.amax(x)])
    plt.ylim([np.amin(y), np.amax(y)])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if np.any(polygon_border):
        plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'k-.', lw=2)
        for i in range(len(polygon_obstacle)):
            plt.plot(polygon_obstacle[i][:, 1], polygon_obstacle[i][:, 0], 'k-.', lw=2)
    return plt.gca()



def plotf(self, v1, v2, title1="mean", title2="cov"):
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(nrows=1, ncols=2)
    ax = fig.add_subplot(gs[0])
    plotf_vector(self.grid[:, 1], self.grid[:, 0], v1,
                 polygon_border=self.g.field.get_polygon_border(),
                 polygon_obstacle=self.g.field.get_polygon_obstacles())
    plt.title(title1)

    ax = fig.add_subplot(gs[1])
    plotf_vector(self.grid[:, 1], self.grid[:, 0], v2,
                 polygon_border=self.g.field.get_polygon_border(),
                 polygon_obstacle=self.g.field.get_polygon_obstacles())
    plt.title(title2)
    plt.show()


class TestGRF(TestCase):

    def setUp(self) -> None:
        self.g = GRF()
        self.grid = self.g.field.get_grid()
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        # mu_prior = 1. - np.exp(- ((x - 1.) ** 2 + (y - .5) ** 2) / .07)
        # mu_prior = (.5 * (1 - np.exp(- ((x - 1.) ** 2 + (y - .5) ** 2) / .07)) +
        #             .5 * (1 - np.exp(- ((x - .0) ** 2 + (y - .5) ** 2) / .07)))
        # mu_prior = mu_prior.reshape(-1, 1)
        # self.g.set_mu(mu_prior)
        self.cov = self.g.get_Sigma()
        self.mu = self.g.get_mu()

    def test_prior_matern_covariance(self):
        print("S1")
        plotf(self, v1=self.g.get_mu(), v2 = np.diag(self.g.get_Sigma()))
        print("END S1")

    def test_assimilate(self):
        # c2: one
        print("S2")
        dataset = np.array([[6000, 8000, 30]])
        self.g.assimilate_data(dataset)
        plotf(self, v1=self.g.get_mu(), v2=np.diag(self.g.get_Sigma()))

        # c3: multiple
        dataset = np.array([[5500, 6000,  35],
                            [6000, 9000, 20],
                            [6200, 8500, 15],
                            [6600, 8800, 20]])
        self.g.assimilate_data(dataset)
        plotf(self, v1=self.g.get_mu(), v2=np.diag(self.g.get_Sigma()))
        print("End S2")

    def test_get_ei_field(self):
        # c1: no data assimilation
        print("S3")
        eibv, ivr = self.g.get_ei_field_total()
        plotf(self, v1=eibv, v2=ivr)

        # c2: with data assimilation
        dataset = np.array([[10000, 9000, 10],
                            [12000, 8000, 15],
                            [8000, 10000, 13],
                            [2000, 2000, 33],
                            [8000, 8000, 26],
                            [4000, 8000, 24]])
        self.g.assimilate_data(dataset)
        eibv, ivr = self.g.get_ei_field_total()
        plotf(self, v1=eibv, v2=ivr)
        plotf(self, v1=self.g.get_mu(), v2=np.diag(self.g.get_Sigma()))
        print("End S3")




from unittest import TestCase
from GMRF.GMRF import GMRF
from WGS import WGS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TestGMRF(TestCase):

    def setUp(self) -> None:
        # self.o1 = [41.065, -8.74]
        self.o2 = [41.053, -8.814]
        self.g = GMRF()
        # self.g.set_xlim([0, 8300])
        # self.g.set_ylim([0, 4800])

        self.g.set_xlim([0, 5100])
        self.g.set_ylim([0, 6200])

        self.g.set_zlim([0.5, -4.5])
        # self.g.set_nx(120)
        # self.g.set_ny(75)
        self.g.set_nx(80)
        self.g.set_ny(95)
        self.g.set_nz(6)
        self.g.construct_rectangular_grid()

    def test_grid_discretisation(self):
        g = self.g.get_grid()
        # ind = np.arange(100)
        # ind = np.where(g[:, 2] == -.5)[0]
        # plt.plot(g[ind, 1], g[ind, 0], 'k.')
        # plt.show()
        # lat, lon = WGS.xy2latlon_with_origin(g[:, 0], g[:, 1], self.o1[0], self.o1[1])
        lat, lon = WGS.xy2latlon_with_origin(g[:, 0], g[:, 1], self.o2[0], self.o2[1])
        ind = np.where(g[:, 2] == .5)[0]
        g[ind, 2] = 0
        grid = np.stack((lat, lon, g[:, 2]), axis=1)
        df = pd.DataFrame(grid, columns=['lat', 'lon', 'depth'])
        df.to_csv("grid2.csv", index=False)

        print(g)

#%%
import os
import pandas as pd

region = pd.read_csv("grid2.csv").to_numpy()
op = pd.read_csv(os.getcwd()+"/../../OperationArea.csv").to_numpy()
# lat = region[:, 0]
# lon = region[:, 1]
# x, y = WGS.latlon2xy_with_origin(lat, lon, lat[0], lon[0])
plt.plot(op[:, 1], op[:, 0], 'k-.')
plt.plot(region[:, 1], region[:, 0], 'r.')
plt.show()

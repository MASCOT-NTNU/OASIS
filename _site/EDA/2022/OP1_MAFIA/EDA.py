"""
EDA visualizes the conditional field over sampling.
"""
from GMRF.GMRF import GMRF
from AUVSimulator.AUVSimulator import AUVSimulator
from Visualiser.Visualiser_EDA import Visualiser
import numpy as np
import pandas as pd
import time
import os


class Agent:
    __counter = 0
    trajectory = np.empty([0, 3])

    def __init__(self) -> None:
        """
        Set up the planning strategies and the AUV simulator for the operation.
        """
        # s1: setup kernel.
        self.gmrf = GMRF()

        # s2: load data
        self.datapath = "csv/samples/"
        self.files = os.listdir(self.datapath)
        self.files.sort()
        print(self.files)

        # s3: data container
        self.mu = self.gmrf.get_mu()
        self.mvar = self.gmrf.get_mvar()
        self.grid = self.gmrf.get_gmrf_grid()
        # d = np.hstack((self.grid, self.mu.reshape(-1, 1)))
        # df = pd.DataFrame(d, columns=['x', 'y', 'z', 'salinity'])
        # df.to_csv('csv/cond/mu/d_00.csv', index=False)
        #
        # d = np.hstack((self.grid, self.mvar.reshape(-1, 1)))
        # df = pd.DataFrame(d, columns=['x', 'y', 'z', 'mvar'])
        # df.to_csv('csv/cond/mvar/d_00.csv', index=False)

        # df = pd.read_csv("csv/")

        # s3: setup Visualiser.
        # self.visualiser = Visualiser(self, figpath=os.getcwd() + "/../../fig/OP1_MAFIA/cond/")

    def run(self):
        """
        Run the autonomous operation according to Sense, Plan, Act philosophy.
        """

        # c1: start the operation from scratch.
        # self.visualiser.plot_agent()

        for i in range(len(self.files)):
            # a1: gather AUV data
            ctd_data = pd.read_csv(self.datapath + self.files[i]).to_numpy()
            self.trajectory = np.append(self.trajectory, ctd_data[:, :-1], axis=0)

            # a2: update GMRF field
            self.gmrf.assimilate_data(ctd_data)
            print("counter: ", self.__counter)
            # self.visualiser.plot_agent()
            self.__counter += 1

            self.mu = self.gmrf.get_mu()
            self.mvar = self.gmrf.get_mvar()
            # d = np.hstack((self.grid, self.mu.reshape(-1, 1)))
            # df = pd.DataFrame(d, columns=['x', 'y', 'z', 'salinity'])
            # df.to_csv('csv/cond/mu/d_{:02d}.csv'.format(self.__counter), index=False)
            #
            # d = np.hstack((self.grid, self.mvar.reshape(-1, 1)))
            # df = pd.DataFrame(d, columns=['x', 'y', 'z', 'mvar'])
            # df.to_csv('csv/cond/mvar/d_{:02d}.csv'.format(self.__counter), index=False)

            df = pd.DataFrame(self.trajectory, columns=['x', 'y', 'z'])
            df.to_csv('csv/trajectory/d_{:02d}.csv'.format(self.__counter), index=False)

    def get_counter(self):
        return self.__counter

    def get_trajectory(self) -> np.ndarray:
        return self.trajectory


if __name__ == "__main__":
    a = Agent()
    a.run()



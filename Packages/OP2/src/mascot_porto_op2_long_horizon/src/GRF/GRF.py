""" GRF object handles GRF-related functions. """

from Field import Field
from Delft3D import Delft3D
from usr_func.vectorize import vectorize
from usr_func.checkfolder import checkfolder
from scipy.spatial.distance import cdist
import numpy as np
from scipy.stats import norm
from usr_func.normalize import normalize
from sys import maxsize
import time
import os
import pandas as pd


class GRF:
    # parameters
    __distance_matrix = None
    __sigma = 2.
    __lateral_range = 3000
    __nugget = .04
    __threshold = 30

    # computed
    __eta = 4.5 / __lateral_range  # decay factor
    __tau = np.sqrt(__nugget)  # measurement noise

    # properties
    __mu = None
    __Sigma = None
    __eibv_field = None
    __ivr_field = None

    # data sources
    __delft3d = Delft3D()

    # field and grid
    field = Field()
    grid = field.get_grid()
    Ngrid = len(grid)
    __Fgrf = np.ones([1, Ngrid])
    __xg = vectorize(grid[:, 0])
    __yg = vectorize(grid[:, 1])

    def __init__(self) -> None:
        # s0: compute matern kernel
        self.__construct_grf_field()

        # s1: update prior mean
        self.__construct_prior_mean()

        # s2: update data folder
        t = int(time.time())
        f = os.getcwd()
        self.foldername = f + "/GRF/data/{:d}/".format(t)
        self.foldername_ctd = f + "/GRF/raw_ctd/{:d}/".format(t)
        # self.foldername_thres = f + "/GRF/threshold/{:d}/".format(t)
        checkfolder(self.foldername)
        checkfolder(self.foldername_ctd)
        self.__cnt = 0
        # checkfolder(self.foldername_thres)

    def __construct_grf_field(self):
        """ Construct distance matrix and thus Covariance matrix for the kernel. """
        self.__distance_matrix = cdist(self.grid, self.grid)
        self.__Sigma = self.__sigma ** 2 * ((1 + self.__eta * self.__distance_matrix) *
                                            np.exp(-self.__eta * self.__distance_matrix))

    def __construct_prior_mean(self):
        # s0: get delft3d dataset
        dataset_delft3d = self.__delft3d.get_dataset()
        # s1: interpolate onto grid.
        dm_grid_delft3d = cdist(self.grid, dataset_delft3d[:, :2])
        ind_close = np.argmin(dm_grid_delft3d, axis=1)
        self.__mu = dataset_delft3d[ind_close, 2].reshape(-1, 1)

    def assimilate_data(self, dataset: np.ndarray) -> None:
        """
        Assimilate dataset to GRF kernel.
        It computes the distance matrix between gmrf grid and dataset grid. Then the values are averged to each cell.
        Args:
            dataset: np.array([x, y, sal])
        """
        # ss1: save raw ctd
        df = pd.DataFrame(dataset, columns=['x', 'y', 'salinity'])
        df.to_csv(self.foldername_ctd + "D_{:03d}.csv".format(self.__cnt))

        # t1 = time.time()
        xd = dataset[:, 0].reshape(-1, 1)
        yd = dataset[:, 1].reshape(-1, 1)
        Fdata = np.ones([dataset.shape[0], 1])
        # t1 = time.time()
        dx = (xd @ self.__Fgrf - Fdata @ self.__xg.T) ** 2
        dy = (yd @ self.__Fgrf - Fdata @ self.__yg.T) ** 2
        dist = dx + dy
        ind_min_distance = np.argmin(dist, axis=1)  # used only for unittest.
        ind_assimilated = np.unique(ind_min_distance)
        salinity_assimilated = np.zeros([len(ind_assimilated), 1])
        for i in range(len(ind_assimilated)):
            ind_selected = np.where(ind_min_distance == ind_assimilated[i])[0]
            salinity_assimilated[i] = np.mean(dataset[ind_selected, -1])
        self.__update(ind_measured=ind_assimilated, salinity_measured=salinity_assimilated)
        # t2 = time.time()
        # print("Data assimilation takes: ", t2 - t1, " seconds")

        # ss2: save assimilated data
        data = np.hstack((ind_assimilated.reshape(-1, 1), salinity_assimilated))
        df = pd.DataFrame(data, columns=['ind', 'salinity'])
        df.to_csv(self.foldername + "D_{:03d}.csv".format(self.__cnt))

    def __update(self, ind_measured: np.ndarray, salinity_measured: np.ndarray):
        """
        Update GRF kernel based on sampled data.
        :param ind_measured: indices where the data is assimilated.
        :param salinity_measured: measurements at sampeld locations, dimension: m x 1
        """
        msamples = salinity_measured.shape[0]
        F = np.zeros([msamples, self.Ngrid])
        for i in range(msamples):
            F[i, ind_measured[i]] = True
        R = np.eye(msamples) * self.__tau ** 2
        C = F @ self.__Sigma @ F.T + R
        self.__mu = self.__mu + self.__Sigma @ F.T @ np.linalg.solve(C, (salinity_measured - F @ self.__mu))
        self.__Sigma = self.__Sigma - self.__Sigma @ F.T @ np.linalg.solve(C, F @ self.__Sigma)

    def get_ei_field_total(self) -> tuple:
        t1 = time.time()
        eibv_field = np.zeros([self.Ngrid])
        ivr_field = np.zeros([self.Ngrid])
        for i in range(self.Ngrid):
            SF = self.__Sigma[:, i].reshape(-1, 1)
            MD = 1 / (self.__Sigma[i, i] + self.__nugget)
            VR = SF @ SF.T * MD
            SP = self.__Sigma - VR
            sigma_diag = np.diag(SP).reshape(-1, 1)
            eibv_field[i] = self.__get_ibv(self.__mu, sigma_diag)
            ivr_field[i] = np.sum(np.diag(VR))
        self.__eibv_field = normalize(eibv_field)
        self.__ivr_field = 1 - normalize(ivr_field)
        t2 = time.time()
        print("Total EI field takes: ", t2 - t1, " seconds.")
        return self.__eibv_field, self.__ivr_field

    def get_ei_field_partial(self, indices: np.ndarray) -> tuple:
        """ Get EI field only for selected indices.
        Only compute EI field for the designated indices. Then the rest EI field is large numbers.
        """
        t1 = time.time()
        eibv_field = np.ones([self.Ngrid]) * maxsize
        ivr_field = np.ones([self.Ngrid]) * maxsize
        for idx in indices:
            SF = self.__Sigma[:, idx].reshape(-1, 1)
            MD = 1 / (self.__Sigma[idx, idx] + self.__nugget)
            VR = SF @ SF.T * MD
            SP = self.__Sigma - VR
            sigma_diag = np.diag(SP).reshape(-1, 1)
            eibv_field[idx] = self.__get_ibv(self.__mu, sigma_diag)
            ivr_field[idx] = np.sum(np.diag(VR))
        eibv_field[indices] = normalize(eibv_field[indices])
        ivr_field[indices] = 1 - normalize(ivr_field[indices])
        self.__eibv_field = eibv_field
        self.__ivr_field = ivr_field
        t2 = time.time()
        print("Partial EI field takes: ", t2 - t1, " seconds.")
        return self.__eibv_field, self.__ivr_field

    def __get_ibv(self, mu: np.ndarray, sigma_diag: np.ndarray):
        """ !!! Be careful with dimensions, it can lead to serious problems.
        :param mu: n x 1 dimension
        :param sigma_diag: n x 1 dimension
        :return:
        """
        p = norm.cdf(self.__threshold, mu, sigma_diag)
        bv = p * (1 - p)
        ibv = np.sum(bv)
        return ibv

    def set_sigma(self, value: float) -> None:
        self.__sigma = value

    def set_lateral_range(self, value: float) -> None:
        self.__lateral_range = value

    def set_nugget(self, value: float) -> None:
        self.__nugget = value

    def set_threshold(self, value: float) -> None:
        self.__threshold = value

    def set_mu(self, value: np.ndarray) -> None:
        self.__mu = value

    def get_sigma(self) -> float:
        return self.__sigma

    def get_lateral_range(self) -> float:
        return self.__lateral_range

    def get_nugget(self) -> float:
        return self.__nugget

    def get_threshold(self) -> float:
        return self.__threshold

    def get_mu(self) -> np.ndarray:
        return self.__mu

    def get_Sigma(self) -> np.ndarray:
        return self.__Sigma

    def get_eibv_field(self) -> np.ndarray:
        """ Return the computed eibv field, given which method to be called. """
        return self.__eibv_field

    def get_ivr_field(self) -> np.ndarray:
        """ Return the computed ivr field, given which method to be called. """
        return self.__ivr_field


if __name__ == "__main__":
    g = GRF()


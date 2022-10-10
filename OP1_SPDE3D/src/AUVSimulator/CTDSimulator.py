"""
CTDSimulator module simulates CTD sensor.
"""
from GMRF.GMRF import GMRF
from usr_func.normalize import normalize
import os
import numpy as np
import pandas as pd
from typing import Union
from scipy.spatial.distance import cdist


class CTDSimulator:
    """
    CTD module handles the simulated truth value at each specific location.
    """
    __field = None
    __field_grid = None
    __field_salinity = None
    __gmrf = GMRF()

    def __init__(self):
        """
        Set up the CTD simulated truth field.
        """
        self.__field_grid = self.__gmrf.get_gmrf_grid()
        value = normalize(self.__field_grid[:, 0] + self.__field_grid[:, 1]) * 3
        self.__field_salinity = self.__gmrf.get_mu() + value
        # path = os.getcwd() + "/AUVSimulator/simulated_truth.csv"
        # self.__field = pd.read_csv(path).to_numpy()
        # self.__field_grid = self.__field[:, :3]
        # self.__field_salinity = self.__field[:, -1]

    def __get_ind_from_location(self, loc: np.ndarray) -> Union[int, np.ndarray, None]:
        """
        Args:
            loc: np.array([xp, yp, zp])
        Returns: index of the closest waypoint.
        """
        if len(loc) > 0:
            dm = loc.ndim
            if dm == 1:
                d = cdist(self.__field_grid, loc.reshape(1, -1))
                return np.argmin(d, axis=0)
            elif dm == 2:
                d = cdist(self.__field_grid, loc)
                return np.argmin(d, axis=0)
            else:
                return None
        else:
            return None

    def get_salinity_at_loc(self, loc: np.ndarray) -> Union[np.ndarray, None]:
        """
        Get CTD measurement at a specific location.

        Args:
            loc: np.array([[x, y, z]])

        Returns:
            salinity value at loc
        """
        ind = self.__get_ind_from_location(loc)
        if ind is not None:
            return self.__field_salinity[ind]
        else:
            return None


if __name__ == "__main__":
    c = CTDSimulator()



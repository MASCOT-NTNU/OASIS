"""
GRF3D module assimilates the data from the CTD measurements and then update the conditional mean and
covariance structure of the field based on Gaussian updating mechanism. It provides an easier way to
access EIBV and other variance-related calculations.
"""
from Penguin.WaypointGraph import WaypointGraph
import numpy as np


class GRF3D:
    polygon = np.array([[0, 0],
                        [0, 1],
                        [1, 1],
                        [1, 0],
                        [0, 0]])

    def __init__(self):
        pass

    def assimilate_data(self):
        pass

    def update(self):
        pass

    def get_eibv_at_location(self):
        pass


if __name__ == "__main__":
    g = GRF3D()


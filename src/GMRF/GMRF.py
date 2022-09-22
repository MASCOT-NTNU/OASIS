from WGS import WGS
import numpy as np
import os


class GMRF:


    def construct_gmrf_grid(self) -> None:
        """
        Construct GMRF grid by converting lats, lons to xy.
        """
        filepath = os.getcwd() + "/GMRF/"

        x, y = WGS.latlon2xy(lat, lon)
        z = depth
        self.__gmrf_grid = np.stack((x, y, z), axis=1)
        self.__N_gmrf_grid = self.__gmrf_grid.shape[0]

        """
        Get the rotation of the grid, used for later plotting.
        """
        box = np.load(filepath + "grid.npy")
        polygon = box[:, 2:]
        polygon = np.stack((WGS.latlon2xy(polygon[:, 0], polygon[:, 1])), axis=1)
        polygon = sort_polygon_vertices(polygon)
        self.__rotated_angle = np.math.atan2(polygon[1, 0] - polygon[0, 0],
                                             polygon[1, 1] - polygon[0, 1])

"""
Delft3D handles data manipulation based on Delft3D data source
"""
from WGS import WGS
from Config import Config
import os
import pickle
import numpy as np
import h5py
from shapely.geometry import Point


class Delft3D:
    """ Load setup. """
    __setup = Config()
    __wind_dir = __setup.get_wind_direction()
    __wind_level = __setup.get_wind_level()
    __polygon_operational_area_shapely = __setup.get_polygon_operational_area_shapely()

    """ Delft3D data manipulation. """
    __datapath_delft3d = os.getcwd() + "/../../../../Data/Porto/OASIS/delft3d/oct_prior.pickle"
    with open(__datapath_delft3d, 'rb') as handle:
        __data_delft3d = pickle.load(handle)

    __lat_delft3d = __data_delft3d['lat']
    __lon_delft3d = __data_delft3d['lon']
    __salinity_delft3d = np.mean(__data_delft3d[__wind_dir][__wind_level], axis=0)
    xd, yd = WGS.latlon2xy(__lat_delft3d, __lon_delft3d)
    __dataset_delft3d = np.stack((xd, yd, __salinity_delft3d), axis=1)

    @staticmethod
    def get_dataset() -> np.ndarray:
        """ Return dataset of Delft3D.
        Example:
             dataset = np.array([[lat, lon, salinity]])
        """
        return Delft3D.__dataset_delft3d


if __name__ == "__main__":
    m = Delft3D()
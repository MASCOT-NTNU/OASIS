"""
MOHID handles the data preparation for the operation day. It imports Delft3D data and
krige the updated field based on the forecast data from MOHID data source.
"""
import os
import pickle
import numpy as np
import h5py
import pandas as pd
from shapely.geometry import Polygon, Point
from WGS import WGS
from scipy.spatial.distance import cdist


class MOHID:
    """ Mission setup. """
    __mission_date = "2022-10-01_2022-10-02"  # needs to have one day ahead.
    __wind_dir = "North"
    __wind_level = "Moderate"
    __clock_start = 10  # expected starting time, at o'clock
    __clock_end = 16  # expected ending time, at o'clock

    """ Operational Area. """
    __polygon_operational_area = pd.read_csv("OPA_GOOGLE.csv").to_numpy()
    __polygon_operational_area_shapely = Polygon(__polygon_operational_area)

    """ Delft3D data manipulation. """
    __datapath_delft3d = os.getcwd() + "/../../../../Data/Porto/OASIS/delft3d/oct_prior.pickle"
    with open(__datapath_delft3d, 'rb') as handle:
        __data_delft3d = pickle.load(handle)

    __lat_delft3d = __data_delft3d['lat']
    __lon_delft3d = __data_delft3d['lon']
    __salinity_delft3d = np.mean(__data_delft3d[__wind_dir][__wind_level], axis=0)
    __dataset_delft3d = np.stack((__lat_delft3d, __lon_delft3d, __salinity_delft3d), axis=1)

    """ MOHID data manipulation. """
    __folderpath_mohid = os.getcwd() + "/../../../../Data/Porto/OASIS/mohid/"
    __files_mohid = os.listdir(__folderpath_mohid); __files_mohid.sort()
    __ind_date = __files_mohid.index(__mission_date)
    __datapath_mohid = __folderpath_mohid + __files_mohid[__ind_date] + "/WaterProperties.hdf5"
    __data_mohid = h5py.File(__datapath_mohid, 'r')
    __grid_mohid = __data_mohid.get('Grid')
    __lat_mohid = np.array(__grid_mohid.get("Latitude"))[:-1, :-1].flatten()
    __lon_mohid = np.array(__grid_mohid.get("Longitude"))[:-1, :-1].flatten()
    __depth_mohid = []
    __salinity_mohid = []
    for i in range(1, 26):
        string_z = "Vertical_{:05d}".format(i)
        string_sal = "salinity_{:05d}".format(i)
        __depth_mohid.append(np.mean(np.array(__grid_mohid.get("VerticalZ").get(string_z)), axis=0))
        __salinity_mohid.append(np.mean(np.array(__data_mohid.get("Results").get("salinity").get(string_sal)), axis=0))
    __depth_mohid = np.array(__depth_mohid)
    __salinity_mohid = np.array(__salinity_mohid)

    # Filter outbound data
    __filter_mohid = []
    for i in range(len(__lat_mohid)):
        __filter_mohid.append(__polygon_operational_area_shapely.contains(Point(__lat_mohid[i], __lon_mohid[i])))
    __ind_legal_mohid = np.where(__filter_mohid)[0]

    __salinity_mohid_time_ave = np.mean(__salinity_mohid[__clock_start:__clock_end, :, :], axis=0).flatten()[
        __ind_legal_mohid]
    __dataset_mohid = np.stack((__lat_mohid, __lon_mohid, __salinity_mohid_time_ave), axis=1)

    @staticmethod
    def set_mission_date(value: str) -> None:
        """ Set mission date with a format 2022-10-01_2022-10-02. """
        MOHID.__mission_date = value

    @staticmethod
    def set_wind_direction(value: str) -> None:
        """ Set wind direction to be North, East, South, West. """
        MOHID.__wind_dir = value

    @staticmethod
    def set_wind_level(value: str) -> None:
        """ Set wind level to be Mild, Moderate, Heavy. """
        MOHID.__wind_level = value

    @staticmethod
    def set_clock_start(value: int) -> None:
        """ Set starting clock to be 0, 1, 2, 3, ..., 24. """
        MOHID.__clock_start = value

    @staticmethod
    def set_clock_end(value: int) -> None:
        """ Set starting clock to be 0, 1, 2, 3, ..., 24. Must be larger than MOHID.__clock_start. """
        MOHID.__clock_end = value

    @staticmethod
    def get_delft3d_dataset() -> np.ndarray:
        """ Return dataset of Delft3D.
        Example:
             dataset = np.array([[lat, lon, salinity]])
        """
        return MOHID.__dataset_delft3d

    @staticmethod
    def get_mohid_dataset() -> np.ndarray:
        """ Return dataset of Delft3D.
        Example:
             dataset = np.array([[lat, lon, salinity]])
        """
        return MOHID.__dataset_mohid


if __name__ == "__main__":
    m = MOHID()



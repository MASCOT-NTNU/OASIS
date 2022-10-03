""" Setup contains all essentials used for operations. """

import numpy as np
from shapely.geometry import Polygon


class Setup:
    """ Mission setup. """
    __mission_date = "2022-10-01_2022-10-02"  # needs to have one day ahead.
    __wind_dir = "North"
    __wind_level = "Moderate"
    __clock_start = 10  # expected starting time, at o'clock
    __clock_end = 16  # expected ending time, at o'clock

    """ Operational Area. """
    __polygon_operational_area = np.array([[41.168745, -8.74571],
                                            [41.10048, -8.741247],
                                            [41.099704, -8.816434],
                                            [41.04068871469593, -8.813846858228182],
                                            [41.06281805019644, -8.68177460501135],
                                            [41.120031055499105, -8.6815804482062],
                                            [41.14559746535853, -8.691982781217751],
                                            [41.168745, -8.74571]])
    __polygon_operational_area_shapely = Polygon(__polygon_operational_area)

    @staticmethod
    def set_mission_date(value: str) -> None:
        """ Set mission date with a format 2022-10-01_2022-10-02. """
        Setup.__mission_date = value

    @staticmethod
    def set_wind_direction(value: str) -> None:
        """ Set wind direction to be North, East, South, West. """
        Setup.__wind_dir = value

    @staticmethod
    def set_wind_level(value: str) -> None:
        """ Set wind level to be Mild, Moderate, Heavy. """
        Setup.__wind_level = value

    @staticmethod
    def set_clock_start(value: int) -> None:
        """ Set starting clock to be 0, 1, 2, 3, ..., 24. """
        Setup.__clock_start = value

    @staticmethod
    def set_clock_end(value: int) -> None:
        """ Set starting clock to be 0, 1, 2, 3, ..., 24. Must be larger than MOHID.__clock_start. """
        Setup.__clock_end = value

    @staticmethod
    def set_polygon_operational_area(value: np.ndarray) -> None:
        """ Set operational area using polygon defined by lat lon coordinates.
        Example:
             value: np.ndarray([[lat1, lon1],
                                [lat2, lon2],
                                ...
                                [latn, lonn]])
        """
        Setup.__polygon_operational_area = value
        Setup.__polygon_operational_area_shapely = Polygon(Setup.__polygon_operational_area)

    @staticmethod
    def get_mission_date() -> str:
        """ Return pre-set mission date string. """
        return Setup.__mission_date

    @staticmethod
    def get_wind_direction() -> str:
        """ Return pre-set wind direction string. """
        return Setup.__wind_dir

    @staticmethod
    def get_wind_level() -> str:
        """ Return pre-set wind level string. """
        return Setup.__wind_level

    @staticmethod
    def get_clock_start() -> int:
        """ Return expected time clock to start the mission. """
        return Setup.__clock_start

    @staticmethod
    def get_clock_end() -> int:
        """ Return expected time clock to end the mission. """
        return Setup.__clock_end

    @staticmethod
    def get_polygon_operational_area() -> np.ndarray:
        """ Return polygon for the oprational area. """
        return Setup.__polygon_operational_area

    @staticmethod
    def get_polygon_operational_area_shapely() -> 'Polygon':
        """ Return shapelized polygon for the operational area. """
        return Setup.__polygon_operational_area_shapely


if __name__ == "__main__":
    s = Setup()

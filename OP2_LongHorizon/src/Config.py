"""
Config handles the most important parameter setting in the long horizon operation in MASCOT Porto mission 2022.
- misssion date: "2022-10-01_2022-10-02"
- wind direction: ["North", "East", "South", "West"]
- wind level: ["Mild", "Moderate", "Heavy"], the corresponding wind speed: [0, 2.5, 6] m/s
- clock_start: 10, when does the operation start: 10 o'clock.
- clock_end: 16, when does the operation end: 16 o'clock.

- polygon_operational_area: the polygon used to define the safe operational area.
- polygon_operational_area_shapely: shapely object to detect collision or border.

- starting location: (lat, lon) used to define the starting location for the long horizon operation.
- home location: (lat, lon) used to define the end location for the home in the long horizon operation.
"""
from WGS import WGS
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, GeometryCollection


class Config:
    """ Mission setup. """
    __mission_date = "2022-10-11_2022-10-12"  # needs to have one day ahead.
    __wind_dir = "North"
    __wind_level = "Moderate"
    __clock_start = 10  # expected starting time, at o'clock
    __clock_end = 16  # expected ending time, at o'clock

    """ Operational Area. """
    __polygon_operational_area = pd.read_csv("OPA_GOOGLE_reduced.csv").to_numpy()
    # __polygon_operational_area = np.array([[41.168745, -8.74571],
    #                                        [41.10048, -8.741247],
    #                                        [41.099704, -8.816434],
    #                                        [41.04068871469593, -8.813846858228182],
    #                                        [41.06281805019644, -8.68177460501135],
    #                                        [41.120031055499105, -8.6815804482062],
    #                                        [41.14559746535853, -8.691982781217751],
    #                                        [41.168745, -8.74571]])
    __polygon_operational_area_shapely = Polygon(__polygon_operational_area)

    """ Starting and end locations. """
    # c1: start in the middle
    # __lat_start = 41.05947
    # __lon_start = -8.77732

    # c2: start on top
    # x = 13000
    # y = 7000
    # lat, lon = WGS.xy2latlon(x, y)
    # __lat_start = lat
    # __lon_start = lon

    # c3: start close to home
    # __lat_start = 41.14
    # __lon_start = -8.7

    __lat_home = 41.12677
    __lon_home = -8.68574

    # c4: start at home
    __lat_start = __lat_home
    __lon_start = __lon_home

    @staticmethod
    def set_mission_date(value: str) -> None:
        """ Set mission date with a format 2022-10-01_2022-10-02. """
        Config.__mission_date = value

    @staticmethod
    def set_wind_direction(value: str) -> None:
        """ Set wind direction to be North, East, South, West. """
        Config.__wind_dir = value

    @staticmethod
    def set_wind_level(value: str) -> None:
        """ Set wind level to be Mild, Moderate, Heavy. """
        Config.__wind_level = value

    @staticmethod
    def set_clock_start(value: int) -> None:
        """ Set starting clock to be 0, 1, 2, 3, ..., 24. """
        Config.__clock_start = value

    @staticmethod
    def set_clock_end(value: int) -> None:
        """ Set starting clock to be 0, 1, 2, 3, ..., 24. Must be larger than MOHID.__clock_start. """
        Config.__clock_end = value

    @staticmethod
    def set_polygon_operational_area(value: np.ndarray) -> None:
        """ Set operational area using polygon defined by lat lon coordinates.
        Example:
             value: np.ndarray([[lat1, lon1],
                                [lat2, lon2],
                                ...
                                [latn, lonn]])
        """
        Config.__polygon_operational_area = value
        Config.__polygon_operational_area_shapely = Polygon(Config.__polygon_operational_area)

    @staticmethod
    def set_loc_start(loc: np.ndarray) -> None:
        """ Set the starting location with (lat,lon). """
        Config.__lat_start, Config.__lon_start = loc

    @staticmethod
    def set_loc_home(loc: np.ndarray) -> None:
        """ Set the home location with (lat, lon). """
        Config.__lat_home, Config.__lon_home = loc

    @staticmethod
    def get_mission_date() -> str:
        """ Return pre-set mission date string. """
        return Config.__mission_date

    @staticmethod
    def get_wind_direction() -> str:
        """ Return pre-set wind direction string. """
        return Config.__wind_dir

    @staticmethod
    def get_wind_level() -> str:
        """ Return pre-set wind level string. """
        return Config.__wind_level

    @staticmethod
    def get_clock_start() -> int:
        """ Return expected time clock to start the mission. """
        return Config.__clock_start

    @staticmethod
    def get_clock_end() -> int:
        """ Return expected time clock to end the mission. """
        return Config.__clock_end

    @staticmethod
    def get_polygon_operational_area() -> np.ndarray:
        """ Return polygon for the oprational area. """
        return Config.__polygon_operational_area

    @staticmethod
    def get_polygon_operational_area_shapely() -> 'Polygon':
        """ Return shapelized polygon for the operational area. """
        return Config.__polygon_operational_area_shapely

    @staticmethod
    def get_loc_start() -> np.ndarray:
        """ Return starting location in (x, y). """
        x, y = WGS.latlon2xy(Config.__lat_start, Config.__lon_start)
        return np.array([x, y])

    @staticmethod
    def get_loc_home() -> np.ndarray:
        """ Return home location in (x, y). """
        x, y = WGS.latlon2xy(Config.__lat_home, Config.__lon_home)
        return np.array([x, y])

    # def update_operational_area(self):
        # x, y = WGS.latlon2xy(__polygon_operational_area[:, 0], __polygon_operational_area[:, 1])
        # box = np.array([])
        # import matplotlib.pyplot as plt
        # plt.plot(__polygon_operational_area[:, 1], __polygon_operational_area[:, 0], 'r-.')
        # plt.show()
        # from matplotlib.patches import Ellipse
        # plt.figure(figsize=(10, 10))
        # xl = 11000
        # yl = 8000
        # cir = Ellipse(xy=(xl, yl), width=10000, height=10000, angle=0,
        #               edgecolor='r', fc='None', lw=2)
        # vertices = cir.get_verts()
        # plg_xy = np.stack((x, y), axis=1)
        # intersection = []
        # intersection.append(Polygon(plg_xy).intersection(Polygon(np.fliplr(vertices))))
        # operational_regions = GeometryCollection(intersection)
        # operational_areas = operational_regions.geoms
        # x_operational_area = operational_areas[0].exterior.xy[0]
        # y_operational_area = operational_areas[0].exterior.xy[1]
        # polygon_opa = np.stack((x_operational_area, y_operational_area), axis=1)
        #
        # lat, lon = WGS.xy2latlon(x_operational_area, y_operational_area)
        # plg_opa = np.stack((lat, lon), axis=1)
        # df = pd.DataFrame(plg_opa, columns=['lat', 'lon'])
        # df.to_csv(os.getcwd() + "OPA_GOOGLE_reduced.csv", index=False)
        # plt.gca().add_patch(cir)
        # plt.plot(y, x, 'k-.')
        # plt.plot(polygon_opa[:, 1], polygon_opa[:, 0], 'g-.')
        # plt.xlim([0, 14000])
        # plt.ylim([0, 14000])
        # plt.axvline(xl)
        # plt.axhline(yl)
        # plt.show()
        # pass


if __name__ == "__main__":
    s = Config()





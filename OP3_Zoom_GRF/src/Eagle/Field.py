"""
Field module contains all essential details about the operational area and parameters for set up.
- polygon border: np.ndarray
"""
from WGS import WGS
import numpy as np
from shapely.geometry import Polygon, Point, LineString
import pandas as pd


class Field:

    __depth_layers = np.array([.5, 1.5, 2.5])

    def __init__(self):
        plg = pd.read_csv("Eagle/OPA_GOOGLE_reduced.csv").to_numpy()
        x, y = WGS.latlon2xy(plg[:, 0], plg[:, 1])
        self.__polygon_border = np.stack((x, y), axis=1)
        self.__polygon_border_shapely = Polygon(self.__polygon_border)

    def set_depth_layers(self, depth_layers: np.ndarray) -> None:
        self.__depth_layers = depth_layers

    def get_depth_layers(self) -> np.ndarray:
        return self.__depth_layers

    def set_polygon_border(self, value: np.ndarray) -> None:
        """ Set the polygon border, only one Nx2 dimension allowed.
        Args:
            value: np.array([[x1, y1],
                             [x2, y2],
                             ...
                             [xn, yn]])
        """
        self.__polygon_border = value

    def get_polygon_border(self) -> np.ndarray:
        """ Get the polygon border
        Returns:
             polygon_border: np.array([[x1, y1],
                                       [x2, y2],
                                       ...
                                       [xn, yn]])
        """
        return self.__polygon_border

    def get_polygon_border_shapely(self) -> 'Polygon':
        """ Get the polygon border
        Returns:
             Polygonized object for polygon border.
        """
        return self.__polygon_border_shapely

    def is_location_within_border(self, loc: np.ndarray) -> bool:
        """ Test if point is within the border polygon.
        Args:
            loc: np.array([[x, y]])
        """
        x, y = loc
        point = Point(x, y)
        return self.__polygon_border_shapely.contains(point)

    def is_border_in_the_way(self, loc_start: np.ndarray, loc_end: np.ndarray) -> bool:
        """ Check if border is in the way between loc_start and loc_end. """
        xs, ys = loc_start
        xe, ye = loc_end
        line = LineString([(xs, ys), (xe, ye)])
        return self.__polygon_border_shapely.intersects(line)


if __name__ == "__main__":
    f = Field()


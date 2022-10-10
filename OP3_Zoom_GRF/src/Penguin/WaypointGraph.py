"""
Discretizes the rectangular field formed by (xrange, yrange) with distance_neighbour.
Sets the boundary and neighbour distance for the discretization under NED coordinate system.
- N: North
- E: East
- D: Down

Args:
    polygon_border: border vertices defined by [[x1, y1], [x2, y2], ..., [xn, yn]].
    polygon_obstacles: multiple obstalce vertices defined by [[[x11, y11], [x21, y21], ... [xn1, yn1]], [[...]]].
    depths: multiple depth layers [d0, d1, d2, ..., dn].
    distance_neighbour: distance between neighbouring waypoints.

The resulting grid will be like:
    _________
   /  .   .  \
  /  .  /\   .\
  \   ./__\   .\
   \.   .   .  /
    \_________/

Get:
    Waypoints: [[x0, y0, z0],
               [x1, y1, z1],
               ...
               [xn, yn, zn]]
    Neighbour hash tables: {0: [1, 2, 3], 1: [0, 2, 3], ..., }
"""
from WGS import WGS
from typing import Any, Union
import numpy as np
from scipy.spatial.distance import cdist
from math import cos, sin, radians
from shapely.geometry import Polygon, Point
from usr_func.is_list_empty import is_list_empty
from usr_func.sort_polygon_vertices import sort_polygon_vertices
import os


class WaypointGraph:
    def __init__(self):
        self.__waypoints = np.empty([0, 3])
        self.__neighbour_distance = 120  # metres between neighbouring locations.
        self.__polygon_border = np.array([[0, 0],
                                          [0, 1000],
                                          [1000, 1000],
                                          [1000, 0],
                                          [0, 0]])
        self.__polygon_border_shapely = Polygon(self.__polygon_border)
        self.__neighbour_hash_table = dict()
        self.__depths = np.array([-0.5, -1, -1.5, -2, -2.5])

    def construct_waypoints(self) -> None:
        """ Construct the waypoint graph based on the instruction given above.
        - Construct regular meshgrid.
        .  .  .  .
        .  .  .  .
        .  .  .  .
        - Then move the even row to the right side.
        .  .  .  .
          .  .  .  .
        .  .  .  .
        - Then remove illegal locations.
        - Then add the depth layers.
        """
        self.__waypoints = np.empty([0, 3])
        xmin, ymin = map(np.amin, [self.__polygon_border[:, 0], self.__polygon_border[:, 1]])
        xmax, ymax = map(np.amax, [self.__polygon_border[:, 0], self.__polygon_border[:, 1]])
        ygap = self.__neighbour_distance * cos(radians(60)) * 2
        xgap = self.__neighbour_distance * sin(radians(60))
        self.__no_depth_layers = len(self.__depths)

        gx = np.arange(xmin, xmax, xgap)  # get [0, x_gap, 2*x_gap, ..., (n-1)*x_gap]
        gy = np.arange(ymin, ymax, ygap)
        grid2d = []
        counter_grid2d = 0
        for i in range(len(gy)):
            for j in range(len(gx)):
                if j % 2 == 0:
                    x = gx[j]
                    y = gy[i] + ygap / 2
                else:
                    x = gx[j]
                    y = gy[i]
                p = Point(x, y)
                if self.__border_contains(p):
                    grid2d.append([x, y])
                    counter_grid2d += 1
        self.multiple_depth_layer = False
        if self.__no_depth_layers > 1:
            self.multiple_depth_layer = True
        for i in range(self.__no_depth_layers):
              for j in range(counter_grid2d):
                self.__waypoints = np.append(self.__waypoints, np.array([grid2d[j][0], grid2d[j][1],
                                                                         self.__depths[i]]).reshape(1, -1), axis=0)
        self.__waypoints = np.array(self.__waypoints)

    def construct_neighbour_hash_table(self) -> None:
        """ Construct the hash table for containing neighbour indices around each waypoint.
        - Get the adjacent depth layers
            - find the current depth layer index, then find the upper and lower depth layer indices.
            - find the corresponding waypoints.
        - Get the lateral neighbour indices for each layer.
        - Append all the neighbour indices for each waypoint.
        """
        # check adjacent depth layers to determine the neighbouring waypoints.
        self.no_waypoint = self.__waypoints.shape[0]
        ERROR_BUFFER = 1
        for i in range(self.no_waypoint):
            # determine ind depth layer
            xy_c = self.__waypoints[i, 0:2].reshape(1, -1)
            d_c = self.__waypoints[i, 2]
            ind_d = np.where(self.__depths == d_c)[0][0]

            # determine ind adjacent layers
            ind_u = ind_d + 1 if ind_d < self.__no_depth_layers - 1 else ind_d
            ind_l = ind_d - 1 if ind_d > 0 else 0

            # compute lateral distance
            id = np.unique([ind_d, ind_l, ind_u])
            ds = self.__depths[id]

            ind_n = []
            for ids in ds:
                ind_id = np.where(self.__waypoints[:, 2] == ids)[0]
                xy = self.__waypoints[ind_id, 0:2]
                dist = cdist(xy, xy_c)
                ind_n_temp = np.where((dist <= self.__neighbour_distance + ERROR_BUFFER) *
                                      (dist >= self.__neighbour_distance - ERROR_BUFFER))[0]
                for idt in ind_n_temp:
                    ind_n.append(ind_id[idt])
            self.__neighbour_hash_table[i] = ind_n

    def set_neighbour_distance(self, value: float) -> None:
        """ Set the neighbour distance """
        self.__neighbour_distance = value

    def set_depth_layers(self, value: np.ndarray) -> None:
        """ Set the depth layers as np.array([-.5, -1.5, -2.5]) """
        self.__depths = value
        self.__no_depth_layers = len(self.__depths)

    def set_polygon_border(self, value: np.ndarray) -> None:
        """ Set the polygon border, only one Nx2 dimension allowed """
        self.__polygon_border = value
        self.__polygon_border_shapely = Polygon(self.__polygon_border)

    def get_neighbour_distance(self) -> float:
        return self.__neighbour_distance

    def get_depth_layers(self) -> np.ndarray:
        return self.__depths

    def get_polygon_border(self) -> np.ndarray:
        return self.__polygon_border

    def get_polygon_border_shapely(self) -> 'Polygon':
        return self.__polygon_border_shapely

    def get_waypoints(self) -> np.ndarray:
        return self.__waypoints

    def get_neighbour_hash_table(self) -> dict:
        """
        Returns: neighbour hash table
        """
        return self.__neighbour_hash_table

    def get_waypoint_from_ind(self, ind: Union[int, list, np.ndarray]) -> np.ndarray:
        """
        Return waypoint locations using ind.
        """
        return self.__waypoints[ind, :]

    def get_ind_from_waypoint(self, waypoint: np.ndarray) -> Union[int, np.ndarray, None]:
        """
        Args:
            waypoint: np.array([xp, yp, zp])
        Returns: index of the closest waypoint.
        """

        if len(waypoint) > 0:
            dm = waypoint.ndim
            if dm == 1:
                d = cdist(self.__waypoints, waypoint.reshape(1, -1))
                return np.argmin(d, axis=0)[0]
            elif dm == 2:
                d = cdist(self.__waypoints, waypoint)
                return np.argmin(d, axis=0)
            else:
                return None
        else:
            return None

    def get_neighbour_indices(self, ind: int) -> list:
        """
        Args:
            ind: waypoint index
        Returns: neighbour indices
        """
        return self.__neighbour_hash_table[ind]

    def get_vector_between_two_waypoints(self, wp1: np.ndarray, wp2: np.ndarray) -> np.ndarray:
        """ Get a vector from wp1 to wp2.

        Args:
            wp1: np.array([x1, y1, z1])
            wp2: np.array([x2, y2, z2])

        Returns:
            vec: np.array([[x2 - x1],
                           [y2 - y1],
                           [z2 - z1]])

        """
        dx = wp2[0] - wp1[0]
        dy = wp2[1] - wp1[1]
        dz = wp2[2] - wp1[2]
        vec = np.vstack((dx, dy, dz))
        return vec

    def __border_contains(self, point: Point) -> bool:
        """ Test if point is within the border polygon """
        return self.__polygon_border_shapely.contains(point)


if __name__ == "__main__":
    w = WaypointGraph()


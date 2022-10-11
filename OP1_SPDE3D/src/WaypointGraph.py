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

# TODO: debug delete
import matplotlib.pyplot as plt


class WaypointGraph:
    # Load set up parameters.
    __waypoints = np.empty([0, 3])
    __box = np.load(os.getcwd() + "/GMRF/models/grid.npy")
    __polygon_border = sort_polygon_vertices(__box[:, 2:])
    __polygon_border = np.stack((WGS.latlon2xy(__polygon_border[:, 0], __polygon_border[:, 1])), axis=1)
    __polygon_border_shapely = Polygon(__polygon_border)
    __xmin, __ymin = map(np.amin, [__polygon_border[:, 0], __polygon_border[:, 1]])
    __xmax, __ymax = map(np.amax, [__polygon_border[:, 0], __polygon_border[:, 1]])
    __depths = np.array([-0.5, -1.5, -2.5])
    no_depth_layers = len(__depths)
    __neighbour_distance = 360  # updated waypoint stepsize
    __neighbour = dict()
    __ygap = __neighbour_distance * cos(radians(60)) * 2
    __xgap = __neighbour_distance * sin(radians(60))

    # TODO: delete debug
    # origin = __polygon_border[0, :]
    # plt.plot(__polygon_border[:, 1], __polygon_border[:, 0], 'k-.')
    # plt.plot(__polygon_border[0, 1], __polygon_border[0, 0], 'r.', markersize=20)
    # plt.plot(__polygon_border[1, 1], __polygon_border[1, 0], 'b.', markersize=20)
    # plt.plot(__polygon_border[2, 1], __polygon_border[2, 0], 'g.', markersize=20)
    # plt.plot(__polygon_border[3, 1], __polygon_border[3, 0], 'y.', markersize=20)
    # plt.show()

    def __init__(self):
        self.__construct_waypoints()
        self.__construct_hash_neighbours()

        # TODO: delete
        # import plotly.graph_objects as go
        # import plotly
        # fig = go.Figure(data=go.Scatter3d(
        #     x=self.__waypoints[:, 1],
        #     y=self.__waypoints[:, 0],
        #     z=self.__waypoints[:, 2],
        #     mode='markers',
        #     marker=dict(
        #         size=2,
        #         color='black',
        #     )
        # ))
        # plotly.offline.plot(fig, filename="/Users/yaolin/Downloads/test.html", auto_open=True)

    @staticmethod
    def __construct_waypoints() -> None:
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
        gx = np.arange(WaypointGraph.__xmin, WaypointGraph.__xmax, WaypointGraph.__xgap)  # get [0, x_gap, 2*x_gap, ..., (n-1)*x_gap]
        gy = np.arange(WaypointGraph.__ymin, WaypointGraph.__ymax, WaypointGraph.__ygap)
        grid2d = []
        counter_grid2d = 0
        for i in range(len(gy)):
            for j in range(len(gx)):
                if j % 2 == 0:
                    x = gx[j]
                    y = gy[i] + WaypointGraph.__ygap / 2
                else:
                    x = gx[j]
                    y = gy[i]
                p = Point(x, y)
                if WaypointGraph.__border_contains(p):
                    grid2d.append([x, y])
                    counter_grid2d += 1
        WaypointGraph.multiple_depth_layer = False
        WaypointGraph.no_depth_layers = len(WaypointGraph.__depths)
        if WaypointGraph.no_depth_layers > 1:
            WaypointGraph.multiple_depth_layer = True
        for i in range(WaypointGraph.no_depth_layers):
              for j in range(counter_grid2d):
                WaypointGraph.__waypoints = np.append(WaypointGraph.__waypoints,
                                                      np.array([grid2d[j][0], grid2d[j][1],
                                                                WaypointGraph.__depths[i]]).reshape(1, -1), axis=0)
        WaypointGraph.__waypoints = np.array(WaypointGraph.__waypoints)

    @staticmethod
    def __construct_hash_neighbours() -> None:
        """ Construct the hash table for containing neighbour indices around each waypoint.
        - Get the adjacent depth layers
            - find the current depth layer index, then find the upper and lower depth layer indices.
            - find the corresponding waypoints.
        - Get the lateral neighbour indices for each layer.
        - Append all the neighbour indices for each waypoint.
        """
        # check adjacent depth layers to determine the neighbouring waypoints.
        WaypointGraph.no_waypoint = WaypointGraph.__waypoints.shape[0]
        ERROR_BUFFER = 1
        for i in range(WaypointGraph.no_waypoint):
            # determine ind depth layer
            xy_c = WaypointGraph.__waypoints[i, 0:2].reshape(1, -1)
            d_c = WaypointGraph.__waypoints[i, 2]
            ind_d = np.where(WaypointGraph.__depths == d_c)[0][0]

            # determine ind adjacent layers
            ind_u = ind_d + 1 if ind_d < WaypointGraph.no_depth_layers - 1 else ind_d
            ind_l = ind_d - 1 if ind_d > 0 else 0

            # compute lateral distance
            id = np.unique([ind_d, ind_l, ind_u])
            ds = WaypointGraph.__depths[id]

            ind_n = []
            for ids in ds:
                ind_id = np.where(WaypointGraph.__waypoints[:, 2] == ids)[0]
                xy = WaypointGraph.__waypoints[ind_id, 0:2]
                dist = cdist(xy, xy_c)
                ind_n_temp = np.where((dist <= WaypointGraph.__neighbour_distance + ERROR_BUFFER) *
                                      (dist >= WaypointGraph.__neighbour_distance - ERROR_BUFFER))[0]
                for idt in ind_n_temp:
                    ind_n.append(ind_id[idt])
            WaypointGraph.__neighbour[i] = ind_n

    @staticmethod
    def set_neighbour_distance(value: float) -> None:
        """ Set the neighbour distance """
        WaypointGraph.__neighbour_distance = value

    @staticmethod
    def set_depth_layers(value: np.ndarray) -> None:
        """ Set the depth layers as np.array([-.5, -1.5, -2.5]) """
        WaypointGraph.__depths = value

    @staticmethod
    def set_polygon_border(value: np.ndarray) -> None:
        """ Set the polygon border, only one Nx2 dimension allowed """
        WaypointGraph.__polygon_border = value
        WaypointGraph.__polygon_border_shapely = Polygon(WaypointGraph.__polygon_border)

    @staticmethod
    def get_neighbour_distance() -> float:
        return WaypointGraph.__neighbour_distance

    @staticmethod
    def get_depth_layers() -> np.ndarray:
        return WaypointGraph.__depths

    @staticmethod
    def get_polygon_border() -> np.ndarray:
        return WaypointGraph.__polygon_border

    @staticmethod
    def get_polygon_border_shapely() -> 'Polygon':
        return WaypointGraph.__polygon_border_shapely

    @staticmethod
    def get_waypoints() -> np.ndarray:
        """
        Returns: waypoints
        """
        return WaypointGraph.__waypoints

    @staticmethod
    def get_neighbour_hash_table():
        """
        Returns: neighbour hash table
        """
        return WaypointGraph.__neighbour

    @staticmethod
    def get_waypoint_from_ind(ind: Union[int, list, np.ndarray]) -> np.ndarray:
        """
        Return waypoint locations using ind.
        """
        return WaypointGraph.__waypoints[ind, :]

    @staticmethod
    def get_ind_from_waypoint(waypoint: np.ndarray) -> Union[int, np.ndarray, None]:
        """
        Args:
            waypoint: np.array([xp, yp, zp])
        Returns: index of the closest waypoint.
        """

        if len(waypoint) > 0:
            dm = waypoint.ndim
            if dm == 1:
                d = cdist(WaypointGraph.__waypoints, waypoint.reshape(1, -1))
                return np.argmin(d, axis=0)[0]
            elif dm == 2:
                d = cdist(WaypointGraph.__waypoints, waypoint)
                return np.argmin(d, axis=0)
            else:
                return None
        else:
            return None

    @staticmethod
    def get_neighbour_indices(ind: int) -> list:
        """
        Args:
            ind: waypoint index
        Returns: neighbour indices
        """
        return WaypointGraph.__neighbour[ind]

    @staticmethod
    def get_vector_between_two_waypoints(wp1: np.ndarray, wp2: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def __border_contains(point: Point) -> bool:
        """ Test if point is within the border polygon """
        return WaypointGraph.__polygon_border_shapely.contains(point)


if __name__ == "__main__":
    w = WaypointGraph()


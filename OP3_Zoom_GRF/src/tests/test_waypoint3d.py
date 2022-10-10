from unittest import TestCase
from Penguin.WaypointGraph import WaypointGraph
from numpy import testing
import numpy as np
import matplotlib.pyplot as plt


class TestWaypoint3D(TestCase):
    def setUp(self) -> None:
        self.wp = WaypointGraph()
        self.wp.construct_waypoints()
        self.wp.construct_neighbour_hash_table()
        self.waypoints = self.wp.get_waypoints()
        self.neighbour_hash_table = self.wp.get_neighbour_hash_table()

    def test_different_setup(self):
        plt.plot(self.waypoints[:, 1], self.waypoints[:, 0], 'k.')
        plt.xlim([0, 1000])
        plt.ylim([0, 1000])
        plt.show()
        neighbour_distance = 50
        self.wp.set_neighbour_distance(neighbour_distance)
        plg = np.array([[0, 0],
                        [0, 500],
                        [500, 500],
                        [500, 0],
                        [0, 0]])
        self.wp.set_polygon_border(plg)
        self.wp.construct_waypoints()
        self.wp.construct_neighbour_hash_table()
        waypoints = self.wp.get_waypoints()
        plt.plot(waypoints[:, 1], waypoints[:, 0], 'k.')
        plt.xlim([0, 1000])
        plt.ylim([0, 1000])
        plt.show()
        pass

    # def test_get_waypoint(self):
    #     self.wp.construct_waypoints()
    #     self.wp.construct_hash_neighbours()
    #     wp = self.wp.get_waypoints()
    #     import plotly
    #     import plotly.graph_objects as go
    #     fig = go.Figure(data=go.Scatter3d(
    #         x=wp[:, 1],
    #         y=wp[:, 0],
    #         z=wp[:, 2],
    #         mode="markers",
    #         marker=dict(
    #             color='black',
    #         )
    #     ))
    #     plotly.offline.plot(fig, filename='/Users/yaoling/Downloads/test_wp.html', auto_open=True)
    #
    #     plt.plot(wp[:, 1], wp[:, 0], 'k.')
    #     plt.show()

    # def test_neighbours_plotting(self):
    #     import matplotlib.pyplot as plt
    #     # plt.plot(self.wp[:, 0], self.wp[:, 1], 'k.')
    #     import plotly
    #     import plotly.graph_objects as go
    #     ind_r = np.random.randint(0, self.waypoints.shape[0])
    #     fig = go.Figure(data=[go.Scatter3d(
    #         x=self.waypoints[:, 0],
    #         y=self.waypoints[:, 1],
    #         z=self.waypoints[:, 2],
    #         mode='markers',
    #         marker=dict(
    #             size=2,
    #             color='black',
    #             opacity=0.8
    #         )
    #     )])
    #     fig.add_trace(go.Scatter3d(
    #         x=[self.waypoints[ind_r, 0]],
    #         y=[self.waypoints[ind_r, 1]],
    #         z=[self.waypoints[ind_r, 2]],
    #         mode='markers',
    #         marker=dict(
    #             size=10,
    #             color='red',
    #             opacity=0.8
    #         )
    #     ))
    #     fig.add_trace(go.Scatter3d(
    #         x=self.waypoints[self.neighbour_hash_table[ind_r], 0],
    #         y=self.waypoints[self.neighbour_hash_table[ind_r], 1],
    #         z=self.waypoints[self.neighbour_hash_table[ind_r], 2],
    #         mode='markers',
    #         marker=dict(
    #             size=10,
    #             color='blue',
    #             opacity=0.8
    #         )
    #     ))
    #     # tight layout
    #     fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    #     plotly.offline.plot(fig, filename='/Users/yaoling/Downloads/test_neighbour.html', auto_open=True)


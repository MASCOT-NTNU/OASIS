""" Unit test for planner

This module tests the planner object.

"""
from unittest import TestCase
from Planner.Planner import Planner
from CostValley.CostValley import CostValley
from Planner.RRTSCV.RRTStarCV import RRTStarCV
from Field import Field
from numpy import testing
import numpy as np
from numpy import testing
import matplotlib.pyplot as plt


class TestPlanner(TestCase):
    """ Common test class for the waypoint graph module
    """

    def setUp(self) -> None:
        self.planner = Planner()
        self.cv = CostValley()
        self.rrtstarcv = RRTStarCV()
        self.stepsize = self.rrtstarcv.get_stepsize()
        self.cv = CostValley()
        self.field = Field()
        self.plg = self.field.get_polygon_border()

    # def test_initial_waypoints(self):
    #     """ Test initial indices to be 0. """
    #     wp_start = self.planner.get_starting_waypoint()
    #     wp_min_cv = self.cv.get_minimum_cost_location()
    #     angle = np.math.atan2(wp_min_cv[0] - wp_start[0],
    #                           wp_min_cv[1] - wp_start[1])
    #     xn = wp_start[0] + self.stepsize * np.sin(angle)
    #     yn = wp_start[1] + self.stepsize * np.cos(angle)
    #     wp_next = np.array([xn, yn])
    #     self.assertIsNone(testing.assert_array_equal(wp_next, self.planner.get_next_waypoint()))
    #
    #     xp = xn + self.stepsize * np.sin(angle)
    #     yp = yn + self.stepsize * np.cos(angle)
    #     wp_pion = np.array([xp, yp])
    #     self.assertIsNone(testing.assert_array_equal(wp_pion, self.planner.get_pioneer_waypoint()))
    #
    #     plt.plot(yp, xp, 'g.')
    #     plt.plot(yn, xn, 'b.')
    #     plt.plot(wp_start[1], wp_start[0], 'r.')
    #     plt.plot(self.plg[:, 1], self.plg[:, 0], 'r-.')
    #     plt.show()

    def test_sense_act_plan(self):
        """ Test update planner method. """
        xn, yn = self.planner.get_current_waypoint()
        xnn, ynn = self.planner.get_next_waypoint()
        xp, yp = self.planner.get_pioneer_waypoint()

        plt.plot(yp, xp, 'g.')
        plt.plot(ynn, xnn, 'b.')
        plt.plot(yn, xn, 'r.')
        plt.plot(self.plg[:, 1], self.plg[:, 0], 'r-.')
        plt.show()

        # s0: update planning trackers
        self.planner.update_planning_trackers()

        # s1: move auv to current location
        xn, yn = self.planner.get_current_waypoint()
        xnn, ynn = self.planner.get_next_waypoint()

        # s2: on the way to the current location, update field.
        ctd_data = np.array([[1000, 2000, 30],
                             [1100, 2000, 33],
                             [1300, 2000, 33]])

        self.planner.update_pioneer_waypoint(ctd_data)
        xp, yp = self.planner.get_pioneer_waypoint()

        plt.plot(yp, xp, 'g.')
        plt.plot(ynn, xnn, 'b.')
        plt.plot(yn, xn, 'r.')
        plt.plot(self.plg[:, 1], self.plg[:, 0], 'r-.')
        plt.show()



        # s0: update planning trackers
        self.planner.update_planning_trackers()

        # s1: move auv to current location
        xn, yn = self.planner.get_current_waypoint()
        xnn, ynn = self.planner.get_next_waypoint()

        # s2: on the way to the current location, update field.
        ctd_data = np.array([[1000, 2500, 35],
                             [2000, 2400, 33],
                             [2100, 2300, 30]])

        self.planner.update_pioneer_waypoint(ctd_data)
        xp, yp = self.planner.get_pioneer_waypoint()

        plt.plot(yp, xp, 'g.')
        plt.plot(ynn, xnn, 'b.')
        plt.plot(yn, xn, 'r.')
        plt.plot(self.plg[:, 1], self.plg[:, 0], 'r-.')
        plt.show()


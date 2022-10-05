""" Unit test for planner

This module tests the planner object.

"""

from unittest import TestCase
from Planner.Planner import Planner
from numpy import testing
import numpy as np


class TestPlanner(TestCase):
    """ Common test class for the waypoint graph module
    """

    def setUp(self) -> None:
        self.planner = Planner()

    def test_initial_waypoints(self):
        """ Test initial indices to be 0. """
        p = self.planner.get_next_waypoint()
        s = np.array([0, 0])
        self.assertIsNone(testing.assert_array_equal(self.planner.get_next_waypoint(), np.array([0, 0])))
        self.assertIsNone(testing.assert_array_equal(self.planner.get_current_waypoint(), np.array([0, 0])))
        self.assertIsNone(testing.assert_array_equal(self.planner.get_pioneer_waypoint(), np.array([0, 0])))

    def test_set_waypoins(self):
        """ Test individual index setting function. """
        wp_next = np.array([10000, 10000])
        wp_now = np.array([8000, 8000])
        wp_pion = np.array([6000, 6000])
        self.planner.set_next_waypoint(wp_next)
        self.planner.set_current_waypoint(wp_now)
        self.planner.set_pioneer_waypoint(wp_pion)

        self.assertIsNone(testing.assert_array_equal(self.planner.get_next_waypoint(), wp_next))
        self.assertIsNone(testing.assert_array_equal(self.planner.get_current_waypoint(), wp_now))
        self.assertIsNone(testing.assert_array_equal(self.planner.get_pioneer_waypoint(), wp_pion))

    def test_update_planner(self):
        """ Test update planner method. """
        wp_pion = self.planner.get_pioneer_waypoint()
        wp_next = self.planner.get_next_waypoint()
        wp_pion_new = np.array([12000, 10000])
        self.planner.update_planner()
        self.planner.set_pioneer_waypoint(wp_pion_new)
        self.assertIsNone(testing.assert_array_equal(wp_pion_new, self.planner.get_pioneer_waypoint()))
        self.assertIsNone(testing.assert_array_equal(wp_pion, self.planner.get_next_waypoint()))
        self.assertIsNone(testing.assert_array_equal(wp_next, self.planner.get_current_waypoint()))


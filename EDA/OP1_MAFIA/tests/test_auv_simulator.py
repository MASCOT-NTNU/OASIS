""" Unit test for AUV Simulator
"""

from unittest import TestCase
from AUVSimulator.AUVSimulator import AUVSimulator
import numpy as np
from numpy import testing
from usr_func.is_list_empty import is_list_empty


def value(x, y, z):
    return 2 * x + 3 * y + 4 * z


class TestAUVSimulator(TestCase):

    def setUp(self) -> None:
        self.auv = AUVSimulator()

    def test_move_to_location(self):
        """
        Test if the AUV moves according to the given direction.
        """
        # c1: starting location
        self.assertIsNone(testing.assert_array_equal(self.auv.get_location(), np.array([0, 0, 0])))
        self.assertIsNone(testing.assert_array_equal(self.auv.get_previous_location(), np.array([0, 0, 0])))

        # c2: move to another location
        loc_new = np.array([10, 10, .5])
        self.auv.move_to_location(loc_new)
        self.assertIsNone(testing.assert_array_equal(self.auv.get_location(), loc_new))
        self.assertIsNone(testing.assert_array_equal(self.auv.get_previous_location(), np.array([0, 0, 0])))

        # c3: move to another location
        loc_new = np.array([20, 20, 1.])
        self.auv.move_to_location(loc_new)
        self.assertIsNone(testing.assert_array_equal(self.auv.get_location(), loc_new))
        self.assertIsNone(testing.assert_array_equal(self.auv.get_previous_location(), np.array([10, 10, .5])))

    def test_data_collection(self):
        # c0: move to original location
        self.auv.move_to_location(np.array([0, 0, 0]))
        ctd = self.auv.get_ctd_data()
        self.assertTrue([True if len(ctd) == 0 else False])

        # c1: after it has moved to a location.
        loc_new = np.array([10, 10, .5])
        self.auv.move_to_location(loc_new)
        df = self.auv.get_ctd_data()

    def test_arrived(self):
        # c1: not arrived
        self.assertFalse(self.auv.is_arrived())

        # c2: arrived
        self.auv.arrive()
        self.assertTrue(self.auv.is_arrived())

        # c3: move
        self.auv.move()
        self.assertFalse(self.auv.is_arrived())



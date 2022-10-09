from unittest import TestCase
from src.Penguin.Myopic3D import Myopic3D
from numpy import testing
import numpy as np


class TestMyopic3D(TestCase):

    def setUp(self) -> None:
        self.mp = Myopic3D()

    def test_get_next_waypoint(self):
        # c1: stable next waypoint
        loc = self.mp.get_next_waypoint()
        self.assertIsNone(testing.assert_array_equal(loc, np.array([1, 1])))

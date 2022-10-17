from unittest import TestCase
from Config import Config
from WGS import WGS
import matplotlib.pyplot as plt
from numpy import testing
import numpy as np


class TestConfig(TestCase):

    def setUp(self) -> None:
        self.c = Config()

    def test_set_home_location(self):
        loc_home = self.c.get_loc_home()
        loc = np.array([41.12677, -8.68574])
        x, y = WGS.latlon2xy(loc[0], loc[1])
        self.assertIsNone(testing.assert_array_equal(loc_home, np.array([x, y])))

    def test_starting_home_location(self):
        loc_home = self.c.get_loc_home()
        loc_start = self.c.get_loc_start()
        plg = self.c.get_polygon_operational_area()
        x, y = WGS.latlon2xy(plg[:, 0], plg[:, 1])
        plt.plot(y, x, 'r-.')
        # plt.plot(plg[:, 1], plg[:, 0], 'r-.')
        plt.plot(loc_start[1], loc_start[0], 'k.')
        plt.plot(loc_home[1], loc_home[0], 'b.')
        plt.show()

    def test_get_resume_state(self):
        # s1: check false state
        Config.set_resume_state(False)
        resume = Config.get_resume_state()
        self.assertFalse(resume)

        # s2: check true state
        Config.set_resume_state(True)
        resume = Config.get_resume_state()
        self.assertTrue(resume)

        # s3: check false again
        Config.set_resume_state(False)
        resume = Config.get_resume_state()
        self.assertFalse(resume)

        # s4: check true
        Config.set_resume_state(True)
        resume = Config.get_resume_state()
        self.assertTrue(resume)

        # s5: check false
        Config.set_resume_state(False)
        resume = Config.get_resume_state()
        self.assertFalse(resume)


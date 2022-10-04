from unittest import TestCase
from Config import Config
from WGS import WGS
import matplotlib.pyplot as plt


class TestConfig(TestCase):

    def setUp(self) -> None:
        self.c = Config()

    # def test_something(self):
    #     pass
        # self.assertEqual(True, False)  # add assertion here

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


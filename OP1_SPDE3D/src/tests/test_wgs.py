from unittest import TestCase
from WGS import WGS


class TestWGS(TestCase):

    def setUp(self) -> None:
        self.wgs = WGS()

    def test_origin(self):
        p = self.wgs.get_origin()
        self.assertIsNotNone(p)
        lato = 41
        lono = 10
        self.wgs.set_origin(lato, lono)
        p1, p2 = self.wgs.get_origin()
        self.assertEqual(lato, p1)
        self.assertEqual(lono, p2)

    def test_get_coordinates(self):
        lat_o, lon_o = self.wgs.get_origin()
        x, y = self.wgs.latlon2xy(lat_o, lon_o)
        self.assertEqual(x, .0)
        self.assertEqual(y, .0)

        lat, lon = 30.2, 10.1
        x, y = self.wgs.latlon2xy(lat, lon)
        lat1, lon1 = self.wgs.xy2latlon(x, y)
        self.assertAlmostEqual(lat, lat1)
        self.assertAlmostEqual(lon, lon1)



from unittest import TestCase
from Eagle.Field import Field
import matplotlib.pyplot as plt


class TestField(TestCase):

    def setUp(self) -> None:
        self.f = Field()

    def test_field_polygon_border(self) -> None:
        # {"type": "Polygon", "coordinates": [[
        #     [-8.74571, 41.168745], [-8.741247, 41.10048], [-8.816434, 41.099704],
        #     [-8.813846858228182, 41.04068871469593], [-8.68177460501135, 41.06281805019644],
        #     [-8.6815804482062, 41.120031055499105], [-8.691982781217751, 41.14559746535853], [-8.74571, 41.168745]]]}
        # p = dict()
        # p["type"] = "Polygon"
        # p[]
        plg = self.f.get_polygon_border()
        plt.plot(plg[:, 1], plg[:, 0], 'r-.')
        plt.show()
        pass



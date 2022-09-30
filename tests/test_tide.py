from unittest import TestCase
from DataHandling.Tide import Tide


class TestTide(TestCase):

    def setUp(self) -> None:
        self.t = Tide()

    def test_tide_cases(self):
        self.t.get_data4month(10)
        pass


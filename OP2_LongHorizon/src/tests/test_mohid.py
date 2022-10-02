from unittest import TestCase
from MOHID import MOHID


class TestMOHID(TestCase):

    def setUp(self) -> None:
        self.m = MOHID()

    def test_pass(self):
        md = self.m.get_mohid_dataset()
        dd = self.m.get_delft3d_dataset()
        pass



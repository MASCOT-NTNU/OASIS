from unittest import TestCase
from Setup import Setup


class TestSetup(TestCase):

    def setUp(self) -> None:
        self.setup = Setup()

    def test_something(self):
        pass
        # self.assertEqual(True, False)  # add assertion here


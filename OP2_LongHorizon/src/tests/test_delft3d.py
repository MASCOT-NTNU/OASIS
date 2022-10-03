from unittest import TestCase
from Delft3D import Delft3D
import matplotlib.pyplot as plt


class TestDelft3D(TestCase):
    def setUp(self) -> None:
        self.d = Delft3D()

    def test(self) -> None:
        dd = self.d.get_delft3d_dataset()

        # plt.scatter(dd[:, 1], dd[:, 0], c=dd[:, 2], cmap="BrBG", vmin=10, vmax=36)
        # plt.colorbar()
        # plt.show()
        dd

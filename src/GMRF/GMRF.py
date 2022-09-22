from WGS import WGS
import numpy as np
import os    


class GMRF:

    __xlim = [0, 1]
    __ylim = [0, 1]
    __zlim = [0, 1]
    __nx = 1
    __ny = 1
    __nz = 1
    __grid = np.empty([0, 3])

    def construct_rectangular_grid(self) -> None:
        """
        Construct rectangular grid for spde discretisation.
        """
        xv = np.linspace(self.__xlim[0], self.__xlim[1], self.__nx)
        yv = np.linspace(self.__ylim[0], self.__ylim[1], self.__ny)
        zv = np.linspace(self.__zlim[0], self.__zlim[1], self.__nz)
        grid = []
        for i in range(self.__nx):
            for j in range(self.__ny):
                for k in range(self.__nz):
                    grid.append([xv[i], yv[j], zv[k]])
        self.__grid = np.array(grid)

    def set_xlim(self, value: list) -> None:
        self.__xlim = value

    def set_ylim(self, value: list) -> None:
        self.__ylim = value

    def set_zlim(self, value: list) -> None:
        self.__zlim = value

    def set_nx(self, value: int) -> None:
        self.__nx = value

    def set_ny(self, value: int) -> None:
        self.__ny = value

    def set_nz(self, value: int) -> None:
        self.__nz = value

    def get_xlim(self):
        return self.__xlim

    def get_ylim(self):
        return self.__ylim

    def get_zlim(self):
        return self.__zlim

    def get_grid(self):
        return self.__grid
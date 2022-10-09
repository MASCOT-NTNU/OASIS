"""
Myopic3D plans the next waypoint based on sense, act, plan ..
"""
import numpy as np


class Myopic3D:

    def __init__(self):
        pass

    def get_next_waypoint(self) -> np.ndarray:
        #x = np.random.rand(1)
        #y = np.random.rand(1)
        x = 1
        y = 1
        return np.array([x, y])
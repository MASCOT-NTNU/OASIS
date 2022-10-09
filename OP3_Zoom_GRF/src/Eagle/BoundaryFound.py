import numpy as np





class BoundaryFound():

    def __init__(self):
        self.threshold = 0

        # In-situ measurments
        self.salinity = np.array([])
        self.depth = np.array([])
        self.x = np.array([])
        self.y = np.array([])

    def is_threshold_found(self):

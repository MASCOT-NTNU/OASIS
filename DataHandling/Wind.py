import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
from numpy import vectorize
from math import radians
from matplotlib.gridspec import GridSpec


class Wind:
    __datapath_wind = os.getcwd() + "/DataSources/wind/raw/wind_data.txt"  # auxiliary data to resolve missing data.

    def __init__(self):
        df_wind = np.loadtxt(self.__datapath_wind, delimiter=',')

    def set_datapath(self, value: str) -> None:
        self.__datapath_wind = value

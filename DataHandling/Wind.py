import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
from numpy import vectorize
from math import radians
from matplotlib.gridspec import GridSpec


@vectorize
def tsp2str(tsp):
    return datetime.fromtimestamp(tsp)


class Wind:
    __datapath_wind_aux = os.getcwd() + "/DataSources/wind/raw/wind_data.txt"  # auxiliary data to resolve missing data.
    __datapath_wind = os.getcwd() + "/DataSources/wind/raw/wind_times_serie_porto_obs_2015_2020.txt"

    def __init__(self):
        df_wind = np.array(pd.read_csv(self.__datapath_wind, sep="\t", engine='python'))
        df_wind = df_wind[:-3, :5]
        yr_wind = df_wind[:, 0]
        hr_wind = df_wind[:, 1]
        timestamp_wind = []
        for i in range(len(yr_wind)):
            year = int(yr_wind[i][6:])
            month = int(yr_wind[i][3:5])
            day = int(yr_wind[i][:2])
            hour = int(hr_wind[i][:2])
            timestamp_wind.append(datetime(year, month, day, hour).timestamp())
        timestamp_wind = np.array(timestamp_wind)
        wind_speed = df_wind[:, 3]
        wind_maxspeed = df_wind[:, 4]
        wind_angle = df_wind[:, 2]

        # check missing data
        dd = np.diff(timestamp_wind)
        ind = np.argmax(dd)


        df_wind_aux = np.loadtxt(self.__datapath_wind_aux, delimiter=',')
        ind_aux = np.where((df_wind_aux[:, 0] >= timestamp_wind[ind]) *
                           (df_wind_aux[:, 1] <= timestamp_wind[ind+1]))[0]
        wind_speed_aux = df_wind_aux[ind_aux, 1]
        wind_angle_aux = df_wind_aux[ind_aux, 2]

        # d_test = datetime(2019, 12, 31, 21, 0).timestamp()
        d_start = datetime(2015, 1, 1, 0, 0).timestamp()
        dist_0 = (d_start -df_wind_aux[:, 0])**2
        id_s = np.argmin(dist_0)
        for t in range(id_s, len(df_wind_aux)):
            d_test = df_wind_aux[t, 0]
            dist1 = (d_test - df_wind_aux[:, 0])**2
            dist2 = (d_test - timestamp_wind)**2
            id1 = np.argmin(dist1)
            id2 = np.argmin(dist2)

            a1 = df_wind_aux[id1, 2]
            a2 = wind_angle[id2]
            fig = plt.figure(figsize=(20, 10))
            gs = GridSpec(nrows=1, ncols=2)
            ax = fig.add_subplot(gs[0])
            s1 = df_wind_aux[id1, 1]
            s2 = wind_speed[id2]
            u1 = s1 * np.cos(radians(a1))
            v1 = s1 * np.sin(radians(a1))
            u2 = s2 * np.cos(radians(a2))
            v2 = s2 * np.sin(radians(a2))
            ax.quiver([0, 0], u1, v1, scale=30)
            ax.set_xlim([-.5, .5])
            ax.set_ylim([-.5, .5])

            ax = fig.add_subplot(gs[1])
            ax.quiver([1, 0], u2, v2, scale=30)
            ax.set_xlim([-.5, .5])
            ax.set_ylim([-.5, .5])
            plt.savefig("/Users/yaoling/Downloads/wind/P_{:04d}.png".format(t))
            plt.close("all")
            # print(datetime.fromtimestamp(df_wind_aux[id1, 0]))
            # print(a1)
            # print(datetime.fromtimestamp(timestamp_wind[id2]))
            # print(a2)
            print(t)
            # break

        # loop to check wind corresponding indices


        # df2 = np.loadtxt(self.__datapath_wind2, delimiter=', ')
        df_wind
        pass

    def set_datapath(self, value: str) -> None:
        self.__datapath_wind = value

"""
Tide module handles the tide organsiation according to raw data.

According to raw data,
# "Preia - Mar" --> high tide
# "Baixa - Mar" --> low tide

Examples
>>> t = Tide()
>>> t.get_data4month(month=11)  # --> to get data for November
"""

import numpy as np
import os
import pandas as pd
from datetime import datetime


class Tide:

    __datapath_tide = os.getcwd() + "/DataSources/tide/raw/"
    __month_of_interest = 11  # October

    # "Preia - Mar" --> high tide
    # "Baixa - Mar" --> low tide

    def set_month(self, value: int) -> None:
        """ Update the month of interest. """
        self.__month_of_interest = value

    def get_month(self) -> int:
        """ Return the month of interest. """
        return self.__month_of_interest

    def get_data4month(self, month: int) -> None:
        """
        Get tide data from raw for the specific month.
        Month is represented by numerical values: 1-Jan, 2-Feb, ..., 10-Oct, ...
        """
        self.__month_of_interest = month
        # s1: reorganise data.
        year = []
        month = []
        day = []
        hour = []
        min = []
        tide_height = np.empty([0, 1])
        tide_type_numerical = np.empty([0, 1])
        for tide_file in os.listdir(self.__datapath_tide):
            print(tide_file)
            temp = pd.read_csv(self.__datapath_tide + tide_file, skiprows=12, sep="\t", header=None,
                               encoding = "unicode-escape")
            temp = pd.DataFrame(temp[0].str.split('  ').tolist())
            year_month_day = np.array(temp.iloc[:, 0])
            hour_min = np.array(temp.iloc[:, 1])
            for i in range(len(hour_min)):
                ind_year = year_month_day[i].index('-')
                year.append(int(year_month_day[i][:ind_year]))
                ind_month = year_month_day[i][ind_year + 1:].index('-')
                month.append(int(year_month_day[i][ind_year + 1:][:ind_month]))
                day.append(int(year_month_day[i][ind_year + 1:][ind_month + 1:]))
                ind_hour = hour_min[i].index(":")
                hour.append(int(hour_min[i][:ind_hour]))
                min.append(int(hour_min[i][ind_hour + 1:]))
            tide_height = np.concatenate(
                (tide_height, np.array(temp.iloc[:, 2]).astype(float).reshape(-1, 1)), axis=0)
            tide_type = temp.iloc[:, 3]  # tide type
            tide_type_numerical = np.concatenate(
                (tide_type_numerical, np.array(tide_type == "Preia-Mar").astype(int).reshape(-1, 1)), axis=0)
        year = np.array(year).reshape(-1, 1)
        month = np.array(month).reshape(-1, 1)
        day = np.array(day).reshape(-1, 1)
        hour = np.array(hour).reshape(-1, 1)
        min = np.array(min).reshape(-1, 1)

        # extract data for the selected month.
        ind_selected = month == self.__month_of_interest  # tide in November months in the historical data
        year_selected = year[ind_selected].reshape(-1, 1)
        month_selected = month[ind_selected].reshape(-1, 1)
        day_selected = day[ind_selected].reshape(-1, 1)
        hour_selected = hour[ind_selected].reshape(-1, 1)
        min_selected = min[ind_selected].reshape(-1, 1)
        tide_height_selected = tide_height[ind_selected].reshape(-1, 1)
        tide_type_numerical_selected = tide_type_numerical[ind_selected].reshape(-1, 1)
        tide_timestamp = []
        for i in range(len(year_selected)):
            tide_timestamp.append(
                datetime(year_selected[i, 0], month_selected[i, 0], day_selected[i, 0],
                         hour_selected[i, 0], min_selected[i, 0]).timestamp())
        tide_timestamp = np.array(tide_timestamp).reshape(-1, 1)
        data = np.hstack((tide_timestamp, tide_height_selected, tide_type_numerical_selected))
        df = pd.DataFrame(data, columns=['timestamp', 'waterlevel', 'tide(1:high tide, 0:low tide)'])
        df.to_csv(self.__datapath_tide + "../Month_"+str(self.__month_of_interest)+".csv", index=False)
        # np.savetxt(self.__datapath_tide[:-5] + "tide.txt", data, delimiter=", ")

        # s3: get filtered ebb phase data.
        ebb_start = []
        ebb_end = []
        for i in range(len(tide_type_numerical_selected)):
            if tide_type_numerical_selected[i] == 1:
                if i < len(tide_type_numerical_selected) - 1:
                    if tide_type_numerical_selected[i + 1] == 0:  # filter undesired state
                        ebb_start.append(tide_timestamp[i, 0])
                        ebb_end.append(tide_timestamp[i + 1, 0])
                    else:
                        pass
                else:
                    pass
        ebb_start = np.array(ebb_start).reshape(-1, 1)
        ebb_end = np.array(ebb_end).reshape(-1, 1)
        ebb = np.hstack((ebb_start, ebb_end))
        df = pd.DataFrame(ebb, columns=['timestamp_ebbstart', 'timestamp_ebbend'])
        df.to_csv(self.__datapath_tide + "../Ebb_Month_"+str(self.__month_of_interest)+".csv", index=False)
        # np.savetxt(self.__datapath_tide[:-5] + "ebb_dec.txt", ebb, delimiter=", ")


if __name__ == "__main__":
    a = Tide()
    a.get_data4month(11)

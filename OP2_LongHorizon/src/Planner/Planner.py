"""
Planner plans the next waypoint according to Sense, Plan, Act process.
It wraps all the essential components together to ease the procedure for the agent during adaptive sampling.

Args:
    _wp_now: current waypoint
    _wp_next: next waypoint
    _wp_pion: pioneer waypoint

"""
from Config import Config
from Planner.RRTSCV.RRTStarCV import RRTStarCV
from Planner.StraightLinePathPlanner import StraightLinePathPlanner
from CostValley.CostValley import CostValley

import numpy as np


class Planner:

    # s0: load configuration
    __config = Config()
    __loc_start = __config.get_loc_start()
    __loc_end = __config.get_loc_home()

    # s1: setup cost valley and kernel.
    __cv = CostValley()
    __Budget = __cv.get_Budget()
    __grf = __cv.get_grf_model()
    __grid = __grf.grid
    __loc_min_cv = __cv.get_minimum_cost_location()

    # s2: set up path planning strategies
    __rrtstar = RRTStarCV()
    __slpp = StraightLinePathPlanner()

    # s3: set planning trackers.
    __wp_now = np.array([0, 0])
    __wp_next = np.array([0, 0])
    __wp_pion = np.array([0, 0])
    __traj = []

    def update_knowledge(self, dataset: np.ndarray) -> None:
        """ Update the field based on the gathered in-situ measurements. """
        
        pass

    def update_planner(self) -> None:
        """
        Update the planner indices by shifting all the remaining indices.
        """
        self.__wp_now = self.__wp_next
        self.__wp_next = self.__wp_pion
        self.__traj.append(self.__wp_now)

    def get_pioneer_waypoint(self):
        return self.__wp_pion

    def get_next_waypoint(self) -> np.ndarray:
        return self.__wp_next

    def get_current_waypoint(self) -> np.ndarray:
        return self.__wp_now

    def get_trajectory(self) -> list:
        return self.__traj

    def set_next_waypoint(self, wp: np.ndarray) -> None:
        self.__wp_next = wp

    def set_current_waypoint(self, wp: np.ndarray) -> None:
        self.__wp_now = wp

    def set_pioneer_waypoint(self, wp: np.ndarray) -> None:
        self.__wp_pion = wp


"""
Planner plans the next waypoint according to Sense, Plan, Act process.

Args:
    _wp_now: current waypoint
    _wp_next: next waypoint
    _wp_pion: pioneer waypoint

"""
import numpy as np


class Planner:
    __wp_now = np.array([0, 0])
    __wp_next = np.array([0, 0])
    __wp_pion = np.array([0, 0])
    __traj = []

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


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
import numpy as np


class Planner:
    # s0: load configuration
    __config = Config()
    __wp_start = __config.get_loc_start()
    __wp_end = __config.get_loc_home()

    # s1: set up path planning strategies
    __rrtstarcv = RRTStarCV()
    __stepsize = __rrtstarcv.get_stepsize()
    __slpp = StraightLinePathPlanner()

    # s1: setup cost valley.
    __cv = __rrtstarcv.get_CostValley()
    __Budget = __cv.get_Budget()
    __wp_min_cv = __cv.get_minimum_cost_location()

    # s2: set up data assimilation kernel
    __grf = __cv.get_grf_model()
    __grid = __grf.grid

    # s3: set planning trackers.
    __wp_now = __wp_start
    __wp_next = __wp_start
    __wp_pion = __wp_start
    __traj = [[__wp_now[0], __wp_now[1]]]

    def __init__(self):
        """ In the initial phase, wp next and wp pion are selected along the direction to the target location. """
        # s0: get minimum cost location.
        self.__wp_min_cv = self.__cv.get_minimum_cost_location()

        # s1: compute angle between the starting location to the minimum cost location.
        angle = np.math.atan2(self.__wp_min_cv[0] - self.__wp_now[0],
                              self.__wp_min_cv[1] - self.__wp_now[1])

        # s2: compute next location and pioneer location.
        xn = self.__wp_now[0] + self.__stepsize * np.sin(angle)
        yn = self.__wp_now[1] + self.__stepsize * np.cos(angle)
        self.__wp_next = np.array([xn, yn])

        xp = xn + self.__stepsize * np.sin(angle)
        yp = yn + self.__stepsize * np.cos(angle)
        self.__wp_pion = np.array([xp, yp])

    def update_planning_trackers(self) -> None:
        """ Move the pointer one step ahead. """
        self.__wp_now = self.__wp_next
        self.__wp_next = self.__wp_pion
        self.__traj.append([self.__wp_now[0], self.__wp_now[1]])

    def update_pioneer_waypoint(self, ctd_data: np.ndarray) -> None:
        """
        Compute the next waypoint:
        - Step I: assimilate data to the kernel.
        - Step II: update the cost valley.
        - Step III: get the minimum cost location in the cost field.
        - Step IV: plan one step ahead.
        """
        # s1: assimilate data to the kernel.
        self.__grf.assimilate_data(ctd_data)

        # s2: update cost valley
        self.__cv.update_cost_valley(self.__wp_next)

        # s3: get minimum cost location.
        self.__wp_min_cv = self.__cv.get_minimum_cost_location()

        # s4: plan one step based on cost valley and rrtstar
        if not self.__Budget.get_go_home_alert():
            self.__wp_pion = self.__rrtstarcv.get_next_waypoint(self.__wp_next, self.__wp_min_cv)
        else:
            self.__wp_pion = self.__slpp.get_waypoint_from_straight_line(self.__wp_next, self.__wp_end)

    def get_starting_waypoint(self) -> np.ndarray:
        """ Return the starting location in the field. """
        return self.__wp_start

    def get_end_waypoint(self) -> np.ndarray:
        """ Return end location for the operation. """
        return self.__wp_end

    def get_pioneer_waypoint(self) -> np.ndarray:
        return self.__wp_pion

    def get_next_waypoint(self) -> np.ndarray:
        return self.__wp_next

    def get_current_waypoint(self) -> np.ndarray:
        return self.__wp_now

    def get_trajectory(self) -> list:
        return self.__traj

    def get_rrstarcv(self) -> 'RRTStarCV':
        return self.__rrtstarcv


if __name__ == "__main__":
    p = Planner()



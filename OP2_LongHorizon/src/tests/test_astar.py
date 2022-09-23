from unittest import TestCase
from WaypointGraph import WaypointGraph

import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, x=0, y=0, z=0, parent=None):
        self.x = x
        self.y = y
        self.z = z
        self.parent = parent

        self.g = 0
        self.h = 0
        self.f = 0

    # def __eq__(self, other):
    #     return (self.x == other.x) and (self.y == other.y) and (self.z == other.z)

figpath = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Projects/OASIS/fig/astar/"

class Astar:
    def __init__(self, wp: 'WaypointGraph'):
        self.wp = wp
        self.neighbour_distance = self.wp.get_neighbour_distance()
        pass

    def search_path(self, start, end):
        xstart, ystart, zstart = start
        xend, yend, zend = end

        start_node = Node(xstart, ystart, zstart, parent=None)
        end_node = Node(xend, yend, zend, parent=None)

        open_list = []
        closed_list = []

        open_list.append(start_node)

        cnt = 0
        NUM_MAX = 100
        waypoints = self.wp.get_waypoints()

        while len(open_list) > 0:
            print(cnt)
            current_node = open_list[0]
            current_index = 0

            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            open_list.pop(current_index)
            closed_list.append(current_node)

            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append([current.x, current.y])
                    current = current.parent
                return path

            wp = np.array([current_node.x, current_node.y, 0])
            ind_current = self.wp.get_ind_from_waypoint(wp)
            ind_neighbours = self.wp.get_ind_neighbours(ind_current)

            children = []
            for idn in ind_neighbours:
                new_loc = self.wp.get_waypoint_from_ind(idn)
                new_node = Node(new_loc[0], new_loc[1], parent=current_node)
                children.append(new_node)

            for child in children:
                for closed_child in closed_list:
                    if child == closed_child:
                        continue

                child.g = current_node.g + self.neighbour_distance
                child.h = ((child.x - end_node.x)**2 +
                           (child.y - end_node.y)**2)
                child.f = child.g + child.h

                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                open_list.append(child)

            plt.figure()
            plt.plot(waypoints[:, 0], waypoints[:, 1], 'k.', alpha=.1)
            for child in children:
                plt.plot(child.x, child.y, 'b.')
            plt.plot(current_node.x, current_node.y, 'r*')
            plt.savefig(figpath + "P_{:03d}.png".format(cnt))
            plt.close("all")

            cnt += 1
            if cnt > NUM_MAX:
                break


class TestAStar(TestCase):

    def setUp(self) -> None:
        self.polygon_border = np.array([[0, 0],
                                        [1, 0],
                                        [1, 1],
                                        [0, 1],
                                        [0, 0]])
        self.polygon_obstacle = [np.array([[.4, .4],
                                           [.7, .6],
                                           [.6, .8],
                                           [.3, .5],
                                           [.4, .4]])]
        self.wp = WaypointGraph()
        self.wp.set_polygon_border(self.polygon_border)
        self.wp.set_polygon_obstacles(self.polygon_obstacle)
        self.wp.set_depth_layers([0])
        self.wp.set_neighbour_distance(.1)
        self.wp.construct_waypoints()
        self.wp.construct_hash_neighbours()

        self.astar = Astar(self.wp)

    def test_wp(self):
        wp = self.wp.get_waypoints()
        loc_start = [0, 0, 0]
        loc_end = [1, 1, 0]
        id_start = self.wp.get_ind_from_waypoint(np.array(loc_start))
        id_end = self.wp.get_ind_from_waypoint(np.array(loc_end))

        wps = self.wp.get_waypoint_from_ind(id_start)
        wpe = self.wp.get_waypoint_from_ind(id_end)
        path = self.astar.search_path(wps, wpe)

        # plt.plot(wp[:, 1], wp[:, 0], 'k.')
        # plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'r-.')
        # plt.plot(self.polygon_obstacle[0][:, 1], self.polygon_obstacle[0][:, 0], 'r-.')
        # plt.show()

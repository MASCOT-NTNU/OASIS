from shapely.geometry import Polygon, Point
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from unittest import TestCase
from WaypointGraph import WaypointGraph
from matplotlib.gridspec import GridSpec


class Node:

    def __init__(self, loc=None, parent=None):
        self.x, self.y = loc
        self.parent = parent
        self.g = np.inf
        self.h = np.inf
        self.f = np.inf


def astar(wp: 'WaypointGraph', loc_start, loc_end, border: np.ndarray, obstacle: np.ndarray):

    figpath = os.getcwd() + "/../../fig/astar/"

    start_node = Node(loc_start, None)
    start_node.cost = 0
    end_node = Node(loc_end, None)
    end_node.cost = 0

    plg_border = Polygon(border)
    plg_obstalce = Polygon(obstacle)
    # stepsize = .1
    maximum_iter = 150
    cnt = 0

    open_list = []
    closed_list = []
    ind_open = []
    ind_closed = []

    open_list.append(start_node)
    wp_temp = np.array([start_node.x, start_node.y, 0])
    ind_open.append(wp.get_ind_from_waypoint(wp_temp))

    while len(open_list) > 0:
        print(cnt)
        node_now = open_list[0]
        ind_now = 0
        for i in range(len(ind_open)):
            if open_list[i].f < node_now.f:
                node_now = open_list[i]
                ind_now = i
        # for index, item in enumerate(open_list):
        #     if item.f < node_now.f:
        #         node_now = item
        #         ind_now = index

        # print("open before: ", open_list)
        open_list.pop(ind_now)
        # print("open after: ", open_list)
        closed_list.append(node_now)
        # print("closed: ", closed_list)

        if np.sqrt((node_now.x - end_node.x)**2 +
                   (node_now.y - end_node.y)**2) <= .01:
            path = []
            pointer = node_now
            while pointer is not None:
                path.append([pointer.x, pointer.y])
                pointer = pointer.parent
            return path[::-1]

        children = []

        wp_now = np.array([node_now.x, node_now.y, 0])
        ind_now_wp = wp.get_ind_from_waypoint(wp_now)
        ind_neighbours = wp.get_ind_neighbours(ind_now_wp)

        for idn in ind_neighbours:
            loc = wp.get_waypoint_from_ind(idn)
            x_new = loc[0]
            y_new = loc[1]
        # angles = np.arange(0, 360, 90)
        # for angle in angles:
        #     x_new = node_now.x + stepsize * np.cos(math.radians(angle))
        #     y_new = node_now.y + stepsize * np.sin(math.radians(angle))

            # point = Point(x_new, y_new)

            # filter obstcle and border
            # if not plg_border.contains(point) or plg_obstalce.contains(point):
            #     continue

            loc_new = np.array([x_new, y_new])
            node_new = Node(loc_new, node_now)

            children.append(node_new)

        for child in children:
            for closed_child in closed_list:
                # if closed_child == child:
                # print(np.sqrt((closed_child.x - child.x)**2 +
                #             (closed_child.y - child.y)**2))
                if (np.sqrt((closed_child.x - child.x)**2 +
                            (closed_child.y - child.y)**2)) <= .099:
                    print("Too close")
                    continue

            child.g = node_now.g + wp.get_neighbour_distance()
            child.h = (child.x - end_node.x)**2 + (child.y - end_node.y)**2
            child.f = child.g + child.h

            for open_node in open_list:
                # if open_node == child and child.g > open_node.g:
                # print(np.sqrt((open_node.x - child.x)**2 + (open_node.y - child.y)**2))
                if (np.sqrt((open_node.x - child.x)**2 +
                            (open_node.y - child.y)**2)) <= .1 and child.g > open_node.g:
                    continue

            open_list.append(child)

        fig = plt.figure(figsize=(35, 15))
        gs = GridSpec(nrows=1, ncols=3)

        ax = fig.add_subplot(gs[0])
        ax.plot(border[:, 0], border[:, 1], 'r-.')
        ax.plot(obstacle[:, 0], obstacle[:, 1], 'r-.')
        plt.plot(loc_start[0], loc_start[1], 'k.')
        plt.plot(loc_end[0], loc_end[1], 'b*')
        for item in open_list:
            ax.plot(item.x, item.y, 'c.', alpha=.2)

        ax = fig.add_subplot(gs[1])
        ax.plot(border[:, 0], border[:, 1], 'r-.')
        ax.plot(obstacle[:, 0], obstacle[:, 1], 'r-.')
        plt.plot(loc_start[0], loc_start[1], 'k.')
        plt.plot(loc_end[0], loc_end[1], 'b*')
        for item in closed_list:
            ax.plot(item.x, item.y, 'k.', alpha=.2)

        ax = fig.add_subplot(gs[2])
        ax.plot(border[:, 0], border[:, 1], 'r-.')
        ax.plot(obstacle[:, 0], obstacle[:, 1], 'r-.')
        plt.plot(loc_start[0], loc_start[1], 'k.')
        plt.plot(loc_end[0], loc_end[1], 'b*')
        for child in children:
            plt.plot(child.x, child.y, 'r.')
        plt.plot(node_now.x, node_now.y, 'g.')

        plt.savefig(figpath + "P_{:03d}.png".format(cnt))
        plt.close("all")

        cnt += 1
        if cnt > maximum_iter:
            print("Cannot converge")
            break


class TestAstar(TestCase):
    def setUp(self) -> None:
        self.plg_border = np.array([[0, 0],
                               [0, 1],
                               [1, 1],
                               [1, 0],
                               [0, 0]])

        self.plg_obstacle = np.array([[.25, .25],
                                 [.65, .25],
                                 [.65, .65],
                                 [.25, .65],
                                 [.25, .25]])
        self.wp = WaypointGraph()
        self.wp.set_polygon_border(self.plg_border)
        self.wp.set_polygon_obstacles([self.plg_obstacle])
        self.wp.set_depth_layers([0])
        self.wp.set_neighbour_distance(.05)
        self.wp.construct_waypoints()
        self.wp.construct_hash_neighbours()
        self.waypoint = self.wp.get_waypoints()
        plt.plot(self.waypoint[:, 1], self.waypoint[:, 0], 'k.')
        plt.show()

    def test_astar(self):
        # pass
        loc_start = np.array([.1, .1, 0])
        loc_end = np.array([.9, .9, 0])
        ids = self.wp.get_ind_from_waypoint(loc_start)
        ide = self.wp.get_ind_from_waypoint(loc_end)
        wps = self.wp.get_waypoint_from_ind(ids)
        wpe = self.wp.get_waypoint_from_ind(ide)
        astar(self.wp, loc_start=wps[:2], loc_end=wpe[:2], border=self.plg_border, obstacle=self.plg_obstacle)











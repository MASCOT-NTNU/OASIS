"""
AgentPlot visualises the agent during the adaptive sampling.
"""

import os
import matplotlib.pyplot as plt
from matplotlib import tri
from matplotlib.cm import get_cmap
from shapely.geometry import Polygon, Point
from matplotlib.gridspec import GridSpec
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20
from Field import Field
field = Field()


class AgentPlot:

    agent = None

    def __init__(self, agent, figpath) -> None:
        self.agent = agent
        self.auv = self.agent.auv
        self.ctd = self.auv.ctd
        self.mu_truth = self.ctd.get_ground_truth()
        self.figpath = figpath
        self.planner = self.agent.planner
        self.rrtstarcv = self.planner.get_rrstarcv()
        self.cv = self.rrtstarcv.get_CostValley()
        self.budget = self.cv.get_Budget()
        self.grf = self.cv.get_grf_model()
        self.field = self.grf.field
        self.grid = self.field.get_grid()
        self.xgrid = self.grid[:, 0]
        self.ygrid = self.grid[:, 1]
        self.plg = self.field.get_polygon_border()

    def plot_agent(self):
        # s0: get updated field
        mu = self.grf.get_mu()

        Sigma = self.grf.get_Sigma()
        threshold = self.grf.get_threshold()
        self.cnt = self.agent.get_counter()
        traj_past = np.array(self.planner.get_trajectory())

        be = self.budget.get_ellipse()

        # s1: get updated waypoints
        wp_now = self.planner.get_current_waypoint()
        wp_next = self.planner.get_next_waypoint()
        wp_pion = self.planner.get_pioneer_waypoint()

        # s2: get cost valley and trees.
        cost_valley = self.cv.get_cost_field()
        cost_eibv = self.cv.get_eibv_field()
        cost_ivr = self.cv.get_ivr_field()
        tree_nodes = self.rrtstarcv.get_tree_nodes()
        rrt_traj = self.rrtstarcv.get_trajectory()

        fig = plt.figure(figsize=(50, 20))
        gs = GridSpec(nrows=2, ncols=3)

        """ truth, mu, sigma, cost, eibv, ivr. """

        """ plot truth"""
        ax = fig.add_subplot(gs[0])
        self.plotf_vector(self.ygrid, self.xgrid, self.mu_truth, title="Ground truth field",
                          cmap=get_cmap("BrBG", 10), vmin=15, vmax=36, cbar_title="Salinity", stepsize=1.5,
                          threshold=threshold)
        ax.plot(self.plg[:, 1], self.plg[:, 0], 'r-.')
        ax.plot(wp_now[1], wp_now[0], 'r.', markersize=10, label="Current waypoint")
        ax.plot(wp_next[1], wp_next[0], 'b.', markersize=10, label="Next waypoint")
        ax.plot(wp_pion[1], wp_pion[0], 'g.', markersize=10, label="Pioneer waypoint")
        ax.plot(traj_past[:, 1], traj_past[:, 0], 'y.-', label="Trajectory")
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")

        """ plot mean """
        ax = fig.add_subplot(gs[1])
        self.plotf_vector(self.ygrid, self.xgrid, mu, title="Conditional salinity field", cmap=get_cmap("BrBG", 10),
                          vmin=15, vmax=36, cbar_title="Salinity", stepsize=1.5, threshold=threshold)
        ax.plot(self.plg[:, 1], self.plg[:, 0], 'r-.')
        ax.plot(wp_now[1], wp_now[0], 'r.', markersize=10, label="Current waypoint")
        ax.plot(wp_next[1], wp_next[0], 'b.', markersize=10, label="Next waypoint")
        ax.plot(wp_pion[1], wp_pion[0], 'g.', markersize=10, label="Pioneer waypoint")
        ax.plot(traj_past[:, 1], traj_past[:, 0], 'y.-', label="Trajectory")
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")

        """ plot var """
        ax = fig.add_subplot(gs[2])
        im = ax.scatter(self.ygrid, self.xgrid, c=np.sqrt(np.diag(Sigma)), s=200,
                        cmap=get_cmap("RdBu", 10), vmin=0, vmax=2)
        plt.title("Conditional uncertainty field")
        plt.colorbar(im)
        # self.plotf_vector(self.ygrid, self.xgrid, np.sqrt(np.diag(Sigma)), title="Conditional uncertainty field",
        #                   cmap=get_cmap("RdBu", 10), cbar_title="Deviation")
        ax.plot(self.plg[:, 1], self.plg[:, 0], 'r-.')
        ax.plot(wp_now[1], wp_now[0], 'r.', markersize=10, label="Current waypoint")
        ax.plot(wp_next[1], wp_next[0], 'b.', markersize=10, label="Next waypoint")
        ax.plot(wp_pion[1], wp_pion[0], 'g.', markersize=10, label="Pioneer waypoint")
        ax.plot(traj_past[:, 1], traj_past[:, 0], 'y.-', label="Trajectory")
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_title("Conditional uncertainty field")

        """ plot cost valley and trees. """
        ax = fig.add_subplot(gs[3])
        self.plotf_vector(self.ygrid, self.xgrid, cost_valley, title="Cost Valley",
                          cmap=get_cmap("GnBu", 10), vmin=0, vmax=1, cbar_title="Cost")
        # im = ax.scatter(self.ygrid, self.xgrid, c=cost_valley, s=200, cmap=get_cmap("BrBG", 10), vmin=0, vmax=4)
        # plt.colorbar(im)
        ax.add_patch(be)
        ax.plot(self.plg[:, 1], self.plg[:, 0], 'r-.')
        ax.plot(wp_now[1], wp_now[0], 'r.', markersize=10, label="Current waypoint")
        ax.plot(wp_next[1], wp_next[0], 'b.', markersize=10, label="Next waypoint")
        ax.plot(wp_pion[1], wp_pion[0], 'g.', markersize=10, label="Pioneer waypoint")
        ax.plot(traj_past[:, 1], traj_past[:, 0], 'y.-', label="Trajectory")
        for node in tree_nodes:
            if node.get_parent() is not None:
                loc = node.get_location()
                loc_p = node.get_parent().get_location()
                ax.plot([loc[1], loc_p[1]],
                         [loc[0], loc_p[0]], "-g")
        ax.plot(rrt_traj[:, 1], rrt_traj[:, 0], 'k-', linewidth=2)
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_title("Cost Valley")

        """ plot eibv field. """
        ax = fig.add_subplot(gs[4])
        self.plotf_vector(self.ygrid, self.xgrid, cost_eibv, title="EIBV cost field",
                          cmap=get_cmap("GnBu", 10), vmin=0, vmax=1, cbar_title="Cost")
        # im = ax.scatter(self.ygrid, self.xgrid, c=cost_valley, s=200, cmap=get_cmap("BrBG", 10), vmin=0, vmax=4)
        # plt.colorbar(im)
        ax.plot(self.plg[:, 1], self.plg[:, 0], 'r-.')
        ax.plot(wp_now[1], wp_now[0], 'r.', markersize=10, label="Current waypoint")
        ax.plot(wp_next[1], wp_next[0], 'b.', markersize=10, label="Next waypoint")
        ax.plot(wp_pion[1], wp_pion[0], 'g.', markersize=10, label="Pioneer waypoint")
        ax.plot(traj_past[:, 1], traj_past[:, 0], 'y.-', label="Trajectory")
        for node in tree_nodes:
            if node.get_parent() is not None:
                loc = node.get_location()
                loc_p = node.get_parent().get_location()
                ax.plot([loc[1], loc_p[1]],
                        [loc[0], loc_p[0]], "-g")
        ax.plot(rrt_traj[:, 1], rrt_traj[:, 0], 'k-', linewidth=2)
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_title("Cost Valley")

        """ plot ivr field. """
        ax = fig.add_subplot(gs[5])
        self.plotf_vector(self.ygrid, self.xgrid, cost_ivr, title="IVR cost field",
                          cmap=get_cmap("GnBu", 10), vmin=0, vmax=1, cbar_title="Cost")
        # im = ax.scatter(self.ygrid, self.xgrid, c=cost_valley, s=200, cmap=get_cmap("BrBG", 10), vmin=0, vmax=4)
        # plt.colorbar(im)
        ax.plot(self.plg[:, 1], self.plg[:, 0], 'r-.')
        ax.plot(wp_now[1], wp_now[0], 'r.', markersize=10, label="Current waypoint")
        ax.plot(wp_next[1], wp_next[0], 'b.', markersize=10, label="Next waypoint")
        ax.plot(wp_pion[1], wp_pion[0], 'g.', markersize=10, label="Pioneer waypoint")
        ax.plot(traj_past[:, 1], traj_past[:, 0], 'y.-', label="Trajectory")
        for node in tree_nodes:
            if node.get_parent() is not None:
                loc = node.get_location()
                loc_p = node.get_parent().get_location()
                ax.plot([loc[1], loc_p[1]],
                        [loc[0], loc_p[0]], "-g")
        ax.plot(rrt_traj[:, 1], rrt_traj[:, 0], 'k-', linewidth=2)
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_title("Cost Valley")


        plt.savefig(self.figpath + "P_{:03d}.png".format(self.cnt))
        # plt.show()
        plt.close("all")

    def plotf_vector(self, xplot, yplot, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10),
                     cbar_title='test', colorbar=True, vmin=None, vmax=None, ticks=None,
                     stepsize=None, threshold=None, polygon_border=None,
                     polygon_obstacle=None, xlabel=None, ylabel=None):
        """ To show threshold as a red line, then vmin, vmax, stepsize, threshold needs to have values. """
        triangulated = tri.Triangulation(xplot, yplot)
        x_triangulated = xplot[triangulated.triangles].mean(axis=1)
        y_triangulated = yplot[triangulated.triangles].mean(axis=1)

        ind_mask = []
        for i in range(len(x_triangulated)):
            ind_mask.append(self.is_masked(y_triangulated[i], x_triangulated[i]))
        triangulated.set_mask(ind_mask)
        refiner = tri.UniformTriRefiner(triangulated)
        triangulated_refined, value_refined = refiner.refine_field(values.flatten(), subdiv=3)

        ax = plt.gca()
        # ax.triplot(triangulated, lw=0.5, color='white')
        if np.any([vmin, vmax]):
            levels = np.arange(vmin, vmax, stepsize)
        else:
            levels = None
        if np.any(levels):
            linewidths = np.ones_like(levels) * .3
            colors = len(levels) * ['black']
            if threshold:
                dist = np.abs(threshold - levels)
                ind = np.where(dist == np.amin(dist))[0]
                linewidths[ind] = 3
                colors[ind[0]] = 'red'
            contourplot = ax.tricontourf(triangulated_refined, value_refined, levels=levels, cmap=cmap, alpha=alpha)
            ax.tricontour(triangulated_refined, value_refined, levels=levels, linewidths=linewidths, colors=colors,
                          alpha=alpha)
        else:
            contourplot = ax.tricontourf(triangulated_refined, value_refined, cmap=cmap, alpha=alpha)
            ax.tricontour(triangulated_refined, value_refined, vmin=vmin, vmax=vmax, alpha=alpha)

        if colorbar:
            cbar = plt.colorbar(contourplot, ax=ax, ticks=ticks)
            cbar.ax.set_title(cbar_title)
        return ax

    @staticmethod
    def is_masked(x, y):
        """
        :param x:
        :param y:
        :return:
        """
        loc = np.array([x, y])
        masked = False
        if not field.border_contains(loc):
            masked = True
        return masked



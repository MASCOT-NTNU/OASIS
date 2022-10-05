from unittest import TestCase
from Agent import Agent


class TestAgent(TestCase):

    def setUp(self) -> None:
        self.ag = Agent()

    def test_agent_run(self):
        self.ag.run()

        # s6: plotting section
        fig = plt.figure(figsize=(30, 10))
        gs = GridSpec(nrows=1, ncols=4)
        ax = fig.add_subplot(gs[0])

        plotf_vector(self.grid[:, 1], self.grid[:, 0], self.grf.get_mu(), cmap=get_cmap("RdBu", 10),
                     vmin=5, vmax=45, stepsize=2.5, threshold=32, cbar_title="Value",
                     title="Ground field", xlabel="East", ylabel="North", polygon_border=field.get_polygon_border())
        goal = self.Budget.get_goal()
        alpha = self.Budget.get_ellipse_rotation_angle()
        mid = self.Budget.get_ellipse_middle_location()
        a = self.Budget.get_ellipse_a()
        b = self.Budget.get_ellipse_b()
        c = self.Budget.get_ellipse_c()
        ellipse = Ellipse(xy=(mid[1], mid[0]), width=2 * a,
                          height=2 * b, angle=math.degrees(alpha),
                          edgecolor='r', fc='None', lw=2)
        plt.gca().add_patch(ellipse)
        plt.plot(loc[1], loc[0], 'y*', markersize=20)

        p_traj = np.array(trajectory)
        ax.plot(p_traj[:, 1], p_traj[:, 0], 'k.-')
        plt.xlim([np.min(self.rrtstar.polygon_border[:, 1]), np.max(self.rrtstar.polygon_border[:, 1])])
        plt.ylim([np.min(self.rrtstar.polygon_border[:, 0]), np.max(self.rrtstar.polygon_border[:, 0])])

        plt.xlabel("East")
        plt.ylabel("North")
        plt.title("Updated mean after step: " + str(j))

        ax = fig.add_subplot(gs[1])
        ax.plot(self.rrtstar.polygon_border[:, 1], self.rrtstar.polygon_border[:, 0], 'k-.')
        ax.plot(self.rrtstar.polygon_obstacle[:, 1], self.rrtstar.polygon_obstacle[:, 0], 'k-.')
        if not self.gohome:
            # for node in self.rrtstar.tree_nodes:
            #     if node.parent is not None:
            #         plt.plot([node.y, node.parent.y],
            #                  [node.x, node.parent.x], "g-")
            # ax.plot(self.rrtstar.path_to_target[:, 1], self.rrtstar.path_to_target[:, 0], 'r')
            ax.plot(self.path_to_target[:, 1], self.path_to_target[:, 0], 'r')
        else:
            if self.obstacle_in_the_way:
                # for node in self.rrthome.tree_nodes:
                #     if node.parent is not None:
                #         plt.plot([node.y, node.parent.y],
                #                  [node.x, node.parent.x], "g-")
                # ax.plot(self.rrthome.path_to_target[:, 1], self.rrthome.path_to_target[:, 0], 'r')
                ax.plot(self.path_to_target[:, 1], self.path_to_target[:, 0], 'r')

        xplot = self.grf_grid[:, 1]
        yplot = self.grf_grid[:, 0]
        im = ax.scatter(xplot, yplot, c=self.CV.cost_valley, s=200, cmap=get_cmap("BrBG", 10), vmin=0, vmax=2, alpha=.5)
        plt.colorbar(im)
        ellipse = Ellipse(xy=(self.CV.budget.y_middle, self.CV.budget.x_middle), width=2 * self.CV.budget.ellipse_a,
                          height=2 * self.CV.budget.ellipse_b, angle=math.degrees(self.CV.budget.angle),
                          edgecolor='r', fc='None', lw=2)
        plt.gca().add_patch(ellipse)
        ax.plot(y_current, x_current, 'bs')
        ax.plot(y_previous, x_previous, 'y^')
        ax.plot(y_next, x_next, 'r*')
        ax.plot(y_pioneer, x_pioneer, 'mP')
        ax.plot(Y_HOME, X_HOME, 'k*')

        p_traj = np.array(trajectory)
        ax.plot(p_traj[:, 1], p_traj[:, 0], 'k.-')
        plt.xlim([np.min(self.rrtstar.polygon_border[:, 1]), np.max(self.rrtstar.polygon_border[:, 1])])
        plt.ylim([np.min(self.rrtstar.polygon_border[:, 0]), np.max(self.rrtstar.polygon_border[:, 0])])

        plt.xlabel("East")
        plt.ylabel("North")

        plt.title("Updated cost valley after step: " + str(j))
        plt.savefig(FILEPATH + "fig/rrtstar/P_{:03d}.jpg".format(j))
        plt.close("all")


        self.tp.update_trees(self.rrtstar.get_nodes())
        self.tp.plot_tree()
        # traj = self.rrtstar.get_trajectory()
        plt.plot(loc[1], loc[0], 'y*', markersize=20)
        plt.plot(self.traj[:, 1], self.traj[:, 0], 'k.-')
        # plt.plot(traj[:, 0], traj[:, 1], 'r-')


        # print("loc: ", loc)
        plt.plot(loc[1], loc[0], 'b*', markersize=20)
        plt.scatter(self.grid[:, 1], self.grid[:, 0], c=self.cv.get_cost_valley(),
                    cmap=get_cmap("BrBG", 10), vmin=0, vmax=4, alpha=.5)
        plt.colorbar()
        plt.savefig(self.__figpath + "/P_{:03d}.png".format(i))
        plt.close("all")




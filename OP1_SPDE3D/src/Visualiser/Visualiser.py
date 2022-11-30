"""
Visualiser object handles the planning visualisation part.
"""

# from Agent import Agent
import os
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from usr_func.interpolate_3d import interpolate_3d
from usr_func.checkfolder import checkfolder
from usr_func.vectorize import vectorize


class Visualiser:

    agent = None

    def __init__(self, agent, figpath) -> None:
        self.agent = agent
        self.figpath = figpath
        checkfolder(self.figpath + "/mu")
        checkfolder(self.figpath + "/mvar")
        self.myopic = self.agent.myopic
        self.gmrf = self.myopic.gmrf
        self.grid = self.myopic.gmrf.get_gmrf_grid()
        self.grid_plot, self.ind_plot = self.interpolate_grid()
        self.xplot = self.grid_plot[:, 1]
        self.yplot = self.grid_plot[:, 0]
        self.zplot = self.grid_plot[:, 2]

    def interpolate_grid(self) -> tuple:
        """
        This function only works for this specific case since the grid from spde is not rectangular for plotting purposes.
        """
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        z = self.grid[:, 2]
        xmin, ymin, zmin = map(np.amin, [x, y, z])
        xmax, ymax, zmax = map(np.amax, [x, y, z])

        xn = np.linspace(xmin, xmax, 25)
        yn = np.linspace(ymin, ymax, 25)
        zn = np.linspace(zmin, zmax, 5)
        grid = []
        ind = []
        t1 = time.time()
        for i in range(xn.shape[0]):
            for j in range(yn.shape[0]):
                for k in range(zn.shape[0]):
                    loc = [xn[i], yn[j], zn[k]]
                    grid.append(loc)
                    ind.append(self.gmrf.get_ind_from_location(np.array(loc)))
        t2 = time.time()
        print("Time for interpolation: ", t2 - t1)
        return np.array(grid), np.array(ind)

    def plot_agent(self):
        mu = self.gmrf.get_mu()
        mvar = self.gmrf.get_mvar()
        self.cnt = self.agent.get_counter()
        mu[mu < 0] = 0

        """ plot mean """
        value = mu[self.ind_plot]
        vmin = 10
        vmax = 33
        filename = self.figpath + "mu/P_{:03d}.html".format(self.cnt)
        self.plot_figure(value, vmin=vmin, vmax=vmax, filename=filename, title="mean", cmap="BrBG")

        """ plot mvar """
        filename = self.figpath + "mvar/P_{:03d}.html".format(self.cnt)
        value = mvar[self.ind_plot]
        vmin = np.amin(value)
        vmax = np.amax(value)
        self.plot_figure(value, vmin=vmin, vmax=vmax, filename=filename, title="marginal variance", cmap="RdBu")

    def plot_figure(self, value, vmin=0, vmax=30, filename=None, title=None, cmap=None):
        # points_grid, values_grid = interpolate_3d(self.xplot, self.yplot, self.zplot, value)
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        # fig.add_trace(go.Scatter3d(
        #     x=self.xplot,
        #     y=self.yplot,
        #     z=self.zplot,
        #     mode="markers",
        #     marker=dict(
        #         size=10,
        #         cmin=vmin,
        #         cmax=vmax,
        #         opacity=.3,
        #         color=value,
        #         colorscale=cmap,
        #         showscale=True,
        #         colorbar=dict(x=0.75, y=0.5, len=.5),
        #     )))
        fig.add_trace(go.Volume(
            x=self.xplot,
            y=self.yplot,
            z=self.zplot,
            value=value,
            isomin=vmin,
            isomax=vmax,
            opacity=.3,
            surface_count=15,
            colorscale=cmap,
            # coloraxis="coloraxis",
            colorbar=dict(x=0.75, y=0.5, len=.5),
            # reversescale=True,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))

        id = self.myopic.get_current_index()
        wp = self.myopic.wp.get_waypoint_from_ind(id)
        # wp = np.dot(self.RR, wp)
        fig.add_trace(go.Scatter3d(
            name="Current waypoint",
            x=[wp[1]],
            y=[wp[0]],
            z=[wp[2]],
            mode='markers',
            marker=dict(
                size=20,
                color="red",
                showscale=False,
            ),
            showlegend=True,  # remove all unnecessary trace names
        ),
            row='all', col='all'
        )

        id = self.myopic.get_next_index()
        wp = self.myopic.wp.get_waypoint_from_ind(id)
        # wp = np.dot(self.RR, wp)
        fig.add_trace(go.Scatter3d(
            name="Next waypoint",
            x=[wp[1]],
            y=[wp[0]],
            z=[wp[2]],
            mode='markers',
            marker=dict(
                size=20,
                color="blue",
                showscale=False,
            ),
            showlegend=True,  # remove all unnecessary trace names
        ),
            row='all', col='all'
        )

        id = self.myopic.get_pioneer_index()
        wp = self.myopic.wp.get_waypoint_from_ind(id)
        # wp = np.dot(self.RR, wp)
        fig.add_trace(go.Scatter3d(
            name="Pioneer waypoint",
            x=[wp[1]],
            y=[wp[0]],
            z=[wp[2]],
            mode='markers',
            marker=dict(
                size=20,
                color="green",
                showscale=False,
            ),
            showlegend=True,  # remove all unnecessary trace names
        ),
            row='all', col='all'
        )

        id = self.myopic.get_trajectory_indices()
        if len(id) > 0:
            wp = self.myopic.wp.get_waypoint_from_ind(id)
            # wp = (self.RR @ wp.T).T
            fig.add_trace(go.Scatter3d(
                name="Trajectory",
                x=wp[:, 1],
                y=wp[:, 0],
                z=wp[:, 2],
                mode='markers+lines',
                marker=dict(
                    size=5,
                    color="black",
                    showscale=False,
                ),
                line=dict(
                    color="yellow",
                    width=3,
                    showscale=False,
                ),
                showlegend=True,
            ),
                row='all', col='all'
            )

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.25, y=-1.25, z=1.25)
        )
        fig.update_layout(
            title={
                'text': "Conditional " + title + " field",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            scene=dict(
                zaxis=dict(nticks=4, range=[-5.5, 0], ),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="East", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="North", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Depth", font=dict(size=18, family="Times New Roman")),
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=.5),
            scene_camera=camera,
        )
        plotly.offline.plot(fig, filename=filename, auto_open=False)

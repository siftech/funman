import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from funman.representation import Box, Interval, ParameterSpace, Point

logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("matplotlib.pyplot").disabled = True


class ParameterSpacePlotter:
    def __init__(
        self,
        parameter_space: ParameterSpace,
        plot_bounds: Box = None,
        title: str = "Feasible Regions",
        color_map: Dict[str, str] = {
            "true": "g",
            "false": "r",
            "unknown": "b",
        },
        shape_map: Dict[str, str] = {"true": "x", "false": "o"},
        alpha=0.2,
        plot_points=False,
        parameters=None,
        dpi=100,
    ):
        if isinstance(parameter_space, ParameterSpace):
            self.ps = parameter_space
        else:
            # FIXME this is a hack to accept ParameterSpace objects from the openapi client
            self.ps = ParameterSpace.model_validate(parameter_space.to_dict())

        # TODO this should be easier to access
        values = []
        true_points = self.ps.true_points()
        false_points = self.ps.false_points()
        if len(true_points) > 0:
            values = true_points[0].values
        elif len(false_points) > 0:
            values = false_points[0].values

        self.parameters = [k for k in values if parameters and k in parameters]
        self.dim = len(self.parameters)
        self.plot_points = plot_points

        self.plot_bounds = plot_bounds if plot_bounds else self.computeBounds()
        self.title = title
        self.color_map = color_map
        self.shape_map = shape_map
        # clear_output(wait=True)
        self.custom_lines = [
            Line2D([0], [0], color="g", lw=4, alpha=alpha),
            Line2D([0], [0], color="r", lw=4, alpha=alpha),
        ]
        self.dpi = dpi

    def computeBounds(self, interval: Interval = Interval(lb=-2000, ub=2000)):
        box = Box(bounds={p: interval for p in self.parameters})
        return box

    def map_param_idx_to_plot_loc(self, i, j, plot_diagonal):
        if plot_diagonal:
            return i, j
        elif i == 0 or j == self.dim - 1:
            return None, None
        else:
            return i - 1, j

    def initialize_figure(self, plot_diagonal):
        if self.dim == 0:
            return

        dim_to_plot = self.dim if plot_diagonal else self.dim - 1
        fig, axs = plt.subplots(
            dim_to_plot,
            dim_to_plot,
            squeeze=False,
            dpi=self.dpi,
            figsize=(2 * dim_to_plot, 2 * dim_to_plot),
        )
        self.fig = fig
        self.axs = axs

        TINY_SIZE = 6
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12

        # plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
        plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=TINY_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=TINY_SIZE)  # fontsize of the tick labels
        # plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
        # plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

        # self.fig.tight_layout(pad=3.0)
        self.fig.tight_layout(pad=1.5)
        self.data = [[None] * self.dim] * self.dim

        for i in range(self.dim):
            for j in range(self.dim):
                i_coord, j_coord = self.map_param_idx_to_plot_loc(
                    i, j, plot_diagonal
                )
                if i_coord is None or j_coord is None:
                    continue

                if j_coord > i_coord:
                    axs[i_coord, j_coord].axis("off")
                else:
                    (self.data[i][j],) = self.axs[i_coord, j_coord].plot(
                        [], []
                    )
                    axs[i_coord, j_coord].set_xlabel(f"{self.parameters[i]}")
                    axs[i_coord, j_coord].set_ylabel(f"{self.parameters[j]}")
        self.fig.suptitle(self.title)
        plt.legend(self.custom_lines, ["true", "false"])

    def plot(self, show=False, plot_diagonal=False):
        self.initialize_figure(plot_diagonal)
        t = "true"
        f = "false"
        for b in self.ps.false_boxes:
            self.plotNDBox(b, self.color_map[f], plot_diagonal=plot_diagonal)
        for b in self.ps.true_boxes:
            self.plotNDBox(b, self.color_map[t], plot_diagonal=plot_diagonal)
        if self.plot_points:
            for p in self.ps.false_points():
                self.plot_add_point(
                    p,
                    self.color_map[f],
                    self.shape_map[f],
                    plot_diagonal=plot_diagonal,
                )
            true_points = self.ps.true_points()
            for p in true_points:
                self.plot_add_point(
                    p,
                    self.color_map[t],
                    self.shape_map[t],
                    plot_diagonal=plot_diagonal,
                )
        if show:
            plt.show(block=False)

    def plot_add_point(
        self,
        point: Point,
        color="r",
        shape="x",
        alpha=0.9,
        plot_diagonal=False,
    ):
        for i in range(self.dim):
            for j in range(self.dim):
                i_coord, j_coord = self.map_param_idx_to_plot_loc(
                    i, j, plot_diagonal
                )
                if i_coord is None or j_coord is None:
                    continue
                if j_coord > i_coord:
                    continue
                yval = (
                    point.values[self.parameters[j]] if self.dim > 1 else 0.0
                )
                self.axs[i_coord, j_coord].scatter(
                    point.values[self.parameters[i]],
                    yval,
                    color=color,
                    marker=shape,
                    alpha=alpha,
                    s=10,
                )
                # self.fig.canvas.draw()
                # self.fig.canvas.flush_events()

    def plotNDBox(self, box, color="g", alpha=0.2, plot_diagonal=False):
        for i in range(self.dim):
            for j in range(self.dim):
                i_coord, j_coord = self.map_param_idx_to_plot_loc(
                    i, j, plot_diagonal
                )
                if i_coord is None or j_coord is None:
                    continue
                if j_coord > i_coord:
                    continue

                x_limits = box.bounds[self.parameters[i]]
                y_limits = box.bounds[self.parameters[j]]

                if i == j:
                    # Plot a line segment
                    self.axs[i_coord, j_coord].plot(
                        [x_limits.lb, x_limits.ub],
                        [x_limits.lb, x_limits.ub],
                        color=color,
                        linewidth=3,
                        alpha=alpha,
                    )
                else:
                    # Plot a box
                    if (
                        abs(float(x_limits.lb)) < 1000
                        and abs(float(x_limits.ub)) < 1000
                    ):
                        x = np.linspace(
                            float(x_limits.lb), float(x_limits.ub), 1000
                        )
                        self.axs[i_coord, j_coord].fill_between(
                            x,
                            y_limits.lb,
                            y_limits.ub,
                            color=color,
                            alpha=alpha,
                        )

import json
import logging
from typing import Dict

import matplotlib.pyplot as plt
from funman_demo.parameter_space_plotter import ParameterSpacePlotter
from IPython.display import clear_output
from matplotlib.lines import Line2D

from funman import Box, Parameter, Point
from funman.representation.parameter_space import ParameterSpace

l = logging.getLogger(__file__)
l.setLevel(logging.INFO)


def plot_parameter_space(ps: ParameterSpace, alpha: float = 0.2, clear=False):
    custom_lines = [
        Line2D([0], [0], color="g", lw=4, alpha=alpha),
        Line2D([0], [0], color="r", lw=4, alpha=alpha),
    ]
    plt.title("Parameter Space")
    plt.xlabel("beta_0")
    plt.ylabel("beta_1")
    plt.legend(custom_lines, ["true", "false"])
    for b1 in ps.true_boxes:
        BoxPlotter.plot2DBoxList(b1, color="g", alpha=alpha)
    for b1 in ps.false_boxes:
        BoxPlotter.plot2DBoxList(b1, color="r", alpha=alpha)
    # plt.show(block=True)
    if clear:
        clear_output(wait=True)


# TODO this behavior could be pulled into search_utils if we
# find reason to pause and restart a search
def plot_cached_search(search_path, alpha: float = 0.2):
    true_boxes = []
    false_boxes = []
    true_points = []
    false_points = []

    with open(search_path) as f:
        for line in f.readlines():
            if len(line) == 0:
                continue
            data = json.loads(line)
            inst = ParameterSpace.decode_labeled_object(data)
            label = inst.label
            if isinstance(inst, Box):
                if label == "true":
                    true_boxes.append(inst)
                elif label == "false":
                    false_boxes.append(inst)
                else:
                    l.info(f"Skipping Box with label: {label}")
            elif isinstance(inst, Point):
                if label == "true":
                    true_points.append(inst)
                elif label == "false":
                    false_points.append(inst)
                else:
                    l.info(f"Skipping Point with label: {label}")
            else:
                l.error(f"Skipping invalid object type: {type(inst)}")
    plot_parameter_space(
        ParameterSpace(true_boxes, false_boxes, true_points, false_points),
        alpha=alpha,
    )


def summarize_results(
    variables,
    results,
    ylabel="Height",
    parameters_to_plot=None,
    label_color={"true": "g", "false": "r"},
) -> str:
    points = results.points()
    boxes = results.parameter_space.boxes()

    point_info = ""
    if points and len(points) > 0:
        point: Point = points[-1]
        parameters: Dict[Parameter, float] = results.point_parameters(point)
        results.plot(
            variables=variables,
            label_marker={"true": ",", "false": ","},
            xlabel="Time",
            ylabel=ylabel,
            legend=variables,
            label_color=label_color,
        )
        parameter_values = {p: point.values[p.name] for p in parameters}
        point_info = f"""Parameters = {parameter_values}
        # {parameters}
        {results.dataframe([point])}
        """
    else:
        # if there are no points, then we have a box that we found without needing points
        box = boxes[0]
        point_info = f"""Found box with no points
        {json.dumps(box.explain(), indent=4)}
        """

    boxes = results.parameter_space.boxes()
    if parameters_to_plot is None:
        parameters_to_plot = results.model._parameter_names() + ["timestep"]
    if len(boxes) > 0 and len(parameters_to_plot) > 1:
        ParameterSpacePlotter(
            results.parameter_space,
            parameters=parameters_to_plot,
            dpi=len(parameters_to_plot) * 20,
            plot_points=True,
        ).plot(show=True)

    divider = "*" * 80

    summary = f"""{divider}
{divider}
* Analysis Summary
{divider}
{len(points)} Points (+:{len(results.parameter_space.true_points())}, -:{len(results.parameter_space.false_points())}), {len(boxes)} Boxes (+:{len(results.parameter_space.true_boxes)}, -:{len(results.parameter_space.false_boxes)})
{point_info}
{divider}
    """

    return summary

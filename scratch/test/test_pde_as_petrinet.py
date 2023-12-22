# This notebook illustrates the halfar ice model

# Import funman related code
import json
import os
import unittest
from typing import Dict

from funman import Box, Parameter, Point
from funman.api.run import Runner

RESOURCES = os.path.join(os.getcwd(), "../../resources")
EXAMPLE_DIR = os.path.join(RESOURCES, "amr", "halfar")
MODEL_PATH = os.path.join(EXAMPLE_DIR, "halfar.json")
REQUEST_PATH = os.path.join(EXAMPLE_DIR, "halfar_request.json")


class TestUseCases(unittest.TestCase):
    def summarize_results(self, variables, results):
        points = results.points()
        boxes = results.parameter_space.boxes()

        print(
            f"{len(points)} Points (+:{len(results.parameter_space.true_points())}, -:{len(results.parameter_space.false_points())}), {len(boxes)} Boxes (+:{len(results.parameter_space.true_boxes)}, -:{len(results.parameter_space.false_boxes)})"
        )
        if points and len(points) > 0:
            point: Point = points[-1]
            parameters: Dict[Parameter, float] = results.point_parameters(
                point
            )
            results.plot(
                variables=variables,
                label_marker={"true": ",", "false": ","},
                xlabel="Time",
                ylabel="Height",
                legend=variables,
                label_color={"true": "g", "false": "r"},
            )
            parameter_values = {p: point.values[p.name] for p in parameters}
            print(f"Parameters = {parameter_values}")
            print(parameters)
            print(results.dataframe([point]))
        else:
            # if there are no points, then we have a box that we found without needing points
            print(f"Found box with no points")
            box = boxes[0]
            print(json.dumps(box.explain(), indent=4))

    def test_advection(self):
        # Advection Model

        num_disc = 5

        MODEL_PATH = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../..",
            f"resources/amr/advection_1d/advection_1d_forward.json",
        )
        # MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", f"resources/amr/advection_1d/advection_1d_backward.json")
        # MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", f"resources/amr/advection_1d/advection_1d_centered.json")

        height_bounds = [
            {
                "name": f"pos_u_{i}",
                "variable": f"u_{i}",
                "interval": {"lb": 0, "ub": 1.2},
            }
            for i in range(num_disc)
        ]

        request_dict = {
            "structure_parameters": [
                {
                    "name": "schedules",
                    "schedules": [{"timepoints": range(0, 10, 1)}],
                },
            ],
            "parameters": [
                {
                    "name": "dx",
                    "label": "any",
                    #  "interval": {"lb":1e-18, "ub":1e-14}}
                    "interval": {"lb": 1e-1, "ub": 1},
                },
                {
                    "name": "a",
                    "label": "any",
                    #  "interval": {"lb":1e-18, "ub":1e-14}}
                    "interval": {"lb": -1, "ub": 0},
                },
            ],
            "constraints": height_bounds
            + [
                # 0 <= dx - a
                # a <= dx
                {
                    "name": "dx_gte_a",
                    "variables": ["dx", "a"],
                    "weights": [1, -1],
                    "additive_bounds": {"lb": 0},
                    # "timepoints": {"lb": 0}
                },
                # {
                #     "name": "preserve_magnitude",
                #     "variable": "u_0",
                #     "interval": {"lb": 0.2},
                #     "timepoints": {"lb": 9},
                # }
            ],
            "config": {
                "use_compartmental_constraints": False,
                "normalization_constant": 1.0,
                "tolerance": 1e-2,
                "verbosity": 10,
                "dreal_mcts": True,
                # "dreal_precision": 1,
                "save_smtlib": "./out",
                "substitute_subformulas": False,
                "series_approximation_threshold": None,
                "dreal_log_level": "none",
                "profile": False,
            },
        }
        variables = [f"u_{d}" for d in range(num_disc)]

        # Use request_dict
        results = Runner().run(
            MODEL_PATH,
            request_dict,
            # REQUEST_PATH,
            description="Halfar demo",
            case_out_dir="./out",
            dump_plot=True,
            parameters_to_plot=["a", "dx", "timestep"],
            point_plot_config={
                "variables": variables,
                "label_marker": {"true": ",", "false": ","},
                "xlabel": "Time",
                "ylabel": "Height",
                "legend": variables
                # ,"label_color":{"true": "g", "false":"r"}
            },
            num_points=None,
        )

        self.summarize_results(variables, results)


if __name__ == "__main__":
    unittest.main()

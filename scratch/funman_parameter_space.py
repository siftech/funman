import json
import os

from funman import Point
from funman.api.run import Runner


def main():
    # Setup Paths

    RESOURCES = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../resources"
    )
    EXAMPLE_DIR = os.path.join(RESOURCES, "amr", "petrinet", "amr-examples")
    MODEL_PATH = os.path.join(EXAMPLE_DIR, "sir.json")
    REQUEST_PATH = os.path.join(EXAMPLE_DIR, "sir_request1.json")

    request_dict = {
        # "query": {
        #     "variable": "I",
        #     "ub": 300
        # },
        "constraints": [
            {
                "name": "I_bounds_A",
                "variable": "I",
                "interval": {"lb": 1},
                "timepoints": {"lb": 55, "ub": 60, "closed_upper_bound": True},
            },
            {
                "name": "I_bounds_B",
                "variable": "I",
                "interval": {"ub": 20},
                "timepoints": {"lb": 0, "ub": 20},
            },
        ],
        "parameters": [
            {
                "name": "beta",
                "interval": {"lb": 1.0e-3, "ub": 3.0e-3},
                "label": "all",
            },
            {
                "name": "gamma",
                "interval": {"lb": 0.1, "ub": 0.18},
                "label": "all",
            },
            {
                "name": "S0",
                "interval": {"lb": 1000, "ub": 1000},
                "label": "any",
            },
            {"name": "I0", "interval": {"lb": 1, "ub": 1}, "label": "any"},
            {"name": "R0", "interval": {"lb": 0, "ub": 0}, "label": "any"},
        ],
        "structure_parameters": [
            # {
            #     "name": "num_steps",
            #     "interval": {"lb": 1, "ub": 2, "closed_upper_bound": True},
            #     "label": "all",
            # },
            # {
            #     "name": "step_size",
            #     "interval": {"lb": 1, "ub": 1, "closed_upper_bound": True},
            #     "label": "all",
            # },
            {
                "name": "schedules",
                "schedules": [
                    {
                        "timepoints": [
                            0,
                            5,
                            10,
                            15,
                            20,
                            30,
                            35,
                            45,
                            50,
                            55,
                            60,
                            100,
                        ]
                    }
                ],
            }
        ],
        "config": {
            "normalization_constant": 1001,
            "use_compartmental_constraints": True,
            "tolerance": 5e-3,
        },
    }

    # Use request_dict
    results = Runner().run(
        MODEL_PATH,
        request_dict,
        description="Basic SIR with simple request",
        case_out_dir="./out",
    )
    points = results.points()
    boxes = results.parameter_space.boxes()

    print(
        f"{len(points)} Points (+:{len(results.parameter_space.true_points)}, -:{len(results.parameter_space.false_points)}), {len(boxes)} Boxes (+:{len(results.parameter_space.true_boxes)}, -:{len(results.parameter_space.false_boxes)})"
    )
    if points and len(points) > 0:
        point: Point = points[-1]
        parameters: Dict[Parameter, float] = results.point_parameters(point)
        print(parameters)
        print(results.dataframe([point]))
    else:
        # if there are no points, then we have a box that we found without needing points

        box = boxes[0]
        print(json.dumps(box.explain(), indent=4))


if __name__ == "__main__":
    main()

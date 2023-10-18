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
            # {
            #     "name": "compartmental bounds",
            #     "additive_bounds": {"lb": 1001, "ub": 1002},
            #     "variables": ["S", "I", "R"]
            # },
            {
                "name": "I_bounds",
                "variable": "I",
                "bounds": {"lb": 0.85},
                "timepoints": {"lb": 1, "ub": 2},
            },
            {
                "name": "I_bounds",
                "variable": "I",
                "bounds": {"lb": 0.8},
                "timepoints": {"lb": 2, "ub": 3},
            },
            {
                "name": "I_bounds",
                "variable": "I",
                "bounds": {"lb": 0.7},
                "timepoints": {"lb": 3, "ub": 4},
            },
            # {
            #     "name": "R_bounds",
            #     "variable" : "R",
            #     "bounds": {"lb":0, "ub":1},
            #     "timepoints": {"lb":0, "ub":2}
            # },
            #   {
            #   "name": "S_bounds",
            #   "variable" : "S",
            #   "bounds": {"lb":980, "ub":1000},
            #   "timepoints": {"lb":4, "ub":5}
            #  }
        ],
        "parameters": [
            {"name": "beta", "lb": 2.6e-7, "ub": 2.8e-7, "label": "all"},
            {"name": "gamma", "lb": 0.1, "ub": 0.18, "label": "all"},
            {"name": "S0", "lb": 1000, "ub": 1000, "label": "any"},
            {"name": "I0", "lb": 1, "ub": 1, "label": "any"},
            {"name": "R0", "lb": 0, "ub": 0, "label": "any"},
        ],
        "structure_parameters": [
            # {"name": "num_steps", "lb": 1, "ub": 10, "label": "all"},
            # {"name": "step_size", "lb": 1, "ub": 1, "label": "all"},
            {
                "name": "schedules",
                "schedules": [{"timepoints": [0, 50, 55, 60, 100]}],
            }
        ],
        "config": {
            "normalize": False,
            "tolerance": 1e-3,
            "simplify_query": False,
            "normalization_constant": 1001,
            # "use_compartmental_constraints" : False,
            # "profile": True
            "save_smtlib": True,
            # "substitute_subformulas": False
            "taylor_series_order": None,
            #   "dreal_log_level": "debug"
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


# Use request file
# results = Runner().run(MODEL_PATH, REQUEST_PATH, description="Basic SIR with simple request", case_out_dir="./out")

if __name__ == "__main__":
    main()

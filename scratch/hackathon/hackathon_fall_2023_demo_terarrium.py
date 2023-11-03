import json
import os

from funman import Point
from funman.api.run import Runner


def main():
    # Setup Paths

    RESOURCES = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../resources"
    )
    EXAMPLE_DIR = os.path.join(RESOURCES, "amr", "petrinet", "mira")
    MODEL_PATH = os.path.join(
        EXAMPLE_DIR, "models", "BIOMD0000000955_askenet.json"
    )
    REQUEST_PATH = os.path.join(
        EXAMPLE_DIR, "requests", "BIOMD0000000955_askenet_request.json"
    )

    request_dict = {
        "parameters": [
            {
                "name": "beta",
                "interval": {
                    "lb": 0.011,
                    "ub": 0.011,
                    "closed_upper_bound": True,
                }
                # "interval": {"lb": 0.008799999999999999, "ub": 0.0132},
            },
            {
                "name": "gamma",
                "interval": {
                    "lb": 0.456,
                    "ub": 0.456,
                    "closed_upper_bound": True,
                }
                #  "interval": {"lb": 0.3648, "ub": 0.5472}
            },
            {
                "name": "delta",
                "interval": {
                    "lb": 0.011,
                    "ub": 0.011,
                    "closed_upper_bound": True,
                }
                # "interval": {"lb": 0.008799999999999999, "ub": 0.0132},
            },
            {
                "name": "alpha",
                "interval": {
                    "lb": 0.57,
                    "ub": 0.57,
                    "closed_upper_bound": True,
                }
                # "interval": {
                #     "lb": 0.45599999999999996,
                #     "ub": 0.6839999999999999,
                # },
            },
            {
                "name": "epsilon",
                "interval": {"lb": 0.1368, "ub": 0.20520000000000002},
                "label": "all",
            },
            {
                "name": "zeta",
                "interval": {
                    "lb": 0.125,
                    "ub": 0.125,
                    "closed_upper_bound": True,
                }
                #  "interval": {"lb": 0.1, "ub": 0.15}
            },
            {
                "name": "lambda",
                "interval": {
                    "lb": 0.034,
                    "ub": 0.034,
                    "closed_upper_bound": True,
                }
                # "interval": {"lb": 0.027200000000000002, "ub": 0.0408},
            },
            {
                "name": "eta",
                "interval": {
                    "lb": 0.125,
                    "ub": 0.125,
                    "closed_upper_bound": True,
                }
                #  "interval": {"lb": 0.1, "ub": 0.15}
            },
            {
                "name": "rho",
                "interval": {
                    "lb": 0.034,
                    "ub": 0.034,
                    "closed_upper_bound": True,
                }
                # "interval": {"lb": 0.027200000000000002, "ub": 0.0408},
            },
            {
                "name": "theta",
                "interval": {"lb": 0.2968, "ub": 0.4452},
                "label": "all",
            },
            {
                "name": "kappa",
                "interval": {
                    "lb": 0.017,
                    "ub": 0.017,
                    "closed_upper_bound": True,
                }
                # "interval": {"lb": 0.013600000000000001, "ub": 0.0204},
            },
            {
                "name": "mu",
                "interval": {
                    "lb": 0.017,
                    "ub": 0.017,
                    "closed_upper_bound": True,
                }
                # "interval": {"lb": 0.013600000000000001, "ub": 0.0204},
            },
            {
                "name": "nu",
                "interval": {
                    "lb": 0.027,
                    "ub": 0.027,
                    "closed_upper_bound": True,
                }
                #   "interval": {"lb": 0.0216, "ub": 0.0324}
            },
            {
                "name": "xi",
                "interval": {
                    "lb": 0.017,
                    "ub": 0.017,
                    "closed_upper_bound": True,
                }
                # "interval": {"lb": 0.013600000000000001, "ub": 0.0204},
            },
            {
                "name": "tau",
                "interval": {
                    "lb": 0.01,
                    "ub": 0.01,
                    "closed_upper_bound": True,
                }
                #  "interval": {"lb": 0.008, "ub": 0.012}
            },
            {
                "name": "sigma",
                "interval": {
                    "lb": 0.017,
                    "ub": 0.017,
                    "closed_upper_bound": True,
                }
                # "interval": {"lb": 0.013600000000000001, "ub": 0.0204},
            },
        ],
        "constraints": [
            {
                "name": "theta_epsilon",
                "additive_bounds": {"lb": 0},
                "variables": ["theta", "epsilon"],
                "weights": [1, -2],
                # No timepoints, because the variables are parameters
            },
            # {
            #     "name": "infected_maximum3",
            #     "variable": "Infected",
            #     "interval": { "ub": 0.7},
            #     "timepoints": {"lb": 130},
            # },
            # {
            #     "name": "infected_maximum1",
            #     "variable": "Infected",
            #     "interval": {  "ub": 0.4},
            #     "timepoints": {"lb": 70, "ub": 75, "closed_upper_bound": True},
            # },
            # {
            #     "name": "infected_maximum2",
            #     "variable": "Infected",
            #     "interval": {"ub": 0.2},
            #     "timepoints": { "ub": 75},
            # },
            # {
            #     "name": "infected_maximum3",
            #     "variable": "Infected",
            #     "interval": {"ub": 0.01},
            #     "timepoints": {"lb": 76},
            # },
        ],
        "structure_parameters": [
            {
                "name": "schedules",
                "schedules": [
                    {"timepoints": [0, 10, 30, 50, 70, 90, 110, 130, 150]}
                    # {"timepoints": [0, 10]}
                ],
            }
        ],
        "config": {
            "use_compartmental_constraints": True,
            "normalization_constant": 1.0,
            "tolerance": 1e-5,
            "verbosity": 10,
            "dreal_mcts": True,
            # "save_smtlib": os.path.join(os.path.realpath(__file__), "./out"),
            "substitute_subformulas": False,
            "series_approximation_threshold": None,
            "dreal_log_level": "none",
            "profile": False,
        },
    }

    # Use request_dict
    results = Runner().run(
        MODEL_PATH,
        request_dict,
        # REQUEST_PATH,
        description="SIDARTHE demo",
        case_out_dir="./out",
    )
    points = results.points()
    boxes = results.parameter_space.boxes()

    print(
        f"{len(points)} Points (+:{len(results.parameter_space.true_points())}, -:{len(results.parameter_space.false_points())}), {len(boxes)} Boxes (+:{len(results.parameter_space.true_boxes)}, -:{len(results.parameter_space.false_boxes)})"
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

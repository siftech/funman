import json
import os

from funman import Point
from funman.api.run import Runner


def main():
    # Setup Paths

    RESOURCES = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../resources"
    )
    EXAMPLE_DIR = os.path.join(RESOURCES, "amr", "halfar")
    MODEL_PATH = os.path.join(EXAMPLE_DIR, "halfar.json")
    REQUEST_PATH = os.path.join(EXAMPLE_DIR, "halfar_request.json")

    request_dict = {
        "structure_parameters": [
            {
                "name": "schedules",
                "schedules": [
                    {"timepoints": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
                ],
            }
        ],
        "config": {
            "use_compartmental_constraints": False,
            "normalization_constant": 1.0,
            "tolerance": 1e-1,
            "verbosity": 10,
            "dreal_mcts": True,
            "save_smtlib": True,
            "substitute_subformulas": False,
            "series_approximation_threshold": None,
            "dreal_log_level": "none",
            "profile": False,
        },
    }

    # Use request_dict
    results: FunmanResults = Runner().run(
        MODEL_PATH,
        request_dict,
        # REQUEST_PATH,
        description="Halfar demo",
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

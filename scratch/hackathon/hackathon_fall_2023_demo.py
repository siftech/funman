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

    # Use request_dict
    results = Runner().run(
        MODEL_PATH,
        # request_dict,
        REQUEST_PATH,
        description="Basic SIR with simple request",
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

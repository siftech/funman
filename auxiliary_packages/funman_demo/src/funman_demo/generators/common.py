import argparse
import os
from typing import Dict, List, Union

from pydantic import BaseModel


class Direction:
    Negative: str = "negative"
    Positive: str = "positive"


class Derivative:
    CENTERED: str = "CENTERED"
    FORWARD: str = "FORWARD"
    BACKWARD: str = "BACKWARD"


class Boundary(BaseModel):
    def id_str(self):
        return "boundary"


class Coordinate(BaseModel):
    """
    Coordinates are N-D points in Cartesian space, as denoted by the vector attribute.  The neighbors are coordinates that are at the next point in each direction.
    """

    vector: List[float]
    id: List[int]
    neighbors: List[Dict[str, Union[List[int], Boundary]]] = []

    def id_str(self):
        return "_".join([str(i) for i in self.id])

    def positive_neighbor(self, dimension, coordinates=None):
        return self.neighbor(
            dimension, Direction.Positive, coordinates=coordinates
        )

    def negative_neighbor(self, dimension, coordinates=None):
        return self.neighbor(
            dimension, Direction.Negative, coordinates=coordinates
        )

    def neighbor(self, dimension, direction, coordinates=None):
        id = self.neighbors[dimension][direction]
        if coordinates:
            if not isinstance(id, Boundary):
                return coordinates.get(id, None)
        return id


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--derivative",
        default=Derivative.CENTERED,
        type=str,
        choices=[Derivative.CENTERED, Derivative.FORWARD, Derivative.BACKWARD],
    )
    parser.add_argument(
        "-d",
        "--dimensions",
        default=2,
        type=int,
        help=f"Number of spatial dimensions",
    )
    parser.add_argument(
        "-t",
        "--discretization-type",
        default=1,
        type=int,
        choices=["centered"],
        help=f"Number of spatial dimensions",
    )
    parser.add_argument(
        "-b",
        "--boundary-slope",
        default=0.1,
        type=float,
        help=f"Time-dependent boundary function parameter f(t) = bt",
    )

    parser.add_argument(
        "-p",
        "--num-discretization-points",
        default=3,
        type=int,
        help=f"Number of discretization points",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default="model.json",
        help=f"Output filename",
    )
    return parser.parse_args()


def main(args, generator, model):
    assert (
        args.num_discretization_points > 2
    ), "Need to have use at least 3 discretization points to properly define the gradients."

    gen = generator()
    mod, semantics = gen.model(args)
    amr_model = model(
        header=gen.header(),
        model=mod,
        semantics=semantics,
    )
    j = amr_model.model_dump_json(indent=4)

    with open(args.outfile, "w") as f:
        print(f"Writing {os.path.join(os.getcwd(), args.outfile)}")
        f.write(j)


if __name__ == "__main__":
    main()

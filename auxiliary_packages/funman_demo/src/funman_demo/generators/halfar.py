"""
This script will generate instances of the Halfar ice model as Petrinet AMR models.  The options control the number of discretization points.
"""

import argparse
import os
from decimal import Decimal
from typing import Dict, List, Tuple

from pydantic import AnyUrl, BaseModel, Field

from funman.model.generated_models.petrinet import (
    Distribution,
    Header,
    Initial,
    Model,
    Model1,
    OdeSemantics,
    Parameter,
    Properties,
    Rate,
    Semantics,
    State,
    Time,
    Transition,
    Unit,
)
from funman.representation.interval import Interval


class HalfarModel(Model):
    pass


class Direction:
    Negative: str = "negative"
    Positive: str = "positive"


class Coordinate(BaseModel):
    """
    Coordinates are N-D points in Cartesian space, as denoted by the vector attribute.  The neighbors are coordinates that are at the next point in each direction.
    """

    vector: List[float]
    id: List[int]
    neighbors: List[Dict[str, List[int]]] = []

    def id_str(self):
        return "_".join([str(i) for i in self.id])


class HalfarGenerator:
    """
    This generator class constructs the AMR instance.  The coordinates are equally spaced across the range.
    """

    range: Interval = Interval(lb=-2.0, ub=2.0)

    def coordinates(self, args) -> List[Coordinate]:
        """
        Generate coordinates for the range.
        """
        coords = []
        step_size = value = self.range.width() / Decimal(
            int(args.num_discretization_points) - 1
        )

        # Discretize each dimension
        axes = [
            [
                self.range.lb + float(step_size * i)
                for i in range(args.num_discretization_points)
            ]
            for d in range(args.dimensions)
        ]

        # Build all points as the cross product of axes
        points = [[]]
        ids = [[]]
        for axis in axes:
            next_points = []
            next_ids = []
            for id, point in zip(ids, points):
                for did, value in enumerate(axis):
                    next_point = point.copy() + [value]
                    next_id = id + [did]
                    next_points.append(next_point)
                    next_ids.append(next_id)
            points = next_points
            ids = next_ids

        coords = {
            tuple(id): Coordinate(vector=point, id=id)
            for (id, point) in zip(ids, points)
        }

        for id, coord in coords.items():
            for dim in range(args.dimensions):
                coord.neighbors.append({})
                prev_id = tuple(
                    [(i if d != dim else i - 1) for d, i in enumerate(id)]
                )
                coord.neighbors[dim][Direction.Negative] = (
                    prev_id if prev_id[dim] >= 0 else None
                )
                next_id = tuple(
                    [(i if d != dim else i + 1) for d, i in enumerate(id)]
                )
                coord.neighbors[dim][Direction.Positive] = (
                    next_id
                    if next_id[dim] < int(args.num_discretization_points)
                    else None
                )
        return coords

    def transition_rate(
        self,
        coordinate: Coordinate,
        dimension: int,
        coordinates: Dict[Tuple, Coordinate],
        args,
    ) -> str:
        """
        Custom rate change
        """
        next_coord_id = coordinate.neighbors[dimension][Direction.Positive]
        prev_coord_id = coordinate.neighbors[dimension][Direction.Negative]

        next_coord = coordinates.get(next_coord_id, None)
        prev_coord = coordinates.get(prev_coord_id, None)

        coord_str = f"h_{coordinate.id_str()}"
        next_str = (
            f"h_{next_coord.id_str()}"
            if next_coord
            else f"({args.boundary_slope}*t)"
        )
        prev_str = (
            f"-h_{prev_coord.id_str()}"
            if prev_coord
            else f"-({args.boundary_slope}*t)"
        )

        gamma = "283701998652.8*A"

        coord_x = coordinate.vector[dimension]
        next_x_dx = (
            next_coord.vector[dimension] - coord_x if next_coord else None
        )
        prev_x_dx = (
            coord_x - prev_coord.vector[dimension] if prev_coord else None
        )

        if next_x_dx is not None and prev_x_dx is not None:
            dx = next_x_dx + prev_x_dx
        elif next_x_dx is not None:
            dx = 2 * next_x_dx
        elif prev_x_dx is not None:
            dx = 2 * prev_x_dx
        else:
            raise Exception(
                "dx is undefined because coordinate has no neighbors"
            )

        return f"({gamma}/{dx})*((abs(({next_str}{prev_str})*0.5)**2)*(({next_str}{prev_str})*0.5)*({coord_str}**5))"

    def centered_difference(self, coordinate: Coordinate, coordinates, args):
        transitions = []
        rates = []
        # Get transition in each dimension
        for dimension, value in enumerate(coordinate.vector):
            # Transition for coordinate is: next_coord -- rate --> prev_coord
            rate = self.transition_rate(
                coordinate, dimension, coordinates, args
            )
            rates.append(
                Rate(
                    target=f"r_{dimension}_{coordinate.id_str()}",
                    expression=rate,
                )
            )
            next_coord_id = coordinate.neighbors[dimension][Direction.Positive]

            prev_coord_id = coordinate.neighbors[dimension][Direction.Negative]
            next_coord = coordinates.get(next_coord_id, None)
            prev_coord = coordinates.get(prev_coord_id, None)

            input_states = [f"h_{next_coord.id_str()}"] if next_coord else []
            output_states = [f"h_{prev_coord.id_str()}"] if prev_coord else []

            transition_name = f"r_{dimension}_{coordinate.id_str()}"
            transition = Transition(
                id=transition_name,
                input=input_states,
                output=output_states,
                properties=Properties(name=transition_name),
            )
            transitions.append(transition)

        return transitions, rates

    def states(self, args, coordinates) -> List[State]:
        # Create a height variable at each coordinate
        states = [
            State(
                id=f"h_{coord.id_str()}",
                name=f"h_{coord.id_str()}",
                description=f"height at {coord.vector}",
            )
            for id, coord in coordinates.items()
        ]
        return states

    def model(self, args) -> Tuple[Model1, Semantics]:
        """
        Generate the AMR Model
        """
        coordinates = self.coordinates(args)

        # Create a height variable at each coordinate
        states = self.states(args, coordinates)

        transitions = []
        rates = []
        for id, coordinate in coordinates.items():
            coord_transitions, trans_rates = self.centered_difference(
                coordinate, coordinates, args
            )
            transitions += coord_transitions
            rates += trans_rates

        time = Time(
            id="t",
            units=Unit(expression="day", expression_mathml="<ci>day</ci>"),
        )

        initials = [
            Initial(
                target=f"h_{c.id_str()}",
                expression=(
                    # "0.0"
                    # if any(
                    #     [
                    #         (
                    #             id == 0
                    #             or id == args.num_discretization_points - 1
                    #         )
                    #         for id in c.id
                    #     ]
                    # )
                    # else
                    f"{1.0/(1.0+max([abs(v) for v in c.vector]))}"
                ),
            )
            for id, c in coordinates.items()
        ]

        parameters = [
            # Parameter(
            #     id="n",
            #     value=3.0,
            #     distribution=Distribution(
            #         type="StandardUniform1",
            #         parameters={"minimum": 3.0, "maximum": 3.0},
            #     ),
            # ),
            # Parameter(
            #     id="rho",
            #     value=910.0,
            #     distribution=Distribution(
            #         type="StandardUniform1",
            #         parameters={"minimum": 910.0, "maximum": 910.0},
            #     ),
            # ),
            # Parameter(
            #     id="g",
            #     value=9.8,
            #     distribution=Distribution(
            #         type="StandardUniform1",
            #         parameters={"minimum": 9.8, "maximum": 9.8},
            #     ),
            # ),
            Parameter(
                id="A",
                value=1e-16,
                distribution=Distribution(
                    type="StandardUniform1",
                    parameters={"minimum": 1e-20, "maximum": 1e-12},
                ),
            ),
        ]

        return Model1(states=states, transitions=transitions), Semantics(
            ode=OdeSemantics(
                rates=rates,
                initials=initials,
                parameters=parameters,
                time=time,
            )
        )


def get_args():
    parser = argparse.ArgumentParser()
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
        default="halfar.json",
        help=f"Output filename",
    )
    return parser.parse_args()


def header():
    return Header(
        name="Halfar Model",
        schema_=AnyUrl(
            "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/petrinet_v0.1/petrinet/petrinet_schema.json"
        ),
        schema_name="petrinet",
        description="Halfar as Petrinet model created by Dan Bryce and Drisana Mosiphir",
        model_version="0.1",
    )


def main():
    args = get_args()

    assert (
        args.num_discretization_points > 2
    ), "Need to have use at least 3 discretization points to properly define the gradients."

    generator = HalfarGenerator()
    model, semantics = generator.model(args)
    halfar_model = HalfarModel(
        header=header(),
        model=model,
        semantics=semantics,
    )
    j = halfar_model.model_dump_json(indent=4)

    with open(args.outfile, "w") as f:
        print(f"Writing {os.path.join(os.getcwd(), args.outfile)}")
        f.write(j)


if __name__ == "__main__":
    main()

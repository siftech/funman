"""
This script will generate instances of the Halfar ice model as Petrinet AMR models.  The options control the number of discretization points.
"""

import argparse
import os
from decimal import Decimal
from typing import Dict, List, Tuple

from pydantic import AnyUrl, BaseModel

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
    id: str
    neighbors: Dict[str, "Coordinate"] = {}


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
        for i in range(args.num_discretization_points):
            value = self.range.lb + float(step_size * i)
            coords.append(Coordinate(vector=[value], id=str(i)))
        for i, coord in enumerate(coords):
            coord.neighbors[Direction.Negative] = (
                coords[i - 1] if i > 0 else None
            )
            coord.neighbors[Direction.Positive] = (
                coords[i + 1] if i < len(coords) - 1 else None
            )
        return coords

    def transition_expression(
        self, n1_name: str, n2_name: str, negative=False
    ) -> str:
        """
        Custom rate change
        """
        prefix = "-1*" if negative else ""
        gamma = "(2/(n+2))*A*(rho*g)**n"
        return f"{prefix}{gamma}*(({n2_name}-{n1_name})**3)*({n1_name}**5)"

    def neighbor_gradient(
        self, coordinate: Coordinate, coordinates: List[Coordinate]
    ) -> Tuple[List[Transition], List[Rate]]:
        """
        Find a triple of coordinates (n0, n1, n2) that are ordered so that dx = n2-n1 and dx = n1-n0 is positive
        """
        if (
            coordinate.neighbors[Direction.Positive]
            and coordinate.neighbors[Direction.Negative]
        ):
            n0 = coordinate.neighbors[Direction.Negative]
        elif (
            coordinate.neighbors[Direction.Positive]
            and not coordinate.neighbors[Direction.Negative]
        ):
            n0 = coordinate
        elif (
            not coordinate.neighbors[Direction.Positive]
            and coordinate.neighbors[Direction.Negative]
        ):
            n0 = coordinate.neighbors[Direction.Negative].neighbors[
                Direction.Negative
            ]
        else:
            raise Exception(
                "Cannot determine the gradient of a coordinate with no neighbors"
            )
        n1 = n0.neighbors[Direction.Positive]
        n2 = n1.neighbors[Direction.Positive]

        w_p_name = f"w_p_{coordinate.id}"
        w_n_name = f"w_n_{coordinate.id}"
        n0_name = f"h_{n0.id}"
        n1_name = f"h_{n1.id}"
        n2_name = f"h_{n2.id}"
        h_name = f"h_{coordinate.id}"

        # tp is the gradient wrt. n2, n1
        tp = Transition(
            id=w_p_name,
            input=[n2_name, n1_name],
            output=[h_name],
            properties=Properties(name=w_p_name),
        )

        # tn is the gradient wrt. n1, n0
        tn = Transition(
            id=w_n_name,
            input=[n1_name, n0_name],
            output=[h_name],
            properties=Properties(name=w_n_name),
        )

        transitions = [tp, tn]

        tpr = Rate(
            target=w_p_name,
            expression=self.transition_expression(n1_name, n2_name),
        )
        tnr = Rate(
            target=w_n_name,
            expression=self.transition_expression(
                n0_name, n1_name, negative=True
            ),
        )

        rates = [tpr, tnr]
        return transitions, rates

    def model(self, args) -> Tuple[Model1, Semantics]:
        """
        Generate the AMR Model
        """
        coordinates = self.coordinates(args)

        # Create a height variable at each coordinate
        states = [
            State(id=f"h_{i}", name=f"h_{i}", description=f"height at {i}")
            for i in range(len(coordinates))
        ]

        transitions = []
        rates = []
        for coordinate in coordinates[1:-1]:
            coord_transitions, trans_rates = self.neighbor_gradient(
                coordinate, coordinates
            )
            transitions += coord_transitions
            rates += trans_rates

        time = Time(
            id="t",
            units=Unit(expression="day", expression_mathml="<ci>day</ci>"),
        )

        initials = [
            Initial(
                target=f"h_{c.id}",
                expression=(
                    "0.0"
                    if (c == coordinates[0] or c == coordinates[-1])
                    else f"{1.0/(1.0+abs(c.vector[0]))}"
                ),
            )
            for c in coordinates
        ]

        parameters = [
            Parameter(
                id="n",
                value=3.0,
                distribution=Distribution(
                    type="StandardUniform1",
                    parameters={"minimum": 3.0, "maximum": 3.0},
                ),
            ),
            Parameter(
                id="rho",
                value=910.0,
                distribution=Distribution(
                    type="StandardUniform1",
                    parameters={"minimum": 910.0, "maximum": 910.0},
                ),
            ),
            Parameter(
                id="g",
                value=9.8,
                distribution=Distribution(
                    type="StandardUniform1",
                    parameters={"minimum": 9.8, "maximum": 9.8},
                ),
            ),
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
    # parser.add_argument(
    #     "-d",
    #     "--dimensions",
    #     default=1,
    #     type=int,
    #     help=f"Number of spatial dimensions",
    # )
    parser.add_argument(
        "-p",
        "--num-discretization-points",
        default=5,
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

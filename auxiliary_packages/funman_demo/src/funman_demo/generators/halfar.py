import argparse
import os
from decimal import Decimal
from typing import Dict, List, Tuple

from pydantic import AnyUrl, field_validator

from funman.model.generated_models.petrinet import *
from funman.representation.interval import Interval


class HalfarModel(Model):
    pass


class Direction:
    Negative: str = "negative"
    Positive: str = "positive"


class Coordinate(BaseModel):
    vector: List[float]
    id: str
    neighbors: Dict[str, "Coordinate"] = {}


class HalfarGenerator:
    range: Interval = Interval(lb=-2.0, ub=2.0)

    def coordinates(self, args) -> List[Coordinate]:
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
        prefix = "-1*" if negative else ""
        return f"{prefix}gamma*({n2_name}-{n1_name})**3*{n1_name}**5"

    def neighbor_gradient(
        self, coordinate: Coordinate, coordinates: List[Coordinate]
    ) -> Tuple[List[Transition], List[Rate]]:
        # find a triple of coordinates (n0, n1, n2) that are ordered so that dx is positive
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

        # tp is the gradient wrt. n2, n1
        tp = Transition(
            id=w_p_name,
            input=[n2_name, n1_name],
            output=[w_p_name],
            properties=Properties(name=w_p_name),
        )

        # tn is the gradient wrt. n1, n0
        tn = Transition(
            id=w_n_name,
            input=[n1_name, n0_name],
            output=[w_n_name],
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
        coordinates = self.coordinates(args)

        # Create a height variable at each coordinate
        states = [
            State(id=f"h_{i}", name=f"h_{i}", description=f"height at {i}")
            for i in range(len(coordinates))
        ]

        transitions = []
        rates = []
        for coordinate in coordinates:
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
                target=f"h_{c.id}", expression=f"{1.0/(1.0+abs(c.vector[0]))}"
            )
            for c in coordinates
        ]

        parameters = [
            Parameter(
                id="gamma",
                value=1.0,
                distribution=Distribution(
                    type="StandardUniform1",
                    parameters={"minimum": 0.0, "maximum": 1.0},
                ),
            )
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
        default=1,
        type=int,
        help=f"Number of spatial dimensions",
    )
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


def main():
    args = get_args()
    generator = HalfarGenerator()
    model, semantics = generator.model(args)
    halfar_model = HalfarModel(
        header=Header(
            name="Halfar Model",
            schema_=AnyUrl(
                "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/petrinet_v0.1/petrinet/petrinet_schema.json"
            ),
            schema_name="petrinet",
            description="Halfar as Petrinet model created by Dan",
            model_version="0.1",
        ),
        model=model,
        semantics=semantics,
    )
    j = halfar_model.model_dump_json(indent=4)
    # print(j)
    with open(args.outfile, "w") as f:
        print(f"Writing {os.path.join(os.getcwd(), args.outfile)}")
        f.write(j)


if __name__ == "__main__":
    main()

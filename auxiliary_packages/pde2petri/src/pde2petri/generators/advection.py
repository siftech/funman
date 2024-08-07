"""
This script will generate instances of the Halfar ice model as Petrinet AMR models.  The options control the number of discretization points.
"""

import sys
from typing import Dict, List, Tuple

from pde2petri.generators.common import (
    Boundary,
    Coordinate,
    Derivative,
    Direction,
    get_args,
)
from pde2petri.generators.common import main as common_main
from pde2petri.generators.generator import Generator
from pde2petri.model.petrinet import (
    Distribution,
    Grounding,
    Header,
    Initial,
    Model,
    Parameter,
)
from pydantic import AnyUrl


class AdvectionModel(Model):
    pass


class AdvectionGenerator(Generator):
    """
    This generator class constructs the AMR instance.  The coordinates are equally spaced across the range.
    """

    variables: List[str] = ["a"]

    def transition_rate(
        self,
        variable: str,
        coordinate: Coordinate,
        dimension: int,
        coordinates: Dict[Tuple, Coordinate],
        args,
    ) -> str:
        """
        Custom rate change
        """
        if not isinstance(coordinate, Boundary):
            coord_str = f"{variable}_{coordinate.id_str()}"
            rate = f"(-u/dx)*({coord_str})"
        else:
            rate = self.boundary_expression(args)
        return rate

    def parameters(self) -> List[Parameter]:
        return [
            Parameter(
                id="u",
                value=1.0,
                distribution=Distribution(
                    type="StandardUniform1",
                    parameters={"minimum": 0.0, "maximum": 1.0},
                ),
            ),
            # Parameter(
            #     id="dt",
            #     value=1,
            #     distribution=Distribution(
            #         type="StandardUniform1",
            #         parameters={"minimum": 0, "maximum": 1},
            #     ),
            # ),
            Parameter(
                id="dx",
                value=1,
                distribution=Distribution(
                    type="StandardUniform1",
                    parameters={"minimum": 0, "maximum": 1},
                ),
            ),
        ]

    def header(self):
        return Header(
            name="Advection Model",
            schema=AnyUrl(
                "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/petrinet_v0.1/petrinet/petrinet_schema.json"
            ),
            schema_name="petrinet",
            description="Advection as Petrinet model created by Dan Bryce and Drisana Iverson (Mosaphir)",
            model_version="0.2",
        )


def main(in_args):
    args = get_args(in_args)
    # args.num_discretization_points = 5
    # args.derivative = Derivative.FORWARD
    # args.derivative = Derivative.BACKWARD
    # args.derivative = Derivative.CENTERED
    # args.boundary_slope = 0.0
    common_main(args, AdvectionGenerator, AdvectionModel)


if __name__ == "__main__":
    main(sys.argv[1:])

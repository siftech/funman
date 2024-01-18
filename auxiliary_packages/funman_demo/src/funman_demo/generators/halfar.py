"""
This script will generate instances of the Halfar ice model as Petrinet AMR models.  The options control the number of discretization points.
"""

import sys
from typing import Dict, List, Tuple

from funman_demo.generators.common import (
    Boundary,
    Coordinate,
    Derivative,
    Direction,
    get_args,
)
from funman_demo.generators.common import main as common_main
from funman_demo.generators.generator import Generator
from funman_demo.generators.model.petrinet import (
    Distribution,
    Grounding,
    Header,
    Initial,
    Model,
    Parameter,
)
from pydantic import AnyUrl


class HalfarModel(Model):
    pass


class HalfarGenerator(Generator):
    """
    This generator class constructs the AMR instance.  The coordinates are equally spaced across the range.
    """

    variables: List[str] = ["h"]

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
        derivative_fn = {
            Derivative.CENTERED: self.inner_centered_difference,
            Derivative.BACKWARD: self.inner_backward_difference,
            Derivative.FORWARD: self.inner_forward_difference,
        }
        if args.derivative in derivative_fn:
            return derivative_fn[args.derivative](
                variable, coordinate, dimension, coordinates, args
            )
        else:
            raise NotImplementedError(
                f"Derivative of type: {args.derivative} not defined."
            )

    def get_expression(self, variable, coordinate, args):
        return (
            f"{variable}_{coordinate.id_str()}"
            if not isinstance(coordinate, Boundary)
            else self.boundary_expression(args)
        )

    def rate_expression(self, variable, source, target, coordinate, args):
        gamma = "283701998652.8*A"
        coord_str = self.get_expression(variable, coordinate, args)
        source_str = self.get_expression(variable, source, args)
        target_str = "-" + self.get_expression(variable, target, args)
        return f"({gamma}/dx)*((abs(({source_str}{target_str})/dx)**2)*(({source_str}{target_str})/dx)*({coord_str}**5))"

    def inner_centered_difference(
        self,
        variable: str,
        coordinate: Coordinate,
        dimension: int,
        coordinates: Dict[Tuple, Coordinate],
        args,
    ) -> str:
        source = (
            coordinate.positive_neighbor(dimension, coordinates=coordinates)
            if not isinstance(coordinate, Boundary)
            else coordinate
        )
        target = (
            coordinate.negative_neighbor(dimension, coordinates=coordinates)
            if not isinstance(coordinate, Boundary)
            else coordinate
        )

        # dx = self.get_dx(dimension, coordinate, source, target, width=2)
        return self.rate_expression(variable, source, target, coordinate, args)

    def inner_forward_difference(
        self,
        variable: str,
        coordinate: Coordinate,
        dimension: int,
        coordinates: Dict[Tuple, Coordinate],
        args,
    ) -> str:
        source = (
            coordinate.positive_neighbor(dimension, coordinates=coordinates)
            if not isinstance(coordinate, Boundary)
            else coordinate
        )
        target = coordinate

        # dx = self.get_dx(dimension, coordinate, source, target, width=1)
        return self.rate_expression(variable, source, target, coordinate, args)

    def inner_backward_difference(
        self,
        variable: str,
        coordinate: Coordinate,
        dimension: int,
        coordinates: Dict[Tuple, Coordinate],
        args,
    ) -> str:
        source = coordinate
        target = (
            coordinate.negative_neighbor(dimension, coordinates=coordinates)
            if not isinstance(coordinate, Boundary)
            else coordinate
        )

        # dx = self.get_dx(dimension, coordinate, source, target, width=1)
        return self.rate_expression(variable, source, target, coordinate, args)

    def parameters(self) -> List[Parameter]:
        return [
            Parameter(
                id="A",
                value=1e-16,
                grounding=Grounding(identifiers={}),
                distribution=Distribution(
                    type="StandardUniform1",
                    parameters={"minimum": 1e-20, "maximum": 1e-12},
                ),
            ),
            # Parameter(
            #     id="dt",
            #     value=1,
            #     distribution=Distribution(
            #         type="StandardUniform1",
            #         parameters={"minimum": 1e-1, "maximum": 1e1},
            #     ),
            # ),
            Parameter(
                id="dx",
                value=1,
                grounding=Grounding(identifiers={}),
                distribution=Distribution(
                    type="StandardUniform1",
                    parameters={"minimum": 1e-1, "maximum": 1e1},
                ),
            ),
        ]

    def header(self):
        return Header(
            name="Halfar Model",
            schema=AnyUrl(
                "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/petrinet_v0.1/petrinet/petrinet_schema.json"
            ),
            schema_name="petrinet",
            description="Halfar as Petrinet model created by Dan Bryce and Drisana Iverson (Mosaphir)",
            model_version="0.2",
        )


def main(args):
    args = get_args(args)
    # args.derivative = Derivative.BACKWARD
    # args.derivative = Derivative.FORWARD
    args.derivative = Derivative.CENTERED
    # args.num_discretization_points = 5
    args.boundary_slope = -0.1
    common_main(args, HalfarGenerator, HalfarModel)


if __name__ == "__main__":
    main(sys.argv[1:])

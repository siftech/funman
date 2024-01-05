"""
This script will generate instances of the Halfar ice model as Petrinet AMR models.  The options control the number of discretization points.
"""

from typing import Dict, List, Tuple

from funman_demo.generators.common import (
    Boundary,
    Coordinate,
    Direction,
    get_args,
)
from funman_demo.generators.common import main as common_main
from funman_demo.generators.generator import Generator
from pydantic import AnyUrl

from funman.model.generated_models.petrinet import (
    Distribution,
    Header,
    Initial,
    Model,
    Parameter,
)


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
        next_coord_id = coordinate.neighbors[dimension][Direction.Positive]
        prev_coord_id = coordinate.neighbors[dimension][Direction.Negative]

        next_coord = (
            coordinates.get(next_coord_id, None)
            if not isinstance(next_coord_id, Boundary)
            else next_coord_id
        )
        prev_coord = (
            coordinates.get(prev_coord_id, None)
            if not isinstance(prev_coord_id, Boundary)
            else prev_coord_id
        )

        coord_str = f"{variable}_{coordinate.id_str()}"
        next_str = (
            f"{variable}_{next_coord.id_str()}"
            if next_coord
            else self.boundary_expression(args)
        )
        prev_str = (
            f"-{variable}_{prev_coord.id_str()}"
            if prev_coord
            else self.boundary_expression(args)
        )

        gamma = "283701998652.8*A"

        coord_x = coordinate.vector[dimension]
        next_x_dx = (
            next_coord.vector[dimension] - coord_x
            if not isinstance(next_coord, Boundary)
            else None
        )
        prev_x_dx = (
            coord_x - prev_coord.vector[dimension]
            if not isinstance(prev_coord, Boundary)
            else None
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

        return f"({gamma}/{dx})*((abs(({next_str}{prev_str})*0.5)**2)*(({next_str}{prev_str})*0.5)*({coord_str}**5))*dt"

    def parameters(self) -> List[Parameter]:
        return [
            Parameter(
                id="A",
                value=1e-16,
                distribution=Distribution(
                    type="StandardUniform1",
                    parameters={"minimum": 1e-20, "maximum": 1e-12},
                ),
            ),
            Parameter(
                id="dt",
                value=1,
                distribution=Distribution(
                    type="StandardUniform1",
                    parameters={"minimum": 1e-1, "maximum": 1e1},
                ),
            ),
        ]

    def header(self):
        return Header(
            name="Halfar Model",
            schema_=AnyUrl(
                "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/petrinet_v0.1/petrinet/petrinet_schema.json"
            ),
            schema_name="petrinet",
            description="Halfar as Petrinet model created by Dan Bryce and Drisana Mosiphir",
            model_version="0.1",
        )


def main(args):
    args = get_args(args)
    # args.num_discretization_points = 5
    common_main(args, HalfarGenerator, HalfarModel)


if __name__ == "__main__":
    main(sys.argv[1:])

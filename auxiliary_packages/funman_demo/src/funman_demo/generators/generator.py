from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, List, Tuple

from funman_demo.generators.common import (
    Boundary,
    Coordinate,
    Derivative,
    Direction,
)

from funman.model.generated_models.petrinet import (
    Initial,
    Model1,
    OdeSemantics,
    Properties,
    Rate,
    Semantics,
    State,
    Time,
    Transition,
    Unit,
)
from funman.representation.interval import Interval


class Generator(ABC):
    range: Interval = Interval(lb=-2.0, ub=2.0)
    variables: List[str] = []

    @abstractmethod
    def transition_rate(
        self,
        variable: str,
        coordinate: Coordinate,
        dimension: int,
        coordinates: Dict[Tuple, Coordinate],
        args,
    ) -> str:
        raise NotImplementedError(
            f"transition_rate undefined for {self.__class__}"
        )

    @abstractmethod
    def parameters(self):
        raise NotImplementedError(f"parameters undefined for {self.__class__}")

    @abstractmethod
    def header(self):
        raise NotImplementedError(f"header undefined for {self.__class__}")

    def initials(self, coordinates) -> List[Initial]:
        boundary = Boundary()
        return [
            Initial(
                target=f"{variable}_{c.id_str()}",
                expression=(f"{1.0/(1.0+max([abs(v) for v in c.vector]))}"),
            )
            for id, c in coordinates.items()
            for variable in self.variables
        ]
        # + [
        #      Initial(
        #         target=f"{variable}_{boundary.id_str()}",
        #         expression="0.01",
        #     )
        #     for variable in self.variables
        # ]

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

        boundary = Boundary()
        for id, coord in coords.items():
            for dim in range(args.dimensions):
                coord.neighbors.append({})
                prev_id = tuple(
                    [(i if d != dim else i - 1) for d, i in enumerate(id)]
                )
                coord.neighbors[dim][Direction.Negative] = (
                    prev_id if prev_id[dim] >= 0 else boundary
                )
                next_id = tuple(
                    [(i if d != dim else i + 1) for d, i in enumerate(id)]
                )
                coord.neighbors[dim][Direction.Positive] = (
                    next_id
                    if next_id[dim] < int(args.num_discretization_points)
                    else boundary
                )
        return coords

    def states(self, args, coordinates) -> List[State]:
        # Create a variable at each coordinate
        boundary = Boundary()
        states = [
            State(
                id=f"{variable}_{coord.id_str()}",
                name=f"{variable}_{coord.id_str()}",
                description=f"{variable} at {coord.vector}",
            )
            for id, coord in coordinates.items()
            for variable in self.variables
        ]
        # + [
        #     State(id=f"{variable}_{boundary.id_str()}",
        #         name=f"{variable}_{boundary.id_str()}",
        #         description=f"{variable} at boundary",)
        #     for variable in self.variables
        # ]
        return states

    def boundary_derivative(self, args):
        return str(args.boundary_slope)

    def boundary_expression(self, args):
        return f"{self.boundary_derivative(args)}*t"

    def make_rate(
        self,
        variable,
        source,
        target,
        dimension,
        coordinate,
        coordinates,
        args,
    ):
        if not isinstance(coordinate, Boundary):
            rate = self.transition_rate(
                variable, coordinate, dimension, coordinates, args
            )
        else:
            rate = self.boundary_expression(args)
        return rate

    def make_transition(
        self,
        variable,
        source,
        target,
        dimension,
        coordinate,
        coordinates,
        args,
    ):
        transition_name = (
            f"r_{variable}_{dimension}_{source.id_str()}_{target.id_str()}"
        )
        rate = self.make_rate(
            variable, source, target, dimension, coordinate, coordinates, args
        )

        input_states = (
            [f"{variable}_{source.id_str()}"]
            if not isinstance(source, Boundary)
            else []
        )
        output_states = (
            [f"{variable}_{target.id_str()}"]
            if not isinstance(target, Boundary)
            else []
        )
        transition = Transition(
            id=transition_name,
            input=input_states,
            output=output_states,
            properties=Properties(name=transition_name),
        )
        return rate, transition

    def centered_difference(
        self, variable: str, coordinate: Coordinate, coordinates, args
    ):
        transitions = []
        rates = []
        # Get transition in each dimension
        for dimension, value in enumerate(coordinate.vector):
            # next_coord_id = coordinate.neighbors[dimension][Direction.Positive]
            # prev_coord_id = coordinate.neighbors[dimension][Direction.Negative]
            # next_coord = coordinates.get(next_coord_id, None) if not isinstance(next_coord_id, Boundary) else boundary
            # # Handle transitions to boundaries as if there is no update to the boundary
            # prev_coord = coordinates.get(prev_coord_id, None) if not isinstance(prev_coord_id, Boundary) else boundary

            source = coordinate.positive_neighbor(
                dimension, coordinates=coordinates
            )
            target = coordinate.negative_neighbor(
                dimension, coordinates=coordinates
            )

            rate, transition = self.make_transition(
                variable,
                source,
                target,
                dimension,
                coordinate,
                coordinates,
                args,
            )
            rate = f"({rate}/2)"
            rates.append(Rate(target=transition.id, expression=rate))
            transitions.append(transition)

            if isinstance(target, Boundary):
                rate, transition = self.make_transition(
                    variable,
                    coordinate,
                    target,
                    dimension,
                    coordinate,
                    coordinates,
                    args,
                )
                rate = f"({rate}/2)"
                rates.append(Rate(target=transition.id, expression=rate))
                transitions.append(transition)

                # # Make transition from coordinate to boundary
                # transition_name = f"r_{variable}_{dimension}_{coordinate.id_str()}_{prev_coord.id_str()}"
                # input_states = [f"{variable}_{coordinate.id_str()}"] if next_coord else []
                # output_states = [] if prev_coord else []

                # rate = self.transition_rate(variable,
                #     coordinate, dimension, coordinates, args
                # )
                # rate = f"({rate}/2)"
                # rates.append(
                #     Rate(
                #         target=transition_name,
                #         expression=rate,
                #     )
                # )
                # transition = Transition(
                #     id=transition_name,
                #     input=input_states,
                #     output=output_states,
                #     properties=Properties(name=transition_name),
                # )
                # transitions.append(transition)
            elif isinstance(source, Boundary):
                rate, transition = self.make_transition(
                    variable,
                    source,
                    coordinate,
                    dimension,
                    source,
                    coordinates,
                    args,
                )
                rate = f"({rate}/2)"
                rates.append(Rate(target=transition.id, expression=rate))
                transitions.append(transition)

                # # Make transition from boundary to coordinate
                # transition_name = f"r_{variable}_{dimension}_{next_coord.id_str()}_{coordinate.id_str()}"
                # input_states = [] if next_coord else []
                # output_states = [f"{variable}_{coordinate.id_str()}"] if prev_coord else []

                # rate = self.boundary_expression(args)
                # rate = f"({rate}/2)"
                # rates.append(
                #     Rate(
                #         target=transition_name,
                #         expression=rate,
                #     )
                # )
                # transition = Transition(
                #     id=transition_name,
                #     input=input_states,
                #     output=output_states,
                #     properties=Properties(name=transition_name),
                # )
                # transitions.append(transition)
        return transitions, rates

    def backward_difference(
        self, variable: str, coordinate: Coordinate, coordinates, args
    ):
        transitions = []
        rates = []
        boundary = Boundary()
        # Get transition in each dimension
        for dimension, value in enumerate(coordinate.vector):
            source = coordinate
            target = coordinate.negative_neighbor(
                dimension, coordinates=coordinates
            )

            rate, transition = self.make_transition(
                variable, source, target, dimension, target, coordinates, args
            )
            rates.append(Rate(target=transition.id, expression=rate))
            transitions.append(transition)

            # prev_coord_id = coordinate.neighbors[dimension][Direction.Negative]

            # # Handle transitions to boundaries as if there is no update to the boundary
            # prev_coord = coordinates.get(prev_coord_id, None) if not isinstance(prev_coord_id, Boundary) else boundary
            # transition_name = f"r_{variable}_{dimension}_{coordinate.id_str()}_{prev_coord.id_str()}"
            # # Transition for coordinate is: coord -- rate --> prev_coord
            # rate = self.transition_rate(variable,
            #     prev_coord, dimension, coordinates, args
            # ) if not isinstance(prev_coord_id, Boundary) else self.boundary_expression(args)
            # rate = f"{rate}"
            # rates.append(
            #     Rate(
            #         target=transition_name,
            #         expression=rate,
            #     )
            # )

            # input_states = [f"{variable}_{coordinate.id_str()}"]
            # output_states = [f"{variable}_{prev_coord.id_str()}"] if not isinstance(prev_coord_id, Boundary) else []

            # transition = Transition(
            #     id=transition_name,
            #     input=input_states,
            #     output=output_states,
            #     properties=Properties(name=transition_name),
            # )
            # transitions.append(transition)

            upper_boundary = coordinate.positive_neighbor(
                dimension, coordinates=coordinates
            )
            if isinstance(upper_boundary, Boundary):
                rate, transition = self.make_transition(
                    variable,
                    upper_boundary,
                    coordinate,
                    dimension,
                    coordinate,
                    coordinates,
                    args,
                )
                rates.append(Rate(target=transition.id, expression=rate))
                transitions.append(transition)
        return transitions, rates

    def forward_difference(
        self, variable: str, coordinate: Coordinate, coordinates, args
    ):
        transitions = []
        rates = []
        boundary = Boundary()
        # Get transition in each dimension
        for dimension, value in enumerate(coordinate.vector):
            source = coordinate.positive_neighbor(
                dimension, coordinates=coordinates
            )
            target = coordinate

            rate, transition = self.make_transition(
                variable, source, target, dimension, source, coordinates, args
            )
            rates.append(Rate(target=transition.id, expression=rate))
            transitions.append(transition)

            # next_coord_id = coordinate.neighbors[dimension][Direction.Positive]
            # next_coord = coordinates.get(next_coord_id, None) if not isinstance(next_coord_id, Boundary) else boundary

            # transition_name = f"r_{variable}_{dimension}_{next_coord.id_str()}_{coordinate.id_str()}"
            # # Transition for coordinate is: next_coord -- rate --> prev_coord
            # rate = self.transition_rate(variable,
            #     next_coord, dimension, coordinates, args
            # ) if not isinstance(next_coord_id, Boundary) else self.boundary_expression(args)
            # rate = f"{rate}"
            # rates.append(
            #     Rate(
            #         target=transition_name,
            #         expression=rate,
            #     )
            # )

            # input_states = [f"{variable}_{next_coord.id_str()}"]  if not isinstance(next_coord_id, Boundary) else []
            # output_states = [f"{variable}_{coordinate.id_str()}"]

            # transition = Transition(
            #     id=transition_name,
            #     input=input_states,
            #     output=output_states,
            #     properties=Properties(name=transition_name),
            # )
            # transitions.append(transition)
            lower_boundary = coordinate.negative_neighbor(
                dimension, coordinates=coordinates
            )
            if isinstance(lower_boundary, Boundary):
                rate, transition = self.make_transition(
                    variable,
                    coordinate,
                    lower_boundary,
                    dimension,
                    coordinate,
                    coordinates,
                    args,
                )
                rates.append(Rate(target=transition.id, expression=rate))
                transitions.append(transition)
        return transitions, rates

    def boundary_rate_transition(self, variable, args):
        # Add boundary transition and rate
        boundary = Boundary()
        transition_name = f"r_{variable}_{boundary.id_str()}"
        boundary_rate = Rate(
            target=transition_name,
            expression=self.boundary_derivative(args),
        )
        boundary_transition = Transition(
            id=transition_name,
            input=[],
            output=[transition_name],
            properties=Properties(name=transition_name),
        )
        return [boundary_transition], [boundary_rate]

    def derivative(self, variable, coordinate, coordinates, args):
        derivative_fn = {
            Derivative.CENTERED: self.centered_difference,
            Derivative.BACKWARD: self.backward_difference,
            Derivative.FORWARD: self.forward_difference,
        }
        if args.derivative in derivative_fn:
            return derivative_fn[args.derivative](
                variable, coordinate, coordinates, args
            )
        else:
            raise NotImplementedError(
                f"Derivative of type: {args.derivative} not defined."
            )

    def model(self, args) -> Tuple[Model1, Semantics]:
        """
        Generate the AMR Model
        """
        coordinates = self.coordinates(args)

        # Create a height variable at each coordinate
        states = self.states(args, coordinates)

        transitions = []
        rates = []
        for variable in self.variables:
            for id, coordinate in coordinates.items():
                coord_transitions, trans_rates = self.derivative(
                    variable, coordinate, coordinates, args
                )
                transitions += coord_transitions
                rates += trans_rates
            # # Get boundary rate and transition
            # trans, rt = self.boundary_rate_transition(variable, args)
            # transitions += trans
            # rates += rt

        time = Time(
            id="t",
            units=Unit(expression="day", expression_mathml="<ci>day</ci>"),
        )

        initials = self.initials(coordinates)

        parameters = self.parameters()

        return Model1(states=states, transitions=transitions), Semantics(
            ode=OdeSemantics(
                rates=rates,
                initials=initials,
                parameters=parameters,
                time=time,
            )
        )

from difflib import SequenceMatcher
from functools import reduce
from typing import Callable, Dict, List, Optional, Union

import graphviz
import matplotlib.pyplot as plt
import pandas as pd
import sympy
from pydantic import BaseModel, ConfigDict
from pysmt.formula import FNode
from pysmt.shortcuts import REAL, Div, Real, Symbol

from funman import to_sympy
from funman.utils.sympy_utils import (
    SympyBoundedSubstituter,
    replace_reserved,
    substitute,
    sympy_to_pysmt,
    to_sympy,
)

from ..representation import Interval
from .generated_models.petrinet import Distribution, Initial
from .generated_models.petrinet import Model
from .generated_models.petrinet import Model as GeneratedPetrinet
from .generated_models.petrinet import (
    Model1,
    Observable,
    OdeSemantics,
    Parameter,
    Properties,
    Rate,
    Semantics,
    State,
    Transition,
)
from .model import FunmanModel


class AbstractPetriNetModel(FunmanModel):
    _is_differentiable: bool = True

    def _num_flow_from_state_to_transition(
        self, state_id: Union[str, int], transition_id: Union[str, int]
    ) -> int:
        return len(
            [
                edge
                for edge in self._input_edges()
                if self._edge_source(edge) == state_id
                and self._edge_target(edge) == transition_id
            ]
        )

    def _num_flow_from_transition_to_state(
        self, state_id: Union[str, int], transition_id: Union[str, int]
    ) -> int:
        return len(
            [
                edge
                for edge in self._output_edges()
                if self._edge_source(edge) == transition_id
                and self._edge_target(edge) == state_id
            ]
        )

    def _num_flow_from_transition(self, transition_id: Union[str, int]) -> int:
        return len(
            [
                edge
                for edge in self._output_edges()
                if self._edge_source(edge) == transition_id
            ]
        )

    def _num_flow_into_transition(self, transition_id: Union[str, int]) -> int:
        return len(
            [
                edge
                for edge in self._input_edges()
                if self._edge_target(edge) == transition_id
            ]
        )

    def _flow_into_state_via_transition(
        self, state_id: Union[str, int], transition_id: Union[str, int]
    ) -> float:
        num_flow_to_transition = self._num_flow_into_transition(transition_id)
        num_inflow = self._num_flow_from_transition_to_state(
            state_id, transition_id
        )
        num_transition_outputs = self._num_flow_from_transition(transition_id)
        if num_transition_outputs > 0:
            return (
                num_inflow / num_transition_outputs
            ) * num_flow_to_transition
        else:
            return 0

    def to_dot(self, values={}):
        """
        Create a dot object for visualizing the graph.

        Returns
        -------
        graphviz.Digraph
            The graph represented by self.
        """
        dot = graphviz.Digraph(
            name=f"petrinet",
            graph_attr={},
        )

        state_vars = self._state_vars()
        state_var_names = self._state_var_names()
        transitions = self._transitions()
        variable_values = self._variable_values()

        # Don't substitute initial state values for state variables, only parameters
        variable_values = {
            k: (None if k in state_var_names else v)
            for k, v in variable_values.items()
        }

        for _, var in enumerate(state_vars):
            state_var_id = self._state_var_id(var)
            state_var_name = self._state_var_name(var)
            for transition in transitions:
                transition_id = self._transition_id(transition)
                transition_parameters = self._transition_rate(transition)
                # transition_parameter_value = [
                #     self._parameter_values()[t] for t in transition_parameters
                # ]
                transition_parameter_value = [
                    substitute(t, variable_values)
                    for t in transition_parameters
                ]
                transition_name = f"{transition_id}({transition_parameters}) = {transition_parameter_value}"
                dot.node(transition_name, _attributes={"shape": "box"})
                # state var to transition
                for edge in self._input_edges():
                    if (
                        self._edge_source(edge) == state_var_id
                        and self._edge_target(edge) == transition_id
                    ):
                        dot.edge(state_var_name, transition_name)
                # transition to state var
                for edge in self._output_edges():
                    if (
                        self._edge_source(edge) == transition_id
                        and self._edge_target(edge) == state_var_id
                    ):
                        flow = self._flow_into_state_via_transition(
                            state_var_id, transition_id
                        ) / self._num_flow_from_transition_to_state(
                            state_var_id, transition_id
                        )
                        dot.edge(
                            transition_name, state_var_name, label=f"{flow}"
                        )

        return dot

    def calculate_normalization_constant(
        self, scenario: "AnalysisScenario", config: "FUNMANConfig"
    ) -> float:
        vars = self._state_var_names()
        values = {v: self._get_init_value(v, scenario, config) for v in vars}
        if all((v is not None and v.is_constant()) for v in values.values()):
            return float(sum(v.constant_value() for v in values.values()))
        else:
            raise Exception(
                f"Cannot calculate the normalization constant for {type(self)} because the initial state variables are not constants. Try setting the 'normalization_constant' in the configuration to constant."
            )

    def compartmental_constraints(
        self, population: Union[float, int], noise: float
    ) -> List["Constraint"]:
        from funman.representation.constraint import LinearConstraint

        vars = self._state_var_names()
        return [
            LinearConstraint(
                name="compartmental_constraint_population",
                additive_bounds={
                    "lb": population - noise,
                    "ub": population + noise,
                    "closed_upper_bound": True,
                },
                variables=vars,
                timepoints=Interval(lb=0.0),
                soft=False,
            )
        ] + [
            LinearConstraint(
                name=f"compartmental_{v}_nonnegative",
                additive_bounds={"lb": 0},
                variables=[v],
                timepoints=Interval(lb=0.0),
                soft=False,
            )
            for v in vars
        ]

    def derivative(
        self,
        var_name,
        t,
        values,
        params,
        var_to_value,
        param_to_value,
        get_lambda=False,
    ):  # var_to_value, param_to_value):
        # param_at_t = {p: pv(t).item() for p, pv in param_to_value.items()}
        # FIXME assumes each transition has only one rate
        # print(f"Calling with args {var_name}; {t}; {values}; {params}")
        if get_lambda:
            pos_rates = [
                self._transition_rate(trans, get_lambda=get_lambda)[0](
                    *values, *params
                )
                for trans in self._transitions()
                for var in trans.output
                if var_name == var
            ]
            neg_rates = [
                self._transition_rate(trans, get_lambda=get_lambda)[0](
                    *values, *params
                )
                for trans in self._transitions()
                for var in trans.input
                if var_name == var
            ]
        else:
            pos_rates = [
                self._transition_rate(trans)[0].evalf(
                    subs={**var_to_value, **param_to_value}, n=10
                )
                for trans in self._transitions()
                for var in trans.output
                if var_name == var
            ]
            neg_rates = [
                self._transition_rate(trans)[0].evalf(
                    subs={**var_to_value, **param_to_value}, n=10
                )
                for trans in self._transitions()
                for var in trans.input
                if var_name == var
            ]
        # print(f"Got rates {pos_rates} {neg_rates}")

        return sum(pos_rates) - sum(neg_rates)

    def gradient(self, t, y, *p):
        # FIXME support time varying paramters by treating parameters as a function
        var_to_value = {
            var: y[i] for i, var in enumerate(self._state_var_names())
        }
        # print(f"y: {y}; t: {t}")
        param_to_value = {
            replace_reserved(param): p[i](t)[()]
            for i, param in enumerate(self._parameter_names())
        }
        param_to_value["timer_t"] = t
        # values = [
        #     y[i] for i, _ in enumerate(self._symbols())
        # ]
        unreserved_symols = [replace_reserved(s) for s in self._symbols()]
        params = [
            param_to_value[str(p)]
            for p in unreserved_symols
            if str(p) in param_to_value
        ]

        grad = [
            self.derivative(
                var,
                t,
                y,
                params,
                var_to_value,
                param_to_value,
                get_lambda=True,
            )  # var_to_value, param_to_value)
            for var in self._state_var_names()
        ]
        # print(f"vars: {self._state_var_names()}")
        # print(f"gradient: {grad}")
        assert not any(
            [not isinstance(v, float) for v in grad]
        ), f"Gradient has a non-float element: {grad}"
        return grad


class GeneratedPetriNetModel(AbstractPetriNetModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    petrinet: GeneratedPetrinet
    _transition_rates_cache: Dict[str, Union[sympy.Expr, str]] = {}
    _observables_cache: Dict[str, Union[str, FNode, sympy.Expr]] = {}
    _transition_rates_lambda_cache: Dict[str, Union[Callable, str]] = {}

    def observables(self):
        return self.petrinet.semantics.ode.observables

    def is_timed_observable(self, observation_id):
        (_, e, _) = self.observable_expression(observation_id)
        vars = [str(e) for e in e.get_free_variables()]
        obs_state_vars = [v for v in vars if v in self._state_var_names()]
        return any(obs_state_vars)

    def observable_expression(self, observation_id):
        if observation_id not in self._observables_cache:
            observable = next(
                iter([o for o in self.observables() if o.id == observation_id])
            )
            sympy_expr = to_sympy(observable.expression, self._symbols())
            self._observables_cache[observation_id] = (
                observable.expression,
                sympy_to_pysmt(sympy_expr),
                sympy_expr,
            )
        return self._observables_cache[observation_id]

    def default_encoder(
        self, config: "FUNMANConfig", scenario: "AnalysisScenario"
    ) -> "Encoder":
        """
        Return the default Encoder for the model

        Returns
        -------
        Encoder
            SMT encoder for model
        """
        from funman.translate.petrinet import PetrinetEncoder

        return PetrinetEncoder(
            config=config,
            scenario=scenario,
        )

    def _time_var(self):
        if (
            hasattr(self.petrinet, "semantics")
            and hasattr(self.petrinet.semantics, "ode")
            and hasattr(self.petrinet.semantics.ode, "time")
        ):
            return self.petrinet.semantics.ode.time
        else:
            return None

    def _time_var_id(self, time_var):
        if time_var:
            return f"timer_{time_var.id}"
        else:
            return None

    def _symbols(self):
        symbols = self._state_var_names() + self._parameter_names()
        if self._time_var():
            symbols += [f"timer_{self._time_var().id}"]
        return symbols

    def _get_init_value(
        self, var: str, scenario: "AnalysisScenario", config: "FUNMANConfig"
    ):
        value = FunmanModel._get_init_value(self, var, scenario, config)
        if value is None:
            if hasattr(self.petrinet.semantics, "ode"):
                initials = self.petrinet.semantics.ode.initials
                value = next(i.expression for i in initials if i.target == var)
            else:
                value = f"{var}0"

        try:  # to cast to float
            value = float(value)
        except:
            pass

        if isinstance(value, float):
            value = Real(value)
        if isinstance(value, int):
            value = Real(float(value))
        elif isinstance(value, str):
            expr = to_sympy(value, self._symbols())
            if expr.is_symbol:
                value = Symbol(value, REAL)
            else:
                value = sympy_to_pysmt(expr)

        if scenario.normalization_constant and config.normalize:
            value = Div(value, Real(scenario.normalization_constant))

        return value

    def _parameter_lb(self, param_name: str):
        return next(
            (
                self._try_float(p.distribution.parameters["minimum"])
                if p.distribution
                else p.value
            )
            for p in self.petrinet.semantics.ode.parameters
            if p.id == param_name
        )

    def _parameter_ub(self, param_name: str):
        return next(
            (
                self._try_float(p.distribution.parameters["maximum"])
                if p.distribution
                else p.value
            )
            for p in self.petrinet.semantics.ode.parameters
            if p.id == param_name
        )

    def _state_vars(self) -> List[State]:
        return self.petrinet.model.states

    def _state_var_names(self) -> List[str]:
        return [self._state_var_name(s) for s in self._state_vars()]

    def _observable_names(self) -> List[str]:
        return [self._observable_name(s) for s in self.observables()]

    def _transitions(self) -> List[Transition]:
        return self.petrinet.model.transitions

    def _state_var_name(self, state_var: State) -> str:
        return state_var.id

    def _observable_name(self, observable: Observable) -> str:
        return observable.id

    def _input_edges(self):
        return [(i, t.id) for t in self._transitions() for i in t.input]

    def _edge_source(self, edge):
        return edge[0]

    def _edge_target(self, edge):
        return edge[1]

    def _output_edges(self):
        return [(t.id, o) for t in self._transitions() for o in t.output]

    def _transition_rate(self, transition, sympify=False, get_lambda=False):
        if hasattr(self.petrinet.semantics, "ode"):
            if transition.id not in self._transition_rates_cache:
                t_rates = [
                    (
                        to_sympy(r.expression, self._symbols())
                        # if not any(
                        #     p == r.expression for p in self._parameter_names()
                        # )
                        # else r.expression
                    )
                    for r in self.petrinet.semantics.ode.rates
                    if r.target == transition.id
                ]
                time_var = self._time_var()
                if time_var:
                    t_rates = [
                        t.subs(
                            {
                                self._time_var().id: f"timer_{self._time_var().id}"
                            }
                        )
                        for t in t_rates
                    ]
                unreserved_symbols = [
                    replace_reserved(s) for s in self._symbols()
                ]
                # convert "t" to "timer_t"
                if unreserved_symbols[-1] == "t":
                    unreserved_symbols[-1] = self._time_var_id(self._time_var())
                t_rates_lambda = [
                    sympy.lambdify(unreserved_symbols, t, cse=True)
                    for t in t_rates
                ]
                self._transition_rates_cache[transition.id] = t_rates
                self._transition_rates_lambda_cache[transition.id] = (
                    t_rates_lambda
                )
            return (
                self._transition_rates_cache[transition.id]
                if not get_lambda
                else self._transition_rates_lambda_cache[transition.id]
            )
        else:
            return transition.id

    def _transition_id(self, transition):
        return transition.id

    def _state_var_id(self, state_var):
        return self._state_var_name(state_var)

    def _parameter_names(self):
        if hasattr(self.petrinet.semantics, "ode"):
            return [p.id for p in self.petrinet.semantics.ode.parameters]
        else:
            # Create a parameter for each transition and initial state variable
            return [t.id for t in self.petrinet.model.transitions] + [
                f"{s.id}0" for s in self.petrinet.model.states
            ]

    def _parameter_values(self):
        if hasattr(self.petrinet.semantics, "ode"):
            return {
                p.id: p.value for p in self.petrinet.semantics.ode.parameters
            }
        else:
            return {}

    def _variable_values(self):
        values = {}
        if hasattr(self.petrinet.semantics, "ode"):
            for p in self.petrinet.semantics.ode.parameters:
                values[p.id] = p.value
            for p in self.petrinet.semantics.ode.initials:
                values[p.target] = p.expression
                try:
                    values[p.target] = float(values[p.target])
                except:
                    pass

        return values

    def contract_parameters(
        self, parameter_bounds: Dict[str, Interval]
    ) -> GeneratedPetrinet:
        contracted_model = self.petrinet.copy(deep=True)

        for param in contracted_model.semantics.ode.parameters:
            new_bounds = parameter_bounds[param.id]
            if param.distribution:
                param.distribution.parameters["minimum"] = max(
                    new_bounds.lb,
                    float(param.distribution.parameters["minimum"]),
                )
                param.distribution.parameters["maximum"] = min(
                    new_bounds.ub,
                    float(param.distribution.parameters["maximum"]),
                )
            else:
                param.distribution = Distribution(
                    parameters={
                        "minimum": new_bounds.lb,
                        "maximum": new_bounds.ub,
                    },
                    type="StandardUniform1",
                )
            if param.distribution.parameters["minimum"] == param.distribution.parameters["maximum"]:
                param.value = param.distribution.parameters["minimum"]
                param.distribution = None
                

        return contracted_model

    def formulate_bounds(self):
        # Reformulate model into bounded version
        # - replace each state S by S_lb and S_ub
        # - replace each transition T by T_lb and T_ub
        # - replace related parameters by their lb and ub, using abstracted values in metadata if present
        # - replace rates by transition type

        bounded_states = {
            s.id: {
                bound: State(
                    id=f"{s.id}_{bound}",
                    name=f"{s.id}_{bound}",
                    description=f"{s.description} {bound}",
                    grounding=s.grounding,
                    units=s.units,
                )
                for bound in ["lb", "ub"]
            }
            for s in self.petrinet.model.states
        }

        abstraction_metadata = self.petrinet.metadata.get("abstraction", {})
        symbols = self._symbols()
        str_to_symbol = {s: sympy.Symbol(s) for s in symbols}
        bound_symbols = {
            s: {"lb": f"{s}_lb", "ub": f"{s}_ub"} for s in symbols
        }
        for _, params in abstraction_metadata.get("parameters", {}).items():
            for p in params:
                bound_symbols[p] = {
                    bound: p.replace("agg", bound) for bound in ["lb", "ub"]
                }
        substituter = SympyBoundedSubstituter(
            bound_symbols=bound_symbols, str_to_symbol=str_to_symbol
        )

        def bound_expression(targets, e, bound, metadata):

            # targets are forced substitutions
            for k, v in targets.items():
                e = e.replace(k, v)

            # # substitute abstracted parameters for lower or upper bounds
            # for k, v in metadata.items():
            #     e = e.replace(k, k.replace("agg", bound))

            return (
                substituter.minimize(targets, e)
                if bound == "lb"
                else substituter.maximize(targets, e)
            )

        def lb_expression(targets, e, symbols, metadata):
            return bound_expression(targets, e, "lb", metadata)

        def ub_expression(targets, e, symbols, metadata):
            return bound_expression(targets, e, "ub", metadata)

        # lb transitions use rates that will:
        #   - decrease ub terms by the least amount
        #   - increase lb terms by the least amount
        # ub transitions use rates that will:
        #   - vice versa wrt. above

        expression_bound_fn = {
            "in": {"lb": ub_expression, "ub": lb_expression},
            "out": {"lb": lb_expression, "ub": ub_expression},
        }

        bounded_transitions = {
            "inputs": {
                # Make transition for each input edge
                t.id: {
                    input_id: {
                        bound: {
                            "transition": Transition(
                                id=f"{t.id}_out_{bound}",
                                input=[f"{input_id}_{bound}"],
                                output=[ ],
                                grounding=t.grounding,
                                properties=Properties(
                                    name=f"{t.id}_in_{bound}",
                                    description=(
                                        f"{t.properties.description} in {bound}"
                                        if t.properties.description
                                        else t.properties.description
                                    ),
                                ),
                            ),
                            "rate": Rate(
                                target=f"{r.target}_in_{bound}",
                                expression=str(
                                    expression_bound_fn["in"][bound](
                                        {
                                            input_id: f"{input_id}_{bound}"
                                        },
                                        r.expression,
                                        symbols,
                                        abstraction_metadata.get(
                                            "parameters", {}
                                        ).get(r.target, {}),
                                    )
                                ),
                            ),
                        }
                        for bound in ["lb", "ub"]
                    }
                    for input_id in t.input
                }
                for t in self.petrinet.model.transitions
                for r in self.petrinet.semantics.ode.rates
                if r.target == t.id
            },
            "outputs": {
                # Make transition for each output edge
                t.id: {
                    output_id: {
                        bound: {
                            "transition": Transition(
                                id=f"{t.id}_in_{bound}",
                                input=[],
                                output=[f"{output_id}_{bound}"],
                                grounding=t.grounding,
                                properties=Properties(
                                    name=f"{t.id}_out_{bound}",
                                    description=(
                                        f"{t.properties.description} in {bound}"
                                        if t.properties.description
                                        else t.properties.description
                                    ),
                                ),
                            ),
                            "rate": Rate(
                                target=f"{r.target}_out_{bound}",
                                expression=str(
                                    expression_bound_fn["out"][bound](
                                        {
                                            output_id: f"{output_id}_{bound}"
                                        },
                                        r.expression,
                                        symbols,
                                        abstraction_metadata.get(
                                            "parameters", {}
                                        ).get(r.target, {}),
                                    )
                                ),
                            ),
                        }
                        for bound in ["lb", "ub"]
                    }
                    for output_id in t.output
                }
                for t in self.petrinet.model.transitions
                for r in self.petrinet.semantics.ode.rates
                if r.target == t.id
            },
        }

        bounded_initials = [
            Initial(
                target=f"{r.target}_{bound}",
                expression=r.expression,
                expression_mathml=r.expression_mathml,
            )
            for r in self.petrinet.semantics.ode.initials
            for bound in ["lb", "ub"]
        ]

        abstract_bounded_parameters = {
            p: {
                p.replace("agg_", "lb_"): bounds["lb"],
                p.replace("agg_", "ub_"): bounds["ub"],
            }
            for t, params in abstraction_metadata.get("parameters", {}).items()
            for p, bounds in params.items()
        }

        bounded_parameters = [
            Parameter(id=f"{r.id}_{bound}", value=r.value, units=r.units)
            for r in self.petrinet.semantics.ode.parameters
            if r.id not in abstract_bounded_parameters
            for bound in ["lb", "ub"]
        ] + [  # Abstracted parameter lb and ub values
            Parameter(id=id, value=value, units=r.units)
            for r in self.petrinet.semantics.ode.parameters
            if r.id in abstract_bounded_parameters
            for id, value in abstract_bounded_parameters[r.id].items()
        ]

        bounded_observables = []

        new_states = [
            bnd for s in bounded_states.values() for bnd in s.values()
        ]

        new_transitions = [
            bnd["transition"]
            for ts in bounded_transitions.values()
            for tid in ts.values()
            for v in tid.values()
            for bnd in v.values()
        ]

        new_rates = [
            bnd["rate"]
            for ts in bounded_transitions.values()
            for tid in ts.values()
            for v in tid.values()
            for bnd in v.values()
        ]

        return GeneratedPetriNetModel(
            petrinet=Model(
                header=self.petrinet.header,
                properties=self.petrinet.properties,
                model=Model1(states=new_states, transitions=new_transitions),
                semantics=Semantics(
                    ode=OdeSemantics(
                        rates=new_rates,
                        initials=bounded_initials,
                        parameters=bounded_parameters,
                        observables=bounded_observables,
                        time=self.petrinet.semantics.ode.time,
                    ),
                    typing=self.petrinet.semantics.typing,
                    span=self.petrinet.semantics.span,
                ),
                metadata=self.petrinet.metadata,
            )
        )

    def stratify(
        self,
        state_var: str,
        strata: List[str],
        strata_parameters: Optional[List[str]] = None,
        strata_transitions=[],
        self_strata_transition=False,
    ):
        """
        Generate a new model that stratifies self.  The 'state_var' will be replaced by one copy
        per entry in the 'strata' list.  The 'strata_parameters', if specified, will be replaced
        by a fresh copy that corresponds to each element of the strata list in each transition
        involving the 'state_var'.  The 'strata_transitions' list includes all transitions
        involving 'state_var' to allow transitioning between the elements of 'strata', if
        possible (e.g., if 'state_var' is both and input and an output of the transition).  The
        'self_strata_transition' flag adds transitions between all pairs of entries in 'strata'.
        When either 'strata_transitions' or 'self_strata_transition' are specified, the rates
        will incorporate a new parameter that defines the probability of transitioning between
        elements of the strata list.  The transition probability parameters are assigned a value
        of 1/len('strata'), treating the transitions as a uniform probability distribution.

        Parameters
        ----------
        self : PetrinetModel
            _description_
        state_var : str
            _description_
        strata : List[str]
            _description_
        strata_parameters : Optional[List[str]], optional
            _description_, by default None
        strata_transitions : list, optional
            _description_, by default []
        self_strata_transition : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """

        # get state variable
        state_vars: List[State] = [
            s
            for s in self._state_vars()
            if self._state_var_name(s) == state_var
        ]
        assert (
            len(state_vars) == 1
        ), "Found more than one State variable for {state_var}"
        original_var = state_vars[0]
        new_vars = [
            State(
                id=f"{original_var.id}_{level}",
                name=f"{original_var.name}_{level}",
                description=f"{original_var.description} Stratified wrt. {level}",
                grounding=original_var.grounding,
                units=original_var.units,
            )
            for level in strata
        ]
        unchanged_vars = [
            s.id for s in self._state_vars() if s != original_var
        ]

        # get new transitions
        transitions: Dict[str, Transition] = {
            t.id: t
            for t in self._transitions()
            if original_var.id in t.input or original_var.id in t.output
        }
        other_transitions = {
            t.id: t for t in self._transitions() if t.id not in transitions
        }

        src_only_transitions: Dict[str, Transition] = {
            t_id: t
            for t_id, t in transitions.items()
            if original_var.id in t.input and original_var.id not in t.output
        }
        dest_only_transitions: Dict[str, Transition] = {
            t_id: t
            for t_id, t in transitions.items()
            if original_var.id not in t.input and original_var.id in t.output
        }
        src_and_dest_transitions: Dict[str, Transition] = {
            t_id: t
            for t_id, t in transitions.items()
            if original_var.id in t.input and original_var.id in t.output
        }

        # Replicate transitions where original_var is in source
        new_src_transitions = [
            Transition(
                id=f"{t.id}_{level}",
                input=[
                    (s if s != original_var.id else f"{s}_{level}")
                    for s in t.input
                ],
                output=t.output,
                grounding=t.grounding,
                properties=Properties(
                    name=f"{t.properties.name}_{level}",
                    description=(
                        f"{t.properties.description} Stratified wrt. {level}"
                        if t.properties.description
                        else t.properties.description
                    ),
                ),
            )
            for t_id, t in src_only_transitions.items()
            for level in strata
        ]

        # Replicate transitions where original_var is in destination
        new_dest_transitions = [
            Transition(
                id=f"{t.id}_{level}",
                input=t.input,
                output=[
                    (s if s != original_var.id else f"{s}_{level}")
                    for s in t.output
                ],
                grounding=t.grounding,
                properties=Properties(
                    name=f"{t.properties.name}_{level}",
                    description=(
                        f"{t.properties.description} Stratified wrt. {level}"
                        if t.properties.description
                        else t.properties.description
                    ),
                ),
            )
            for t_id, t in dest_only_transitions.items()
            for level in strata
        ]

        # Replicate transitions where original_var is in source and destination
        new_src_dest_transitions = [
            Transition(
                id=f"{t.id}_{level_s}_{level_t}",
                input=[
                    (s if s != original_var.id else f"{s}_{level_s}")
                    for s in t.input
                ],
                output=[
                    (s if s != original_var.id else f"{s}_{level_t}")
                    for s in t.output
                ],
                grounding=t.grounding,
                properties=Properties(
                    name=f"{t.properties.name}_{level_s}_{level_t}",
                    description=(
                        f"{t.properties.description} Stratified wrt. {level_s}, {level_t}."
                        if t.properties.description
                        else t.properties.description
                    ),
                ),
            )
            for t_id, t in src_and_dest_transitions.items()
            for level_s in strata
            for level_t in strata
            if t_id in strata_transitions or level_s == level_t
        ]

        new_transitions = (
            new_src_transitions
            + new_dest_transitions
            + new_src_dest_transitions
        )

        # Modify rates by substituting fresh versions of the strata_parameters
        old_rates = {
            t_id: self._transition_rate(t) for t_id, t in transitions.items()
        }
        other_rates = {
            r.target: r
            for r in self.petrinet.semantics.ode.rates
            if r.target in other_transitions
        }

        src_only_rates = [
            Rate(
                target=f"{t_id}_{level}",
                expression=(
                    reduce(
                        lambda x, y: x.replace(y, f"{y}_{level}"),
                        strata_parameters,
                        str(r[0]),
                    )
                    if strata_parameters
                    else str(r[0])
                ).replace(state_var, f"{state_var}_{level}"),
            )
            for t_id, r in old_rates.items()
            if t_id in src_only_transitions
            for level in strata
        ]

        dest_only_rates = [
            Rate(
                target=f"{t_id}_{level}",
                expression=(
                    reduce(
                        lambda x, y: x.replace(y, f"{y}_{level}"),
                        strata_parameters,
                        str(r[0]),
                    )
                    if strata_parameters
                    else str(r[0])
                ).replace(state_var, f"{state_var}_{level}"),
            )
            for t_id, r in old_rates.items()
            if t_id in dest_only_transitions
            for level in strata
        ]

        src_and_dest_rates = [
            Rate(
                target=f"{t_id}_{level_s}_{level_t}",
                expression=f"p_{state_var}_{level_s}_{level_t}*"
                + (
                    reduce(
                        lambda x, y: x.replace(y, f"{y}_{level_s}_{level_t}"),
                        strata_parameters,
                        str(r[0]),
                    )
                    if strata_parameters
                    else str(r[0])
                ).replace(state_var, f"{state_var}_{level_s}"),
            )
            for t_id, r in old_rates.items()
            if t_id in src_and_dest_transitions
            for level_s in strata
            for level_t in strata
        ]

        new_rates = src_only_rates + dest_only_rates + src_and_dest_rates

        new_states = new_vars + [
            s for s in self.petrinet.model.states.root if s not in state_vars
        ]

        # update with new states by splitting old state values
        original_init_value = to_sympy(
            next(
                i.expression
                for i in self.petrinet.semantics.ode.initials
                if i.target == original_var.id
            ),
            {},
        )

        new_initials = [
            i
            for i in self.petrinet.semantics.ode.initials
            if i.target in unchanged_vars
        ] + [
            Initial(
                target=n.id,
                expression=str(original_init_value / float(len(new_vars))),
            )
            for n in new_vars
        ]

        if strata_parameters is not None:
            original_parameter_values = {
                p: self._parameter_values()[p] for p in strata_parameters
            }
            original_parameters = {
                sp: next(
                    iter(
                        [
                            p
                            for p in self.petrinet.semantics.ode.parameters
                            if p.id == sp
                        ]
                    )
                )
                for sp in strata_parameters
            }
            unchanged_parameters = [
                p
                for p in self.petrinet.semantics.ode.parameters
                if p.id not in strata_parameters
            ]
            src_only_parameters = [
                Parameter(
                    id=f"{sp}_{level}",
                    name=f"{sp}_{level}",
                    description=f"{original_parameters[sp].description} stratified as {sp}_{level}",
                    value=original_parameter_values[sp],
                    distribution=original_parameters[sp].distribution,
                    units=original_parameters[sp].units,
                    grounding=original_parameters[sp].grounding,
                )
                for t_id, r in old_rates.items()
                for sp in strata_parameters
                if t_id in src_only_transitions
                and sympy.Symbol(sp) in old_rates[t_id][0].free_symbols
                for level in strata
            ]

            dest_only_parameters = [
                Parameter(
                    id=f"{sp}_{level}",
                    name=f"{sp}_{level}",
                    description=f"{original_parameters[sp].description} stratified as {sp}_{level}",
                    value=original_parameter_values[sp],
                    distribution=original_parameters[sp].distribution,
                    units=original_parameters[sp].units,
                    grounding=original_parameters[sp].grounding,
                )
                for t_id, r in old_rates.items()
                for sp in strata_parameters
                if t_id in dest_only_transitions
                and sympy.Symbol(sp) in old_rates[t_id][0].free_symbols
                for level in strata
            ]

            src_and_dest_parameters = [
                Parameter(
                    id=f"{sp}_{level_s}_{level_t}",
                    name=f"{sp}_{level_s}_{level_t}",
                    description=f"{original_parameters[sp].description} stratified as {sp}_{level_s}_{level_t}",
                    value=original_parameter_values[sp],
                    distribution=original_parameters[sp].distribution,
                    units=original_parameters[sp].units,
                    grounding=original_parameters[sp].grounding,
                )
                for t_id, r in old_rates.items()
                for sp in strata_parameters
                if t_id in src_and_dest_transitions
                and sympy.Symbol(sp) in old_rates[t_id][0].free_symbols
                for level_s in strata
                for level_t in strata
            ]
            transition_probability_parameters = [
                Parameter(
                    id=f"p_{state_var}_{level_s}_{level_t}",
                    name=f"p_{state_var}_{level_s}_{level_t}",
                    description=f"p_{state_var}_{level_s}_{level_t}",
                    value=str(1.0 / float(len(strata))),
                    distribution=None,
                    units=None,
                    grounding=None,
                )
                for t_id, r in old_rates.items()
                if t_id in src_and_dest_transitions
                for level_s in strata
                for level_t in strata
            ]
            new_parameters = (
                unchanged_parameters
                + src_only_parameters
                + dest_only_parameters
                + src_and_dest_parameters
                + transition_probability_parameters
            )
        else:
            new_parameters = self.petrinet.semantics.ode.parameters

        # FIXME update with splits
        new_observables = self.petrinet.semantics.ode.observables

        if self_strata_transition:
            self_strata_transitions = [
                Transition(
                    id=f"p_{state_var}_{level_s}_{level_t}",
                    input=[f"{state_var}_{level_s}"],
                    output=[f"{state_var}_{level_t}"],
                    grounding=None,
                    properties={"name": f"p_{state_var}_{level_s}_{level_t}"},
                )
                for level_s in strata
                for level_t in strata
                if level_s != level_t
            ]
            self_strata_rates = [
                Rate(
                    target=f"p_{state_var}_{level_s}_{level_t}",
                    expression=f"{state_var}_{level_s}*p_{state_var}_{level_s}_{level_t}",
                )
                for level_s in strata
                for level_t in strata
                if level_s != level_t
            ]
            self_strata_parameters = (
                [
                    Parameter(
                        id=f"p_{state_var}_{level_s}_{level_t}",
                        name=f"p_{state_var}_{level_s}_{level_t}",
                        description="Transition rate parameter between {state_var} strata {level_s} and {level_t}.",
                        value=1.0 / float(len(strata)),
                    )
                    for level_s in strata
                    for level_t in strata
                    # if level_s != level_t
                ]
                if len(transition_probability_parameters) == 0
                else []
            )
        else:
            self_strata_transitions = []
            self_strata_rates = []
            self_strata_parameters = []

        new_model = GeneratedPetriNetModel(
            petrinet=Model(
                header=self.petrinet.header,
                properties=self.petrinet.properties,
                model=Model1(
                    states=new_states,
                    transitions=[
                        *new_transitions,
                        *other_transitions.values(),
                        *self_strata_transitions,
                    ],
                ),
                semantics=Semantics(
                    ode=OdeSemantics(
                        rates=[
                            *new_rates,
                            *other_rates.values(),
                            *self_strata_rates,
                        ],
                        initials=new_initials,
                        parameters=new_parameters + self_strata_parameters,
                        observables=new_observables,
                        time=self.petrinet.semantics.ode.time,
                    ),
                    typing=self.petrinet.semantics.typing,
                    span=self.petrinet.semantics.span,
                ),
                metadata=self.petrinet.metadata,
            )
        )

        return new_model  # new_rates, transitions, new_transitions # dest_only_rates #original_var, new_vars, new_transitions

    def abstract(self, state_abstraction: Dict[str, str]):
        # Get existing state variables
        state_objs = {s.id: s for s in self._state_vars()}

        # Check that there is a state variable for each key in the state_abstraction
        assert all(
            {k in state_objs for k in state_abstraction.keys()}
        ), f"There are unknown states in the state_abstraction keys: {[k for k in state_abstraction.keys() if k not in state_objs]}"

        # Check that the state_abstraction maps the keys to a state variable that is not in the state_objs
        assert not any(
            {v in state_objs for v in state_abstraction.values()}
        ), f"There are unknown states in the state_abstraction values: {[v for v in state_abstraction.values() if v in state_objs]}"

        # Create states for values in state_abstraction
        new_state_objs = {
            v: State(
                id=v, name=v, description=None, grounding=None, units=None
            )
            for v in set(state_abstraction.values())
        }

        new_states = [
            *[
                v
                for k, v in state_objs.items()
                if k not in state_abstraction.keys()
            ],
            *new_state_objs.values(),
        ]

        # Replace states in the transitions
        subbed_state_ids = set(state_abstraction.keys())
        subbed_transitions = [  # transitions not involved in abstraction
            t
            for t in self.petrinet.model.transitions
            if not any(
                [s for s in t.input + t.output if s in subbed_state_ids]
            )
        ] + [  # transitions with substitutions
            Transition(
                id=t.id,
                input=[
                    (state_abstraction[i] if i in state_abstraction else i)
                    for i in t.input
                ],
                output=[
                    (state_abstraction[i] if i in state_abstraction else i)
                    for i in t.output
                ],
                grounding=t.grounding,
                properties=t.properties,
            )
            for t in self.petrinet.model.transitions
            if any([s for s in t.input + t.output if s in subbed_state_ids])
        ]
        grouped_transitions = []
        for t in subbed_transitions:
            # Find group in grouped_transitions for t
            try:
                matching_group = next(
                    iter(
                        [
                            i
                            for i, g in enumerate(grouped_transitions)
                            if any(
                                [
                                    t.input == gt.input
                                    and t.output == gt.output
                                    for gt in g
                                ]
                            )
                        ]
                    )
                )
                # print("append {t}")
                grouped_transitions[matching_group].append(t)
            except StopIteration:
                # If no group for t, then make a new group
                grouped_transitions.append([t])

        grouped_rates = [
            [
                next(
                    Rate(
                        target=r.target,
                        expression=str(
                            to_sympy(r.expression, self._symbols())
                        ),
                        expression_mathml=None,
                    )
                    for r in self.petrinet.semantics.ode.rates
                    if r.target == t.id
                )
                for t in g
            ]
            for g in grouped_transitions
        ]

        # Convert grouped transitions into a single transition
        consolidated_transitions = []
        for g in grouped_transitions:
            if len(g) == 1:
                consolidated_transitions.append(g[0])
            else:
                sub_sequences = set(
                    SequenceMatcher(
                        None, g[0].id, g[1].id
                    ).get_matching_blocks()
                )
                s_sub = list(sub_sequences)
                s_sub.sort(key=lambda x: min(x.a, x.b))
                sub = "".join(
                    [g[0].id[s.a : s.a + s.size] for s in s_sub if s.size > 0]
                )
                for i, t in enumerate(g[2:]):
                    sub_sequences = set(
                        SequenceMatcher(None, sub, t.id).get_matching_blocks()
                    )
                    s_sub = list(sub_sequences)
                    s_sub.sort(key=lambda x: min(x.a, x.b))
                    sub = "".join(
                        [
                            g[0].id[s.a : s.a + s.size]
                            for s in s_sub
                            if s.size > 0
                        ]
                    )
                if sub.endswith("_"):
                    sub = sub[:-1]

                consolidated_transitions.append(
                    Transition(
                        id=sub,
                        input=g[0].input,
                        output=g[0].output,
                        grounding=g[0].grounding,
                        properties=g[0].properties,
                    )
                )

        ## Remove self transitions
        new_transitions = [
            t
            for t in consolidated_transitions
            if not (t.input == t.output and len(t.input) == 1)
        ]

        def get_rate(
            self, target: str, rates: List[Rate], max_rate=False
        ) -> Rate:
            # Identify which variables in the rate expressions are distinct
            # Verify that by replacing the different variables that the expressions are identical
            expressions = [
                to_sympy(r.expression, self._symbols()) for r in rates
            ]
            expression_symbols = [e.free_symbols for e in expressions]
            all_symbols = reduce(
                lambda x, y: x.union(y),
                expression_symbols[1:],
                set(expression_symbols[0]),
            )
            common_symbols = reduce(
                lambda x, y: x.intersection(y),
                expression_symbols[1:],
                set(expression_symbols[0]),
            )
            parameter_map = {}
            if all_symbols != common_symbols:
                # Need to find the min
                variable_value = {
                    str(s): self._parameter_values()[str(s)]
                    for s in all_symbols
                    if s not in common_symbols
                }

                if max_rate:
                    min_value = max(variable_value.values())
                    new_parameter_id = (
                        "max("
                        + ",".join(
                            [
                                str(s)
                                for s in all_symbols
                                if s not in common_symbols
                            ]
                        )
                        + ")"
                    )

                else:
                    min_value = min(variable_value.values())
                    new_parameter_id = (
                        "min("
                        + ",".join(
                            [
                                str(s)
                                for s in all_symbols
                                if s not in common_symbols
                            ]
                        )
                        + ")"
                    )

                new_parameter_symbol = sympy.Symbol(new_parameter_id)
                new_expression = str(
                    expressions[0].subs(
                        {
                            old_symbol: new_parameter_symbol
                            for old_symbol in all_symbols
                            if old_symbol not in common_symbols
                        }
                    )
                )
                parameter_map = {
                    str(s): new_parameter_id
                    for s in all_symbols
                    if s not in common_symbols
                }
                # new_parameter = Parameter(id=new_parmeter_id, name=new_parameter_id, description="", value=min_value )
            else:
                new_expression = str(expressions[0])

            return (
                Rate(target=target, expression=new_expression),
                parameter_map,
            )

        def aggregate_rates(self, rates, abstraction):
            expressions = [
                to_sympy(r.expression, self._symbols()) for r in rates
            ]
            starting_expression = reduce(lambda x, y: x + y, expressions)

            # invert the abstraction
            i_abstraction = {
                k: [v for v in abstraction.keys() if abstraction[v] == k]
                for k in abstraction.values()
            }

            # abstraction implies that abstract variable is sum of variables mapped to it
            abstraction_substitution = {
                sympy.Symbol(i_abstraction[k][0]): to_sympy(
                    f"{k}-{'-'.join(i_abstraction[k][1:])}",
                    self._symbols() + list(abstraction.values()),
                )
                for k in i_abstraction.keys()
            }
            # abstraction_substitution = {Symbol(k): [Symbol(v) for v in v_list] for k, v_list in abstraction_substitution.items()}

            expression_symbols = [e.free_symbols for e in expressions]
            all_symbols = reduce(
                lambda x, y: x.union(y),
                expression_symbols[1:],
                set(expression_symbols[0]),
            )
            common_symbols = reduce(
                lambda x, y: x.intersection(y),
                expression_symbols[1:],
                set(expression_symbols[0]),
            )
            unique_symbols = [
                s
                for s in all_symbols
                if s not in common_symbols and str(s) not in abstraction
            ]

            # FIXME need to remove double counting of parameters p_I_I and beta_I_I
            parameter_minimization = {
                str(s): f"agg_{'_'.join([str(us) for us in unique_symbols])}"
                for s in unique_symbols
            }

            # substitute abstraction into starting expression
            abstract_expression = sympy.expand(
                starting_expression.subs(
                    {**abstraction_substitution, **parameter_minimization}
                )
            )

            # abstract_expression1 = sympy.expand(sympy.expand(starting_expression.subs(abstraction_substitution )).subs(parameter_minimization))
            return {
                "rate": str(abstract_expression),
                "parameters": parameter_minimization,
            }

        aggregated_rates_and_parameters = [
            aggregate_rates(
                self, g, state_abstraction
            )  # reduce(lambda x, y: x+y, [to_sympy(r.expression, self._symbols()) for r in g]) #"+".join([f"({r.expression})" for r in g])
            for i, g in enumerate(grouped_rates)
            if i < len(new_transitions)
        ]

        # New Rates
        new_rates = [
            Rate(
                target=new_transitions[i].id,
                expression=aggregated_rates_and_parameters[i]["rate"],
            )
            for i, g in enumerate(grouped_rates)
            if i < len(new_transitions)
        ]

        new_parameters = [
            p
            for p in self.petrinet.semantics.ode.parameters
            if not any(
                [
                    p.id in rp["parameters"]
                    for rp in aggregated_rates_and_parameters
                ]
            )
            and any([p.id in r.expression for r in new_rates])
        ] + [
            Parameter(
                id=p,
                name=p,
                description=p,
                v=0.0,
                grounding=None,
                distribution=None,
                units=None,
            )
            for arp in aggregated_rates_and_parameters
            for p in set(arp["parameters"].values())
        ]

        new_initials = [
            # Initial.model_copy(st)
            next(
                i
                for i in self.petrinet.semantics.ode.initials
                if i.target == st.id
            )
            for st in new_states
            if st.id not in new_state_objs
        ] + [
            Initial(
                target=s_id,
                expression=str(
                    reduce(
                        lambda x, y: x + y,
                        [
                            to_sympy(o_s.expression, self._symbols())
                            for o_s in [
                                next(
                                    i
                                    for i in self.petrinet.semantics.ode.initials
                                    if i.target == o_s_id
                                )
                                for o_s_id in [
                                    k
                                    for k, v in state_abstraction.items()
                                    if v == s_id
                                ]
                            ]
                        ],
                        0.0,
                    )
                ),
            )
            for s_id, st in new_state_objs.items()
            if s_id in new_state_objs
        ]

        new_metadata = self.petrinet.metadata.copy()
        new_metadata["abstraction"] = {
            # Need to know which parameter to replace by the min or max value, as well as the min and max value
            # parameters -> transition_id -> parameter_id -> [lb,ub]
            "parameters": {
                new_rates[i].target: {
                    p: {
                        "lb": min(
                            [
                                next(
                                    p2
                                    for p2 in self.petrinet.semantics.ode.parameters
                                    if p2.id == p1
                                ).value
                                for p1, v in arp["parameters"].items()
                                if v == p
                            ]
                        ),
                        "ub": max(
                            [
                                next(
                                    p2
                                    for p2 in self.petrinet.semantics.ode.parameters
                                    if p2.id == p1
                                ).value
                                for p1, v in arp["parameters"].items()
                                if v == p
                            ]
                        ),
                    }
                    for p in set(arp["parameters"].values())
                }
                for i, arp in enumerate(aggregated_rates_and_parameters)
                if len(arp["parameters"]) > 0
            }
        }

        new_model = GeneratedPetriNetModel(
            petrinet=Model(
                header=self.petrinet.header,
                properties=self.petrinet.properties,
                model=Model1(
                    states=new_states,
                    transitions=new_transitions,  # [*new_transitions, *other_transitions.values(), *self_strata_transitions]
                ),
                semantics=Semantics(
                    ode=OdeSemantics(
                        rates=new_rates,  # [*new_rates, *other_rates.values(), *self_strata_rates],
                        initials=new_initials,  # new_initials,
                        parameters=new_parameters,
                        observables=None,  # new_observables,
                        time=self.petrinet.semantics.ode.time,
                    ),
                    typing=self.petrinet.semantics.typing,
                    span=self.petrinet.semantics.span,
                ),
                metadata=new_metadata,
            )
        )
        return new_model


class PetrinetDynamics(BaseModel):
    json_graph: Dict[
        str,
        List[
            Dict[
                str,
                Optional[
                    Union[
                        Dict[str, Optional[Union[str, bool, float]]],
                        int,
                        str,
                        float,
                    ]
                ],
            ]
        ],
    ]

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     self.json_graph = kwargs["json_graph"]
    #     self._initialize_from_json()


class PetrinetModel(AbstractPetriNetModel):
    petrinet: PetrinetDynamics

    def default_encoder(self, config: "FUNMANConfig") -> "Encoder":
        """
        Return the default Encoder for the model

        Returns
        -------
        Encoder
            SMT encoder for model
        """
        return PetrinetEncoder(
            config=config,
            model=self,
        )

    def _state_vars(self):
        return self.petrinet.json_graph["S"]

    def _state_var_names(self):
        return [self._state_var_name(s) for s in self.petrinet.json_graph["S"]]

    def _state_var_name(self, state_var: Dict) -> str:
        return state_var["sname"]

    def _transitions(self):
        return self.petrinet.json_graph["T"]

    def _input_edges(self):
        return self.petrinet.json_graph["I"]

    def _output_edges(self):
        return self.petrinet.json_graph["O"]

    def _edge_source(self, edge):
        return edge["is"] if "is" in edge else edge["ot"]

    def _edge_target(self, edge):
        return edge["it"] if "it" in edge else edge["os"]

    def _transition_rate(self, transition):
        return self._encode_state_var(transition["tprop"]["parameter_name"])

    def _transition_id(self, transition):
        return next(
            i + 1
            for i, s in enumerate(self._transitions())
            if s["tname"] == transition["tname"]
        )

    def _state_var_id(self, state_var):
        return next(
            i + 1
            for i, s in enumerate(self._state_vars())
            if s["sname"] == state_var["sname"]
        )

    def _parameter_names(self):
        return [t["tprop"]["parameter_name"] for t in self._transitions()]

    def _parameter_values(self):
        return {
            t["tprop"]["parameter_name"]: t["tprop"]["parameter_value"]
            for t in self._transitions()
        }

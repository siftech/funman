import copy
import itertools
import logging
from collections import Counter
from difflib import SequenceMatcher
from functools import reduce
from math import prod
from typing import Callable, Dict, List, Optional, Set, Union

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
    _logger: logging.Logger = logging.getLogger(__name__)

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


class StratumAttributeValue(BaseModel):
    name: str

    def __hash__(self):
        return self.name.__hash__()

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (
            isinstance(other, StratumAttributeValue)
            and self.name == other.name
        )


class StratumAttribute(BaseModel):
    name: str
    values: Set[StratumAttributeValue]

    def __hash__(self):
        return self.name.__hash__()

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


class StratumAttributeValueSet(BaseModel):
    values: Set[StratumAttributeValue]

    def __hash__(self):
        return sum(hash(v) for v in self.values)

    def __str__(self):
        return "_".join((str(v) for v in self.values))

    def __repr__(self):
        return str(self)

    def is_subset(self, other, strict=False):
        return self.values.issubset(other.values) and (
            not strict or not other.values.issubset(self.values)
        )


class StratumValuation(BaseModel):
    values: Dict[StratumAttribute, StratumAttributeValueSet] = {}


    def __hash__(self):
        return sum(hash(k)*hash(v) for k,v in self.values.items())

    def __str__(self):
        return "_".join([f"{a}_{av}" for a, av in self.values.items()])

    def __repr__(self):
        return str(self)

    def __getitem__(self, key):
        if key in self.values:
            return self.values[key]
        else:
            return key.values

    def __setitem__(self, key, value):
        self.values[key] = value

    def expand_valuations(self, attr: StratumAttribute):
        exp_vals = []
        for val in attr.values:
            c = self.model_copy(deep=True)
            c[attr] = StratumAttributeValueSet(values={val})
            exp_vals.append(c)
        return exp_vals

    def num_interpretations(self, attributes: Set[StratumAttribute]):
        unassigned_attributes = attributes.difference(set(self.values.keys()))
        num_specified_interpretations = prod([len(v.values) for v in self.values.values()])
        num_unspecified_interpretations = prod(
            [len(a.values) for a in unassigned_attributes]
        )
        return  num_specified_interpretations * num_unspecified_interpretations

    def is_subset(self, other, strict=False):
        # cases:
        #  - attr not in self, attr not in other -> not strict
        #  - attr in self, attr not in other -> not strict or self[attr] != domain(attr)
        #  - attr not in self, attr in other -> not strict or other[attr] == domain(attr)
        #  - attr in self, attr in other -> self[attr].is_subset(other[atter])
        #  --

        attrs = set(self.values.keys()).union(set(other.values.keys()))
        return (len(attrs) == 0 and not strict) or (
            len(attrs) > 0
            and all(
                [
                    (
                        attr not in self.values
                        and attr not in other.values
                        and not strict
                    )
                    or (
                        attr in self.values
                        and attr not in other.values
                        and (not strict or self.values[attr] != attr.values)
                    )
                    or (
                        attr not in self.values
                        and attr in other.values
                        and (not strict or other.values[attr] == attr.values)
                    )
                    or (
                        attr in self.values
                        and attr in other.values
                        and self.values[attr].is_subset(
                            other.values[attr], strict=strict
                        )
                    )
                    for attr in attrs
                ]
            )
        )

        # return (len(self.values)==0 and not strict and len(other.values)== 0) or \
        #        all((
        #     a not in other.values.keys() or
        #     av.is_subset(other.values[a], strict=strict)
        #     for a, av in self.values.items()))


class Stratum(BaseModel):
    values: Dict[StratumAttribute, Set[StratumAttributeValueSet]] = {}

    def valuations(self) -> List[StratumValuation]:
        value_list = list(self.values.items())
        cross_vals = itertools.product(*[a_val[1] for a_val in value_list])
        valuation = [
            StratumValuation(
                values={
                    val_list[0]: val for (val_list, val) in zip(value_list, v)
                }
            )
            for v in cross_vals
        ]
        return valuation

    def __hash__(self):
        return reduce(
            lambda v1, v2: hash(v1)
            + reduce(lambda a, b: hash(a) + hash(b), self.values[v2], 0),
            self.values,
            0,
        )

    def __str__(self):
        return (
            "["
            + ",".join([str(k) + "=" + str(v) for k, v in self.values.items()])
            + "]"
        )


class Stratification(BaseModel):
    base_state: str
    base_parameters: List[str] = []
    stratum: Stratum  # interpreted as cross product over attribute values
    self_strata_transitions: bool = False
    cross_strata_transitions: bool = False
    only_natural_transitions: bool = (
        True  # only stratify transitions that are not persistence
    )
    _new_vars: Optional[List[State]]=None
    _new_vars_strata: Optional[Dict[str, Stratum]] = None
    _state_ancestors: Dict[str, str] = {}
    _transition_ancestors: Dict[str, str] = {}
    _parameter_ancestors: Dict[str, str] = {}


    def _ancestors(self):
        return {
            **self._state_ancestors,
            **self._transition_ancestors,
            **self._parameter_ancestors
        }

    def stratifies(self, variable: str) -> bool:
        return any(
            [
                s
                for s in self.strata
                if f"{self.base_state}_{s.name}" == variable
            ]
        )


class Abstraction(BaseModel):
    abstraction: Dict[str, str]
    parameters: Dict[str, Dict[str, Interval]] = {}
    base_states: Dict[str, State] = {}

    _state_ancestors: Dict[str, List[str]] = {}
    _transition_ancestors: Dict[str, List[str]] = {}
    _parameter_ancestors: Dict[str, List[str]] = {}

    def _ancestors(self):
        return {
            **self._state_ancestors,
            **self._transition_ancestors,
            **self._parameter_ancestors
        }

    def keys(self):
        return self.abstraction.keys()

    def values(self):
        return self.abstraction.values()

    def items(self):
        return self.abstraction.items()

    def __getitem__(self, key):
        return self.abstraction[key]

    def __setitem__(self, key, value):
        self.abstraction[key] = value

    def get(self, key, default=None):
        # Custom logic here
        if key in self:
            return self.abstraction[key]
        else:
            return default

    def is_transition_abstracted(self, t: Transition) -> bool:
        return any([s for s in t.input + t.output if s in self.keys()])

    def abstract_transition(self, t: Transition) -> Transition:
        return Transition(
            id=t.id,
            input=[self.abstraction.get(i, i) for i in t.input],
            output=[self.abstraction.get(i, i) for i in t.output],
            grounding=t.grounding,
            properties=t.properties,
        )

    def abstract_states(self):
        abstract_state_ids = set(
            [self[bs] for bs in self.base_states if bs in self.abstraction]
        )
        return {
            ab: State(
                id=ab, name=ab, description=None, grounding=None, units=None
            )
            for ab in abstract_state_ids
        }

    def set_parameters(
        self, base_param_bounds, new_rates, aggregated_rates_and_parameters
    ):
        # abstraction_params = set([
        #     p
        #     for arp in aggregated_rates_and_parameters
        #     for p in arp["parameters"].values])

        self.parameters = {
            new_rates[i].target: {
                p: {
                    "lb": min(
                        [
                            base_param_bounds[p1].lb
                            for p1, v in arp["parameters"].items()
                            if v == p
                        ]
                    ),
                    "ub": max(
                        [
                            base_param_bounds[p1].ub
                            for p1, v in arp["parameters"].items()
                            if v == p
                        ]
                    ),
                }
                for p in arp["parameters"].values()
            }
            for i, arp in enumerate(aggregated_rates_and_parameters)
            if len(arp["parameters"]) > 0
        }
        # self.parameters = {
        #         new_rates[i].target: {
        #             p: {
        #                 "lb": min(
        #                     [
        #                         base_model.parameter(p1).value
        #                         for p1, v in arp["parameters"].items()
        #                         if v == p
        #                     ]
        #                 ),
        #                 "ub": max(
        #                     [
        #                         base_model.parameter(p1).value
        #                         for p1, v in arp["parameters"].items()
        #                         if v == p
        #                     ]
        #                 ),
        #             }
        #             for p in arp["parameters"].values()
        #         }
        #         for i, arp in enumerate(aggregated_rates_and_parameters)
        #         if len(arp["parameters"]) > 0
        #     }
        # bounded_params = {
        #     p
        #     for trans_id, p_bounds in base_model.parameters.items()
        #     for p in p_bounds
        # }
        # param_min_value = {
        #     p: min(
        #         [
        #             bounds["lb"]
        #             for trans_id, p_bounds in base_model.parameters.items()
        #             for p1, bounds in p_bounds.items()
        #             if p1 == p
        #         ]
        #     )
        #     for p in bounded_params
        # }
        # param_max_value = {
        #     p: max(
        #         [
        #             bounds["ub"]
        #             for trans_id, p_bounds in self.parameters.items()
        #             for p1, bounds in p_bounds.items()
        #             if p1 == p
        #         ]
        #     )
        #     for p in bounded_params
        # }
        # self.parameters = {
        #     trans_id: {
        #         p1: (
        #             bounds
        #             if p1 not in param_min_value
        #             else {"lb": param_min_value[p1], "ub": param_max_value[p1]}
        #         )
        #         for p1, bounds in p_bounds.items()
        #     }
        #     for trans_id, p_bounds in self.parameters.items()
        # }

    def abstract_strata(self, state, state_strata):
        # Combine the strata for states that map to state in self
        mapped_state_valuations = {
            k: state_strata[k]
            for k, v in self.abstraction.items()
            if v == state
        }
        valuation_attrs = {
            k
            for v in mapped_state_valuations.values()
            for k in v.values.keys()
        }
        valuation = StratumValuation(
            values={
                attr: StratumAttributeValueSet(
                    values={
                        v
                        for s, strat in mapped_state_valuations.items()
                        for v in strat.values[attr].values
                    }
                )
                for attr in valuation_attrs
            }
        )
        return valuation

        # if need to combine v1, v2 that have different attrs, then
        # v1 = {a1: [v1, v2]}, v2 = {a2: [v3, v4]}, intrepret as:
        # v1 = {a1: [v1, v2], a2: all(vals(a2))}, v2 = {a1: all(vals(a1)), a2: [v3, v4]}


class StrataTransition(BaseModel):
    input_stratum: Optional[StratumValuation] = None
    output_stratum: Optional[StratumValuation] = None

    def __str__(self):
        return f'_{self.input_stratum}_to_{self.output_stratum}'

    def __repr__(self):
        return str(self)

    def input_attributes(self):
        in_attrs = set(self.input_stratum.values.keys())if self.input_stratum else set()
        return in_attrs

    def output_attributes(self):
        out_attrs = set(self.output_stratum.values.keys())if self.output_stratum else set()
        return out_attrs

    def attributes(self):
        in_attrs = self.input_attributes()
        out_attrs = self.output_attributes()
        attributes = in_attrs.union(out_attrs)
        return attributes

    def num_input_interpretations(self, strata_attributes=set([])):
        num_strata = self.input_stratum.num_interpretations(self.input_attributes().union(strata_attributes))
        return num_strata 

    def num_output_interpretations(self, strata_attributes=set([])):
        num_strata = self.output_stratum.num_interpretations(self.output_attributes().union(strata_attributes))
        return num_strata  


class StateTransition(BaseModel):
    input: Optional[State] = None
    input_stratum: Optional[StratumValuation] = None
    output: Optional[State] = None
    output_stratum: Optional[StratumValuation] = None
    strata_transition: StrataTransition

    def __str__(self):
        return f"{self.input.id}({self.strata_transition.input_stratum}) -> {self.output.id}({self.strata_transition.output_stratum})"

    def __repr__(self):
        return str(self)

    def id(self, state_vars:List[str]=None):
        input_id = self.input_id() if state_vars is None or  self.input_id() in state_vars else "_"
        output_id = self.output_id() if state_vars is None or self.output_id() in state_vars else "_"
        return f"{input_id}_to_{output_id}"

    def input_id(self):
        # return f"{self.input.id}_{self.input_stratum}"
        return f"{self.input.id}"

    def output_id(self):
        # return f"{self.output.id}_{self.output_stratum}"
        return f"{self.output.id}"

    def is_natural_transition(self):
        return self.input != self.output

    def stratify(self, var, strata_transition):
        # (src, params, dest) = strata_transition
        new_st = self.model_copy(deep=True)
        if (
            self.input
            and self.input == var
            and strata_transition
            and strata_transition.input_stratum.is_subset(new_st.strata_transition.input_stratum)
        ):
            new_st.input = State(
                id=f"{new_st.input.id}_{strata_transition.input_stratum}",
                name=f"{new_st.input.name}_{strata_transition.input_stratum}",
            )
            new_st.strata_transition.input_stratum = strata_transition.input_stratum
        if (
            self.output
            and self.output == var
            and strata_transition.output_stratum
            and strata_transition.output_stratum.is_subset(new_st.strata_transition.output_stratum)
        ):
            new_st.output = State(
                id=f"{new_st.output.id}_{strata_transition.output_stratum}",
                name=f"{new_st.output.name}_{strata_transition.output_stratum}",
            )
            new_st.strata_transition.output_stratum = strata_transition.output_stratum
        return new_st

    def num_input_interpretations(self):
        return self.strata_transition.num_input_interpretations()

    def num_output_interpretations(self):
        return self.strata_transition.num_output_interpretations()  

    def stratification_allowed(self, strata_transition):
        # allowed if strata_transition src and dest are consistent with input and output
        # (src, params, dest) = strata_transition
        return (strata_transition.input_stratum is None or strata_transition.input_stratum.is_subset(self.strata_transition.input_stratum)) and (
            strata_transition.output_stratum is None or strata_transition.output_stratum.is_subset(self.strata_transition.output_stratum)
        )

    def abstract_to_concrete(self):
        # if either i/o stratum is None, then its all strata
        # cases:
        # - both None -> false
        # - output None -> false
        # - input None -> true
        # - neither None -> is_subset
        #
        # predicate: input is None or (output is not None and is_subset)
        return (
            self.strata_transition.input_stratum is None and self.strata_transition.output_stratum is not None
        ) or (
            self.strata_transition.input_stratum is not None
            and self.strata_transition.output_stratum is not None
            and self.strata_transition.output_stratum.is_subset(self.strata_transition.input_stratum, strict=True)
        )


class TransitionMap(BaseModel):
    state_transitions: List[StateTransition] = []
    cross_stratam_transition_parameters: List[Parameter] = []
    # abstract_input_parameters: List[Parameter] = []
    transition_id: str = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(
        self, transition: Transition, model: "GeneratedPetrinetModel"
    ):
        self.transition_id = transition.id
        i_o_vars = set(transition.input).union(set(transition.output))
        remaining_inputs = transition.input.copy()
        remaining_outputs = transition.output.copy()

        # Identify variables that are persistence
        for v in transition.input:
            if v in remaining_outputs:
                self.state_transitions.append(
                    StateTransition(
                        input=model.state_var(v),
                        output=model.state_var(v),
                        strata_transition=StrataTransition(
                            input_stratum=model.state_strata(v),
                            output_stratum=model.state_strata(v)),
                    )
                )
                remaining_inputs.remove(v)
                remaining_outputs.remove(v)

        # Lexically associate remaining variables as transition pairs
        for i in range(0, max(len(remaining_inputs), len(remaining_outputs))):
            input_var = (
                model.state_var(remaining_inputs[i])
                if i < len(remaining_inputs)
                else None
            )
            output_var = (
                model.state_var(remaining_outputs[i])
                if i < len(remaining_outputs)
                else None
            )
            st = StateTransition(
                input=input_var,
                output=output_var,
                strata_transition=StrataTransition(
                    input_stratum=model.state_strata(input_var),
                    output_stratum=model.state_strata(output_var)
                )
            )
            self.state_transitions.append(st)

        # Identify relevant abstract input and cross-strata transition probabilities
        rates = model._transition_rate(transition)
        if len(rates) > 0:
            rate = str(rates[0])
            params = model._semantics_parameters()
            for (
                parameter_id,
                parameter,
            ) in params.items():
                if parameter_id in rate:
                    # if "p_abstract" in parameter_id:
                    #     self.abstract_input_parameters.append(parameter)
                    # el
                    if "p_cross" in parameter_id:
                        self.cross_stratam_transition_parameters.append(
                            parameter
                        )

    # def abstract_input_probability(self, state_var):
    #     return f"p_abstract_{self.transition_id}_{state_var.id}"

    def id(self, state_transitions, state_vars:List[str]=None):
        return "___".join(s.id(state_vars=state_vars) for s in state_transitions)

    def cross_strata_transition_probability(self, state_transitions):
        return f"p_cross_{self.id(state_transitions, state_vars=None)}"

    def inputs(self) -> List[State]:
        return [st.input.id for st in self.state_transitions]

    def outputs(self) -> List[State]:
        return [st.output.id for st in self.state_transitions]

    def var_ids(self) -> Set[State]:
        return set(self.inputs()).union(set(self.outputs()))

    def stratify(
        self, stratification, var, strata_transitions, transition_probability
    ):
        # new_sts = []
        # cst_params = []
        # ai_parameters = []
        tm = TransitionMap()
        for strata_transition, state_transition in zip(
            strata_transitions, self.state_transitions
        ):
            if strata_transition:
                if state_transition.stratification_allowed(strata_transition):
                    str_st = state_transition.stratify(var, strata_transition)
                    # FIXME modify ai_parameters
                    # if str_st.abstract_to_concrete():
                    #     abstract_parameter_name = (
                    #         self.abstract_input_probability(str_st.input)
                    #     )
                    #     num_interpretations = (
                    #         strata_transition.num_input_interpretations(strata_attributes=strata_transition.attributes())
                    #     )
                    #     abstract_parameter = Parameter(
                    #         id=abstract_parameter_name,
                    #         name=abstract_parameter_name,
                    #         description=abstract_parameter_name,
                    #         value=1.0 / float(num_interpretations),
                    #     )
                    #     tm.abstract_input_parameters.append(abstract_parameter)
                    #     stratification._parameter_ancestors[abstract_parameter_name] = self.transition_id
                    tm.state_transitions.append(str_st)
                else:
                    return None
            else:
                tm.state_transitions.append(state_transition)
        # FIXME modify cst_parameters
        if transition_probability < 1.0:
            cst_param_name = self.cross_strata_transition_probability(
                tm.state_transitions
            )
            num_interpretations = str_st.num_output_interpretations()
            tm.cross_stratam_transition_parameters.append(
                Parameter(
                    id=cst_param_name,
                    description=cst_param_name,
                    value=transition_probability
                )
            )
            stratification._parameter_ancestors[cst_param_name] = self.transition_id
        return tm

    # def abstract_to_concrete(self):
    #     return [
    #         st for st in self.state_transitions if st.abstract_to_concrete()
    #     ]


class GeneratedPetriNetModel(AbstractPetriNetModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    petrinet: GeneratedPetrinet
    _transition_rates_cache: Dict[str, Union[sympy.Expr, str]] = {}
    _observables_cache: Dict[str, Union[str, FNode, sympy.Expr]] = {}
    _transition_rates_lambda_cache: Dict[str, Union[Callable, str]] = {}
    _transition_maps: Dict[str, TransitionMap] = {}

    def transition_map(self, transition: Transition) -> TransitionMap:
        if transition.id not in self._transition_maps:
            transition_map = TransitionMap()
            transition_map.initialize(transition, self)
            self._transition_maps[transition.id] = transition_map
        return self._transition_maps[transition.id]

    def num_elements(self):
        num_elts = (
            len(self._state_var_names())
            + len(self._parameter_names())
            + len(self.observables())
            + len(list(self._transitions()))
        )
        return num_elts

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

    def _semantics_parameters(self):
        return (
            {p.id: p for p in self.petrinet.semantics.ode.parameters}
            if self.petrinet.semantics
            else {}
        )

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
                # # remove dups, preserve order
                # dedup = []
                # for s in unreserved_symbols:
                #     if s not in dedup:
                #         dedup.append(s)
                # unreserved_symbols = dedup

                # convert "t" to "timer_t"
                if unreserved_symbols[-1] == "t":
                    unreserved_symbols[-1] = self._time_var_id(
                        self._time_var()
                    )
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

    def state_var(self, state_var_id):
        try:
            return next(
                iter(
                    [
                        s
                        for s in self._state_vars()
                        if self._state_var_id(s) == state_var_id
                    ]
                )
            )
        except StopIteration:
            raise Exception(
                f"There is no state_var in model with id: {state_var_id}"
            )

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

    def _parameter_bounds(self) -> Dict[str, Interval]:
        if hasattr(self.petrinet.semantics, "ode"):
            return {
                p.id: Interval(
                    lb=self._parameter_lb(p.id), ub=self._parameter_ub(p.id)
                )
                for p in self.petrinet.semantics.ode.parameters
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
        contracted_model = self.petrinet.model_copy(deep=True)

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
            if (
                param.distribution.parameters["minimum"]
                == param.distribution.parameters["maximum"]
            ):
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

        abstracted_parameters = self.petrinet.metadata.get(
            "abstracted_parameters", {}
        )
        symbols = self._symbols()
        str_to_symbol = {s: sympy.Symbol(s) for s in symbols}
        bound_symbols = {
            sympy.Symbol(s): {
                bound: sympy.Symbol(f"{s}_{bound}") for bound in ["lb", "ub"]
            }
            for s in symbols
        }
        for _, params in abstracted_parameters.items():
            for p in params:
                bound_symbols[sympy.Symbol(p)] = {
                    bound: sympy.Symbol(f"{p}_{bound}")
                    for bound in ["lb", "ub"]
                }
        substituter = SympyBoundedSubstituter(
            bound_symbols=bound_symbols, str_to_symbol=str_to_symbol
        )

        def bound_expression(targets, e, bound, metadata):

            targets = {
                sympy.Symbol(k): sympy.Symbol(v) for k, v in targets.items()
            }
            e_s = sympy.sympify(e, substituter.str_to_symbol)

            # targets are forced substitutions
            # for k, v in targets.items():
            e_s = e_s.subs(targets)

            # # substitute abstracted parameters for lower or upper bounds
            # for k, v in metadata.items():
            #     e = e.replace(k, k.replace("agg", bound))

            return (
                substituter.minimize(targets, e_s)
                if bound == "lb"
                else substituter.maximize(targets, e_s)
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
            "in": {"lb": lb_expression, "ub": ub_expression},
            "out": {"lb": lb_expression, "ub": ub_expression},
        }

        bounded_transitions = {
            "inputs": {
                # Make transition for each input edge
                t.id: {
                    input_id: {
                        bound: {
                            "transition": Transition(
                                id=f"{t.id}_in_{input_id}_{bound}",
                                input=[
                                    f"{input_id}_{('lb' if bound == 'ub' else 'ub')}"
                                    for i in range(num_input)
                                ],
                                output=[],
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
                                target=f"{r.target}_in_{input_id}_{bound}",
                                expression=str(
                                    expression_bound_fn["in"][bound](
                                        {
                                            input_id: f"{input_id}_{('lb' if bound == 'ub' else 'ub')}"
                                        },
                                        r.expression,
                                        symbols,
                                        abstracted_parameters.get(
                                            r.target, {}
                                        ),
                                    )
                                ),
                            ),
                        }
                        for bound in ["lb", "ub"]
                    }
                    for input_id, num_input in Counter(t.input).items()
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
                                id=f"{t.id}_out_{output_id}_{bound}",
                                input=[],
                                output=[
                                    f"{output_id}_{bound}"
                                    for i in range(num_output)
                                ],
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
                                target=f"{r.target}_out_{output_id}_{bound}",
                                expression=str(
                                    expression_bound_fn["out"][bound](
                                        {output_id: f"{output_id}_{bound}"},
                                        r.expression,
                                        symbols,
                                        abstracted_parameters.get(
                                            r.target, {}
                                        ),
                                    )
                                ),
                            ),
                        }
                        for bound in ["lb", "ub"]
                    }
                    for output_id, num_output in Counter(t.output).items()
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
                f"{p}_lb": bounds["lb"],
                f"{p}_ub": bounds["ub"],
            }
            for t, params in abstracted_parameters.items()
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

    # def stratum(self, s: State) -> Stratum:
    #     # Actual Stratum of s
    #     s_stratum = Stratum()
    #     pass

    # def strata(self, s: State) -> List[Stratum]:
    #     # Stratum and Peer strata of s
    #     pass

    def stratified_trans_id(self, transition, strata_transitions):
        return f"{transition.id}_{'_'.join([str(st) for st in strata_transitions])}"

    def stratified_state_id(self, state_var, index, strata):
        return f"{state_var}_{'_'.join([str(s) for s in strata])}_{index}"

    def stratified_parameter_id(self, parameter, strata_transition):
        return f"{parameter}__{'_'.join([str(st) for st in strata_transitions])}"

    def transformations(self):
        return (
            self.petrinet.metadata["transformations"]
            if "transformations" in self.petrinet.metadata
            else []
        )

    def state_strata(self, state: Union[State, str]) -> StratumValuation:
        s_id = state if isinstance(state, str) else state.id
        if (
            self.petrinet.metadata
            and "state_strata" in self.petrinet.metadata
            and s_id in self.petrinet.metadata["state_strata"]
        ):
            return self.petrinet.metadata["state_strata"][s_id]
        else:
            return StratumValuation()

    def strata_self_transitions(self, stratification, state_var, strata):
        transition = Transition(
            id=f"self_{state_var.id}",
            input=[state_var.id],
            output=[state_var.id],
            grounding=None,
            properties={"name": f"self_{state_var.id}"},
        )
        strata_levels = strata.valuations()
        strata_transitions = [
            StrataTransition(input_stratum=input_strata, output_stratum=output_strata)
            for input_strata in strata_levels
            for output_strata in strata_levels
            if input_strata != output_strata
        ]
        tr_map = self.transition_map(transition)
        new_transitions = []
        new_transition_maps = []
        new_rates = []
        new_parameters = []
        old_parameters = self._semantics_parameters()
        state_strata_transitions = list(
            itertools.product(
                *[
                    (
                        strata_transitions
                        if (state_var == st.input or state_var == st.output)
                        else [None]
                    )
                    for st in tr_map.state_transitions
                ]
            )
        )
        transition_probability_value = 1.0/float(len(state_strata_transitions))
        for state_strata_transition in state_strata_transitions:
            strat_tr_map = tr_map.stratify(
                stratification, state_var, state_strata_transition, transition_probability_value
            )
            if strat_tr_map is None:
                continue
            new_transition_maps.append(strat_tr_map)
            new_id = self.stratified_trans_id(
                transition, state_strata_transition
            )
            new_t = Transition(
                id=new_id,
                input=strat_tr_map.inputs(),
                output=strat_tr_map.outputs(),
                grounding=transition.grounding,
                properties=Properties(
                    name=new_id,
                    description=(
                        f"{transition.properties.description} Stratified."
                        if transition.properties.description
                        else transition.properties.description
                    ),
                ),
            )
            new_transitions.append(new_t)
            try:
                transition_probability = next(
                    iter(strat_tr_map.cross_stratam_transition_parameters)
                )
            except StopIteration as e:
                self._logger.exception(
                    f"Did not generate a cross strata transition probability as expected, {e}"
                )
                raise e
            new_rate = Rate(
                target=new_id, expression=transition_probability.id
            )
            new_rates.append(new_rate)

            new_parameters.append(transition_probability)

        return new_transitions, new_rates, new_parameters

    def stratify_transition(
        self, transition: Transition, stratification: Stratification
    ):

        state_var = self.state_var(stratification.base_state)
        strata = stratification.stratum
        strata_parameters = stratification.base_parameters
        cross_strata_transitions = stratification.cross_strata_transitions

        assert (
            len(strata.values.keys()) == 1
        ), f"Only support stratification by one attribute at a time, got: {list(strata.keys())}"
        strata_attr = next(iter(strata.values.keys()))

        # Need to determine which strata transitions occur and involving what variables.
        strata_levels = strata.valuations()
        tr_map = self.transition_map(transition)
        possible_strata_transitions = []

        for st in tr_map.state_transitions:
            input_stratum = self.state_strata(st.input)
            output_stratum = self.state_strata(st.output)
            if strata_attr in input_stratum.values:
                # Already have a valuation for attribute
                current_input_stratum = [input_stratum]
            elif state_var == st.input:
                # Don't have a valuation, so consider all values
                current_input_stratum = input_stratum.expand_valuations(
                    strata_attr
                )
            else:
                current_input_stratum = [input_stratum]

            if strata_attr in output_stratum.values:
                # Already have a valuation for attribute
                current_output_stratum = [output_stratum]
            elif state_var == st.output:
                # Don't have a valuation, so consider all values
                current_output_stratum = output_stratum.expand_valuations(
                    strata_attr
                )
            else:
                current_output_stratum = [output_stratum]

            # Can only make more concrete with those levels consistent with the current stratum
            possible_input_levels = (
                [
                    s
                    for s in strata_levels
                    if any(
                        [s1 for s1 in current_input_stratum if s1.is_subset(s)]
                    )
                ]
                if state_var == st.input
                else current_input_stratum
            )
            possible_output_levels = (
                [
                    s
                    for s in strata_levels
                    if any(
                        [
                            s1
                            for s1 in current_output_stratum
                            if s1.is_subset(s)
                        ]
                    )
                ]
                if state_var == st.output
                else current_output_stratum
            )

            legal_strata_transitions = []
            for input_level in possible_input_levels:
                for output_level in possible_output_levels:

                    if cross_strata_transitions and st.is_natural_transition():
                        # allow levels to be different
                        # param_transition = (input_level if state_var == st.input else None, output_level if state_var == st.output else None)
                        # strat_params =  {p:self.stratified_parameter_id(p, [param_transition]) for p in strata_parameters if not (param_transition[0] is None and param_transition[1] is None)}
                        legal_strata_transitions.append(
                            (input_level, strata_parameters, output_level)
                        )
                    elif (
                        input_level == output_level
                        or len(input_level.values) == 0
                        or len(output_level.values) == 0
                    ):
                        # levels must be the same
                        # param_transition = (input_level if state_var == st.input else None, output_level if state_var == st.output else None)
                        # strat_params = {p:self.stratified_parameter_id(p, [param_transition]) for p in strata_parameters if not (param_transition[0] is None and param_transition[1] is None)}
                        legal_strata_transitions.append(
                            (input_level, strata_parameters, output_level)
                        )
            possible_strata_transitions.append(legal_strata_transitions)

        new_transitions = []
        new_transition_maps = []
        new_rates = []
        new_parameters = []
        new_strata_transitions = []
        # grouped_cross_transition_parameters = {}
        # grouped_abstract_transition_parameters = {}
        old_rate = self._transition_rate(transition)[0]
        old_parameters = self._semantics_parameters()
        state_strata_transitions = list(
            itertools.product(*possible_strata_transitions)
        )
        strata_transitions = [tuple(StrataTransition(input_stratum=tr[0], output_stratum=tr[2]) for tr in sst) for sst in state_strata_transitions ]

        # Need to know the probability of each transition.  For those with the same input states, 
        # we need to distribute the probability over the edges.
        # num_cross_strata_transitions = len(possible_output_levels) 
        #if len(tr_map.cross_stratam_transition_parameters) == 0 else int(1.0/tr_map.cross_stratam_transition_parameters[0].value)
        
        strata_transitions_by_input = {}
        for sst in strata_transitions:
            input_key = tuple(tr.input_stratum for tr in sst)
            input_key_transitions = strata_transitions_by_input.get(input_key, [])
            input_key_transitions.append(sst)
            strata_transitions_by_input[input_key] = input_key_transitions
            
        
        output_strata_attributes = set(a for st_tr in strata_transitions for t in st_tr for a in t.output_attributes())
        # Assume that all combinations of strata transitions are possible for now, and will normalize after processing all transitions
        total_interpretations = pow(prod(len(a.values) for a in output_strata_attributes), len(tr_map.state_transitions))
        
        strata_transition_probability_by_input = {}
        for input_key, input_strata_transitions in strata_transitions_by_input.items():
            num_interpretations = [prod([float(st.num_output_interpretations()) for st in sts]) for sts in input_strata_transitions ]
            transition_probability = [val/total_interpretations for val in num_interpretations]
            strata_transition_probability_by_input[input_key] = transition_probability


        for input_key in strata_transitions_by_input.keys():
            for state_strata_transition, transition_probability in zip(strata_transitions_by_input[input_key], strata_transition_probability_by_input[input_key]):            
                strat_tr_map = tr_map.stratify(
                    stratification,
                    state_var,
                    state_strata_transition,
                    transition_probability,
                )
                if strat_tr_map is None:
                    continue
                new_transition_maps.append(strat_tr_map)
                new_strata_transitions.append(state_strata_transition)

                relevant_new_state_var_ids = [s.id for s in stratification._new_vars if s.id in strat_tr_map.var_ids() ]

                new_id = self.stratified_trans_id(
                    transition, state_strata_transition
                )
                new_t = Transition(
                    id=new_id,
                    input=strat_tr_map.inputs(),
                    output=strat_tr_map.outputs(),
                    grounding=transition.grounding,
                    properties=Properties(
                        name=new_id,
                        description=(
                            f"{transition.properties.description} Stratified."
                            if transition.properties.description
                            else transition.properties.description
                        ),
                    ),
                )
                new_transitions.append(new_t)

                # abstract_transition_probabilities = {
                #     st.input.id: sympy.Symbol(
                #         f"p_abstract_{transition.id}_{st.input.id}"
                #     )
                #     for st in strat_tr_map.abstract_to_concrete()
                # }
                # cross_strata_transition_probability = {
                #     st.input.id: sympy.Symbol(f"p_cross_{transition.id}_{st.id()}")
                #     for st in strat_tr_map.state_transitions
                #     if (
                #         cross_strata_transitions
                #         and ((state_var.id in transition.output) or
                #               state_var.id in transition.input and "p_cross_" in str(old_rate))
                #         and (
                #             st.is_natural_transition()
                #             or not stratification.only_natural_transitions
                #         )
                #     )
                # }

                # for trans, arp in zip(
                #     new_transitions, aggregated_rates_and_parameters
                # ):

                # for i, p in cross_strata_transition_probability.items():
                #     grouped_cross_transition_parameters[i] = (
                #         grouped_cross_transition_parameters.get(i, set({})).union(
                #             set([p])
                #         )
                #     )
                # for i, p in abstract_transition_probabilities.items():
                #     grouped_abstract_transition_parameters[i] = (
                #         grouped_abstract_transition_parameters.get(
                #             i, set({})
                #         ).union(set([p]))
                #     )

                # Stratify parameters
                to_be_stratified_parameters = set(strata_parameters)
                # {
                #     k for sst in state_strata_transition for k in sst[1]
                # }
                param_subs = {}
                for p in to_be_stratified_parameters:
                    param_subs[p] = (
                        f"{p}_{strat_tr_map.id(strat_tr_map.state_transitions, state_vars=relevant_new_state_var_ids)}"
                    )
                    new_parameters.append(
                        Parameter(
                            id=param_subs[p],
                            name=param_subs[p],
                            description=f"{p} stratified as {param_subs[p]}",
                            value=old_parameters[p].value,
                            distribution=old_parameters[p].distribution,
                            units=old_parameters[p].units,
                            grounding=old_parameters[p].grounding,
                        )
                    )

                stratification._parameter_ancestors.update({v: k for k, v in param_subs.items()})

                input_subs = {
                    tr_map.state_transitions[i]
                    .input.id: strat_tr_map.state_transitions[i]
                    .input.id
                    for i in range(len(tr_map.state_transitions))
                    if tr_map.state_transitions[i].input.id
                    != strat_tr_map.state_transitions[i].input.id
                }

                abstract_params_not_needed = {
                    s: 1
                    for s in old_rate.free_symbols
                    if str(s).startswith("p_abstract_")
                    and len(strat_tr_map.abstract_input_parameters) == 0
                }
                cross_params_not_needed = {
                    s: 1
                    for s in old_rate.free_symbols
                    if str(s).startswith("p_cross_")
                    and len(strat_tr_map.cross_stratam_transition_parameters) > 0
                }
                all_sub = {
                    **input_subs,
                    **param_subs,
                    **abstract_params_not_needed,
                    **cross_params_not_needed,
                }
                rate_expr = old_rate.subs(all_sub)
                # for p in strat_tr_map.abstract_input_parameters:
                #     if p not in new_parameters:
                #         new_parameters.append(p)
                #     rate_expr = rate_expr * sympy.Symbol(p.id)
                for p in strat_tr_map.cross_stratam_transition_parameters:
                    new_parameters.append(p)
                    rate_expr = rate_expr * sympy.Symbol(p.id)
                new_rate = Rate(target=new_id, expression=str(rate_expr))
                new_rates.append(new_rate)

        # for parameter_groups in [
        #     grouped_abstract_transition_parameters,
        #     grouped_cross_transition_parameters,
        # ]:
        #     for input_state, group in parameter_groups.items():
        #         for parameter in group:
        #             new_parameters.append(
        #                 Parameter(
        #                     id=str(parameter),
        #                     name=str(parameter),
        #                     description=str(parameter),
        #                     value=str(1.0 / float(len(group))),
        #                     distribution=None,
        #                     units=None,
        #                     grounding=None,
        #                 )
        #             )

        return new_transitions, new_rates, new_parameters, new_strata_transitions

    def stratify_state(self, original_var, valuations):
        new_vars = []
        new_vars_strata = {}
        for valuation in valuations:
            valuation_str = "_".join(
                [
                    f"{attribute}_{values}"
                    for attribute, values in valuation.values.items()
                ]
            )
            new_var_id = f"{original_var.id}_{valuation_str}"
            new_var = State(
                id=new_var_id,
                name=new_var_id,
                description=f"{original_var.description} Stratified wrt. {valuation_str}",
                grounding=original_var.grounding,
                units=original_var.units,
            )
            new_vars.append(new_var)
            new_vars_strata[new_var.id] = valuation
        return new_vars, new_vars_strata

    def normalize_stratified_transitions(self, stratified_transitions_rates_params):
        # group transitions based upon their input states
        # normalize the cross strata transition probability within each group\

        strata_transitions_by_input = {}
        cross_strata_parameter_by_transition = {}
        for strps in stratified_transitions_rates_params.values():
            trs, rates, params, strata_transitions = strps
            for trans, rate in zip(trs, rates):
                input_key = tuple(trans.input)
                input_key_transitions = strata_transitions_by_input.get(input_key, [])
                input_key_transitions.append(trans)
                strata_transitions_by_input[input_key] = input_key_transitions

                for param in params:
                    if param.id in rate.expression and param.id.startswith("p_cross_"):
                        cross_strata_parameter_by_transition[trans.id] = param


        if len(cross_strata_parameter_by_transition) > 0:
            strata_transition_probability_by_input = {}
            for input_key, input_strata_transitions in strata_transitions_by_input.items():
                unnormalized_probabilities = [cross_strata_parameter_by_transition[t.id].value for t in input_strata_transitions if t.id in cross_strata_parameter_by_transition]
                norm = sum(unnormalized_probabilities)
                normalized_probabilities = [p/norm for p in unnormalized_probabilities]
                for t, p in zip(input_strata_transitions, normalized_probabilities):
                    cross_strata_parameter_by_transition[t.id].value = p
        return stratified_transitions_rates_params

    def stratify(self, stratification: Stratification):
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
        state_var = stratification.base_state
        stratum = stratification.stratum
        strata_parameters = stratification.base_parameters
        cross_strata_transitions = stratification.cross_strata_transitions
        self_strata_transition = stratification.self_strata_transitions

        # # get state variable
        # state_vars: List[State] = [
        #     s
        #     for s in self._state_vars()
        #     if self._state_var_name(s) == state_var
        # ]
        # assert (
        #     len(state_vars) == 1
        # ), "Found more than one State variable for {state_var}"
        # original_var = state_vars[0]
        original_var = self.state_var(state_var)

        valuations = stratum.valuations()
        new_vars, new_vars_strata = self.stratify_state(
            original_var, valuations
        )
        stratification._new_vars = new_vars
        stratification._new_vars_strata = new_vars_strata
        stratification._state_ancestors ={new_var.id: stratification.base_state for new_var in new_vars}

        # get new transitions
        transitions_to_stratify: List[Transition] = [
            t
            for t in self._transitions()
            if original_var.id in t.input or original_var.id in t.output
        ]
        transitions_to_stratify_ids = [t.id for t in transitions_to_stratify]
        other_transitions = {
            t.id: t
            for t in self._transitions()
            if t.id not in transitions_to_stratify_ids
        }

        stratified_transitions_rates_params = {
            t_id: self.stratify_transition(t, stratification)
            for t_id, t in zip(
                transitions_to_stratify_ids, transitions_to_stratify
            )
        }

        # Normalize the cross strata transition probabilities across the new transitions.
        # Its possible to generate transitions with the same input, but different output
        # from different pre-stratification transitions
        normalized_stratified_transitions_rates_params = self.normalize_stratified_transitions(stratified_transitions_rates_params)


        # Transitions
        stratification._transition_ancestors = {st.id: t_id for t_id, t in normalized_stratified_transitions_rates_params.items() for st in t[0]}
        new_transitions = [
            tr for t in normalized_stratified_transitions_rates_params.values() for tr in t[0]
        ]
        new_rates = [
            r for t in normalized_stratified_transitions_rates_params.values() for r in t[1]
        ]
        new_parameters = [
            p for t in normalized_stratified_transitions_rates_params.values() for p in t[2]
        ]        

        # There may be duplicate transition probability parameters between strata when there are multiple transitions that are stratified
        # This is ugly because Parameter does not have a hash function
        unique_params = []
        for p in new_parameters:
            if p not in unique_params:
                unique_params.append(p)
        new_parameters = unique_params

        other_rates = {
            r.target: r
            for r in self.petrinet.semantics.ode.rates
            if r.target in other_transitions
        }

        new_states = new_vars + [
            s for s in self.petrinet.model.states.root if s != original_var
        ]

        new_model = GeneratedPetriNetModel(
            petrinet=Model(
                header=self.petrinet.header,
                properties=self.petrinet.properties,
                model=Model1(
                    states=new_states,
                    transitions=new_transitions
                    + list(other_transitions.values()),
                ),
            )
        )

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
            if i.target != original_var.id
        ] + [
            Initial(
                target=n.id,
                expression=str(original_init_value / float(len(new_vars))),
            )
            for n in new_vars
        ]

        if len(strata_parameters) > 0:
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

            new_parameters = (
                unchanged_parameters
                + new_parameters
                # + src_only_parameters
                # + dest_only_parameters
                # + src_and_dest_parameters
                # + transition_probability_parameters
            )
        else:
            new_parameters += self.petrinet.semantics.ode.parameters

        # FIXME update with splits
        new_observables = self.petrinet.semantics.ode.observables

        new_model.petrinet.semantics = Semantics(
            ode=OdeSemantics(
                rates=[*new_rates, *other_rates.values()],
                initials=new_initials,
                parameters=new_parameters,
                observables=new_observables,
                time=self.petrinet.semantics.ode.time,
            ),
            typing=self.petrinet.semantics.typing,
            span=self.petrinet.semantics.span,
        )

        if self_strata_transition:
            (
                self_strata_transitions,
                self_strata_rates,
                self_strata_parameters,
            ) = self.strata_self_transitions(stratification, original_var, stratum)
            new_model.petrinet.model.transitions.root += (
                self_strata_transitions
            )
            new_model.petrinet.semantics.ode.rates += self_strata_rates
            new_model.petrinet.semantics.ode.parameters += (
                self_strata_parameters
            )

        # Remove parameters not appearing in a rate
        new_model.petrinet.semantics.ode.parameters = [
            p
            for p in new_model.petrinet.semantics.ode.parameters
            if any(
                [
                    p.id in r.expression
                    for r in new_model.petrinet.semantics.ode.rates
                ]
            )
        ]

        new_metadata = copy.deepcopy(self.petrinet.metadata)
        transformations = new_metadata.get("transformations", [])
        transformations.append(stratification)
        new_metadata["transformations"] = transformations
        state_strata = new_metadata.get("state_strata", {})
        state_strata.update(new_vars_strata)
        new_metadata["state_strata"] = state_strata
        
        # ancestors are reltations between states, transitions, and parameters
        ancestors = new_metadata.get("ancestors", [])
        ancestors.append(stratification._ancestors())
        new_metadata['ancestors'] = ancestors

        # Create new entries for parameter bounds of new parameters
        abstracted_parameters = new_metadata.get("abstracted_parameters", {})
        for i, t_id in enumerate(transitions_to_stratify_ids):
            old_param_bounds = abstracted_parameters.get(t_id, None)
            if old_param_bounds:
                for j, nt in enumerate(
                    stratified_transitions_rates_params[i][0]
                ):
                    abstracted_parameters[nt.id] = copy.deepcopy(
                        old_param_bounds
                    )
                    for bp in stratification.base_parameters:
                        # Add stratified parameters, inheriting bounds
                        if j < len(stratified_transitions_rates_params[2]):
                            abstracted_parameters[nt.id][
                                stratified_transitions_rates_params[2][j]
                            ] = abstracted_parameters[nt.id][bp]
                        # Remove base parameters
                        if bp in abstracted_parameters[nt.id]:
                            del abstracted_parameters[nt.id][bp]
        new_metadata["abstracted_parameters"] = abstracted_parameters
        new_model.petrinet.metadata = new_metadata

        # new_model = GeneratedPetriNetModel(
        #     petrinet=Model(
        #         header=self.petrinet.header,
        #         properties=self.petrinet.properties,
        #         model=Model1(
        #             states=new_states,
        #             transitions=[
        #                 *new_transitions,
        #                 *other_transitions.values(),
        #                 *self_strata_transitions,
        #             ],
        #         ),
        #         semantics=Semantics(
        #             ode=OdeSemantics(
        #                 rates=[
        #                     *new_rates,
        #                     *other_rates.values(),
        #                     *self_strata_rates,
        #                 ],
        #                 initials=new_initials,
        #                 parameters=new_parameters + self_strata_parameters,
        #                 observables=new_observables,
        #                 time=self.petrinet.semantics.ode.time,
        #             ),
        #             typing=self.petrinet.semantics.typing,
        #             span=self.petrinet.semantics.span,
        #         ),
        #         metadata=new_metadata,
        #     )
        # )

        return new_model  # new_rates, transitions, new_transitions # dest_only_rates #original_var, new_vars, new_transitions

    def group_abstract_transitions(self, subbed_transitions):
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
        return grouped_transitions, grouped_rates

    def consolidate_grouped_transitions(self, grouped_transitions):
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
        return consolidated_transitions

    # def transition_probability(self, trans, transformations):
    #     try:
    #         tr_map = self.transition_map(trans)
    #         ambiguous_transitions = tr_map.abstract_to_concrete()
    #         # a2c_outp =  next(iter([outp for inp in trans.input for outp in trans.output
    #         #                             if inp in new_abstract_states and not outp in new_abstract_states and maps_to(inp, outp)]))
    #         a2c_outp = [
    #             tr_map.abstract_input_probability(at.input) for at in ambiguous_transitions
    #         ]
    #     except StopIteration:
    #         a2c_outp = None
    #     return a2c_outp

    def has_common_ancestor(self, s1, s2):
        # Determine if s1 and s2 have a common ancestor in the stratification/abstraction lattice
        if s1 == s2:
            return True
        
        ancestors = self.petrinet.metadata.get("ancestors", [])
        c1 = str(s1)
        c2 = str(s2)
        for level in reversed(ancestors):
            p1 = level.get(c1, c1)
            p2 = level.get(c2, c2)
            if p1 == p2:
                return True
            c1 = p1
            c2 = p2

        return False

    def abstract(self, abstraction: Abstraction):
        # Get existing state variables
        abstraction.base_states = {s.id: s for s in self._state_vars()}
        # Check that there is a state variable or parameter for each key in the state_abstraction
        assert all(
            {
                (k in abstraction.base_states or k in self._parameter_names())
                for k in abstraction.keys()
            }
        ), f"There are unknown states in the state_abstraction keys: {[k for k in abstraction.keys() if not (k in abstraction.base_states or k in self._parameter_names())]}"

        # Check that the state_abstraction maps the keys to a state variable that is not in the abstraction.base_states
        assert not any(
            {
                (v in abstraction.base_states or v in self._parameter_names())
                for v in abstraction.values()
            }
        ), f"There are unknown states in the state_abstraction values: {[v for v in abstraction.values() if (v in abstraction.base_states or v in self._parameter_names()) ]}"

        # Create states for values in state_abstraction
        new_abstract_states = abstraction.abstract_states()
        
        abstraction._state_ancestors = {s: [
            k
            for k in abstraction.base_states
            if k  in abstraction.keys()
        ] for s in new_abstract_states}
        old_untouched_states = [
            v
            for k, v in abstraction.base_states.items()
            if k not in abstraction.keys()
        ]
        new_states = [
            *old_untouched_states,
            *new_abstract_states.values(),
        ]

        new_model = GeneratedPetriNetModel(
            petrinet=Model(
                header=self.petrinet.header,
                properties=self.petrinet.properties,
                model=Model1(states=new_states, transitions=[]),
            )
        )
        new_metadata = copy.deepcopy(self.petrinet.metadata)
        new_model.petrinet.metadata = new_metadata
        model_transformations = new_metadata.get("transformations", [])
        model_transformations.append(abstraction)

        # Replace states in the transitions
        subbed_state_ids = set(abstraction.keys())
        old_untouched_transitions = (
            [  # transitions not involved in abstraction
                t
                for t in self.petrinet.model.transitions
                if not any(
                    [s for s in t.input + t.output if s in subbed_state_ids]
                )
            ]
        )
        new_to_be_abstracted_transitions = [  # transitions with substitutions
            abstraction.abstract_transition(t)
            for t in self.petrinet.model.transitions
            if abstraction.is_transition_abstracted(t)
        ]
        subbed_transitions = (
            old_untouched_transitions + new_to_be_abstracted_transitions
        )
        grouped_transitions, grouped_rates = self.group_abstract_transitions(
            subbed_transitions
        )

        # Convert grouped transitions into a single transition
        consolidated_transitions = self.consolidate_grouped_transitions(
            grouped_transitions
        )

        abstraction._transition_ancestors = {ct.id: [gt.id for gt in gts] for ct, gts in zip(consolidated_transitions, grouped_transitions)  if len(gts)>1}

        ## Remove self transitions
        new_transitions = [
            t
            for t in consolidated_transitions
            if not (t.input == t.output and len(t.input) == 1)
        ]

        new_model.petrinet.model.transitions = new_transitions

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



        def aggregate_rates(
            self, rates, abstraction
            #, abstract_to_concrete_transition
        ):
            expressions = [
                to_sympy(r.expression, self._symbols()) for r in rates
            ]
            starting_expression = reduce(lambda x, y: x + y, expressions)

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

            # invert the abstraction for all vars in expressions
            i_abstraction = {
                k: [
                    v
                    for v in abstraction.keys()
                    if abstraction[v] == k
                    and any([sympy.Symbol(v) in e for e in expression_symbols])
                ]
                for k in abstraction.values()
            }
            # remove empty i_abstractions
            i_abstraction = {
                k: v for k, v in i_abstraction.items() if len(v) > 0
            }

            state_ids = self._state_var_names()

            # abstraction implies that abstract variable is sum of variables mapped to it
            abstraction_substitution = {
                sympy.Symbol(v[0]): to_sympy(
                    f"{k}{'-' if len(v)>1 else ''}{'-'.join(v[1:])}",
                    self._symbols() + list(abstraction.values()),
                )
                for k, v in i_abstraction.items()
                if any([vs in state_ids for vs in v])
            }
            
            # abstraction_substitution = {Symbol(k): [Symbol(v) for v in v_list] for k, v_list in abstraction_substitution.items()}

            parameter_names = self._parameter_names()

            # Symbols that differ across rates and are not part of the abstraction
            unique_symbols = [
                s
                for s in all_symbols
                if s not in common_symbols and str(s) in parameter_names
            ]
            # Group symbols if they have a common ancestor in abstraction/stratification chain
            unique_symbol_groups = []
            for us in unique_symbols:
                match_index = -1
                for i, group in enumerate(unique_symbol_groups):
                    if self.has_common_ancestor(us, group[0]):
                        match_index = i
                        break
                if match_index < 0:
                    unique_symbol_groups.append([us])
                else:
                    unique_symbol_groups[match_index].append(us)


            state_var_names = self._state_var_names()
            parameter_names = self._parameter_names()

            if len(rates) > 1:
                # When have more than one rate that we're aggregating, then we identify parameters that can be aggregated

                # starting_expression = I*S_unvac*beta_vac_unvac_1/N + I*S_vac*beta_vac_unvac_0/N
                #                     = I/N * (S_unvac*beta_vac_unvac_1 + S_vac*beta_vac_unvac_0)
                #                     = I/N * (S_unvac*agg_beta + S_vac*agg_beta)
                #                     = I*agg_beta/N * (S_unvac + S_vac)
                #                     = I*agg_beta*S/N
                #
                #                     = I/N * (S_unvac*beta_vac_unvac_1 + S_vac*beta_vac_unvac_0)
                #                     = I/N * (S_unvac*beta_vac_unvac_1 + (S-S_unvac)*(agg_beta-beta_vac_unvac_1))

                # FIXME need to remove double counting of parameters p_I_I and beta_I_I
                parameter_values = self._parameter_values()
                parameter_minimization = {
                    str(s):  f"agg_{'_'.join([str(us) for us in unique_symbols])}"
                    for unique_symbols in unique_symbol_groups
                    for s in unique_symbols
                    if str(s) in parameter_names  #and not str(s).startswith("p_cross_")
                }
                constant_substitution = {}
                # {
                #   str(s): parameter_values[str(s)] 
                #     for unique_symbols in unique_symbol_groups
                #     for s in unique_symbols
                #     if str(s) in parameter_names  and str(s).startswith("p_cross_")
                # }
                # substitute abstraction into starting expression
                abstract_expression = sympy.expand(
                    starting_expression.subs(
                        {**abstraction_substitution, **parameter_minimization, **constant_substitution}
                    )
                )
            else:
                # When have one rate that we're aggregating, then we need to be told if a parameter is being replaced.  It is assumed to be in the susbstitution.

                # There are no symbols that differ among rates (or there is only one rate)
                abstract_expression = sympy.expand(
                    starting_expression.subs(abstraction_substitution)
                )

                parameter_minimization = {
                    k: str(v)
                    for k, v in abstraction_substitution.items()
                    if str(k) in parameter_names
                }

            abstracted_states = {
                k: str(v)
                for k, v in abstraction_substitution.items()
                if str(k) in state_var_names
            }

            # Need to introduce a new parameter for probability of transition going from abstract input to a concrete output
            transition_parameters = []
            # if abstract_to_concrete_transition is not None:
            #     for atc in abstract_to_concrete_transition:
            #         trans_sym = sympy.Symbol(atc)
            #         abstract_expression *= trans_sym
            #         # parameter_minimization[trans_sym] = str(trans_sym)
            #         transition_parameters.append(trans_sym)

            abstraction._parameter_ancestors = {k: v for k, v in i_abstraction.items() if all([p in parameter_names for p in v])}
            # abstract_expression1 = sympy.expand(sympy.expand(starting_expression.subs(abstraction_substitution )).subs(parameter_minimization))
            return {
                "rate": str(abstract_expression),
                "parameters": parameter_minimization,
                "transition_parameters": transition_parameters,
                "states": abstracted_states,
            }

        aggregated_rates_and_parameters = [
            aggregate_rates(
                self,
                g,
                abstraction,
                # new_model.transition_probability(
                #     consolidated_transitions[i], self.transformations()
                # ),
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

        unchanged_parameters = [
            p
            for p in self.petrinet.semantics.ode.parameters
            if not any(
                [
                    p.id in rp["parameters"]
                    for rp in aggregated_rates_and_parameters
                ]
            )
            and any([p.id in r.expression for r in new_rates])
        ]

        aggregated_parameters = [
            Parameter(
                id=p,
                name=p,
                description=p,
                value=0.0,
                grounding=None,
                distribution=None,
                units=None,
            )
            for p in set(
                [
                    pn
                    for arp in aggregated_rates_and_parameters
                    for k, pn in arp["parameters"].items()
                    if str(k) not in self._state_var_names() 
                ]
            )
        ]

        # When introducing transition probabilities, we need to determine which are part of the same distribution.  Those corresponding to transitions with the same inputs (i.e., are applicable to the same states) must sum to 1.0.
        grouped_transition_parameters = {}
        for trans, arp in zip(
            new_transitions, aggregated_rates_and_parameters
        ):
            if len(arp["transition_parameters"]) > 0:
                related_parameters = grouped_transition_parameters.get(
                    tuple(trans.input), set({})
                )
                related_parameters = related_parameters.union(
                    set(arp["transition_parameters"])
                )
                grouped_transition_parameters[tuple(trans.input)] = (
                    related_parameters
                )

        transition_parameters = [
            Parameter(
                id=str(p),
                name=str(p),
                description=str(p),
                value=1.0 / float(len(param_group)),
                grounding=None,
                distribution=None,
                units=None,
            )
            for param_group in grouped_transition_parameters.values()
            for p in param_group
        ]

        new_parameters = (
            unchanged_parameters
            + aggregated_parameters
            + transition_parameters
        )

        new_initials = [
            # Initial.model_copy(st)
            next(
                i
                for i in self.petrinet.semantics.ode.initials
                if i.target == st.id
            )
            for st in new_states
            if st.id not in new_abstract_states
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
                                    for k, v in abstraction.items()
                                    if (
                                        v == s_id
                                        and k not in self._parameter_names()
                                    )
                                ]
                            ]
                        ],
                        0.0,
                    )
                ),
            )
            for s_id, st in new_abstract_states.items()
            if s_id in new_abstract_states
        ]

        new_model.petrinet.semantics = Semantics(
            ode=OdeSemantics(
                rates=new_rates,  # [*new_rates, *other_rates.values(), *self_strata_rates],
                initials=new_initials,  # new_initials,
                parameters=new_parameters,
                observables=None,  # new_observables,
                time=self.petrinet.semantics.ode.time,
            ),
            typing=self.petrinet.semantics.typing,
            span=self.petrinet.semantics.span,
        )

        param_bounds = self._parameter_bounds()
        param_bounds.update(new_model._parameter_bounds())
        abstraction.set_parameters(
            param_bounds, new_rates, aggregated_rates_and_parameters
        )
        state_strata = new_metadata.get("state_strata", {})
        new_vars_strata = {
            s: abstraction.abstract_strata(s, state_strata)
            for s in new_abstract_states
        }
        for s in abstraction.abstraction:
            if s in state_strata:
                del state_strata[s]

        state_strata.update(new_vars_strata)

        # Remove abstracted parameter bounds from metadata
        # Store bounds on abstracted parameters
        if abstraction.parameters:
            abstracted_parameters = new_metadata.get(
                "abstracted_parameters", {}
            )
            to_remove = []
            for p in abstraction.abstraction.keys():
                for t, ps in abstracted_parameters.items():
                    if p in ps:
                        to_remove.append((t, p))
            for t, p in to_remove:
                del abstracted_parameters[t][p]
            for t, p in to_remove:
                if len(abstracted_parameters[t]) == 0:
                    del abstracted_parameters[t]

            abstracted_parameters.update(abstraction.parameters)
            new_metadata["abstracted_parameters"] = abstracted_parameters

        new_metadata["state_strata"] = state_strata
        ancestors = new_metadata.get("ancestors", [])
        ancestors.append(abstraction._ancestors())
        new_metadata['ancestors'] = ancestors
        # new_metadata["abstraction"] = {
        #     # Need to know which parameter to replace by the min or max value, as well as the min and max value
        #     # parameters -> transition_id -> parameter_id -> [lb,ub]
        #     "parameters": {
        #         new_rates[i].target: {
        #             p:
        #                 {"lb": min(
        #                     [
        #                         next(
        #                             p2
        #                             for p2 in self.petrinet.semantics.ode.parameters
        #                             if p2.id == str(p1)
        #                         ).value
        #                         for p1, v in arp["parameters"].items()
        #                         if v == p
        #                     ]
        #                 ),
        #                 "ub": max(
        #                     [
        #                         next(
        #                             p2
        #                             for p2 in self.petrinet.semantics.ode.parameters
        #                             if p2.id == str(p1)
        #                         ).value
        #                         for p1, v in arp["parameters"].items()
        #                         if v == p
        #                     ]
        #                 ),
        #             }
        #             for p in set(
        #                 [
        #                     p
        #                     for k, p in arp["parameters"].items()
        #                     if str(k) in self._parameter_names()
        #                 ]
        #             )
        #         }
        #         for i, arp in enumerate(aggregated_rates_and_parameters)
        #         if len(arp["parameters"]) > 0
        #     }
        # }
        ## Consolidate bounds on a variable that is shared across multiple transitions
        # bounded_params = {
        #     p
        #     for trans_id, p_bounds in new_metadata["abstraction"][
        #         "parameters"
        #     ].items()
        #     for p in p_bounds
        # }
        # param_min_value = {
        #     p: min(
        #         [
        #             bounds["lb"]
        #             for trans_id, p_bounds in new_metadata["abstraction"][
        #                 "parameters"
        #             ].items()
        #             for p1, bounds in p_bounds.items()
        #             if p1 == p
        #         ]
        #     )
        #     for p in bounded_params
        # }
        # param_max_value = {
        #     p: max(
        #         [
        #             bounds["ub"]
        #             for trans_id, p_bounds in new_metadata["abstraction"][
        #                 "parameters"
        #             ].items()
        #             for p1, bounds in p_bounds.items()
        #             if p1 == p
        #         ]
        #     )
        #     for p in bounded_params
        # }
        # new_metadata["abstraction"]["parameters"] = {
        #     trans_id: {
        #         p1: (
        #             bounds
        #             if p1 not in param_min_value
        #             else {"lb": param_min_value[p1], "ub": param_max_value[p1]}
        #         )
        #         for p1, bounds in p_bounds.items()
        #     }
        #     for trans_id, p_bounds in new_metadata["abstraction"][
        #         "parameters"
        #     ].items()
        # }

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

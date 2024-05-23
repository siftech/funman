"""
This module defines the abstract base classes for the model encoder classes in funman.translate package.
"""

import logging
from abc import ABC
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict
from pysmt.constants import Numeral
from pysmt.formula import FNode
from pysmt.shortcuts import (  # type: ignore
    BOOL,
    GE,
    LE,
    LT,
    REAL,
    TRUE,
    And,
    Div,
    Equals,
    Iff,
    Implies,
    Minus,
    Plus,
    Real,
    Symbol,
    Times,
    get_env,
)
from pysmt.solvers.solver import Model as pysmtModel

from funman.config import FUNMANConfig
from funman.constants import NEG_INFINITY, POS_INFINITY
from funman.model.model import FunmanModel
from funman.model.query import (
    QueryAnd,
    QueryEncoded,
    QueryGE,
    QueryLE,
    QueryTrue,
)
from funman.representation.constraint import (
    Constraint,
    LinearConstraint,
    ModelConstraint,
    ParameterConstraint,
    QueryConstraint,
    StateVariableConstraint,
    TimeseriesConstraint,
)
from funman.translate.simplifier import FUNMANSimplifier
from funman.utils import math_utils
from funman.utils.sympy_utils import (
    FUNMANFormulaManager,
    sympy_to_pysmt,
    to_sympy,
)

from ..representation import Interval, Point, Timepoint, Timestep, TimestepSize
from ..representation.assumption import Assumption
from ..representation.box import Box
from ..representation.parameter import ModelParameter, Schedules
from ..representation.representation import EncodingSchedule
from ..representation.symbol import ModelSymbol
from .encoding import *

l = logging.getLogger(__name__)


StateTimepoints = List[Timepoint]
TransitionTimepoints = List[Timepoint]
TimeStepConstraints = Dict[Tuple[Timestep, TimestepSize], FNode]
TimeStepSubstitutions = Dict[Tuple[Timestep, TimestepSize], Dict[FNode, FNode]]
TimedParameters = Dict[Tuple[Timestep, TimestepSize], List["Parameter"]]


class Encoder:
    pass


EncodedSymbolTimeStampedDict = Dict[str, Dict[str, FNode]]
EncodedSymbolList = List[FNode]
EncodedSymbols = Union[EncodedSymbolList, EncodedSymbolTimeStampedDict]
EncodedFormula = Tuple[FNode, EncodedSymbols]
EncodingLayer = Dict[Constraint, EncodedFormula]


class LayeredEncoding(BaseModel):
    """
    An encoding comprises a formula over a set of symbols.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    _layers: List[EncodingLayer] = []
    _encoder: Encoder

    # @validator("formula")
    # def set_symbols(cls, v: FNode):
    #     cls.symbols = Symbol(v, REAL)

    def construct_encoding(
        self,
        scenario: "AnalysisScenario",
        constraint,
        options: EncodingOptions,
        layers=None,
        box: Box = None,
        assumptions: Optional[List[Assumption]] = None,
    ) -> FNode:
        layers_to_encode = (
            layers if layers is not None else range(self._layers)
        )
        return And(
            [
                self._get_or_create_layer(
                    scenario,
                    constraint,
                    i,
                    options,
                    box=box,
                    assumptions=assumptions,
                )[0]
                for i in layers_to_encode
            ]
        )

    def _get_or_create_layer(
        self,
        scenario,
        constraint,
        layer_idx: int,
        options: EncodingOptions,
        box: Box = None,
        assumptions: List[Assumption] = None,
    ) -> FNode:
        if constraint not in self._layers[layer_idx]:
            layer = self._encoder.encode_constraint(
                scenario,
                constraint,
                options,
                layer_idx,
                assumptions=assumptions,
            )
            self._layers[layer_idx][constraint] = layer
        return self._layers[layer_idx][constraint]

    def assume(self, assumption: List[FNode], layers=None):
        for i, l in enumerate(self._layers):
            (f, s) = l
            self._layers[i] = (
                (Iff(assumption[i], f), s)
                if not layers or i in layers
                else (f, s)
            )

    def substitute(self, substitutions: Dict[FNode, FNode]):
        self._layers = [
            {
                constraint: (l[0].substitute(substitutions), l[1])
                for constraint, l in layer.items()
            }
            for layer in self._layers
        ]

    def simplify(self):
        self._layers = [
            {
                constraint: (l[0].simplify(), l[1])
                for constraint, l in layer.items()
            }
            for layer in self._layers
        ]

    def symbols(self):
        return {
            k: v
            for layer in self._layers
            for constraint, l in layer.items()
            for k, v in l[1].items()
        }


class Encoder(ABC, BaseModel):
    """
    An Encoder translates a Model into an SMTLib formula.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: "FUNMANConfig"
    _timed_model_elements: Dict = None
    _min_time_point: int
    _min_step_size: int
    _untimed_symbols: Set[str] = set([])
    _timed_symbols: Set[str] = set([])
    _untimed_constraints: FNode
    # _assignments: Dict[str, float] = {}
    _env = get_env()
    _env._simplifier = FUNMANSimplifier(_env)
    _constraint_encoder_handler: Dict[Constraint, Callable] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        scenario = kwargs["scenario"]
        env = get_env()
        if not isinstance(env._formula_manager, FUNMANFormulaManager):
            env._formula_manager = FUNMANFormulaManager(env._formula_manager)
            # Before calling substitute, need to replace the formula_manager
            env._substituter.mgr = env.formula_manager

        # Need to initialize pysmt symbols for parameters to help with parsing custom rate equations
        variables = [
            p.name for p in scenario.model._parameters()
        ] + scenario.model._state_var_names()
        variable_symbols = [self._encode_state_var(p) for p in variables]

        self._encode_timed_model_elements(scenario)

        self._constraint_encoder_handler = {
            ModelConstraint: self.encode_model_layer,
            ParameterConstraint: self.encode_parameter,
            StateVariableConstraint: self.encode_query_layer,
            QueryConstraint: self.encode_query_layer,
            LinearConstraint: self.encode_linear_constraint,
            TimeseriesConstraint: self.encode_timeseries,
        }

    def step_size_index(self, step_size: int) -> int:
        return self._timed_model_elements["step_sizes"].index(step_size)

    def substitutions(self, schedule: EncodingSchedule) -> Dict[FNode, FNode]:
        return self._timed_model_elements["time_step_substitutions"][schedule]

    def state_timepoint(
        self, schedule: EncodingSchedule, layer_idx: int
    ) -> int:
        step_size_idx = self.step_size_index(step_size)
        return self._timed_model_elements["state_timepoints"][step_size_idx][
            layer_idx
        ]

    def time_step_constraints(
        self, timepoint: Timepoint, stepsize: TimestepSize
    ):
        return self._timed_model_elements["time_step_constraints"][
            (timepoint, stepsize)
        ]

    def set_time_step_constraints(
        self, layer_idx: int, step_size: TimestepSize, c: FNode
    ):
        self._timed_model_elements["time_step_constraints"][
            (layer_idx, step_size)
        ] = c

    def set_step_substitutions(
        self, schedule: EncodingSchedule, substitutions: Dict[FNode, FNode]
    ):
        self._timed_model_elements["time_step_substitutions"][
            schedule
        ] = substitutions

    def encode_constraint(
        self,
        scenario: "AnalysisScenario",
        constraint: Constraint,
        options: EncodingOptions,
        layer_idx: int = 0,
        assumptions: List[Assumption] = [],
    ) -> EncodedFormula:
        try:
            handler = self._constraint_encoder_handler[type(constraint)]
        except Exception as e:
            raise NotImplementedError(
                f"Could not encode constraint of type {type(constraint)}"
            )
        try:
            encoded_constraint_layer = handler(
                scenario, constraint, layer_idx, options, assumptions
            )
            if constraint.soft:
                encoded_constraint_layer = self.encode_assumed_constraint(
                    encoded_constraint_layer,
                    constraint,
                    assumptions,
                    options,
                    layer_idx=layer_idx,
                )

            return encoded_constraint_layer
        except Exception as e:
            raise e

    def encode_assumed_constraint(
        self,
        encoded_constraint: EncodedFormula,
        constraint: Constraint,
        assumptions: List[Assumption],
        options: EncodingOptions,
        layer_idx: int = 0,
    ) -> EncodedFormula:
        assumption = next(a for a in assumptions if a.constraint == constraint)
        assumption_symbol = self.encode_assumption(assumption, options)

        # The assumption is timed if the constraint has a timed variable
        if constraint.time_dependent():
            timed_assumption_symbol = self.encode_assumption(
                assumption, options, layer_idx=layer_idx
            )

            # Assumption is the same at all timepoints and it is equisatisfiable with the encoded_constraint
            assumed_constraint = And(
                # Iff(assumption_symbol, timed_assumption_symbol),
                Implies(assumption_symbol, timed_assumption_symbol),
                Iff(timed_assumption_symbol, encoded_constraint[0]),
            )
        else:
            assumed_constraint = And(
                Iff(assumption_symbol, encoded_constraint[0]),
            )
        symbols = {k: v for k, v in encoded_constraint[1].items()}
        symbols[assumption_symbol.symbol_name()] = assumption_symbol
        return (assumed_constraint, symbols)

    def encode_assumption(
        self,
        assumption: Assumption,
        options: EncodingOptions,
        layer_idx: int = None,
    ) -> FNode:
        time = (
            options.schedule.time_at_step(layer_idx)
            if layer_idx is not None
            else None
        )

        formula = self._encode_state_var(
            str(assumption), time=time, symbol_type=BOOL
        )
        if time is not None:
            self._timed_symbols.add(str(assumption))
        else:
            self._untimed_symbols.add(str(assumption))
        return formula

    def can_encode():
        """
        Return boolean indicating if the scenario can be encoded with the FUNMANConfig
        """
        return True

    def _symbols(self, vars: List[FNode]) -> Dict[str, Dict[str, FNode]]:
        symbols = {}
        # vars.sort(key=lambda x: x.symbol_name())
        for var in vars.values():
            var_name, timepoint = self._split_symbol(var)
            if timepoint:
                if var_name not in symbols:
                    symbols[var_name] = {}
                symbols[var_name][timepoint] = var
        return symbols

    def initialize_encodings(self, scenario, num_steps):
        encoding = LayeredEncoding()
        encoding._layers = [{} for i in range(num_steps + 1)]
        encoding._encoder = self

        return encoding

    def _encode_simplified(self, model, query, step_size_idx: int):
        formula = And(model, query)
        if self.config.substitute_subformulas:
            sub_formula = formula.substitute(
                self._timed_model_elements["time_step_substitutions"][
                    step_size_idx
                ]
            ).simplify()
            return sub_formula
        else:
            return formula

    def _encode_next_step(
        self, model: FunmanModel, step: int, next_step: int, substitutions={}
    ) -> FNode:
        pass

    def encode_assumptions(
        self, assumptions: List[Assumption], options: EncodingOptions
    ) -> Dict[Assumption, FNode]:
        encoded_assumptions = {
            a: self.encode_assumption(a, options) for a in assumptions
        }
        return encoded_assumptions

    def encode_model_layer(
        self,
        scenario: "AnalysisScenario",
        constraint: ModelConstraint,
        layer_idx: int,
        options: EncodingOptions,
        assumptions: List[Assumption],
    ) -> EncodedFormula:
        if layer_idx == 0:
            return self.encode_init_layer()
        else:
            return self.encode_transition_layer(scenario, layer_idx, options)

    def encode_init_layer(self) -> EncodedFormula:
        initial_state = self._timed_model_elements["init"]
        initial_symbols = initial_state.get_free_variables()

        return (initial_state, {s.symbol_name(): s for s in initial_symbols})

    def encode_transition_layer(
        self,
        scenario: "AnalysisScenario",
        layer_idx: int,
        options: EncodingOptions,
    ) -> EncodedFormula:
        stepsize: TimestepSize = options.schedule.stepsize_at_step(
            layer_idx - 1
        )
        timepoint: Timepoint = options.schedule.time_at_step(layer_idx - 1)
        c = self.time_step_constraints(timepoint, stepsize)

        substitutions = self.substitutions(options.schedule)

        if c is None:
            next_timepoint = options.schedule.time_at_step(layer_idx)
            c, substitutions = self._encode_next_step(
                scenario,
                timepoint,
                next_timepoint,
                substitutions=substitutions,
            )
            self.set_time_step_constraints(timepoint, stepsize, c)
            self.set_step_substitutions(options.schedule, substitutions)

        return (c, {str(s): s for s in c.get_free_variables()})

    def encode_model_timed(
        self, scenario: "AnalysisScenario", num_steps: int, step_size: int
    ) -> Encoding:
        """
        Encode a model into an SMTLib formula.

        Parameters
        ----------
        model : FunmanModel
            model to encode
        num_steps: int
            number of encoding steps (e.g., time steps)
        step_size: int
            size of a step

        Returns
        -------
        Encoding
            formula and symbols for the encoding
        """
        init_layer = self.encode_init_layer()
        layers = [init_layer]

        for i in range(num_steps + 1):
            layer = self.encode_transition_layer(
                scenario, i + 1, step_size=step_size
            )
            layers.append(layer)

        encoding = LayeredEncoding(
            substitutions=self.substitutions,
        )
        encoding._layers = layers
        return encoding

    def _initialize_substitutions(
        self, scenario: "AnalysisScenario", normalization=True
    ):
        # Add parameter assignments
        parameters = scenario.model_parameters()

        # Normalize if constant value
        parameter_assignments = {
            self._encode_state_var(k.name): Real(float(k.interval.lb))
            for k in parameters
            if k.interval.lb == k.interval.ub
        }

        return parameter_assignments

    def parameter_values(
        self, model: FunmanModel, pysmtModel: pysmtModel
    ) -> Dict[str, List[Union[float, None]]]:
        """
        Gather values assigned to model parameters.

        Parameters
        ----------
        model : FunmanModel
            model encoded by self
        pysmtModel : pysmt.solvers.solver.Model
            the assignment to symbols

        Returns
        -------
        Dict[str, List[Union[float, None]]]
            mapping from parameter symbol name to value
        """
        try:
            parameters = {
                parameter.name: pysmtModel[parameter.name]
                for parameter in model._parameters()
            }
            return parameters
        except OverflowError as e:
            l.warning(e)
            return {}

    def _get_timed_symbols(self, model: FunmanModel) -> Set[str]:
        """
        Get the names of the state (i.e., timed) variables of the model.

        Parameters
        ----------
        model : FunmanModel
            The petrinet model

        Returns
        -------
        List[str]
            state variable names
        """
        pass

    def _get_untimed_symbols(self, model: FunmanModel) -> Set[str]:
        untimed_symbols = set(["timestep"])
        # All flux nodes correspond to untimed symbols
        for var_name in model._parameter_names():
            untimed_symbols.add(var_name)
        return untimed_symbols

    def _encode_state_var(
        self, var: str, time: int = None, symbol_type=REAL
    ) -> Symbol:
        timing = f"_{time}" if time is not None else ""
        return Symbol(f"{var}{timing}", symbol_type)

    def _get_structural_configurations(self, scenario: "AnalysisScenario"):
        configurations: List[Dict[str, int]] = []
        structure_parameters = scenario.structure_parameters()
        if len(structure_parameters) == 0:
            self._min_time_point = 0
            self._min_step_size = 1
            num_steps = [1, self.config.num_steps]
            step_size = [1, self.config.step_size]
            max_step_size = self.config.step_size
            max_step_index = self.config.num_steps * max_step_size
            configurations.append(
                {
                    "num_steps": self.config.num_steps,
                    "step_size": self.config.step_size,
                }
            )
        else:
            num_steps = scenario.structure_parameter("num_steps")
            step_size = scenario.structure_parameter("step_size")
            self._min_time_point = int(num_steps.lb)
            self._min_step_size = int(step_size.lb)
            max_step_size = int(step_size.ub)
            max_step_index = int(num_steps.ub) * int(max_step_size)
            configurations += [
                {"num_steps": ns, "step_size": ss}
                for ns in range(int(num_steps.lb), int(num_steps.ub) + 1)
                for ss in range(int(step_size.lb), int(step_size.ub) + 1)
            ]
        return configurations, max_step_index, max_step_size

    def _define_init_term(
        self,
        scenario: "AnalysisScenario",
        var: str,
        init_time: int,
        substitutions=None,
    ):
        value = scenario.model._get_init_value(var, scenario, self.config)

        init_term = None
        substitution = ()

        if isinstance(value, FNode):
            value_expr = to_sympy(value, scenario.model._symbols())
            if self.config.substitute_subformulas and substitutions:
                value_expr = (
                    sympy_to_pysmt(value_expr)
                    .substitute(substitutions)
                    .simplify()
                )
                # value_expr = FUNMANSimplifier.sympy_simplify(
                #             value_expr,
                #             parameters=scenario.model_parameters(),
                #             substitutions=substitutions,
                #             threshold=self.config.series_approximation_threshold,
                #             taylor_series_order=self.config.taylor_series_order,
                #         )
            else:
                value_expr = sympy_to_pysmt(value_expr)

            substitution = (
                self._encode_state_var(var, time=init_time),
                value_expr,
            )
            init_term = Equals(*substitution)
            return init_term, substitution
        elif isinstance(value, list):
            substitution = None
            init_term = And(
                GE(
                    self._encode_state_var(var, time=init_time),
                    Real(value[0]),
                ),
                LT(
                    self._encode_state_var(var, time=init_time),
                    Real(value[1]),
                ),
            )
            return init_term, substitution
        else:
            return TRUE(), None

    def _define_init(
        self, scenario: "AnalysisScenario", init_time: int = 0
    ) -> FNode:
        # Generate Parameter symbols and assignments
        substitutions = self._initialize_substitutions(scenario)
        initial_state = And([Equals(k, v) for k, v in substitutions.items()])

        state_var_names = scenario.model._state_var_names()

        time_var = scenario.model._time_var()
        if time_var is not None:
            time_var_name = scenario.model._time_var_id(time_var)
            time_symbol = self._encode_state_var(
                time_var_name, time=0
            )  # Needed so that there is a pysmt symbol for 't'

            substitutions[time_symbol] = Real(0.0)
            time_var_init = Equals(time_symbol, Real(0.0))
        else:
            time_var_init = TRUE()

        initial_state_vars_and_subs = [
            self._define_init_term(
                scenario, var, init_time, substitutions=substitutions
            )
            for var in state_var_names
        ]

        substitutions = {
            **substitutions,
            **{
                sv[1][0]: sv[1][1]
                for sv in initial_state_vars_and_subs
                if sv[1]
            },
        }
        initial_state = And(
            And([sv[0] for sv in initial_state_vars_and_subs]),
            time_var_init,
            initial_state,
        )

        substitutions = self._propagate_substitutions(substitutions)

        return initial_state, substitutions

    def _propagate_substitutions(self, substitutions):
        change = True
        while change:
            next_subs = {}
            for var in substitutions:
                next_subs[var] = (
                    substitutions[var].substitute(substitutions).simplify()
                )
                change = change or next_subs[var] != substitutions[var]
            substitutions = next_subs
            change = False
        return substitutions

    def encode_parameter(
        self,
        scenario: "AnalysisScenario",
        constraint: ModelConstraint,
        layer_idx: int,
        options: EncodingOptions,
        assumptions: List[Assumption],
    ) -> Optional[EncodedFormula]:
        parameter = constraint.parameter
        if parameter in scenario.model_parameters():
            formula = self.interval_to_smt(
                parameter._escaped_name,
                parameter.interval.model_copy(deep=True),
                # closed_upper_bound=False,
                infinity_constraints=False,
            )
            return (
                formula,
                {s.symbol_name(): s for s in formula.get_free_variables()},
            )
        else:
            return None

    def encode_timeseries(
        self,
        scenario: "AnalysisScenario",
        constraint: TimeseriesConstraint,
        layer_idx: int,
        options: EncodingOptions,
        assumptions: List[Assumption],
    ) -> Optional[EncodedFormula]:
        timepoint = constraint.timeseries["time"][layer_idx]
        formulas = []
        for sv in constraint.timeseries.columns:
            if sv == "time":
                continue

            sv_formula = Equals(
                self._encode_state_var(sv, time=int(timepoint)),
                Real(constraint.timeseries[sv][layer_idx]),
            )
            formulas.append(sv_formula)
        formula = And(formulas)
        return (formula, {str(s): s for s in formula.get_free_variables()})

    def encode_linear_constraint(
        self,
        scenario: "AnalysisScenario",
        constraint: LinearConstraint,
        layer_idx: int,
        options: EncodingOptions,
        assumptions: List[Assumption],
    ) -> Optional[EncodedFormula]:
        bounds: Interval = constraint.additive_bounds
        vars: List[str] = constraint.variables
        weights: List[int | float] = constraint.weights
        timestep = options.schedule.time_at_step(layer_idx)
        parameters = scenario.parameters
        encoded_vars = [
            (
                self._encode_state_var(v)
                if len([p for p in parameters if p.name == v]) > 0
                else self._encode_state_var(v, time=timestep)
            )
            for v in vars
        ]
        if constraint.derivative:
            next_timestep = constraint.next_timestep(
                layer_idx, options.schedule
            )
            encoded_dt_vars = [
                (
                    self._encode_state_var(v)
                    if len([p for p in parameters if p.name == v]) > 0
                    else self._encode_state_var(
                        v, time=options.schedule.time_at_step(next_timestep)
                    )
                )
                for v in vars
            ]
            curr = (
                encoded_dt_vars if next_timestep < layer_idx else encoded_vars
            )
            curr_t = (
                options.schedule.time_at_step(next_timestep)
                if next_timestep < layer_idx
                else timestep
            )
            nxt = (
                encoded_dt_vars if next_timestep > layer_idx else encoded_vars
            )
            nxt_t = (
                options.schedule.time_at_step(next_timestep)
                if next_timestep > layer_idx
                else timestep
            )
            expression = Div(
                Minus(
                    Plus(
                        [
                            Times(
                                Real(weights[i]),
                                nxt[i],
                            )
                            for i in range(len(vars))
                        ]
                    ),
                    Plus(
                        [
                            Times(
                                Real(weights[i]),
                                curr[i],
                            )
                            for i in range(len(vars))
                        ]
                    ),
                ),
                Minus(Real(nxt_t), Real(curr_t)),
            ).simplify()
            pass
        else:
            expression = Plus(
                [
                    Times(
                        Real(weights[i]),
                        encoded_vars[i],
                    )
                    for i in range(len(vars))
                ]
            )

        if bounds.lb != NEG_INFINITY:
            lb = Real(bounds.lb)
        else:
            lb = None

        if bounds.ub != POS_INFINITY:
            ub = Real(bounds.ub)
        else:
            ub = None

        if options.normalize:
            expression = Div(expression, Real(options.normalization_constant))
            if lb is not None:
                lb = Div(lb, Real(options.normalization_constant))
            if ub is not None:
                ub = Div(ub, Real(options.normalization_constant))

        if lb is not None:
            lbe = GE(expression, lb)
        else:
            lbe = TRUE()
        if ub is not None:
            ube = (
                LE(expression, ub)
                if constraint.additive_bounds.closed_upper_bound
                else LT(expression, ub)
            )
        else:
            ube = TRUE()
        formula = And(lbe, ube).simplify()
        return (formula, {str(s): s for s in formula.get_free_variables()})

    def _encode_untimed_constraints(
        self, scenario: "AnalysisScenario"
    ) -> FNode:
        untimed_constraints = []
        parameters = [
            p
            for p in scenario.model._parameters()
            if p not in scenario.parameters
        ] + scenario.parameters

        # Create bounds on parameters, but not necessarily synthesize the parameters
        untimed_constraints.append(
            self.box_to_smt(
                Box(
                    bounds={
                        p.name: p.interval.model_copy()
                        for p in parameters
                        if isinstance(p, ModelParameter)
                    },
                    closed_upper_bound=True,
                )
            )
        )

        return And(untimed_constraints).simplify()

    def _encode_timed_model_elements(self, scenario: "AnalysisScenario"):
        model = scenario.model
        self._timed_symbols = self._get_timed_symbols(model)
        self._untimed_symbols = self._get_untimed_symbols(model)
        step_sizes = scenario.structure_parameter("step_size")
        num_steps = scenario.structure_parameter("num_steps")
        schedules = scenario.structure_parameter("schedules")

        state_timepoints: StateTimepoints = []
        transition_timepoints: TransitionTimepoints = []
        time_step_constraints: TimeStepConstraints = {}
        time_step_substitutions: TimeStepSubstitutions = {}
        timed_parameters: TimedParameters = {}
        initial_state, initial_substitutions = self._define_init(scenario)
        if step_sizes and num_steps:
            schedules: List[EncodingSchedule] = []
            for i, step_size in enumerate(
                range(
                    int(step_sizes.interval.lb),
                    int(step_sizes.interval.ub) + 1,
                )
            ):
                s_timepoints, t_timepoints = self._get_timepoints(
                    num_steps.interval.ub, step_size
                )
                schedule = EncodingSchedule(timepoints=s_timepoints)
                schedules.append(schedule)
            schedules = Schedules(schedules=schedules)

            # state_timepoints.append(s_timepoints)
            # transition_timepoints.append(t_timepoints)

            # (
            #     configurations,
            #     max_step_index,
            #     max_step_size,
            # ) = self._get_structural_configurations(scenario)
            # step_sizes = list(
            #     range(int(step_sizes.lb), int(step_sizes.ub + 1))
            # )
            # time_step_constraints = [
            #     [None for i in range(max_step_size)]
            #     for j in range(max_step_index)
            # ]
            # time_step_substitutions = [
            #     initial_substitutions.copy() for i in range(max_step_size)
            # ]
            # timed_parameters = [
            #     [None for i in range(max_step_size)]
            #     for j in range(max_step_index)
            # ]
            # schedules = [
            #     list(
            #         range(
            #             0,
            #         )
            #     )
            #     for step_size in step_sizes
            # ]
        elif schedules:
            pass
        else:
            raise Exception(
                "Cannot translate scenario without either a step_list, or step_sizes and num_steps structural parameters"
            )

        for schedule in schedules.schedules:
            state_timepoints.append(schedule.timepoints)
            transition_timepoints.append(
                schedule.timepoints[:-1]
            )  # Last timepoint is a state with no transition
            time_step_constraints.update(
                {
                    (t1, t2 - t1): None
                    for s1, t1 in enumerate(schedule.timepoints)
                    for s2, t2 in enumerate(schedule.timepoints)
                    if s1 + 1 == s2
                }
            )
            time_step_substitutions[schedule] = initial_substitutions.copy()
            timed_parameters.update(
                {
                    (t1, t2 - t1): None
                    for s1, t1 in enumerate(schedule.timepoints)
                    for s2, t2 in enumerate(schedule.timepoints)
                    if s1 + 1 == s2
                }
            )

        # configurations = None
        # max_step_index = None
        # max_step_size = None
        # step_sizes = None

        self._timed_model_elements = {
            "schedules": schedules,
            "state_timepoints": state_timepoints,
            "transition_timepoints": transition_timepoints,
            "init": initial_state,
            "time_step_constraints": time_step_constraints,
            "time_step_substitutions": time_step_substitutions,
            # "configurations": configurations,
            "untimed_constraints": self._encode_untimed_constraints(scenario),
            "timed_parameters": timed_parameters,
        }

    def _get_timepoints(
        self, num_steps: int, step_size: int
    ) -> Tuple[List[int], List[int]]:
        state_timepoints = range(
            0,
            int(step_size) * int(num_steps) + 1,
            int(step_size),
        )

        if len(list(state_timepoints)) == 0:
            raise Exception(
                f"Could not identify timepoints from step_size = {step_size} and num_steps = {num_steps}"
            )

        transition_timepoints = range(
            0, int(step_size) * int(num_steps), int(step_size)
        )
        return list(state_timepoints), list(transition_timepoints)

    def encode_query_layer(
        self,
        scenario: "AnalysisScenario",
        constraint: Constraint,
        layer_idx: int,
        options: EncodingOptions,
        assumptions: List[Assumption],
    ):
        """
        Encode a query into an SMTLib formula.

        Parameters
        ----------
        model : FunmanModel
            model to encode

        Returns
        -------

            formula and symbols for the encoding
        """
        query = (
            constraint.query
            if isinstance(constraint, QueryConstraint)
            else constraint
        )

        query_handlers = {
            QueryAnd: self._encode_query_and,
            QueryLE: self._encode_query_le,
            QueryGE: self._encode_query_ge,
            QueryTrue: self._encode_query_true,
            QueryEncoded: self._return_encoded_query,
            StateVariableConstraint: self._encode_state_variable_constraint,
        }

        if type(query) in query_handlers:
            layer = query_handlers[type(query)](
                scenario, query, layer_idx, options, assumptions
            )
            return layer
            # encoded_query.substitute(substitutions)
            # encoded_query.simplify()
            # return encoded_query
        else:
            raise NotImplementedError(
                f"Do not know how to encode query of type {type(query)}"
            )

    def _return_encoded_query(
        self,
        scenario: "AnalysisScenario",
        query: QueryEncoded,
        layer_idx: int,
        options: EncodingOptions,
    ):
        return (
            query._formula,
            {str(v): v for v in query._formula.get_free_variables()},
        )

    def _query_variable_name(self, query):
        return (
            query.variable
            if not isinstance(query.variable, ModelSymbol)
            else str(query.variable)
        )

    def _encode_query_and(
        self,
        scenario: "AnalysisScenario",
        query: QueryAnd,
        layer_idx: int,
        options: EncodingOptions,
        assumptions: List[Assumption],
    ):
        queries = [
            self.encode_query_layer(
                scenario, q, layer_idx, options, assumptions
            )
            for q in query.queries
        ]

        layer = (
            And([q[0] for q in queries]),
            {str(s): s for q in queries for k, s in q[1].items()},
        )

        return layer

    def _normalize(self, scenario: "AnalysisScenario", value):
        return sympy_to_pysmt(
            to_sympy(
                Div(value, Real(scenario.normalization_constant)),
                scenario.model._symbols(),
            )
        )

    def _encode_state_variable_constraint(
        self,
        scenario: "AnalysisScenario",
        query: StateVariableConstraint,
        layer_idx: int,
        options: EncodingOptions,
        assumptions: List[Assumption],
    ):
        time = options.schedule.time_at_step(layer_idx)

        if query.contains_time(time):
            bounds = (
                query.interval.normalize(options.normalization_constant)
                if options.normalize
                else query.interval
            )
            symbol = self._encode_state_var(var=query.variable, time=time)

            norm = (
                scenario.normalization_constant if options.normalize else 1.0
            )

            lb = (
                math_utils.div(query.interval.lb, norm)
                if query.interval.lb != NEG_INFINITY
                else query.interval.lb
            )
            ub = (
                math_utils.div(query.interval.ub, norm)
                if query.interval.ub != POS_INFINITY
                else query.interval.ub
            )

            formula = self.interval_to_smt(
                query.variable,
                Interval(
                    lb=lb,
                    ub=ub,
                    closed_upper_bound=query.interval.closed_upper_bound,
                ),
                time=time,
                # closed_upper_bound=query.interval.closed_upper_bound,
                infinity_constraints=False,
            )
        else:
            formula = TRUE()

        return (
            formula,
            {v.symbol_name(): v for v in formula.get_free_variables()},
        )

    def _encode_query_le(
        self,
        scenario: "AnalysisScenario",
        query: QueryLE,
        layer_idx: int,
        options: EncodingOptions,
        assumptions: List[Assumption],
    ):
        time = options.schedule.time_at_step(layer_idx)
        if options.normalize:
            ub = Div(Real(query.ub), Real(scenario.normalization_constant))
        else:
            ub = Real(query.ub)
        q = LE(
            self._encode_state_var(var=query.variable, time=time),
            ub,
        )

        return (q, {str(v): v for v in q.get_free_variables()})

    def _encode_query_ge(
        self,
        scenario: "AnalysisScenario",
        query: QueryGE,
        layer_idx: int,
        options: EncodingOptions,
        assumptions: List[Assumption],
    ):
        time = options.schedule.time_at_step(layer_idx)
        if options.normalize:
            lb = Div(Real(query.lb), Real(scenario.normalization_constant))
        else:
            lb = Real(query.lb)
        q = GE(
            self._encode_state_var(var=query.variable, time=time),
            lb,
        )
        return (q, {str(v): v for v in q.get_free_variables()})

    def _encode_query_true(
        self,
        scenario: "AnalysisScenario",
        query: QueryTrue,
        layer_idx: int,
        options: EncodingOptions,
        assumptions: List[Assumption],
    ):
        return (TRUE(), {})

    def symbol_timeseries(
        self, model_encoding, pysmtModel: pysmtModel
    ) -> Dict[str, List[Union[float, None]]]:
        """
        Generate a symbol (str) to timeseries (list) of values

        Parameters
        ----------
        pysmtModel : pysmt.solvers.solver.Model
            variable assignment
        """
        series = self.symbol_values(model_encoding, pysmtModel)
        a_series = {}  # timeseries as array/list
        max_t = max(
            [
                max([int(k) for k in tps.keys() if k.isdigit()] + [0])
                for _, tps in series.items()
            ]
        )
        a_series["index"] = list(range(0, max_t + 1))
        for var, tps in series.items():
            vals = [None] * (int(max_t) + 1)
            for t, v in tps.items():
                if t.isdigit():
                    vals[int(t)] = v
            a_series[var] = vals
        return a_series

    def symbol_values(
        self, model_encoding: Encoding, pysmtModel: pysmtModel
    ) -> Dict[str, Dict[str, float]]:
        """
         Get the value assigned to each symbol in the pysmtModel.

        Parameters
        ----------
        model_encoding : Encoding
            encoding using the symbols
        pysmtModel : pysmt.solvers.solver.Model
            assignment to symbols

        Returns
        -------
        Dict[str, Dict[str, float]]
            mapping from symbol and timepoint to value
        """

        vars = self._symbols(model_encoding.symbols())
        vals = {}
        for var in vars:
            vals[var] = {}
            for t in vars[var]:
                try:
                    symbol = vars[var][t]
                    value = pysmtModel.get_py_value(symbol)
                    if isinstance(value, Numeral):
                        value = 0.0
                    vals[var][t] = float(value)
                except OverflowError as e:
                    l.warning(e)
        return vals

    def interval_to_smt(
        self,
        p: str,
        i: Interval,
        time: int = None,
        # closed_upper_bound: bool = False,
        infinity_constraints=False,
    ) -> FNode:
        """
        Convert the interval into contraints on parameter p.

        Parameters
        ----------
        p : funman.representation.parameter.Parameter
            parameter to constrain
        closed_upper_bound : bool, optional
            interpret interval as closed (i.e., p <= ub), by default False

        Returns
        -------
        FNode
            formula constraining p to the interval
        """

        symbol = self._encode_state_var(p, time=time)
        if i.lb == i.ub and i.lb != NEG_INFINITY and i.lb != POS_INFINITY:
            return Equals(symbol, Real(i.lb))
        else:
            lower = (
                GE(symbol, Real(i.lb))
                if i.lb != NEG_INFINITY or infinity_constraints
                else TRUE()
            )
            upper_ineq = LE if i.closed_upper_bound else LT
            upper = (
                upper_ineq(symbol, Real(i.ub))
                if i.ub != POS_INFINITY or infinity_constraints
                else TRUE()
            )
            return And(
                lower,
                upper,
            ).simplify()

    def point_to_smt(self, pt: Point):
        return And(
            [
                Equals(Symbol(p, REAL), Real(value))
                for p, value in pt.values.items()
            ]
        )

    def box_to_smt(
        self,
        box: Box,
        closed_upper_bound: bool = False,
        infinity_constraints=False,
    ):
        """
        Compile the interval for each parameter into SMT constraints on the corresponding parameter.

        Parameters
        ----------
        closed_upper_bound : bool, optional
            use closed upper bounds for each interval, by default False

        Returns
        -------
        FNode
            formula representing the box as a conjunction of interval constraints.
        """
        return And(
            [
                self.interval_to_smt(
                    p,
                    interval,
                    # closed_upper_bound=closed_upper_bound,
                    infinity_constraints=infinity_constraints,
                )
                for p, interval in box.bounds.items()
            ]
        )

    def _split_symbol(self, symbol: FNode) -> Tuple[str, str]:
        if symbol.symbol_name() in self._untimed_symbols:
            return symbol.symbol_name(), None
        else:
            s, t = symbol.symbol_name().rsplit("_", 1)
            if s not in self._timed_symbols or not t.isdigit():
                raise Exception(
                    f"Cannot determine if symbol {symbol} is timed."
                )
            return s, t


class DefaultEncoder(Encoder):
    """
    The DefaultEncoder will not actually encode a model as SMT.  It is used to provide an Encoder for SimulatorModel objects, but the encoder will not be used.
    """

    pass

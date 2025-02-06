import logging
import threading
from abc import ABC, abstractclassmethod, abstractmethod
from decimal import Decimal
from typing import Dict, List, Optional, Union

import numpy as np
import sympy
from pydantic import BaseModel, ConfigDict
from pysmt.shortcuts import TRUE, And, Solver
from pysmt.logics import QF_NRA

from funman import (
    NEG_INFINITY,
    POS_INFINITY,
    Assumption,
    BilayerModel,
    Box,
    DecapodeModel,
    EncodedModel,
    GeneratedPetriNetModel,
    GeneratedRegnetModel,
    Interval,
    ModelConstraint,
    ModelParameter,
    Parameter,
    ParameterConstraint,
    PetrinetModel,
    QueryAnd,
    QueryConstraint,
    QueryEncoded,
    QueryFunction,
    QueryGE,
    QueryLE,
    QueryTrue,
    StructureParameter,
)
from funman.constants import LABEL_TRUE, LABEL_UNKNOWN
from funman.model.ensemble import EnsembleModel
from funman.model.petrinet import GeneratedPetriNetModel
from funman.model.regnet import GeneratedRegnetModel, RegnetModel
from funman.representation import Point
from funman.representation.constraint import (
    FunmanUserConstraint,
    TimeseriesConstraint,
)
from funman.representation.parameter import NumSteps, Schedules, StepSize
from funman.search.simulate import Simulator, Timeseries
from funman.translate.translate import EncodingOptions
from funman.utils import math_utils
from funman.utils.sympy_utils import replace_reserved, to_sympy

from ..representation import Point, Timepoint

l = logging.getLogger(__name__)


class AnalysisScenario(ABC, BaseModel):
    """
    Abstract class for Analysis Scenarios.
    """

    parameters: List[Union[Parameter, ModelParameter, StructureParameter]]
    normalization_constant: Optional[float] = None
    constraints: Optional[List[Union[FunmanUserConstraint]]] = None
    """True if its okay when the volume of the search space is empty (e.g., when it is a point)"""
    empty_volume_ok: bool = False
    model_config = ConfigDict(extra="forbid")
    init_time: Timepoint = 0.0

    model: Union[
        GeneratedPetriNetModel,
        GeneratedRegnetModel,
        RegnetModel,
        PetrinetModel,
        DecapodeModel,
        BilayerModel,
        EncodedModel,
        EnsembleModel,
    ]
    query: Union[
        QueryAnd, QueryGE, QueryLE, QueryEncoded, QueryFunction, QueryTrue
    ] = QueryTrue()
    _assumptions: List[Assumption] = []
    _smt_encoder: Optional["Encoder"] = None
    # Encoding for different step sizes (key)
    _encodings: Optional[Dict["Schedule", "Encoding"]] = {}
    _original_parameter_widths: Dict[str, Decimal] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # create default constraints
        if self.constraints is None:
            self.constraints = [
                ModelConstraint(name="model_dynamics", model=self.model)
            ]

        if not any(
            c for c in self.constraints if isinstance(c, ModelConstraint)
        ):
            self.constraints.append(
                ModelConstraint(name="model_dynamics", model=self.model)
            )

        # create assumptions for each constraint that may be assumed.
        if self.constraints is not None:
            for constraint in self.constraints:
                if constraint.soft:
                    self._assumptions.append(Assumption(constraint=constraint))

    @abstractclassmethod
    def get_kind(cls) -> str:
        pass

    @abstractmethod
    def solve(
        self, config: "FUNMANConfig", haltEvent: Optional[threading.Event]
    ):
        pass

    @abstractmethod
    def get_search(config: "FUNMANConfig") -> "Search":
        pass

    def initialize(self, config: "FUNMANConfig") -> "Search":
        search = self.get_search(config)
        self._process_parameters()

        self.constraints += [
            ParameterConstraint(name=parameter.name, parameter=parameter)
            for parameter in self.model_parameters()
        ]

        self._set_normalization(config)
        if config.normalize:
            self._normalize_parameters()

        if config.use_compartmental_constraints:
            capacity = self.normalization_constant
            # 1.0 if config.normalize else self.normalization_constant
            ccs = self.model.compartmental_constraints(
                capacity, config.compartmental_constraint_noise
            )
            if ccs is not None:
                self.constraints += ccs
                for cc in ccs:
                    if cc.soft:
                        self._assumptions.append(Assumption(constraint=cc))

        self._initialize_encodings(config)

        self._original_parameter_widths = {
            p.name: p.interval.original_width for p in self.model_parameters()
        }

        return search

    def parameter_map(self) -> Dict[str, Parameter]:
        return {p.name: p for p in self.parameters}

    def escaped_parameter_map(self) -> Dict[str, str]:
        rmap = {}
        for p in self.parameters:
            if hasattr(p, "_escaped_name"):
                rmap[p._escaped_name] = p.name
            else:
                rmap[p.name] = p.name
        return rmap

    def _initialize_encodings(self, config: "FUNMANConfig"):
        # self._assume_model = Symbol("assume_model")
        self._smt_encoder = self.model.default_encoder(config, self)
        assert self._smt_encoder._timed_model_elements

        # Initialize Assumptions
        # Maintain backward support for query as a single constraint
        if self.query is not None and not isinstance(self.query, QueryTrue):
            query_constraint = QueryConstraint(
                name="query", query=self.query, timepoints=Interval(lb=0)
            )
            self.constraints += [query_constraint]
            self._assumptions.append(Assumption(constraint=query_constraint))

        for schedule in self._smt_encoder._timed_model_elements[
            "schedules"
        ].schedules:
            encoding = self._smt_encoder.initialize_encodings(
                self, len(schedule.timepoints)
            )
            self._encodings[schedule] = encoding

    def num_dimensions(self):
        """
        Return the number of parameters (dimensions) that are synthesized.  A parameter is synthesized if it has a domain with width greater than zero and it is either labeled as LABEL_ALL or is a structural parameter (which are LABEL_ALL by default).
        """
        return len(self.parameters)

    def num_timepoints(self) -> int:
        schedules = self.parameters_of_type(Schedules)
        if len(schedules) == 1:
            num_timepoints = sum(
                len(schedule.timepoints) - 1
                for schedule in schedules[0].schedules
            )
        else:
            # use num_steps and step_size
            num_steps = self.parameters_of_type(NumSteps)
            step_size = self.parameters_of_type(StepSize)
            num_timepoints = (num_steps[0].width() + 1) * (
                step_size[0].width() + 1
            )
        return num_timepoints

    def search_space_volume(self, normalize: bool = False) -> Decimal:
        bounds = {}
        for param in self.model_parameters():
            bounds[param.name] = param.interval
        space_box = Box(bounds=bounds)

        if len(bounds) > 0:
            # Normalized volume for a timeslice is 1.0, but compute anyway to verify
            space_time_slice_volume = (
                space_box.volume(normalize=self._original_parameter_widths)
                if normalize
                else space_box.volume()
            )
        else:
            space_time_slice_volume = Decimal(1.0)
        assert (
            self.empty_volume_ok
            or not normalize
            or abs(space_time_slice_volume - Decimal(1.0)) <= Decimal(1e-8)
        ), f"Normalized space volume is not 1.0, computed = {space_time_slice_volume}"
        space_volume = (
            space_time_slice_volume
            if normalize
            else self.num_timepoints() * space_time_slice_volume
        )
        return space_volume

    def representable_space_volume(self) -> Decimal:
        bounds = {}
        for param in self.model_parameters():
            bounds[param.name] = Interval(lb=NEG_INFINITY, ub=POS_INFINITY)
        space_box = Box(bounds=bounds)
        space_time_slice_volume = space_box.volume()
        space_volume = self.num_timepoints() * space_time_slice_volume
        return space_volume

    def structure_parameters(self):
        return self.parameters_of_type(StructureParameter)

    def model_parameters(self):
        return self.parameters_of_type(ModelParameter)

    def synthesized_model_parameters(self):
        return [p for p in self.model_parameters() if p.is_synthesized()]

    def synthesized_parameters(self):
        return [p for p in self.parameters if p.is_synthesized()]

    def parameters_of_type(self, parameter_type) -> List[Parameter]:
        return [p for p in self.parameters if isinstance(p, parameter_type)]

    def structure_parameter(self, name: str) -> StructureParameter:
        try:
            return next(p for p in self.parameters if p.name == name)
        except StopIteration:
            return None

    def _process_parameters(self):
        if len(self.structure_parameters()) == 0:
            # either undeclared or wrong type
            # if wrong type, recover structure parameters
            self.parameters = [
                (
                    NumSteps(name=p.name, interval=Interval(lb=p.lb, ub=p.ub))
                    if (p.name == "num_steps")
                    else p
                )
                for p in self.parameters
            ] + [
                (
                    StepSize(name=p.name, interval=Interval(lb=p.lb, ub=p.ub))
                    if (p.name == "step_size")
                    else p
                )
                for p in self.parameters
            ]
            if len(self.structure_parameters()) == 0:
                # Add the structure parameters if still missing
                self.parameters += [
                    NumSteps(name="num_steps", interval=Interval(lb=0, ub=0)),
                    StepSize(name="step_size", interval=Interval(lb=1, ub=1)),
                ]

        self._extract_non_overriden_parameters()
        self._filter_parameters()

    def _extract_non_overriden_parameters(self):
        from funman.constants import LABEL_ANY

        # If a model has parameters that are not overridden by the scenario, then add them to the scenario
        model_parameters = self.model._parameter_names()
        model_parameter_values = self.model._parameter_values()
        model_parameters = [] if model_parameters is None else model_parameters
        non_overriden_parameters = []
        for p in [
            param
            for param in model_parameters
            if param
            not in [
                overridden_param.name for overridden_param in self.parameters
            ]
        ]:
            bounds = {}
            lb = self.model._parameter_lb(p)
            ub = self.model._parameter_ub(p)
            if ub is not None and lb is not None:
                bounds["ub"] = ub
                bounds["lb"] = lb
            elif model_parameter_values[p]:
                value = model_parameter_values[p]
                bounds["lb"] = bounds["ub"] = value
            else:
                bounds = {}
            non_overriden_parameters.append(
                ModelParameter(
                    name=p, interval=Interval(**bounds), label=LABEL_ANY
                )
            )

        self.parameters += non_overriden_parameters

    def _filter_parameters(self):
        # If the scenario has parameters that are not in the model, then remove them from the scenario
        model_parameters = self.model._parameter_names()

        if model_parameters is not None:
            filtered_parameters = [
                p
                for p in self.parameters
                if p.name in model_parameters
                or isinstance(p, StructureParameter)
            ]
            self.parameters = filtered_parameters

    def _normalize_parameters(self):
        for p in self.parameters:
            if p.name == "N":
                p.normalize_bounds(self.normalization_constant)

    def _set_normalization(self, config):
        if config.normalization_constant is not None:
            self.normalization_constant = config.normalization_constant
        elif config.normalize:
            self.normalization_constant = (
                self.model.calculate_normalization_constant(self, config)
            )
        else:
            self.normalization_constant = 1.0
            l.warning("Warning: The scenario is not normalized!")

    def run_scenario_simulation(
        self, init, parameters, tvect
    ) -> Optional[Timeseries]:
        simulator = Simulator(
            model=self.model, init=init, parameters=parameters, tvect=tvect
        )
        timeseries = simulator.sim()
        return timeseries

    def run_point_simulation(
        self, point: Point, tvect
    ) -> Optional[Timeseries]:
        init = {
            var: value
            for var, value in point.values_at(
                point.schedule.timepoints[0], self.model
            ).items()
            if var != "timer_t"
        }

        parameters = {
            p: point.value_of(p) for p in self.model._parameter_names()
        }
        simulator = Simulator(
            model=self.model, init=init, parameters=parameters, tvect=tvect
        )
        timeseries = simulator.sim()

        if timeseries:
            observable_timeseries = self.compute_observables(
                timeseries, parameters
            )
            for k, v in observable_timeseries.items():
                timeseries.data.append(v)
                timeseries.columns.append(k)
            
        # timeseries = np.array([[tvect[t], timeseries[t]] for t in range(len(tvect))])
        return timeseries

    def simulation_tvects(self, config) -> List[Union[float, int]]:
        num_steps = self.structure_parameter("num_steps")
        step_size = self.structure_parameter("step_size")
        schedules = self.structure_parameter("schedules")

        tvects = []
        if schedules:
            for s in schedules.schedules:
                tvects.append(s.timepoints)
        else:
            min_steps = num_steps.interval.lb
            max_steps = num_steps.interval.ub
            min_size = step_size.interval.lb
            max_size = step_size.interval.ub
            for ss in range(int(min_size), int(max_size) + 1):
                tvects.append(np.arange(0, int(max_steps * ss) + 1, int(ss)))

        return tvects

    def compute_observables(self, timeseries, parameters):
        observables = self.model.observables()
        timepoints = timeseries.data[0]
        data = {}
        unreseved_parameters = {
            replace_reserved(k): v for k, v in parameters.items()
        }

        if observables:
            for o in observables:
                o_name = self.model._observable_name(o)
                # o_fn = o.expression
                o_fn = self.model.observable_expression(o_name)
                # Evaluate o_fn for each time in timeseries
                if self.model.is_timed_observable(o_name):
                    values = []
                    for ti, t in enumerate(timepoints):
                        # state_at_t = [timeseries.data[ci][ti] for ci, c in enumerate(timeseries.columns)]
                        state_at_t = {
                            c: timeseries.data[ci][ti]
                            for ci, c in enumerate(timeseries.columns)
                            if c != "time"
                        }
                        value = o_fn[2].evalf(
                            subs={**state_at_t, **parameters}
                        )
                        values.append(float(value))
                    data[o_name] = values
                else:
                    value = o_fn[2].evalf(subs={**unreseved_parameters})
                    data[o_name] = float(value)
        return data

    def simulate_scenario(self, config: "FUNMANConfig") -> Point:

        init = {
            var: float(
                self.model._get_init_value(var, self, config).constant_value()
            )
            for var in self.model._state_var_names()
        }
        parameters = {
            p: pm.interval.lb
            for p in self.model._parameter_names()
            for pm in self.parameters
            if pm.name == p
        }
        # parameters = {
        #     p.name: p.interval.lb
        #     for p in self.parameters
        #     }
        # timestamped_variables ={f"{var}_{tp}": float(self.model._get_init_value(var, self, config).constant_value()) for var in self.model._state_var_names()}
        schedule = self.structure_parameter("schedules").schedules[0]
        timepoints = schedule.timepoints

        timeseries = self.run_scenario_simulation(init, parameters, timepoints)

        observable_timeseries = self.compute_observables(
            timeseries, parameters
        )
        for k, v in observable_timeseries.items():
            timeseries.data.append(v)
            timeseries.columns.append(k)

        values = {
            **{
                f"{var}_{str(timepoint)}": timeseries.data[var_idx + 1][
                    timestep
                ]
                for var_idx, var in enumerate(timeseries.columns[1:])
                for timestep, timepoint in enumerate(timeseries.data[0])
                if isinstance(timeseries.data[var_idx + 1], list)
            },
            **{
                var: timeseries.data[var_idx + 1]
                for var_idx, var in enumerate(timeseries.columns[1:])
                for timestep, timepoint in enumerate(timeseries.data[0])
                if not isinstance(timeseries.data[var_idx + 1], list)
            },
            **parameters,
            **{"timestep": len(timepoints) - 1},
        }
        point = Point(
            values=values,
            label=LABEL_TRUE,
            schedule=schedule,
            simulation=timeseries,
        )

        # with Solver() as solver:
        #         sim_encoding = self.encode_timeseries_verification(
        #             point, timeseries
        #         )
        #         solver.add_assertion(sim_encoding)
        #         result = solver.solve()
        #         if result:
        #             l.info("simulation passed verification")
        #         else:
        #             l.info("simulation failed verification")
        #             return False

        return point

    def check_simulation(
        self, config: "FUNMANConfig", results: "AnalysisScenarioResult"
    ):
        # Check solution with simulation
        sim_results = []

        points = results.parameter_space.points()
        # [Point(label=LABEL_UNKNOWN, values={**{p.name: p.interval.lb for p in self.parameters}, **{f"{var}_0": float(self.model._get_init_value(var, self, config).constant_value()) for var in self.model._state_var_names()},**{f"{var}_{tp}": float(self.model._get_init_value(var, self, config).constant_value()) for var in self.model._state_var_names()}})] if results is None else

        for point in points:
            timeseries = self.run_point_simulation(
                point, point.relevant_timepoints(self.model)
            ) if not point.simulation else point.simulation
            sim_results.append((point, timeseries))
            point.simulation = timeseries

        for point, timeseries in sim_results:
            if timeseries is None:
                l.warning(
                    f"Skipping point validation because there is no timeseries ..."
                )
                continue

            if config.solver == "dreal":
                opts = {
                    "dreal_precision": config.dreal_precision,
                    "dreal_log_level": config.dreal_log_level,
                    "dreal_mcts": config.dreal_mcts,
                    "preferred": (
                        config.dreal_prefer_parameters
                        if config.dreal_prefer_parameters
                        else [p.name for p in self.parameters]
                    ),
                    "random_seed": config.random_seed,
                }
            else:
                opts = {}

            with Solver(
                name=config.solver,
                logic=QF_NRA,
                solver_options=opts,
            ) as solver:
                sim_encoding = self.encode_timeseries_verification(
                    point, timeseries
                )
                solver.add_assertion(sim_encoding)
                result = solver.solve()
                if result:
                    l.info("simulation passed verification")
                else:
                    l.info("simulation failed verification")
                    return False

        return True

    def encode_timeseries_verification(
        self, point: Point, timeseries: Timeseries
    ):
        # Get constraints needed to check timeseries
        ts_constraint = TimeseriesConstraint(
            name="simulation", timeseries=timeseries
        )
        timeseries_constraints = [
            TimeseriesConstraint(name="simulation", timeseries=timeseries)
        ] + [c for c in self.constraints if not isinstance(c, ModelConstraint)]

        encoded_constraints = []
        timepoints = timeseries["time"]
        encoding = self._smt_encoder.initialize_encodings(
            self, len(point.schedule.timepoints)
        )
        for c in timeseries_constraints:
            for timestep, timepoint in enumerate(timepoints):
                if c.encodable() and c.relevant_at_time(timepoint):
                    encoded_constraints.append(
                        encoding.construct_encoding(
                            self,
                            c,
                            EncodingOptions(schedule=point.schedule),
                            layers=[timestep],
                            assumptions=self._assumptions,
                        )
                    )
        formula = And(encoded_constraints)

        return formula


class AnalysisScenarioResult(ABC):
    """
    Abstract class for AnalysisScenario result data.
    """

    @abstractmethod
    def plot(self, **kwargs):
        pass


class AnalysisScenarioResultException(BaseModel, AnalysisScenarioResult):
    exception: str

    def plot(self, **kwargs):
        raise NotImplemented(
            "AnalysisScenarioResultException cannot be plotted with plot()"
        )

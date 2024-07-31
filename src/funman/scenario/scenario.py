import logging
import threading
from abc import ABC, abstractclassmethod, abstractmethod
from decimal import Decimal
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

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
from funman.model.ensemble import EnsembleModel
from funman.model.petrinet import GeneratedPetriNetModel
from funman.model.regnet import GeneratedRegnetModel, RegnetModel
from funman.representation.constraint import FunmanUserConstraint
from funman.representation.parameter import NumSteps, Schedules, StepSize
from funman.utils import math_utils

l = logging.getLogger(__name__)


class AnalysisScenario(ABC, BaseModel):
    """
    Abstract class for Analysis Scenarios.
    """

    parameters: List[Parameter]
    normalization_constant: Optional[float] = None
    constraints: Optional[List[Union[FunmanUserConstraint]]] = None
    """True if its okay when the volume of the search space is empty (e.g., when it is a point)"""
    empty_volume_ok: bool = False
    model_config = ConfigDict(extra="forbid")

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
            assert (
                0 in schedule.timepoints
            ), "Schedule for encoding does not include a timepoint 0"
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

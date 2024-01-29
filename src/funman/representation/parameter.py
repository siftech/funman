import copy
import logging
from decimal import Decimal
from typing import List, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pysmt.fnode import FNode
from pysmt.shortcuts import REAL, Symbol

from funman import LABEL_ALL, LABEL_ANY

from .encoding_schedule import EncodingSchedule
from .interval import Interval
from .symbol import ModelSymbol

l: logging.Logger = logging.getLogger(__name__)


class Parameter(BaseModel):
    name: Union[str, ModelSymbol]
    interval: Interval = Interval()

    def width(self) -> Decimal:
        return self.interval.width()

    def is_unbound(self) -> bool:
        return self.interval.is_unbound()

    def __hash__(self):
        return abs(hash(self.name))

    @model_validator(mode="after")
    def set_interval_original_width(self) -> str:
        self.interval.original_width = self.interval.width()
        return self


class LabeledParameter(Parameter):
    label: Literal["any", "all"] = LABEL_ANY

    def is_synthesized(self) -> bool:
        return self.label == LABEL_ALL and self.width() > 0.0


class StructureParameter(LabeledParameter):
    def is_synthesized(self):
        return True


class NumSteps(StructureParameter):
    @field_validator("name")
    @classmethod
    def check_name(cls, name: str, info: ValidationInfo):
        assert name == "num_steps", "NumSteps.name must be 'num_steps'"
        return name


class StepSize(StructureParameter):
    @field_validator("name")
    @classmethod
    def check_name(cls, name: str, info: ValidationInfo):
        assert name == "step_size", "StepSize.name must be 'step_size'"
        return name


StepListValue = List[Union[float, int]]


class Schedules(StructureParameter):
    schedules: List[EncodingSchedule]

    @field_validator("name")
    @classmethod
    def check_name(cls, name: str, info: ValidationInfo):
        assert name == "schedules", "StepList.name must be 'step_list'"
        return name

    @model_validator(mode="before")
    def check_empty_name(self) -> str:
        if ("name" not in self) or (self["name"] is None):
            self["name"] = "schedules"

        return self


class ModelParameter(LabeledParameter):
    """
    A parameter is a free variable for a Model.  It has the following attributes:

    * lb: lower bound

    * ub: upper bound

    * symbol: a pysmt FNode corresponding to the parameter variable

    """

    model_config = ConfigDict(extra="forbid")

    _symbol: FNode = None

    def symbol(self):
        """
        Get a pysmt Symbol for the parameter

        Returns
        -------
        pysmt.fnode.FNode
            _description_
        """
        if not self._symbol:
            self._symbol = Symbol(self.name, REAL)
        return self._symbol

    def timed_copy(self, timepoint: int):
        """
        Create a time-stamped copy of a parameter.  E.g., beta becomes beta_t for a timepoint t

        Parameters
        ----------
        timepoint : int
            Integer timepoint

        Returns
        -------
        Parameter
            A time-stamped copy of self.
        """
        timed_parameter = copy.deepcopy(self)
        timed_parameter.name = f"{timed_parameter.name}_{timepoint}"
        return timed_parameter

    def __eq__(self, other):
        if not isinstance(other, ModelParameter):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.name == other.name

    def __hash__(self):
        # necessary for instances to behave sanely in dicts and sets.
        return hash(self.name)

    def __repr__(self) -> str:
        return f"{self.name}[{self.interval.lb}, {self.interval.ub})"

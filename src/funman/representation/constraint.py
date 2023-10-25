from typing import List, Optional, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    WrapValidator,
    field_validator,
)
from typing_extensions import Annotated

from funman.model import Model
from funman.model.query import Query

from .interval import Interval
from .parameter import ModelParameter, StructureParameter


class Constraint(BaseModel):
    soft: bool = True
    name: str

    # model_config = ConfigDict(extra="forbid")

    def __hash__(self) -> int:
        return 1

    def encodable(self) -> bool:
        return True

    def relevant_at_time(self, time: int) -> bool:
        return True


class TimedConstraint(Constraint):
    timepoints: Optional["Interval"] = None

    def contains_time(self, time: Union[float, int]) -> bool:
        return self.timepoints is None or self.timepoints.contains_value(time)

    def relevant_at_time(self, time: int) -> bool:
        return self.contains_time(time)


class ModelConstraint(Constraint):
    soft: bool = False
    model: Model

    model_config = ConfigDict(extra="forbid")

    def __hash__(self) -> int:
        return 2


class ParameterConstraint(Constraint):
    soft: bool = False
    parameter: Union[ModelParameter, StructureParameter]

    model_config = ConfigDict(extra="forbid")

    def __hash__(self) -> int:
        return 1

    def encodable(self) -> bool:
        return not isinstance(self.parameter, StructureParameter)

    def relevant_at_time(self, time: int) -> bool:
        return time == 0


class QueryConstraint(Constraint):
    soft: bool = True
    query: Query

    model_config = ConfigDict(extra="forbid")

    def __hash__(self) -> int:
        return 4


class StateVariableConstraint(TimedConstraint):
    variable: str
    interval: "Interval" = None

    model_config = ConfigDict(extra="forbid")

    def __hash__(self) -> int:
        return 3


class LinearConstraint(TimedConstraint):
    soft: bool = True
    additive_bounds: "Interval"
    variables: List[str]
    weights: Annotated[
        Optional[List[Union[int, float]]], Field(validate_default=True)
    ] = None
    closed_upper_bound: bool = False

    model_config = ConfigDict(extra="forbid")

    @field_validator("weights")
    @classmethod
    def check_weights(
        cls, weights: Optional[List[Union[int, float]]], info: ValidationInfo
    ):
        assert (
            "variables" in info.data
        ), "LinearConstraint must have a list of variables."

        if weights is None:
            weights = [1.0] * len(info.data["variables"])
        return weights

    def __hash__(self) -> int:
        return 4

    def contains_time(self, time: Union[float, int]) -> bool:
        return self.timepoints is None or self.timepoints.contains_value(time)

    def relevant_at_time(self, time: int) -> bool:
        return self.contains_time(time)


FunmanConstraint = Union[
    ModelConstraint,
    ParameterConstraint,
    StateVariableConstraint,
    LinearConstraint,
    QueryConstraint,
]

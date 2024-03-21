from typing import Union

from pydantic import BaseModel

from . import Timepoint
from .constraint import FunmanConstraint


class Assumption(BaseModel):
    constraint: Union[FunmanConstraint]

    def relevant_at_time(self, timepoint: Timepoint) -> bool:
        return self.constraint.relevant_at_time(timepoint)

    def __str__(self) -> str:
        assert (
            self.constraint._escaped_name is not None
        ), f"Assumption {self.name} does not have an '_escaped_name'."
        if hasattr(self.constraint, "_escaped_name"):
            return f"assume_{self.constraint._escaped_name}"
        else:
            return f"assume_{str(self.constraint)}"

    def __hash__(self) -> int:
        return hash(self.constraint)

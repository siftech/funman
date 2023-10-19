from typing import Union

from pydantic import BaseModel

from . import Timepoint
from .constraint import FunmanConstraint


class Assumption(BaseModel):
    constraint: Union[FunmanConstraint]

    def relevant_at_time(self, timepoint: Timepoint) -> bool:
        return self.constraint.relevant_at_time(timepoint)

    def __str__(self) -> str:
        if hasattr(self.constraint, "name"):
            return f"assume_{self.constraint.name}"
        else:
            return f"assume_{str(self.constraint)}"

    def __hash__(self) -> int:
        return hash(self.constraint)

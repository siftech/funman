import logging
from typing import List, Union

from pydantic import BaseModel

from . import Timepoint, TimestepSize

l = logging.getLogger(__name__)


class EncodingSchedule(BaseModel):
    timepoints: List[Timepoint]

    def time_at_step(self, step: int) -> Timepoint:
        return self.timepoints[step]

    def stepsize_at_step(self, step: int) -> TimestepSize:
        if step < len(self.timepoints) - 1:
            return self.time_at_step(step + 1) - self.time_at_step(step)
        elif step == len(self.timepoints) - 1:
            return 0
        else:
            raise Exception(
                f"Step {step} is not in timepoints (|timepoints|={len(timepoints)})"
            )
        return self.timepoints[step]

    def __hash__(self):
        return int(sum(self.timepoints))

    def __eq__(self, other) -> bool:
        if isinstance(other, EncodingSchedule):
            return self.timepoints == other.timepoints
        return False

    def __len__(self) -> int:
        return 1

    @staticmethod
    def from_steps(
        num_steps: int, step_size: Union[int, float]
    ) -> "EncodingSchedule":
        timepoints = list(range(0, num_steps, step_size))
        return EncodingSchedule(timepoints=timepoints)

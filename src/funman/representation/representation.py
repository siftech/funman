"""
This submodule contains definitions for the classes used
during the configuration and execution of a search.
"""

import logging
import math
from typing import Dict, List, Literal, Optional, Set

from pydantic import BaseModel

from funman import to_sympy
from funman.constants import LABEL_UNKNOWN, NEG_INFINITY, POS_INFINITY, Label

from . import Timepoint

l = logging.getLogger(__name__)


from . import EncodingSchedule, PointValue


class Timeseries(BaseModel):
    data: List[List[float]]
    columns: List[str]

    def __getitem__(self, key):
        return self.data[self.columns.index(key)]


class Point(BaseModel):
    type: Literal["point"] = "point"
    label: Label = LABEL_UNKNOWN
    values: Dict[str, PointValue]
    normalized_values: Optional[Dict[str, float]] = None
    schedule: Optional[EncodingSchedule] = None

    # def __init__(self, **kw) -> None:
    #     super().__init__(**kw)
    #     self.values = kw['values']

    def timestep(self) -> int:
        return int(self.values["timestep"]) if "timestep" in self.values else 0

    def __str__(self):
        return f"Point({self.model_dump()})"

    def __repr__(self) -> str:
        return str(self.model_dump())

    def values_at(self, tp: Timepoint) -> Dict[str, float]:
        v = {
            k.rsplit("_", 1)[0]: v
            for k, v in self.values.items()
            if self._is_state_variable(k) and int(k.rsplit("_", 1)[-1]) == tp
        }
        return v

    def value_of(self, var) -> float:
        return self.values[var] if var in self.values else None

    def relevant_timesteps(self) -> Set[int]:
        steps = {
            int(k.rsplit("_", 1)[-1])
            for k, v in self.values.items()
            if k.startswith("solve_step") and v == 1.0
        }
        return steps

    def _is_state_variable(self, s: str) -> bool:
        return (
            not s.startswith("solve_step")
            and not s.startswith("assume_")
            and "_" in s
        )

    def state_values(self) -> Dict[str, float]:
        return {
            k: v for k, v in self.values.items() if self._is_state_variable(k)
        }

    def relevant_timepoints(self) -> List[int]:
        steps = list(
            {
                int(k.rsplit("_", 1)[-1])
                for k, v in self.values.items()
                if self._is_state_variable(k)
            }
        )
        steps.sort()
        return steps

    def remove_irrelevant_steps(self, untimed_symbols: Set[str]):
        relevant = self.relevant_timesteps()
        relevant_timepoints = [
            self.schedule.time_at_step(ts) for ts in relevant
        ]
        self.values = {
            k: v
            for step in relevant_timepoints
            for k, v in self.values.items()
            if (k.endswith(f"_{step}") or k in untimed_symbols)
        }

    def denormalize(self, scenario):
        if scenario.normalization_constant:
            norm = to_sympy(
                scenario.normalization_constant, scenario.model._symbols()
            )
            denormalized_values = {
                k: (v * norm if scenario.model._is_normalized(k) else v)
                for k, v in self.values.items()
            }
            denormalized_point = Point(
                label=self.label,
                values=denormalized_values,
                normalized_values=self.values,
                type=self.type,
                schedule=self.schedule,
            )
            return denormalized_point
        else:
            return self

    def __hash__(self):
        my_hash = sum(
            [
                v
                for _, v in self.values.items()
                if not isinstance(v, EncodingSchedule)
                and v != POS_INFINITY
                and v != NEG_INFINITY
            ]
        )

        return int(my_hash) if not math.isinf(my_hash) else 0

    def __eq__(self, other):
        if isinstance(other, Point):
            return all(
                [
                    p in other.values and self.values[p] == other.values[p]
                    for p in self.values.keys()
                ]
            )
        else:
            return False

"""
This submodule contains definitions for the classes used
during the configuration and execution of a search.
"""
import logging
import math
import sys
from typing import Dict, Literal, Optional

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pydantic import BaseModel, Field

import funman.utils.math_utils as math_utils
from funman import to_sympy
from funman.constants import LABEL_UNKNOWN, Label
from funman.representation import Timestep

l = logging.getLogger(__name__)


from . import EncodingSchedule, PointValue


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
        return int(self.values["timestep"])

    def __str__(self):
        return f"Point({self.model_dump()})"

    def __repr__(self) -> str:
        return str(self.model_dump())

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
            )
            return denormalized_point
        else:
            return self

    def __hash__(self):
        return int(
            sum(
                [
                    v
                    for _, v in self.values.items()
                    if not isinstance(v, EncodingSchedule)
                    and v != sys.float_info.max
                    and not math.isinf(v)
                ]
            )
        )

    def __eq__(self, other):
        if isinstance(other, Point):
            return all(
                [self.values[p] == other.values[p] for p in self.values.keys()]
            )
        else:
            return False

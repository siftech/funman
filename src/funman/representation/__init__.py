"""
Classes for representing analysis elements, such as parameter, intervals, boxes, and parmaeter spaces.
"""

from typing import Union, Optional

Timepoint = Union[int, float]
Timestep = int
TimestepSize = Union[int, float]

from .encoding_schedule import EncodingSchedule

PointValue = Optional[Union[float, int, EncodingSchedule]]
from .parameter import *
from .assumption import *
from .constraint import *
from .explanation import *
from .representation import *
from .box import *
from .symbol import *
from .parameter_space import *

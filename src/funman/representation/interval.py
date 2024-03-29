import logging
from decimal import Decimal
from typing import List, Optional, Union

from numpy import average, nextafter
from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    field_serializer,
    model_validator,
)
from pydantic.functional_serializers import WrapSerializer
from typing_extensions import Annotated

import funman.utils.math_utils as math_utils
from funman.constants import NEG_INFINITY, POS_INFINITY

l = logging.Logger(__name__)


class Interval(BaseModel):
    """
    An interval is a pair [lb, ub) that is open (i.e., an interval specifies all points x where lb <= x and ub < x).
    """

    lb: Optional[Union[float, str]] = NEG_INFINITY
    ub: Optional[Union[float, str]] = POS_INFINITY
    closed_upper_bound: bool = False
    original_width: Optional[Decimal] = None
    normalized: bool = False
    unnormalized_lb: Optional[Union[float, str]] = None
    unnormalized_ub: Optional[Union[float, str]] = None

    @field_serializer("original_width")
    def ser_wrap(self, v: Decimal, _info) -> float:
        return float(v)

    def normalize_bounds(self, normalization_constant):
        assert (
            not self.normalized
        ), f"Cannot normalize an interval twice: {self}"

        return Interval(
            lb=math_utils.div(self.lb, normalization_constant),
            ub=math_utils.div(self.ub, normalization_constant),
            normalized=True,
            unnormalized_lb=self.lb,
            unnormalized_ub=self.ub,
            original_width=self.original_width
            / Decimal(normalization_constant),
            closed_upper_bound=self.closed_upper_bound,
        )

    @staticmethod
    def from_value(v: Union[float, str]):
        return Interval(lb=v, ub=v, closed_upper_bound=True)

    def __hash__(self):
        return int(math_utils.plus(self.lb, self.ub))

    def disjoint(self, other: "Interval") -> bool:
        """
        Is self disjoint (non overlapping) with other?

        Parameters
        ----------
        other : Interval
            other interval

        Returns
        -------
        bool
            are the intervals disjoint?
        """
        return math_utils.lte(self.ub, other.lb) or math_utils.gte(
            self.lb, other.ub
        )

    def is_point(self) -> bool:
        return self.lb == self.ub and self.closed_upper_bound

    def normalized_width(self) -> Decimal:
        assert (
            self.original_width is not None
        ), f"{self} cannot be normalized without knowing the original width."
        return (
            self.width() / self.original_width
            if self.original_width > 0.0
            else 0.0
        )

    def width(self, normalize=False) -> Decimal:
        """
        The width of an interval is ub - lb.

        Returns
        -------
        float
            ub - lb
        """
        if normalize:
            return self.normalized_width()
        else:
            return Decimal(self.ub) - Decimal(self.lb)

    def is_unbound(self) -> bool:
        return self.lb == NEG_INFINITY and self.ub == POS_INFINITY

    def normalize(self, normalization_factor: Union[float, int]) -> "Interval":
        return Interval(
            lb=math_utils.div(self.lb, normalization_factor),
            ub=math_utils.div(self.lb, normalization_factor),
        )

    def __lt__(self, other):
        if isinstance(other, Interval):
            return self.width() < other.width()
        else:
            raise Exception(
                f"Cannot compare __lt__() Interval to {type(other)}"
            )

    def __eq__(self, other):
        if isinstance(other, Interval):
            return self.lb == other.lb and self.ub == other.ub
        else:
            return False

    def __repr__(self):
        return str(self.model_dump())

    def __str__(self):
        ub = "]" if self.closed_upper_bound else ")"
        return f"[{self.lb:.5f}, {self.ub:.5f}{ub}"

    def meets(self, other: "Interval") -> bool:
        """
        Does self meet other?

        Parameters
        ----------
        other : Interval
            another inteval

        Returns
        -------
        bool
            Does self meet other?
        """
        return (
            (self.ub == other.lb and not self.closed_upper_bound)
            or (
                nextafter(self.ub, POS_INFINITY) == other.lb
                and self.closed_upper_bound
            )
            or (self.lb == other.ub and not other.closed_upper_bound)
            or (
                self.ub == nextafter(other.lb, POS_INFINITY)
                and other.closed_upper_bound
            )
        )

    def finite(self) -> bool:
        """
        Are the lower and upper bounds finite?

        Returns
        -------
        bool
            bounds are finite
        """
        return self.lb != NEG_INFINITY and self.ub != POS_INFINITY

    def contains(self, other: "Interval") -> bool:
        """
        Does self contain other interval?

        Parameters
        ----------
        other : Interval
            interval to check for containment

        Returns
        -------
        bool
            self contains other
        """
        lhs = (
            (self.lb == NEG_INFINITY or other.lb != NEG_INFINITY)
            if (self.lb == NEG_INFINITY or other.lb == NEG_INFINITY)
            else self.lb <= other.lb
        )
        rhs = (
            (other.ub != POS_INFINITY or self.ub == POS_INFINITY)
            if (self.ub == POS_INFINITY or other.ub == POS_INFINITY)
            else other.ub <= self.ub
        )
        return lhs and rhs

    def intersects(self, other: "Interval") -> bool:
        return (
            self.contains_value(other.lb)
            or other.contains_value(self.lb)
            or (
                self.contains_value(other.ub)
                and math_utils.gt(other.ub, self.lb)
            )
            or (
                other.contains_value(self.ub)
                and math_utils.gt(self.ub, other.lb)
            )
        )

    def intersection(self, b: "Interval") -> Optional["Interval"]:
        """
        Given an interval b with self = [a0,a1] and b=[b0,b1], check whether they intersect.  If they do, return interval with their intersection.

        Parameters
        ----------
        b : Interval
            interval to check for intersection

        Returns
        -------
        Interval
            intersection of self and b
        """
        if self == b:
            return self
        else:
            if self.lb == b.lb:
                if math_utils.lt(self.ub, b.ub):
                    minArray = self
                    maxArray = b
                else:
                    minArray = b
                    maxArray = self
            elif math_utils.lt(self.lb, b.lb):
                minArray = self
                maxArray = b
            else:
                minArray = b
                maxArray = self
            if math_utils.gt(
                minArray.ub, maxArray.lb
            ):  ## has nonempty intersection. return intersection
                result = Interval()
                result.lb = float(maxArray.lb)
                result.ub = float(minArray.ub)
                return result
            else:  ## no intersection.
                return None

    def subtract(self, b: "Interval") -> "Interval":
        """
        Given 2 intervals self = [a0,a1] and b=[b0,b1], return the part of self that does not intersect with b.

        Parameters
        ----------
        b : Interval
            interval to subtract from self

        Returns
        -------
        Interval
            self - b
        """

        if math_utils.lt(self.lb, b.lb):
            return Interval(lb=self.lb, ub=b.lb)

        if math_utils.gt(self.lb, b.lb):
            return Interval(lb=b.ub, ub=self.ub)

        return None

    def midpoint(self, points: List[List["Point"]] = None):
        """
        Compute the midpoint of the interval.

        Parameters
        ----------
        points : List[Point], optional
            if specified, compute midpoint as average of points in the interval, by default None

        Returns
        -------
        float
            midpoint
        """

        if points:
            # Find mean of groups and mid point bisects the means
            means = [float(average(grp)) for grp in points]
            mid = float(average(means))
            return mid

        if self.lb == NEG_INFINITY and self.ub == POS_INFINITY:
            return 0
        elif self.lb == NEG_INFINITY:
            return self.ub - BIG_NUMBER
        if self.ub == POS_INFINITY:
            return self.lb + BIG_NUMBER
        else:
            return ((self.ub - self.lb) / 2) + self.lb

    def union(self, other: "Interval") -> List["Interval"]:
        """
        Union other interval with self.

        Returns
        -------
        List[Interval]
            union of intervals
        """
        if self == other:  ## intervals are equal, so return original interval
            ans = self
            # total_height = [self.width()]
            return [ans]
        else:  ## intervals are not the same. start by identifying the lower and higher intervals.
            if self.lb == other.lb:
                if math_utils.lt(self.ub, other.lb):
                    minInterval = self
                    maxInterval = other
                else:  ## other.ub > self.ub
                    minInterval = other
                    maxInterval = self
            elif math_utils.lt(self.lb, other.lb):
                minInterval = self
                maxInterval = other
            else:
                minInterval = other
                maxInterval = self
        if math_utils.gte(
            minInterval.ub, maxInterval.lb
        ):  ## intervals intersect.
            ans = Interval(lb=minInterval.lb, ub=maxInterval.ub)
            # total_height = ans.width()
            return [ans]
        elif math_utils.lt(
            minInterval.ub, maxInterval.lb
        ):  ## intervals are disjoint.
            ans = [minInterval, maxInterval]
            # total_height = [
            #     math_utils.plus(minInterval.width(), maxInterval.width())
            # ]
            return ans

    def contains_value(
        self, value: float, denormalize_bounds: bool = False
    ) -> bool:
        """
        Does the interval include a value?

        Parameters
        ----------
        value : float
            value to check for containment
        denormalize_bounds : bool
            if true, and self has unnormalized_lb and unormalized_ub, use these insead of lb and ub.

        Returns
        -------
        bool
            the value is in the interval
        """
        lb = (
            self.lb
            if not self.unnormalized_lb or not denormalize_bounds
            else self.unnormalized_lb
        )
        ub = (
            self.ub
            if not self.unnormalized_ub or not denormalize_bounds
            else self.unnormalized_ub
        )

        lb_sat = math_utils.gte(value, lb)
        ub_sat = (
            math_utils.lte(value, ub)
            if self.closed_upper_bound
            else math_utils.lt(value, ub)
        )
        return lb_sat and ub_sat

    def _denormalize(self):
        self.lb = self.lb if not self.unnormalized_lb else self.unnormalized_lb
        self.unnormalized_lb = None
        self.ub = self.ub if not self.unnormalized_ub else self.unnormalized_ub
        self.unnormalized_ub = None

    @model_validator(mode="after")
    def check_interval(self) -> str:
        if self.lb is None:
            self.lb = NEG_INFINITY
        if self.ub is None:
            self.ub = POS_INFINITY

        # Assume that intervals where lb == ub imply closed_upper_bound
        if self.lb == self.ub and not self.closed_upper_bound:
            l.warning(
                f"{self} has equal lower and upper bounds, so assuming the upper bound is closed.  (I.e., [lb, ub) is actually [lb, ub])"
            )
            self.closed_upper_bound = True
        if self.original_width is None:
            self.original_width = self.width()

        return self

import copy
import logging
from decimal import ROUND_CEILING, Decimal
from functools import reduce
from math import log2
from pickle import FALSE
from statistics import mean as average
from typing import Dict, List, Literal, Optional, Union

from numpy import nextafter
from pydantic import BaseModel, Field

import funman.utils.math_utils as math_utils
from funman.constants import LABEL_FALSE, LABEL_TRUE, LABEL_UNKNOWN, Label

from . import EncodingSchedule, Interval, Point, Timestep
from .explanation import BoxExplanation
from .interval import Interval
from .parameter import ModelParameter, Parameter
from .symbol import ModelSymbol

l = logging.getLogger(__name__)


# @total_ordering
class Box(BaseModel):
    """
    A Box maps n parameters to intervals, representing an n-dimensional connected open subset of R^n.
    """

    type: Literal["box"] = "box"
    label: Label = LABEL_UNKNOWN
    bounds: Dict[str, Interval] = {}
    explanation: Optional[BoxExplanation] = None
    schedule: Optional[EncodingSchedule] = None
    corner_points: List[Point] = []
    points: List[Point] = []
    _points_at_step: Dict[Timestep, List[Point]] = {}
    _prioritize_entropy: bool = False

    @staticmethod
    def from_point(
        point: Point,
        parameters: Dict[str, Parameter] = None,
        radius: float = None,
        radius_vars=None,
    ) -> "Box":
        box = Box()
        if radius is not None and radius_vars is not None:
            assert (
                radius > 0
            ), "Cannot create a bounding box around a point using a non-positive radius."
            box.bounds = {
                p: (
                    Interval(lb=v - radius, ub=v + radius)
                    if p in radius_vars
                    and (parameters is None or p in parameters)
                    else Interval.from_value(v)
                )
                for p, v in point.values.items()
            }

        else:
            box.bounds = {
                p: Interval.from_value(v)
                for p, v in point.values.items()
                if parameters is None or p in parameters
            }
        if "timestep" in point.values:
            box.bounds["timestep"] = Interval.from_value(
                point.values["timestep"]
            )
        box.points.append(point)
        box.schedule = point.schedule
        box.label = point.label
        return box

    def add_point(self, point: Point) -> None:
        if point not in self.points:
            timestep = point.timestep()
            step_points = self._points_at_step.get(timestep, [])
            step_points.append(point)
            self._points_at_step[timestep] = step_points
            self.points.append(point)

    def true_points(self, step=None) -> List[Point]:
        return [
            p
            for p in self.points
            if p.label == LABEL_TRUE and (step is None or p.timestep() == step)
        ]

    def false_points(self, step=None) -> List[Point]:
        return [
            p
            for p in self.points
            if p.label == LABEL_FALSE
            and (step is None or p.timestep() == step)
        ]

    def explain(self) -> "BoxExplanation":
        expl = {"box": {k: v.model_dump() for k, v in self.bounds.items()}}
        expl.update(self.explanation.explain())
        return expl

    def timestep(self) -> Interval:
        return (
            self.bounds["timestep"]
            if "timestep" in self.bounds
            else Interval(lb=0, ub=0, closed_upper_bound=True)
        )

    def __hash__(self):
        return int(sum([i.__hash__() for _, i in self.bounds.items()]))

    def advance(self):
        # Advancing a box means that we move the time step forward until it exhausts the possible number of steps
        if self.timestep().lb == self.timestep().ub:
            return None
        else:
            box: Box = self.model_copy(deep=True)
            box.timestep().lb += 1
            # Remove points because they correspond to prior timesteps
            box.points = [
                pt
                for pt in self.points
                if box.timestep().contains_value(pt.timestep())
            ]
            box._points_at_step = {
                step: [p for p in pts if p in box.points]
                for step, pts in box._points_at_step.items()
            }
            return box

    def corners(self, parameters: List[Parameter] = None) -> List[Point]:
        points: List[Point] = [Point(values={})]
        parameter_names = (
            [p.name for p in parameters] if parameters is not None else []
        )
        for p, interval in self.bounds.items():
            if p not in parameter_names:
                continue
            if interval.is_point():
                for pt in points:
                    pt.values[p] = interval.lb
            else:
                lb_points = [pt.model_copy(deep=True) for pt in points]
                ub_points = [pt.model_copy(deep=True) for pt in points]

                nextbefore_ub = (
                    interval.ub
                    if interval.closed_upper_bound
                    else float(nextafter(interval.ub, interval.lb))
                )

                for pt in lb_points:
                    pt.values[p] = interval.lb
                for pt in ub_points:
                    pt.values[p] = nextbefore_ub
                points = lb_points + ub_points
        return points

    def current_step(self) -> "Box":
        # Restrict bounds on num_steps to the lower bound (i.e., the current step)
        curr = self.model_copy(deep=True)
        timestep = curr.timestep()
        timestep.closed_upper_bound = True
        timestep.ub = timestep.lb

        return curr

    def project(self, vars: Union[List[ModelParameter], List[str]]) -> "Box":
        """
        Takes a subset of selected variables (vars_list) of a given box (b) and returns another box that is given by b's values for only the selected variables.

        Parameters
        ----------
        vars : Union[List[ModelParameter], List[str]]
            variables to project onto

        Returns
        -------
        Box
            projected box

        """
        bp = copy.deepcopy(self)
        if len(vars) > 0:
            if isinstance(vars[0], str):
                bp.bounds = {k: v for k, v in bp.bounds.items() if k in vars}
            elif isinstance(vars[0], ModelParameter):
                vars_str = [v.name for v in vars]
                bp.bounds = {
                    k: v for k, v in bp.bounds.items() if k in vars_str
                }
            else:
                raise Exception(
                    f"Unknown type {type(vars[0])} used as intput to Box.project()"
                )
        else:
            bp.bounds = {}
        return bp

    def _merge(self, other: "Box") -> "Box":
        """
        Merge two boxes.  This function assumes the boxes meet in one dimension and are equal in all others.

        Parameters
        ----------
        other : Box
            other box

        Returns
        -------
        Box
            merge of two boxes that meet in one dimension
        """
        bounds = {p: None for p in self.bounds.keys()}
        for p in bounds:
            if self.bounds[p].meets(other.bounds[p]) or other.bounds[p].meets(
                self.bounds[p]
            ):
                bounds[p] = Interval(
                    lb=min(self.bounds[p].lb, other.bounds[p].lb),
                    ub=max(self.bounds[p].ub, other.bounds[p].ub),
                )
            else:
                bounds[p] = Interval(
                    lb=self.bounds[p].lb,
                    ub=self.bounds[p].ub,
                    closed_upper_bound=self.bounds[p].closed_upper_bound,
                )

        merged = self.model_copy(deep=True)
        merged.bounds = bounds
        return merged

    def _get_merge_candidates(self, boxes: Dict[ModelParameter, List["Box"]]):
        equals_set = set([])
        meets_set = set([])
        disqualified_set = set([])
        for p in boxes:
            sorted = boxes[p]
            # find boxes in sorted that meet or equal self in dimension p
            self_index = sorted.index(self)
            # sorted is sorted by upper bound, and candidate boxes are either
            # before or after self in the list
            # search backward
            for r in [
                reversed(range(self_index)),  # search forward
                range(self_index + 1, len(boxes[p])),  # search backward
            ]:
                for i in r:
                    if sorted[i] == self:
                        continue

                    if (
                        (
                            sorted[i].bounds[p].meets(self.bounds[p])
                            or self.bounds[p].meets(sorted[i].bounds[p])
                        )
                        and sorted[i] not in disqualified_set
                        and sorted[i].schedule == self.schedule
                    ):
                        if sorted[i] in meets_set:
                            # Need exactly one dimension where they meet, so disqualified
                            meets_set.remove(sorted[i])
                            disqualified_set.add(sorted[i])
                        else:
                            meets_set.add(sorted[i])
                    elif (
                        sorted[i].bounds[p] == self.bounds[p]
                        and sorted[i] not in disqualified_set
                        and sorted[i].schedule == self.schedule
                    ):
                        equals_set.add(sorted[i])
                    else:
                        if sorted[i] in meets_set:
                            meets_set.remove(sorted[i])
                        if sorted[i] in equals_set:
                            equals_set.remove(sorted[i])
                        disqualified_set.add(sorted[i])
                    if sorted[i].bounds[p].disjoint(self.bounds[p]) and not (
                        sorted[i].bounds[p].meets(self.bounds[p])
                        or self.bounds[p].meets(sorted[i].bounds[p])
                    ):
                        break  # Because sorted, no further checking needed
        if len(boxes.keys()) == 1:  # 1D
            candidates = meets_set
        else:  # > 1D
            candidates = meets_set.intersection(equals_set)
        return candidates

    def point_entropy(self, bias=1.0) -> float:
        """
        Calculate the entropy of a box in terms of the point labels.  Assumes only binary labels, so that p = |true|/(|true|+|false|), and the entropy is H = -(p log p) - ((1-p) log (1-p))

        bias: inverse weight given to positive points

        Returns
        -------
        float
            Entropy of the box
        """
        tp = len(self.true_points()) * bias
        fp = len(self.false_points())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.5

        if p == 0.0:
            H = -((1.0 - p) * log2(1.0 - p))
        elif p == 1.0:
            H = -(p * log2(p))
        else:
            H = -(p * log2(p)) - ((1.0 - p) * log2(1.0 - p))
        return H

    def _lt_base_(self, other):
        # print("b")
        s_t = len(self.true_points())
        o_t = len(other.true_points())

        s_ts = self.timestep().lb
        o_ts = other.timestep().lb

        s_nv = self.normalized_volume()
        o_nv = other.normalized_volume()

        if s_t == o_t:
            if s_ts == o_ts:
                return s_nv > o_nv
            else:
                return s_ts > o_ts
        else:
            return s_t > o_t

    def _lt_entropy_(self, other):
        # print("e")

        s_t = len(self.true_points())
        o_t = len(other.true_points())

        s_et = (1.0 - self.point_entropy()) * (s_t + 1)
        o_et = (1.0 - other.point_entropy()) * (o_t + 1)

        s_nv = self.normalized_volume()
        o_nv = other.normalized_volume()

        s_ts = self.timestep().lb
        o_ts = other.timestep().lb
        # return s_t < o_t
        # if self.timestep().lb == other.timestep().lb:
        #     if s_t == o_t:
        if s_et == o_et:
            if s_ts == o_ts:
                return s_nv > o_nv
            else:
                return s_ts > o_ts
        else:
            return s_et > o_et

        # if s_t == o_t:
        #     if s_nv == o_nv:
        #         return s_ts > o_ts
        #     else:
        #         return s_nv > o_nv
        # else:
        #     return s_t > o_t
        #     else:
        #         return s_t > o_t
        # else:
        #     return self.timestep().lb > other.timestep().lb

    def __lt__(self, other):
        if isinstance(other, Box):
            return (
                self._lt_entropy_(other)
                if self._prioritize_entropy
                else self._lt_base_(other)
            )
        else:
            raise Exception(f"Cannot compare __lt__() Box to {type(other)}")

    def __eq__(self, other):
        if isinstance(other, Box):
            return all(
                [self.bounds[p] == other.bounds[p] for p in self.bounds.keys()]
            )
        else:
            return False

    def __repr__(self):
        return str(self.model_dump())

    def __str__(self):
        bounds_str = "\n".join(
            [
                f"{k}:\t{str(v)}\t({v.normalized_width():.5f})"
                for k, v in self.bounds.items()
            ]
        )
        box_str = f"Box(\n|+pts|: {len(self.true_points())}\n|-pts|: {len(self.false_points())}\nlabel: {self.label}\nwidth: {self.width()}\nnorm-width: {self.width(normalize=True)}\nvolume: {self.volume()}\nnorm-volume: {self.volume(normalize=True)},\ntimepoints: {Interval(lb=self.schedule.time_at_step(int(self.timestep().lb)), ub=self.schedule.time_at_step(int(self.timestep().ub)), closed_upper_bound=True)},\n{bounds_str}\n)"
        return box_str
        # return f"Box(t_{self.timestep()}={Interval(lb=self.schedule.time_at_step(int(self.timestep().lb)), ub=self.schedule.time_at_step(int(self.timestep().ub)), closed_upper_bound=True)} {self.bounds}), width = {self.width()}"

    def finite(self) -> bool:
        """
        Are all parameter intervals finite?

        Returns
        -------
        bool
            all parameter intervals are finite
        """
        return all([i.finite() for _, i in self.bounds.items()])

    def contains(self, other: "Box") -> bool:
        """
        Does the interval for each parameter in self contain the interval for the corresponding parameter in other?

        Parameters
        ----------
        other : Box
            other box

        Returns
        -------
        bool
            self contains other
        """
        return all(
            [
                interval.contains(other.bounds[p])
                for p, interval in self.bounds.items()
            ]
        )

    def _denormalize(self):
        for p, interval in self.bounds.items():
            interval._denormalize()

    def contains_point(
        self, point: Point, denormalize_bounds: bool = False
    ) -> bool:
        """
        Does the box contain a point?

        Parameters
        ----------
        point : Point
            a point
        denormalize_bounds : bool
            if true, and self has unnormalized_lb and unormalized_ub, use these insead of lb and ub.

        Returns
        -------
        bool
            the box contains the point
        """
        return all(
            [
                interval.contains_value(
                    point.values[p], denormalize_bounds=denormalize_bounds
                )
                for p, interval in self.bounds.items()
            ]
        )

    def equal(
        self, b2: "Box", param_list: List[str] = None
    ) -> bool:  ## added 11/27/22 DMI
        ## FIXME @dmosaphir use Parameter instead of str for param_list
        """
        Are two boxes equal, considering only parameters in param_list?

        Parameters
        ----------
        b1 : Box
            box 1
        b2 : Box
            box 2
        param_list : list
            parameters over which to restrict the comparison

        Returns
        -------
        bool
            boxes are equal
        """

        if param_list:
            result = []
            for p1 in param_list:
                for b in self.bounds:
                    if b.name == p1:
                        b1_bounds = [b.lb, b.ub]
                for b in b2.bounds:
                    if b.name == p1:
                        b2_bounds = [b.lb, b.ub]
                if b1_bounds == b2_bounds:
                    result.append(True)
                else:
                    result.append(False)
            return all(result)
        else:
            return self == b

    def intersects(self, other: "Box") -> bool:
        """
        Does self and other intersect? I.e., do all parameter intervals instersect?

        Parameters
        ----------
        other : Box
            other box

        Returns
        -------
        bool
            self intersects other
        """
        return all(
            [
                interval.intersects(other.bounds[p])
                for p, interval in self.bounds.items()
            ]
        )

    def intersection(self, other: "Box", param_list=None) -> Optional["Box"]:
        """
        Return the intersection of two boxes (which is also a box)

        Parameters
        ----------
        other : Box
            other box

        Returns
        -------
        Box
            self intersected with other
        """
        result = Box()
        result_bounds = {}
        if self.label == other.label:
            for p, interval in self.bounds.items():
                if param_list is None or p in param_list:
                    result_bounds[p] = interval.intersection(other.bounds[p])
            result.label = self.label
            result.bounds = result_bounds
        for (
            p,
            interval,
        ) in (
            result.bounds.items()
        ):  # If any intervals are empty, there is no intersection
            if interval is None:
                return None
        return (
            result  # FIXME should this also include points and unknown boxes?
        )

    def _get_max_width_point_Parameter(
        self, points: List[List[Point]], parameters: List[Parameter]
    ):
        """
        Get the parameter that has the maximum average distance from the center point for each parameter and the value for the parameter assigned by each point.

        Parameters
        ----------
        points : List[Point]
            Points in the box

        Returns
        -------
        Parameter
            parameter (dimension of box) where points are most distant from the center of the box.
        """
        parameter_names = [p.name for p in parameters if p.is_synthesized()]
        group_centers = {
            p: [
                average([pt.values[p] for pt in grp])
                for grp in points
                if len(grp) > 0
            ]
            for p in self.bounds
            if p in parameter_names
        }
        centers = {
            p: average(grp) for p, grp in group_centers.items() if len(grp) > 0
        }
        # print(points)
        # print(centers)
        point_distances = [
            {
                p: Decimal(abs(pt.values[p] - centers[p]))
                for p in pt.values
                if p in centers
            }
            for grp in points
            for pt in grp
            if len(grp) > 0
        ]

        parameter_widths = {
            p: average([pt[p] for pt in point_distances])
            for p in self.bounds
            if p in parameter_names
        }
        parameter_widths = {
            p: (
                v / self.bounds[p].original_width
                if self.bounds[p].original_width > 0.0
                else 0.0
            )
            for p, v in parameter_widths.items()
        }

        # normalized_parameter_widths = {
        #     p: average([pt[p] for pt in point_distances])
        #     / (self.bounds[p].width())
        #     for p in self.bounds
        #     if self.bounds[p].width() > 0
        # }
        max_width_parameter = max(
            parameter_widths, key=lambda k: parameter_widths[k]
        )
        if parameter_widths[max_width_parameter] == 0.0:
            return None
        else:
            return max_width_parameter

    def _get_max_width_Parameter(
        self, normalize=False, parameters: List[ModelParameter] = None
    ) -> Union[str, ModelSymbol]:
        if parameters:
            widths = {
                parameter.name: (
                    self.bounds[parameter.name].width(normalize=normalize)
                )
                for parameter in parameters
                if parameter.is_synthesized()
            }
        else:
            widths = {
                p: self.bounds[p].width(normalize=normalize)
                for p in self.bounds
            }
        if "timestep" in widths:
            del widths["timestep"]
        max_width = max(widths, key=widths.get)

        return max_width

    def _get_min_width_Parameter(
        self, normalize=FALSE, parameters: List[ModelParameter] = None
    ) -> Union[str, ModelSymbol]:
        if parameters:
            widths = {
                parameter.name: (
                    self.bounds[parameter.name].width(normalize=normalize)
                )
                for parameter in parameters
            }
        else:
            widths = {
                p: (self.bounds[p].width(normalize=normalize))
                for p in self.bounds
            }
        min_width = min(widths, key=widths.get)

        return min_width

    def volume(
        self,
        normalize=False,
        parameters: List[ModelParameter] = None,
        *,
        ignore_zero_width_dimensions=True,
    ) -> Decimal:
        # construct a list of parameter names to consider
        # if no parameters are requested then use all of the bounds
        if parameters is None:
            pnames = list(self.bounds.keys())
        else:
            pnames = [
                p.name if isinstance(p.name, str) else p.name.name
                for p in parameters
                if isinstance(p, ModelParameter)
            ]

        # handle the volume of zero dimensions
        if len(pnames) <= 0:
            return Decimal("nan")

        # get a mapping of parameters to widths
        # use normalize.get(p.name, None) to select between default behavior and normalization
        widths = {p: self.bounds[p].width(normalize=normalize) for p in pnames}
        if ignore_zero_width_dimensions:
            # filter widths of zero from the
            widths = {p: w for p, w in widths.items() if w != 0.0}

        # TODO in there a 'class' of parameters that we can identify
        # that need this same treatment. Specifically looking for
        # strings 'num_steps' and 'step_size' is brittle.
        num_timepoints = 1
        if self.schedule:
            num_timepoints = Decimal(
                int(self.timestep().ub) + 1 - int(self.timestep().lb)
            ).to_integral_exact(rounding=ROUND_CEILING)
            if normalize is not None:
                num_timepoints = num_timepoints / Decimal(
                    len(self.schedule.timepoints)
                )
        elif "num_steps" in widths:
            del widths["num_steps"]
            # TODO this timepoint computation could use more thought
            # for the moment it just takes the ceil(width) + 1.0
            # so num steps 1.0 to 2.5 would result in:
            # ceil(2.5 - 1.0) + 1.0 = 3.0
            num_timepoints = Decimal(
                self.bounds["num_steps"].ub
            ).to_integral_exact(rounding=ROUND_CEILING)
            num_timepoints += 1

        if "step_size" in widths:
            del widths["step_size"]

        if len(widths) <= 0:
            # TODO handle volume of a point
            return Decimal(0.0)

        # compute product
        product = Decimal(1.0)
        for param_width in widths.values():
            if param_width < 0:
                raise Exception("Negative parameter width")
            product *= Decimal(param_width)
        product *= num_timepoints
        return product

    def normalized_volume(self, parameters: List[ModelParameter] = None):
        params = self.bounds.keys()
        if parameters:
            params = [p.name for p in parameters]

        norm_volume = reduce(
            lambda a, b: a * b,
            [Decimal(self.bounds[p].normalized_width()) for p in params],
            Decimal(1.0),
        )
        return norm_volume

    def normalized_width(self, parameters: List[ModelParameter] = None):
        p = self._get_max_width_Parameter(
            normalize=True, parameters=parameters
        )
        norm_width = self.bounds[p].normalized_width()
        return norm_width

    def width(
        self,
        normalize=False,
        parameters: List[ModelParameter] = None,
    ) -> float:
        """
        The width of a box is the maximum width of a parameter interval.

        Returns
        -------
        float
            Max{p: parameter}(p.ub-p.lb)
        """
        if normalize:
            return self.normalized_width(parameters=parameters)
        else:
            p = self._get_max_width_Parameter(
                normalize=normalize, parameters=parameters
            )
            return self.bounds[p].width(normalize=normalize)

    def variance(self, overwrite_cache=False) -> float:
        """
        The variance of a box is the maximum variance of a parameter interval.
        STUB for Milestone 8 sensitivity analysis

        Returns
        -------
        float
            Variance{p: parameter}
        """
        pass

    def split(
        self,
        points: List[List[Point]] = None,
        normalize: Dict[str, float] = {},
        parameters=[],
    ):
        """
        Split box along max width dimension. If points are provided, then pick the axis where the points are maximally distant.

        Parameters
        ----------
        points : List[Point], optional
            solution points that the split will separate, by default None

        Returns
        -------
        List[Box]
            Boxes resulting from the split.
        """
        p = None
        if (
            points
            and len(points) > 1
            and all([len(grp) > 0 for grp in points])
        ):
            p = self._get_max_width_point_Parameter(
                points, parameters=parameters
            )
            if (
                p is not None
                and self.bounds[p].normalized_width()
                < Decimal(0.5) * self.normalized_width()
            ):
                # Discard selected parameter if its width is much smaller than box width
                p = None
            if p is not None:
                mid = self.bounds[p].midpoint(
                    points=[[pt.values[p] for pt in grp] for grp in points]
                )
                if mid == self.bounds[p].lb or mid == self.bounds[p].ub:
                    # Fall back to box midpoint if point-based mid is degenerate
                    p = self._get_max_width_Parameter(parameter=parameters)
                    mid = self.bounds[p].midpoint()

        if p is None:
            p = self._get_max_width_Parameter(
                normalize=normalize, parameters=parameters
            )
            mid = self.bounds[p].midpoint()

        b1 = self.model_copy(deep=True)
        b2 = self.model_copy(deep=True)

        # b1 is lower half
        assert math_utils.lte(b1.bounds[p].lb, mid)
        b1.bounds[p] = Interval(lb=b1.bounds[p].lb, ub=mid)
        b1.bounds[p].original_width = self.bounds[p].original_width
        b1.points = [pt for pt in b1.points if b1.contains_point(pt)]
        b1._points_at_step = {
            step: [p for p in pts if p in b1.points]
            for step, pts in b1._points_at_step.items()
        }

        # b2 is upper half
        assert math_utils.lte(mid, b2.bounds[p].ub)
        b2.bounds[p] = Interval(lb=mid, ub=b2.bounds[p].ub)
        b2.bounds[p].original_width = self.bounds[p].original_width
        b2.points = [pt for pt in b2.points if b2.contains_point(pt)]
        b2._points_at_step = {
            step: [p for p in pts if p in b2.points]
            for step, pts in b2._points_at_step.items()
        }

        l.debug(
            f"Split[{self.timestep()}]({p}[{self.bounds[p].lb, mid}][{mid, self.bounds[p].ub}])"
        )
        l.debug(
            f"widths: {self.width(parameters=parameters):.5f} -> {b1.width(parameters=parameters):.5f} {b2.width():.5f} (raw), {self.normalized_width(parameters=parameters):.5f} -> {b1.normalized_width(parameters=parameters):.5f} {b2.normalized_width(parameters=parameters):.5f} (norm)"
        )
        return [b2, b1]

    def symm_diff(b1: "Box", b2: "Box"):
        result = []
        ## First check that the two boxes have the same variables
        vars_b1 = set([b for b in b1.bounds])
        vars_b2 = set([b for b in b2.bounds])
        if vars_b1 == vars_b2:
            vars_list = list(vars_b1)
            print("symm diff in progress")
        else:
            print(
                "cannot take the symmetric difference of two boxes that do not have the same variables."
            )
            raise Exception(
                "Cannot take symmetric difference since the two boxes do not have the same variables"
            )
        ### Find intersection
        desired_vars_list = list(vars_b1)
        intersection = b1.intersection(b2, param_list=desired_vars_list)
        ### Calculate symmetric difference based on intersection
        if (
            intersection == None
        ):  ## No intersection, so symmetric difference is just the original boxes
            return [b1, b2]
        else:  ## Calculate symmetric difference
            unknown_boxes = [b1, b2]
            false_boxes = []
            true_boxes = []
            while len(unknown_boxes) > 0:
                b = unknown_boxes.pop()
                if Box.contains(intersection, b) == True:
                    false_boxes.append(b)
                elif Box.contains(b, intersection) == True:
                    new_boxes = Box.split(b)
                    for i in range(len(new_boxes)):
                        unknown_boxes.append(new_boxes[i])
                else:
                    true_boxes.append(b)
            return true_boxes

    @staticmethod
    def _subtract_two_1d_boxes(a, b):
        """Given 2 intervals a = [a0,a1] and b=[b0,b1], return the part of a that does not intersect with b."""
        if intersect_two_1d_boxes(a, b) == None:
            return a
        else:
            if a.lb == b.lb:
                if math_utils.lt(a.ub, b.ub):
                    minArray = a
                    maxArray = b
                else:
                    minArray = b
                    maxArray = a
            elif math_utils.lt(a.lb, b.lb):
                minArray = a
                maxArray = b
            else:
                minArray = b
                maxArray = a
            if math_utils.gt(
                minArray.ub, maxArray.lb
            ):  ## has nonempty intersection. return intersection
                return [float(maxArray.lb), float(minArray.ub)]
            else:  ## no intersection.
                return []

        return [lhs, rhs]

import logging
from typing import Dict, List, Union

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pydantic import BaseModel, Field

import funman.utils.math_utils as math_utils
from funman import to_sympy
from funman.constants import LABEL_FALSE, LABEL_TRUE, LABEL_UNKNOWN, Label

from . import Interval, Point
from .box import Box
from .explanation import BoxExplanation
from .interval import Interval
from .parameter import ModelParameter

l = logging.getLogger(__name__)


class ParameterSpace(BaseModel):
    """
    This class defines the representation of the parameter space that can be
    returned by the parameter synthesis feature of FUNMAN. These parameter spaces
    are represented as a collection of boxes that are either known to be true or
    known to be false.
    """

    num_dimensions: int = None
    true_boxes: List[Box] = []
    false_boxes: List[Box] = []
    unknown_points: List[Point] = []

    def true_points(self) -> List[Point]:
        return [pt for b in self.true_boxes for pt in b.true_points()]

    def false_points(self) -> List[Point]:
        return [pt for b in self.false_boxes for pt in b.false_points()]

    def points(self) -> List[Point]:
        return self.true_points() + self.false_points()

    def boxes(self) -> List[Box]:
        return self.true_boxes + self.false_boxes

    def explain(self) -> "ParameterSpaceExplanation":
        from .explanation import ParameterSpaceExplanation

        true_explanations = [box.explain() for box in self.true_boxes]
        false_explanations = [box.explain() for box in self.false_boxes]
        return ParameterSpaceExplanation(
            true_explanations=true_explanations,
            false_explanations=false_explanations,
        )

    @staticmethod
    def _from_configurations(
        configurations: List[Dict[str, Union[int, "ParameterSpace"]]]
    ) -> "ParameterSpace":
        all_ps = ParameterSpace()
        for result in configurations:
            ps = result["parameter_space"]
            num_steps = result["num_steps"]
            step_size = result["step_size"]
            for box in ps.true_boxes:
                box.bounds["num_steps"] = Interval(
                    lb=num_steps, ub=num_steps + 1
                )
                box.bounds["step_size"] = Interval(
                    lb=step_size, ub=step_size + 1
                )
                all_ps.true_boxes.append(box)
            for box in ps.false_boxes:
                box.bounds["num_steps"] = Interval(
                    lb=num_steps, ub=num_steps + 1
                )
                box.bounds["step_size"] = Interval(
                    lb=step_size, ub=step_size + 1
                )
                all_ps.false_boxes.append(box)
            for point in ps.true_points:
                point.values["num_steps"] = num_steps
                point.values["step_size"] = step_size
                all_ps.true_points.append(point)
            for point in ps.false_points:
                point.values["num_steps"] = num_steps
                point.values["step_size"] = step_size
                all_ps.false_points.append(point)
        return all_ps

    # STUB project parameter space onto a parameter
    @staticmethod
    def project() -> "ParameterSpace":
        raise NotImplementedError()
        return ParameterSpace()

    @staticmethod
    def _intersect_boxes(b1s, b2s):
        results_list = []
        for box1 in b1s:
            for box2 in b2s:
                subresult = Box.__intersect_two_boxes(box1, box2)
                if subresult != None:
                    results_list.append(subresult)
        return results_list

    # STUB intersect parameters spaces
    @staticmethod
    def intersect(ps1, ps2):
        return ParameterSpace(
            ParameterSpace._intersect_boxes(ps1.true_boxes, ps2.true_boxes),
            ParameterSpace._intersect_boxes(ps1.false_boxes, ps2.false_boxes),
        )

    @staticmethod
    def symmetric_difference(ps1: "ParameterSpace", ps2: "ParameterSpace"):
        return ParameterSpace(
            ParameterSpace._symmetric_difference(
                ps1.true_boxes, ps2.true_boxes
            ),
            ParameterSpace._symmetric_difference(
                ps1.false_boxes, ps2.false_boxes
            ),
        )

    @staticmethod
    def _symmetric_difference(ps1: List[Box], ps2: List[Box]) -> List[Box]:
        results_list = []

        for box2 in ps2:
            box2_results = []
            should_extend = True
            for box1 in ps1:
                subresult = Box._symmetric_difference_two_boxes(box2, box1)
                if subresult != None:
                    box2_results.extend(subresult)
                else:
                    should_extend = False
                    break
            if should_extend:
                results_list.extend(box2_results)

        for box1 in ps1:
            box1_results = []
            should_extend = True
            for box2 in ps2:
                subresult = Box._symmetric_difference_two_boxes(box1, box2)
                if subresult != None:
                    box1_results.extend(subresult)
                else:
                    should_extend = False
                    break
            if should_extend:
                results_list.extend(box1_results)

        return results_list

    # STUB construct space where all parameters are equal
    @staticmethod
    def construct_all_equal(ps) -> "ParameterSpace":
        raise NotImplementedError()
        return ParameterSpace()

    # STUB compare parameter spaces for equality
    @staticmethod
    def compare(ps1, ps2) -> bool:
        raise NotImplementedError()

    def plot(self, color="b", alpha=0.2):
        import logging

        from funman_demo import BoxPlotter

        # remove matplotlib debugging
        logging.getLogger("matplotlib.font_manager").disabled = True
        logging.getLogger("matplotlib.pyplot").disabled = True
        logging.getLogger("funman.translate.translate").setLevel(logging.DEBUG)

        custom_lines = [
            Line2D([0], [0], color="g", lw=4, alpha=alpha),
            Line2D([0], [0], color="r", lw=4, alpha=alpha),
        ]
        plt.title("Parameter Space")
        plt.xlabel("beta_0")
        plt.ylabel("beta_1")
        plt.legend(custom_lines, ["true", "false"])
        for b1 in self.true_boxes:
            BoxPlotter.plot2DBoxList(b1, color="g")
        for b1 in self.false_boxes:
            BoxPlotter.plot2DBoxList(b1, color="r")
        # plt.show(block=True)

    @staticmethod
    def decode_labeled_object(obj: dict):
        if not isinstance(obj, dict):
            raise Exception("obj is not a dict")

        try:
            return Point.model_validate(obj)
        except:
            pass

        try:
            return Box.model_validate(obj)
        except:
            pass

        raise Exception(f"obj of type {obj['type']}")

    def __repr__(self) -> str:
        return str(self.model_dump())

    def append_result(self, result: dict):
        inst = ParameterSpace.decode_labeled_object(result)
        label = inst.label
        if isinstance(inst, Box):
            if label == "true":
                self.true_boxes.append(inst)
            elif label == "false":
                self.false_boxes.append(inst)
            else:
                l.info(f"Skipping Box with label: {label}")
        elif isinstance(inst, Point):
            if label == "true":
                self.true_points.append(inst)
            elif label == "false":
                self.false_points.append(inst)
            else:
                l.info(f"Skipping Point with label: {label}")
        else:
            l.error(f"Skipping invalid object type: {type(inst)}")

    def consistent(self) -> bool:
        """
        Check that the parameter space is consistent:

        * All boxes are disjoint

        * All points are in a respective box

        * No point is both true and false
        """
        boxes = self.true_boxes + self.false_boxes
        for i1, b1 in enumerate(boxes):
            for i2, b2 in enumerate(boxes[i1 + 1 :]):
                if b1.intersects(b2):
                    l.exception(f"Parameter Space Boxes intersect: {b1} {b2}")
                    return False
        for tp in self.true_points():
            if not any([b.contains_point(tp) for b in self.true_boxes]):
                return False
        for fp in self.false_points():
            if not any([b.contains_point(fp) for b in self.false_boxes]):
                return False

        if (
            len(set(self.true_points()).intersection(set(self.false_points())))
            > 0
        ):
            return False
        return True

    def _reassign_point_labels(self) -> None:
        """
        For every point, update the label based on the box that contains it.
        """
        points: List[Point] = list(
            set(self.true_points() + self.false_points())
        )
        boxes: List[Box] = self.true_boxes + self.false_boxes
        assigned_points = set([])
        for box in boxes:
            box.points = []
            for point in iter(
                pt for pt in points if pt not in assigned_points
            ):
                if box.contains_point(point):
                    point.label = box.label
                    box.points.append(point)
                    assigned_points.add(point)

        self.unknown_points = [
            pt for pt in points if pt not in assigned_points
        ]

    def _compact(self):
        """
        Compact the boxes by joining boxes that can create a box
        """
        self.true_boxes = self._box_list_compact(self.true_boxes)
        self.false_boxes = self._box_list_compact(self.false_boxes)

    def labeled_volume(self):
        self._compact()
        labeled_vol = 0
        # TODO should actually be able to compact the true and false boxes together, since they are both labeled.
        # TODO can calculate the percentage of the total parameter space.  Is there an efficient way to get the initial PS so we can find the volume of that box? or to access unknown boxes?
        for box in self.true_boxes:
            true_volume = box.volume()
            labeled_vol += true_volume

        for box in self.false_boxes:
            false_volume = box.volume()
            labeled_vol += false_volume
        return labeled_vol

    def max_true_volume(self):
        self.true_boxes = self._box_list_compact(self.true_boxes)
        max_vol = 0
        max_box = (self.true_boxes)[0]
        for box in self.true_boxes:
            box_vol = box.volume()
            if box_vol > max_vol:
                max_vol = box_vol
                max_box = box

        return max_vol, max_box

    def _box_list_compact(self, group: List[Box]) -> List[Box]:
        """
        Attempt to union adjacent boxes and remove duplicate points.
        """
        # Boxes of dimension N can be merged if they are equal in N-1 dimensions and meet in one dimension.
        # Sort the boxes in each dimension by upper bound.
        # Interate through boxes in order wrt. one of the dimensions. For each box, scan the dimensions, counting the number of dimensions that each box meeting in at least one dimension, meets.
        # Merging a dimension where lb(I) = ub(I'), results in an interval I'' = [lb(I), lb(I')].

        if len(group) <= 0:
            return []

        dimensions = group[0].bounds.keys()
        # keep a sorted list of boxes by dimension based upon the upper bound in the dimension
        sorted_dimensions = {p: [b for b in group] for p in dimensions}
        for p, boxes in sorted_dimensions.items():
            boxes.sort(key=lambda x: x.bounds[p].ub)
        dim = next(iter(sorted_dimensions.keys()))
        merged = True
        while merged:
            merged = False
            for b in sorted_dimensions[dim]:
                # candidates for merge are all boxes that meet or are equal in a dimension
                candidates = b._get_merge_candidates(sorted_dimensions)
                # pick first candidate
                if len(candidates) > 0:
                    c = next(iter(candidates))
                    m = b._merge(c)
                    sorted_dimensions = {
                        p: [
                            box if box != b else m for box in boxes if box != c
                        ]
                        for p, boxes in sorted_dimensions.items()
                    }
                    merged = True
                    break

        return sorted_dimensions[dim]

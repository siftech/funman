import unittest

from funman.representation import Box, Interval, ParameterSpace


class TestCompilation(unittest.TestCase):
    def test_interval_intersection(self):
        interval_1 = Interval(lb=0, ub=3)
        interval_2 = Interval(lb=2, ub=5)
        assert interval_1.intersection(interval_2) == Interval(lb=2.0, ub=3.0)

    def test_box_intersection_identical(self):
        box1 = Box(
            bounds={
                "x": Interval(lb=0, ub=2),
                "timestep": Interval(lb=0, ub=0),
            },
            label="true",
        )
        box2 = Box(
            bounds={
                "x": Interval(lb=0, ub=2),
                "timestep": Interval(lb=0, ub=0),
            },
            label="true",
        )
        result = box1.intersection(box2)
        print("box intersection result:", result.__dict__)
        assert (result == box1) and (
            result == box2
        )  # Since these boxes are identical, their intersection will be equal to each original box

    def test_box_intersection_identical_diff_labels(self):
        box1 = Box(
            bounds={
                "x": Interval(lb=0, ub=2),
                "timestep": Interval(lb=0, ub=0),
            },
            label="true",
        )
        box2 = Box(
            bounds={
                "x": Interval(lb=0, ub=2),
                "timestep": Interval(lb=0, ub=0),
            },
            label="false",
        )
        result = box1.intersection(box2)
        assert result.bounds == {}  # Empty intersection

    def test_box_intersection_nontrivial(
        self,
    ):  # Not intersecting on all dimensions, so intersection will be empty
        box1 = Box(
            bounds={
                "x": Interval(lb=0, ub=2),
                "y": Interval(lb=0, ub=2),
                "timestep": Interval(lb=0, ub=0),
            },
            label="true",
        )
        box2 = Box(
            bounds={
                "x": Interval(lb=11, ub=13),
                "y": Interval(lb=0, ub=2),
                "timestep": Interval(lb=0, ub=0),
            },
            label="true",
        )
        result = box1.intersection(box2)
        assert result is None

    def test_box_intersection_nonidentical(self):  # Non-empty, non-identical
        box1 = Box(
            bounds={
                "x": Interval(lb=0, ub=2),
                "y": Interval(lb=0, ub=2),
                "timestep": Interval(lb=0, ub=0),
            },
            label="true",
        )
        box2 = Box(
            bounds={
                "x": Interval(lb=1, ub=3),
                "y": Interval(lb=1, ub=3),
                "timestep": Interval(lb=0, ub=0),
            },
            label="true",
        )
        result = box1.intersection(box2)
        assert result.bounds["x"] == result.bounds["y"]

    def test_ps_intersection(self):
        # Set up two parameter spaces to intersect
        ps1 = ParameterSpace(num_dimensions=2)
        ps2 = ParameterSpace(num_dimensions=2)
        ps1.append_result(
            {
                "type": "box",
                "label": "true",
                "bounds": {
                    "inf_o_o": {"lb": 0.0, "ub": 0.3},
                    "rec_o_o": {"lb": 0.0, "ub": 0.5},
                    "timestep": {"lb": 1.0, "ub": 1.0},
                },
            }
        )
        ps1.append_result(
            {
                "type": "box",
                "label": "true",
                "bounds": {
                    "inf_o_o": {"lb": 0.5, "ub": 1.0},
                    "rec_o_o": {"lb": 0.5, "ub": 1.0},
                    "timestep": {"lb": 1.0, "ub": 1.0},
                },
            }
        )
        ps2.append_result(
            {
                "type": "box",
                "label": "true",
                "bounds": {
                    "inf_o_o": {"lb": 0.0, "ub": 0.3},
                    "rec_o_o": {"lb": 0.0, "ub": 0.5},
                    "timestep": {"lb": 1.0, "ub": 1.0},
                },
            }
        )
        ps2.append_result(
            {
                "type": "box",
                "label": "true",
                "bounds": {
                    "inf_o_o": {"lb": 0.75, "ub": 1.25},
                    "rec_o_o": {"lb": 0.75, "ub": 1.25},
                    "timestep": {"lb": 1.0, "ub": 1.0},
                },
            }
        )

        result = ps1.intersection(ps2)  ## should consist of 2 boxes
        print(result.__dict__)
        assert result


if __name__ == "__main__":
    unittest.main()

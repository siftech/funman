import unittest

from funman.representation import Box, Interval, ParameterSpace


class TestCompilation(unittest.TestCase):
    def test_interval_intersection(self):
        interval_1 = Interval(lb=0, ub=3)
        interval_2 = Interval(lb=2, ub=5)
        assert interval_1.intersection(interval_2) == [2.0, 3.0]

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
                },
            }
        )
        result = ps1.intersection(ps2)  ## should consist of 2 boxes
        assert result


if __name__ == "__main__":
    unittest.main()

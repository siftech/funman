import logging
import sys
import unittest

import sympy

from funman.utils.sympy_utils import SympyBoundedSubstituter, to_sympy


class TestUseCases(unittest.TestCase):
    l = logging.Logger(__name__)

    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        self.l.level = logging.getLogger().level
        self.l.handlers.append(logging.StreamHandler(sys.stdout))

    def test_minimize_expression(self):
        tests = [
            {
                "input": ["S", "- S * I * beta/N"],
                "bound": "lb",
                "expected_output": "-I_ub*S_lb*beta_ub/N_lb",
            },
            {
                "input": ["S", "- S * I * beta/N"],
                "bound": "ub",
                "expected_output": "-I_lb*S_ub*beta_lb/N_ub",
            },
            {
                "input": ["S", "- S * I * beta"],
                "bound": "lb",
                "expected_output": "-I_ub*S_lb*beta_ub",
            },
            {
                "input": ["S", "- S * I * beta"],
                "bound": "ub",
                "expected_output": "-I_lb*S_ub*beta_lb",
            },
            {
                "input": ["I", "S * I * beta - I * gamma"],
                "bound": "lb",
                "expected_output": "I_lb*S_lb*beta_lb - I_lb*gamma_ub",
            },
            {
                "input": ["I", "S * I * beta - I * gamma"],
                "bound": "ub",
                "expected_output": "I_ub*S_ub*beta_ub - I_ub*gamma_lb",
            },
            {
                "input": ["R", "I * gamma"],
                "bound": "lb",
                "expected_output": "I_lb*gamma_lb",
            },
            {
                "input": ["I", "I * gamma"],
                "bound": "ub",
                "expected_output": "I_ub*gamma_ub",
            },
        ]

        str_symbols = ["S", "I", "R", "beta", "gamma", "N"]
        symbols = {s: sympy.Symbol(s) for s in str_symbols}
        bound_symbols = {
            s: {"lb": f"{s}_lb", "ub": f"{s}_ub"} for s in str_symbols
        }
        substituter = SympyBoundedSubstituter(
            bound_symbols=bound_symbols, str_to_symbol=symbols
        )

        for test in tests:
            with self.subTest(f"{test['bound']}({test['input']})"):
                test_fn = (
                    substituter.minimize
                    if test["bound"] == "lb"
                    else substituter.maximize
                )
                test_output = test_fn(*test["input"])
                # self.l.debug(f"Minimized: [{infection_rate}], to get expression: [{test_output}]")
                assert (
                    str(test_output) == test["expected_output"]
                ), f"Failed to create the expected expression: [{test['expected_output']}], got [{test_output}]"


if __name__ == "__main__":
    unittest.main()

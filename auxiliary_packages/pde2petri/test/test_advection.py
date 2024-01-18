import logging
import sys
import unittest

import pde2petri

l = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)


class TestAdvection(unittest.TestCase):
    def test_advection_cartesian(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logging.getLogger().addHandler(stream_handler)

        converter = pde2petri.PDE2Petri()

        constants = {
            "g": 9.8101,
            "rho": 910.0,
            "n": 3.0,
        }
        gamma = "(2/(n + 2))*A*(g*rho)**n"
        w = "(h**(n+2) *Abs(Derivative(h, x))**(n - 1)) *Derivative(h, x)"
        expression = f"Eq(Derivative(h, t),({gamma}*Derivative({w},x)))"

        sympy_expression = converter.discretize(
            expression, constants=constants
        )
        l.debug(f"pde2petri.discretize() returned {sympy_expression}")
        assert (
            sympy_expression
        ), "Could not generate sympy expression for discretization"

    def test_advection_to_amr(self):
        converter = pde2petri.PDE2Petri()
        amr = converter.to_amr()
        assert amr, "Could not create an AMR model."


if __name__ == "__main__":
    unittest.main()

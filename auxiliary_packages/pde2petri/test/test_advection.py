import logging
import unittest

import pde2petri

l = logging.getLogger(__name__)
l.setLevel(logging.DEBUG)


class TestAdvection(unittest.TestCase):
    def test_advection_cartesian(self):
        converter = pde2petri.PDE2Petri()
        sympy_expression = converter.discretize(
            "Eq(Derivative(h, t),(((g*rho)/A)**n*Derivative((h**(n+2) *Abs(Derivative(h, x))**(n - 1)) *Derivative(h, x),x))*(2/(n + 2)))"
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

import logging
import sys
import unittest

import pde2petri
from sympy.parsing.latex import parse_latex

l = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)


class TestAdvection(unittest.TestCase):
    def test_advection_cartesian(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logging.getLogger().addHandler(stream_handler)

        converter = pde2petri.PDE2Petri()

        constants = {"a": 0.5}
        latex_eqn = (
            r"\frac{\partial h}{\partial t} + a\frac{\partial h}{\partial x} = 0"
        )

        eqn_parse = parse_latex(latex_eqn)

        sympy_expression = converter.discretize(eqn_parse, constants=constants)
        l.debug(f"pde2petri.discretize() returned {sympy_expression[0]}")
        l.debug(f"pde2petri returned Petri Net component {sympy_expression[1]}")
        assert (
            sympy_expression
        ), "Could not generate sympy expression for discretization"


#    def test_advection_to_amr(self):
#        converter = pde2petri.PDE2Petri()
#        amr = converter.to_amr()
#        assert amr, "Could not create an AMR model."


if __name__ == "__main__":
    unittest.main()

import unittest

import pde2petri

import logging
l = logging.getLogger(__name__)
l.setLevel(logging.DEBUG)

class TestAdvection(unittest.TestCase):

    def test_advection_cartesian(self):
        converter = pde2petri.PDE2Petri()
        sympy_expression = converter.discretize("dx/dt")
        l.debug(f"pde2petri.discretize() returned {sympy_expression}")
        assert sympy_expression, "Could not generate sympy expression for discretization"
        

if __name__ == "__main__":
    unittest.main()
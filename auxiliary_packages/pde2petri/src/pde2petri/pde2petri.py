import sympy

class PDE2Petri(object):
    def __init__(self, *args, **kwargs):
        pass

    def discretize(self, pde: str):
        return sympy.sympify(pde)
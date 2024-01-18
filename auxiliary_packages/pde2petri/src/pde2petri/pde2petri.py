from typing import List

import sympy
from pydantic import BaseModel

from .model.petrinet import *


class SymbolicPetri(BaseModel):
    symbols: List[str]
    boundary: str
    initial: str
    state_update: str


class PDE2Petri(object):
    def __init__(self, *args, **kwargs):
        pass

    def derivative_to_finite_difference(self, expr: sympy.Expr):
        """
        Recursively visit expression expr to replace Derivative terms with their finite_difference.

        Parameters
        ----------
        expr : sympy.Expr
            The expression to modify
        """
        print(expr)
        print(expr.args)
        if isinstance(expr, sympy.core.function.Derivative):
            new_args = [
                self.derivative_to_finite_difference(expr.args[0]),
                expr.args[1],
            ]
            new_expr = expr.func(*new_args)
            finite_difference = new_expr.as_finite_difference(
                [expr.args[1][0] + 1, expr.args[1][0]]
            )
            return finite_difference
        elif isinstance(expr, sympy.core.containers.Tuple) or isinstance(
            expr, bool
        ):
            return expr
        elif isinstance(expr, sympy.core.symbol.Symbol):
            return expr
        elif (
            expr.is_Function
            or expr.is_Relational
            or expr.is_Mul
            or expr.is_Add
            or expr.is_Pow
        ):
            new_args = [
                self.derivative_to_finite_difference(arg) for arg in expr.args
            ]
            new_expr = expr.func(*new_args)
            return new_expr
        else:
            return expr

    def extract_dims(self, dims):
        time_dim = "t" if "t" in dims else None
        non_time_dims = [d for d in dims if d != time_dim]
        return time_dim, non_time_dims

    def discretize(self, pde_str: str, vars=["h"], dims=["x", "t"]):
        pde = sympy.sympify(pde_str)

        time_dim, non_time_dims = self.extract_dims(dims)

        dims_str = ",".join(dims)
        var_to_var_dims = {var: f"{var}({dims_str})" for var in vars}
        pde_dims = pde.subs(var_to_var_dims)
        pde_dims_discrete = self.derivative_to_finite_difference(pde_dims)

        step_updates = {}
        for var in vars:
            next_state = f"{var}({non_time_dims}, {time_dim}+1)"
            step_update = sympy.solve(
                pde_dims_discrete, sympy.sympify("h(x, t+1)")
            )[0]
            step_updates[next_state] = step_update
        return step_updates

    def instantiate(self, points):
        pass

    def to_amr(self):
        model = Model(
            header=Header(
                name="name",
                schema_=AnyUrl(
                    "https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/petrinet_v0.1/petrinet/petrinet_schema.json"
                ),
                description="Petrinet model created by Dan Bryce and Drisana Mosiphir",
                model_version="0.1",
            ),
            model=Model1(
                states=States(root=[]), transitions=Transitions(root=[])
            ),
            semantics=Semantics(ode=OdeSemantics()),
        )
        return model

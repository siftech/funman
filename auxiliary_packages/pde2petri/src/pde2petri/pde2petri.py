import logging
from typing import List

import sympy
from pydantic import BaseModel

from .model.petrinet import *

l = logging.getLogger(__name__)


class SymbolicPetri(BaseModel):
    symbols: List[str]
    boundary: str
    initial: str
    state_update: str


class PDE2Petri(object):
    def __init__(self, *args, **kwargs):
        pass

    def derivative_to_finite_difference(self, expr: sympy.Expr, points=None):
        """
        Recursively visit expression expr to replace Derivative terms with their finite_difference.

        Parameters
        ----------
        expr : sympy.Expr
            The expression to modify
        """
        # print(expr)
        # print(expr.args)
        if isinstance(expr, sympy.core.function.Derivative):
            new_args = [
                self.derivative_to_finite_difference(
                    expr.args[0], points=points
                ),
                expr.args[1],
            ]
            new_expr = expr.func(*new_args)
            differential = expr.args[1][0]
            if points:
                difference_points = [
                    differential + p for p in points[str(differential)]
                ]
            else:
                difference_points = [expr.args[1][0] + 1, expr.args[1][0]]
            finite_difference = new_expr.as_finite_difference(
                difference_points
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
                self.derivative_to_finite_difference(arg, points=points)
                for arg in expr.args
            ]
            new_expr = expr.func(*new_args)
            return new_expr
        else:
            return expr

    def extract_dims(self, dims):
        time_dim = "t" if "t" in dims else None
        non_time_dims = [d for d in dims if d != time_dim]
        return time_dim, non_time_dims

    def discretize(
        self, pde_str: str, constants={}, vars=["h"], dims=["x", "t"]
    ):
        dt = sympy.sympify("dt")
        pde = sympy.sympify(pde_str)

        l.info(f"Discretizing: {pde}\n[\n{sympy.latex(pde)}\n]")

        time_dim, non_time_dims = self.extract_dims(dims)
        l.info(f"Discretizing with non-time dimensions: {non_time_dims}")

        dims_str = ",".join(dims)
        var_to_var_dims = {var: f"{var}({dims_str})" for var in vars}
        pde_dims = pde.subs(var_to_var_dims)
        l.info(
            f"With dimension terms subbed: {pde_dims}\n[\n{sympy.latex(pde_dims)}\n]"
        )

        pde_w_constants = pde_dims.subs(constants)
        l.info(
            f"With constants subbed: {pde_w_constants}\n[\n{sympy.latex(pde_w_constants)}\n]"
        )

        pde_dims_discrete = self.derivative_to_finite_difference(
            pde_w_constants, points={"x": [-1, 0], "t": [dt, 0]}
        )
        l.info(
            f"Finite Difference: {pde_dims_discrete} \n[\n{sympy.latex(pde_dims_discrete)}\n]"
        )

        step_updates = {}
        for var in vars:
            next_state = f"{var}({','.join(non_time_dims)}, {time_dim}+{dt})"
            step_update = (
                sympy.solve(pde_dims_discrete, sympy.sympify(next_state))[0]
                .expand()
                .collect(dt)
            )
            step_updates[next_state] = step_update
            l.info(
                f"Step-update: {next_state}: {step_update} \n[\n{sympy.latex(next_state)}:\n {sympy.latex(step_update)}\n]"
            )
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

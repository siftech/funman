from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import sympy
from pydantic import BaseModel
from scipy.integrate import odeint

from funman import FunmanModel

from ..representation.representation import Timeseries

numeric = Union[int, float]


class Simulator(BaseModel):
    model: FunmanModel
    init: Dict[str, Union[float, str]]
    parameters: Dict[str, float]
    tvect: List[numeric]

    def model_args(self) -> List[float]:
        def make_param_func(pname, value):
            def pfunc(t):
                return np.piecewise(t, [t >= 0], [value])

            pfunc.__name__ = pname
            return pfunc

        params = [make_param_func(p, pv) for p, pv in self.parameters.items()]
        return tuple(params)

    def initial_state(self) -> List[float]:
        init_state = [
            (
                sympy.sympify(self.init[var]).evalf(subs=self.parameters)
                if isinstance(self.init[var], str)
                else self.init[var]
            )
            for var in self.model._state_var_names()
        ]
        return tuple(init_state)

    def sim(self) -> Optional[Timeseries]:
        # gradient_fn = partial(self.model.gradient, self.model) # hide the self reference to self.model from odeint

        if self.model._is_differentiable:
            timeseries = odeint(
                self.model.gradient,
                self.initial_state(),
                self.tvect,
                args=self.model_args(),
            )

            ts = Timeseries(
                data=[self.tvect] + timeseries.T.tolist(),
                columns=["time"] + self.model._state_var_names(),
            )
        else:
            ts = None
        return ts

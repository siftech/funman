from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import sympy
from pydantic import BaseModel
from scipy.integrate import odeint, solve_ivp

from funman import FunmanModel

from ..representation.representation import Timeseries

numeric = Union[int, float]

import logging

l = logging.getLogger(__name__)


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
            full_output = 1
            use_odeint = True
            if use_odeint:
                timeseries = odeint(
                    self.model.gradient,
                    self.initial_state(),
                    self.tvect,
                    args=self.model_args(),
                    full_output=full_output,
                    tfirst=True,
                )
                if full_output == 1:
                    timeseries, output = timeseries

                l.debug(f"odeint output: {output}")
                data = (
                    timeseries.T.tolist()
                    if len(timeseries) > 0
                    else [[v] for v in self.initial_state()]
                )
            else:
                result = solve_ivp(
                    self.model.gradient,
                    (self.tvect[0], self.tvect[-1]),
                    self.initial_state(),
                    args=self.model_args(),
                    t_eval=self.tvect,
                    # first_step=1.0,
                    # max_step=1.0,
                    # rtol=1.0,
                    # atol=1.0,
                )
                timeseries = result.y

                data = (
                    timeseries.tolist()
                    if len(timeseries) > 0
                    else [[v] for v in self.initial_state()]
                )
            ts = Timeseries(
                data=[self.tvect] + data,
                columns=["time"] + self.model._state_var_names(),
            )
        else:
            ts = None
        return ts

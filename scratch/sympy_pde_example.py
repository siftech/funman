from sympy import Derivative, E, Eq, Function, pde_separate, symbols
from sympy.parsing.latex import parse_latex

### Symbol initialization
x, y, z, t, i, n, dx, dt, a = symbols("x y z t i n dx dt a")
init_printing(use_unicode=True)

u = symbols("u", cls=Function)
u = Function("u")
dudx = u(x, t).diff(x)
dudt = u(x, t).diff(t)

### Solve for discrete derivatives - other arbitrary schemes for discretizing time and space can be brought in here.  This is general, so these can be substituted into other PDEs.

backward_space_derivative = dudx.as_finite_difference(
    [dx * (i - 1), dx * i]
).subs(t, dt * i)
forward_space_derivative = dudx.as_finite_difference(
    [dx * (i), dx * (i + 1)]
).subs(t, dt * i)
forward_time_derivative = dudt.as_finite_difference(
    [dt * (n), dt * (n + 1)]
).subs(x, dx * i)

### Parse advection equation.  This one is from LaTeX but can use other sources too.

advection_parse = parse_latex(
    r"\frac{\partial q}{\partial t} + a\frac{\partial q}{\partial x} = 0"
)


### Substitute into parsed equation - input definitions for u and discrete derivatives

advection_parse = advection_parse.subs(Symbol("q"), u(x, t))

advection_parse = advection_parse.subs(
    Derivative(u(x, t), t), forward_time_derivative
)
advection_parse = advection_parse.subs(
    Derivative(u(x, t), x), backward_space_derivative
)

print(
    advection_parse
)  ### Advection equation in terms of discrete time and space points

### Solve for u at next timepoint ("n+1") in terms of previous points

next_time_step_advection_parse = solve(
    advection_parse, u(dx * i, dt * (n + 1))
)

print(
    next_time_step_advection_parse
)  ## matches FTBS on upwind scheme page of https://indico.ictp.it/event/a06220/session/18/contribution/10/material/0/2.pdf for a > 0

# PDE to Petri converter
A package for converting PDEs into Petrinet AMRs

This package contains manual and automated approaches to convert vector space PDEs into Petrinets with custom rate laws.

1) Manual: The script [scripts/generate-pde-amr.py](scripts/generate-pde-amr.py) will write several directories to the current working directory.  Each directory contains alternative formulations of a PDE problem as a Petrinet, including:
    - `halfar`: ice dome model 
    - `advection`: advection model for incompressible flows 

    The files in each subdirectory use the naming scheme, as follows:

    `{problem}_{derivative}_{dimensions}_{boundary_slope}_{num_disc}.json`

    where 

    - `problem`: the name of the problem (e.g., `advection`)
    - `derivative`: the method used to compute spatial derivatives (e.g., `forward`, `backward`, and `centered`)
    - `dimensions`: the number of spatial dimensions (1, 2, or 3)
    - `boundary_slope`: the coefficient for boundary conditions, expressed as `u(x, t) = kt`, where `k = boundary_slope`, `t` is the time (relative to starting time at 0), `u` is a state variable, and `x` is a boundary position.
    - `num_disc`: the number of discrete points in each dimension (e.g., if `dimension = 2` and `num_disc = 5`, then there will be `5^2 = 25` positions, not including boundaries).
    
    A notebook illustrating the results of FUNMAN analyzing the models is available [here](https://github.com/siftech/funman/blob/pde-amr-examples/notebooks/pde_as_petrinet.ipynb).

2) Automated: The [./test](./test) directory includes tests that illustrate automatically converting a vector space PDE to a Petrinet.  

    The code in [./src](./src) corresponds to the `pde2petri` package, which can be installed with `pip` (e.g., `pip install .` where the current directory is the same directory containing this README).  

    The automated approach involves parsing an expression with the sympy package, discretizing the PDE, and isolating the "next state" variables to determine the Petrinet states and rate laws for the transitions.  The [document](./doc/discretization/main.pdf) describes the general approach taken in more detail, and with examples.

---
Authored by dbryce@sift.net and dmosaphir@sift.net
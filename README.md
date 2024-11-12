# Terarium Notes
Terarium will build and publish `funman-base` to GHCR that includes the dReal, ibex and other auxiliaries.
This build uses CircleCI, see [pipeline results](https://app.circleci.com/pipelines/github/DARPA-ASKEM) 

Terarium then builds a `funman-taskrunner` that bootstraps off `funman-base`. The taskrunner provides an ad-hoc invocation of Funman using MQ and named-pipes as the messaging mechanism. The taskrunner will do a source install of this repository. See here for more [information](https://github.com/DARPA-ASKEM/terarium/tree/main/packages/funman).


# `funman`: Functional Model Analysis tool

<!-- Outline:
- Introduction (problem solved, major packages, prerequisites)
- Quickstart (get docker container, use commmand line or Jupyter)
- Model Examples (focus on Petrinet)
- Request structure and examples (input, plotting) (constraints, parameters, flags)
- Show how to run examples with: command line, API, code
- Dev Setup (needs cleanup)
    - Code layout
    - Obtain pre-built container 
    - Building Dev container
    - Run dev container with vscode
 -->

![Funman synthesizes parameters for ODE and PDE systems](https://github.com/siftech/funman/blob/v1.8.0-rc/fig/funman-diagram.png?raw=true)

The `funman` package performs Functional Model Analysis by processing a request and model pair that describes an analysis scenario.  `funman` answers the question: 

    "Under which parameter assignements will my model behave well?"

`funman` encodes and reasons about parameterized models in the SMT framework.  As illustrated in the figure above, `funman` labels regions of a parameter space that satisfy (green, "true") or do not satisfy (red, "false") a number of constraints on the model dynamics (dashed lines).  Each point in the parameter space corresponds (black arrows) to a trajectory (green and red curves).  

# Quickstart

This section explains how to run `funman`, and the following section describes the analysis request in more detail.

We recommend using a pre-built docker image to run or develop the `funman` source, due to a few build dependencies.  To run `funman` via its API, (in the `funman` root directory) start the API with the script:

```bash
sh terarium/scripts/run-api-in-docker.sh
```

This will pull the latest release of the `funman` API image and start a new container.  Then, to submit analysis requests to the API, run the following script:

```bash
terarium/run_example.sh petrinet-sir
```

This script will run a request, as specified in `resources/terarium-tests.json`.

Running the example will POST a model analysis request, wait, and then GET the results.  `funman` analysis runs in an anytime fashion, and the GET request may return only partial results.  The example will generate the following output:

```
Running example 'petrinet-sir':
{
  "name": "petrinet-sir",
  "model-path": "amr/petrinet/terrarium-tests/sir.json",
  "request-path": "amr/petrinet/terrarium-tests/sir_request.json"
}
################################################################################
################################################################################
Making POST request to http://127.0.0.1:8190/api/queries with contents:
{
    "model": <Contents of /home/danbryce/funman/resources/amr/petrinet/terrarium-tests/sir.json>
    "request": <Contents of /home/danbryce/funman/resources/amr/petrinet/terrarium-tests/sir_request.json>
}
################################################################################
################################################################################
Response for query:
{
  "id": "cfabd91c-4a6a-42aa-8d10-e88b4ca2dd5f",
  "model": "Removed for brevity",
  "request": "Removed for brevity"
}
################################################################################
################################################################################
Work Id is 'cfabd91c-4a6a-42aa-8d10-e88b4ca2dd5f'
Waiting for 5 seconds...
################################################################################
################################################################################
Making GET request to http://127.0.0.1:8190/api/queries/cfabd91c-4a6a-42aa-8d10-e88b4ca2dd5f:
{
  "id": "cfabd91c-4a6a-42aa-8d10-e88b4ca2dd5f",
  "model": "Removed for brevity",
  "progress": {
    "progress": 0.6918088739568541,
    "coverage_of_search_space": 0.6918088739568541
  },
  "request": "Removed for brevity",
  "parameter_space": {
    "num_dimensions": 6,
    "true_boxes": [ {"Removed for brevity"}, {"Removed for brevity"}, ...]
    "false_boxes": [ {"Removed for brevity"}, {"Removed for brevity"}, ...]
  }
}
```

The response is the status of the query, which includes the model and request used in the query, as well as progress and the current parameter space.  

# Quickstart Example Description

The quickstart example performs an analysis of the SIR model:

$\dot{S} = -\beta SI$

$\dot{I} = \beta SI - \gamma I$

$\dot{R} = \gamma I$

where $\beta$ and $\gamma$ are parameters.  The SIR model captures how a susceptible $S$ population becomes infected $I$ at a rate $\beta SI$, and an infected population becomes recovered $R$ with a rate $\gamma I$.  The $\beta$ parameter describes the transmissibility of a pathogen and the $\gamma$ parameter, the impact (recovery time) of the pathogen on individuals.  The example defines an initial condition where $S(0) = 0.99$, $I(0) = 0.01$, and $R(0) = 0.0$.  

In the example, `funman` analyzes which values of 

$\beta \in [0.08, 0.1)$ and 

$\gamma \in [0.02, 0.03)$ 

will satisfy the constraint 

$0.15 \leq I(t) < 1.0$, for $t \in [50, 50)$.

In this analysis request the analyst would like to know what assumptions on $\beta$ and $\gamma$ are needed so that the infected population is greater than 0.15, 50 days from today.  

Solving this analysis problem generates a parameter space, as summarized by the following plot.

![Funman synthesizes parameters the SIR model](https://github.com/siftech/funman/blob/v1.8.0-rc/fig/parameter-space.png?raw=true)

The plot illustrates the parameter space as a matrix of 2D plots for each pair of parameters. (This example includes parameters $S0$, $I0$, and $R0$ for the initial state assignment).  `funman` introduces the `timestep` as an additional parameter to support reasoning about different time spans.  This `timestep` is important for many scenarios because the constraints may be satsified or not under different assumptions about the scenario time span.    

The upper left plot for $\beta$-$\gamma$ projects all boxes onto $\beta$ and $\gamma$, indicating that there are values of the other parameters where any assignment to the pair will satisfy the constraint.  Unsurprisingly, larger values for $\beta$ and smaller values for $\gamma$ result in more scenario time spans where the constraints are satisfied (darker green regions include more true regions that are stacked/projected onto $\beta$ and $\gamma$). 

The lower left plots for $\beta$-timestep and $\gamma$-timestep help to interpret the $\beta$-$\gamma$ plot.  Timesteps 5-10 (corresponding to 50-100 days) include several false regions (also stacked) where the constraints are not satisfied, and a few true regions where they are satisfied.  

Plotting the points generated while creating the parameter space, will result in the following trajectories.  

![Funman creates multiple trajectories for the SIR model](https://github.com/siftech/funman/blob/v1.8.0-rc/fig/trajectories.png?raw=true)

The plot illustrates $S$, $I$, and $R$ for all points that `funman` generates to construct the parameter space.  The green lines correspond to true points, and red, false.  The lines that begin near 1.0 a time 0 correspond to $S$.  The group of lines near 0.2 at time 50 correspond to $I$, and the remaining group, to $R$.  The false trajectories are truncated at time 50 because they violate the constraint and will not satisfy it if extended to longer time spans.  (This is also the reason that the parameter space includes many false regions for time steps 5 and greater.)  

# `funman` inputs

`funman` requires two inputs: a model and an analysis request.  There are several ways to configure these inputs, but the most simple is to write a pair of json files.  It is also possible to define these inputs via python objects, as illustrated by tests in [tests/test_use_cases.py](https://github.com/siftech/funman/blob/v1.8.0-rc/test/test_use_cases.py).

`funman` supports a number of model formats, which correspond to the classes in `funman.models`:

- [GeneratedPetriNet](#funman.model.generated_models.petrinet.Model): AMR model for petri nets generated 
- [GeneratedRegNet](#funman.model.generated_models.regnet.Model): AMR model for regulatory networks
- [RegnetModel](#funman.model.regnet.RegnetModel): MIRA model for regulatory networks
- [PetrinetModel](#funman.model.petrinet.PetrinetModel): MIRA model for petri nets
- [BilayerModel](#funman.model.bilayer.BilayerModel): ASKE model for bilayer networks



Requests correspond to the class `funman.server.query.FunmanWorkRequest` and specify the following keys:
- [query](#funman.model.query): a soft constraint that a model must satisfy (legacy support, deprecated)
- [constraints](#funman.representation.constraint): a list of hard or soft constraints that a model must satisfy
- parameters: a list of bounded parameters that `funman` will either synthesize ("label": "all") or satsify ("label": "any").  `funman` will check  "all" values within the parameter bounds or if "any" within the bounds permit the model to satisfy the query and constraints.
- [config](#funman.config): A dictionary of `funman` configuration options. 
 label regions of the parameter space as satisfying the query and constraints, if synthesized, or find any legal value if asked to satsify.  
- [structure_parameters](#funman.representation.parameter): parameters shaping the way that `funman` structures its analysis.  `funman` requires that either the `num_steps` and `step_size` parameters are specified, or the `schedules` parameter is specified.  If all are omitted, then `funman` defaults to checking one unit-sized step.

The example illustrated in the quickstart uses the request [resources/amr/petrinet/terrarium-tests/sir_request.json](https://github.com/siftech/funman/blob/v1.8.0-rc/resources/amr/petrinet/terrarium-tests/sir_request.json)

```json
{
    "constraints": [
        {
            "name": "I",
            "variable": "I",
            "interval": {
                "lb": 0.15,
                "ub": 1.0
            },
            "timepoints": {
                "lb": 50,
                "ub": 50,
                "closed_upper_bound": true
            }
        }
    ],
    "parameters": [
        {
            "name": "beta",
            "interval": {
                "lb": 0.08,
                "ub": 0.1
            },
            "label": "all"
        },
        {
            "name": "gamma",
            "interval": {
                "lb": 0.02,
                "ub": 0.03
            },
            "label": "all"
        },
        {
            "name": "S0",
            "interval": {
                "lb": 0.99,
                "ub": 0.99
            },
            "label": "all"
        },
        {
            "name": "I0",
            "interval": {
                "lb": 0.01,
                "ub": 0.01
            },
            "label": "all"
        },
        {
            "name": "R0",
            "interval": {
                "lb": 0,
                "ub": 0
            },
            "label": "all"
        }
    ],
    "structure_parameters": [
        {
            "name": "schedules",
            "schedules": [
                {
                    "timepoints": [
                        0,
                        10,
                        20,
                        30,
                        40,
                        50,
                        60,
                        70,
                        80,
                        90,
                        100
                    ]
                }
            ]
        }
    ],
    "config": {
        "use_compartmental_constraints": true,
        "normalization_constant": 1,
        "tolerance": 0.02
    }
}
```

## **Running `funman`**

There are multiple ways to run `funman` on a model and request pair. We recommend running `funman` in Docker, due to some complex dependencies on dReal (and its dependency on ibex).  `funman` has a Makefile that supports building three Docker use cases: 1) run a development container that mounts the source code, 2) run a container with a jupyter server, and 3) run a container with uvicorn serving a REST API. Examples of running each of these cases are illustrated by the tests (`test/test_api.py`, and `test/test_terarium.py`).  It is also possible to pull a pre-generated image that will run the API, as described in `terarium/README.md`.

# **Use cases**

The documentation at [https://siftech.github.io/funman/use_cases.html](https://siftech.github.io/funman/use_cases.html) describes several epidemiology use cases for `funman`.  These use cases are also implemented by tests in [tests/test_use_cases.py](https://github.com/siftech/funman/blob/v1.8.0-rc/test/test_use_cases.py)




# Development Setup

## Code layout

- .config: linting configuration
- .devcontainer: dev container configuration
- .github/workflows: workflow configuration (create docker images)
- artifacts: analysis artifacts
- auxiliary_packages:
  - funman_benchmarks: tools to benchmark configurations of `funman`  
  - funman_demo: tools related to specific demos
  - funman_dreal: extensions to pysmt configuration that include dreal  
  - funman_plot: plotting helpers
  - ibex_tests: ibex usage testing
  - pde2petri: convert PDE models to petri nets
- docker: docker files for building containers
- docs: sphinx/read-the-docs static content and configuration
- notebooks: usage notebooks
- notes: development notes
- resources: models and requests 
- scratch: old demos and prototyping
- src/funman: 
  - api: REST API
  - config.py: flags for request `configuration` dictionary
  - constants.py: constants 
  - funman.py: main entrypoint
  - __init__.py: global imports
  - model: input model representations
  - representation: internal and output representations
  - scenario: scenario definitions
  - search: search methods to generate parameter spaces
  - server: API server
  - translate: methods for encoding to SMT 
  - utils: utilities
  - _version.py: current version
- terarium: scripts and examples for terarium integration
- test: unit tests
- tools: helper scripts 
- .gitignore
- .pre-commit-config.yaml: precommit hook config
- .pylintrc: pylint config
- LICENSE
- Makefile
- Makefile.uncharted: Uncharted's custom make
- README.md
- pyproject.toml: project configuration
- requirements-dev-extras.txt: extra development dependencies
- requirements-dev.txt: development dependencies
- setup.py: pip install configuration


### Pre-commit hooks
`funman` has a set of pre-commit hooks to help with code uniformity. These hooks
will share the same rules as any automated testing so it is recommended to
install the hooks locally to ease the development process.

To use the pre-commit hooks you with need the tools listed in
`requirements-dev.txt`. These should be installed in the same environment where
you git tooling operates.
```bash
pip install -r requirements-dev.txt
```

Once install you should be able to run the following from the root of the repo:
```bash
make install-pre-commit-hooks
```

Once installed you should begin to receive isort/black feedback when
committing. These should not alter the code during a commit but instead just
fail and prevent a commit if the code does not conform to the specification.

To autoformat the entire repo you can use:
```bash
make format
```

### Code coverage
Pytest is configured to generate code coverage, and requires the `pytest-cov`
package to be installed.  The `pytest-cov` package is included in the
`requirements-dev.txt` (see above) and will be installed with the other dev
dependencies.  The code coverage will be displayed in the pytest output (i.e.,
`term`) and saved to the `coverage.xml` file.  The `Coverage Gutters` VSCode
plugin will use the `coverage.xml` to display code coverage highlighting over
the source files.

### Development Setup: Docker dev container
`funman` provides tooling to build a Docker image that can be used as a
development container. The image builds on either arm64 or amd64 architectures.

The dev container itself is meant to be ephemeral. The `launch-dev-container`
command will delete the existing dev container if an newer image has been made
available in the local cache. Any data that is meant to be retained from the
dev-container must be kept in one of the mounted volumes.

The dev container supports editing and rebuilding of dreal4 as well. This
requires that a dreal4 repository is cloned as a sibling to the `funman`
directory (../dreal4). So long as that directory is present, the next time the
funman-dev container is created will also result in a bind mount of the dreal4
directory to the container.

# Build the image:
```bash
# For building with your local arch
make build
```
```bash
# For building to a target arch
TARGET_ARCH=arm64 make build
```

# Launch the dev container:
```bash
make launch-dev-container
```

# If building a local dreal4 instead of the built-in version:
```bash
# from within the container
update-dreal
```

### (DEPRECATED) Development Setup: Ubuntu 20.04
```bash
# install python 3.9
sudo apt install python3.9 python3.9-dev
# install dev dependencies
sudo apt install make
pip install --user pipenv
# install pygraphviz dependencies
sudo apt install graphviz libgraphviz-dev pkg-config
# Initialize development environment
make setup-dev-env
```

### (DEPRECATED) Development Setup: OSX M1

```bash
# install python 3.9
brew install python@3.9 
# install dev dependencies
brew install make
pip3 install --user pipenv
# install pygraphviz dependencies
brew install graphviz pkg-config
# install z3
brew install z3
# install miniconda
brew install miniforge
# Initialize development environment
make setup-conda-dev-env
```

#### **Z3 issue**

On the M1, installing with conda gets pysmt with z3 for the wrong architecture. To fix this, if it happens, replace the `z3lib.dylib` in your virtual environment (in my case this was `.venv/lib/python3.9/site-packages/z3/lib/libz3.dylib`) with a symbolic link to the library that you get from your brew install of z3.  For example

    ln -s /opt/homebrew/Cellar/z3/4.11.0/lib/libz3.dylib ~/projects/askem/funman/.venv/lib/python3.9/site-packages/z3/lib/

---
#### **Pipenv issue and conda**

When I (rpg) tried to set up the environment with only pipenv (`make setup-dev-env`), it didn't work because `pip` tried to build the pygraphviz wheel and when it did, it used the Xcode compiler, which did not work with the version of graphviz I had installed with brew.

Suggest dealing with this by using `setup-CONDA-dev-env` [caps for emphasis] instead.

---
#### **Error during setup: "Could not find a version that matches"**
Try updating pipenv: `pip install pipenv --upgrade`

# Generating docs
```bash
pipenv run pip install sphinx sphinx_rtd_theme matplotlib

# Needed only if the gh-pages branch is not at origin
make init-pages 

# Run sphinx and pyreverse on source, generate docs/
make docs 

# push docs/ to origin
make deploy-pages 
```

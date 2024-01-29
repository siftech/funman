# funman: Functional Model Analysis tool

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

Plotting the points generated while creating the parameter space, will result in the following trajectories.

![Funman creates multiple trajectories for the SIR model](https://github.com/siftech/funman/blob/v1.8.0-rc/fig/trajectories.png?raw=true)

Funman supports a number of model formats, which correspond to the classes in `funman.models`:

- [GeneratedPetriNet](#funman.model.generated_models.petrinet.Model): AMR model for petri nets generated 
- [GeneratedRegNet](#funman.model.generated_models.regnet.Model): AMR model for regulatory networks
- [RegnetModel](#funman.model.regnet.RegnetModel): MIRA model for regulatory networks
- [PetrinetModel](#funman.model.petrinet.PetrinetModel): MIRA model for petri nets
- [BilayerModel](#funman.model.bilayer.BilayerModel): ASKE model for bilayer networks

Requests correspond to the class `funman.server.query.FunmanWorkRequest` and specify the following keys:
- [query](#funman.model.query): a soft constraint that a model must satisfy (legacy support, deprecated)
- [constraints](#funman.representation.constraint): a list of hard or soft constraints that a model must satisfy
- parameters: a list of bounded parameters that funman will either synthesize ("label": "all") or satsify ("label": "any").  Funman will check  "all" values within the parameter bounds or if "any" within the bounds permit the model to satisfy the query and constraints.
- [config](#funman.config): A dictionary of funman configuration options. 
 label regions of the parameter space as satisfying the query and constraints, if synthesized, or find any legal value if asked to satsify.  
- [structure_parameters](#funman.representation.parameter): parameters shaping the way that funman structures its analysis.  Funman requires that either the `num_steps` and `step_size` parameters are specified, or the `schedules` parameter is specified.  If all are omitted, then funman defaults to checking one unit-sized step.

## **Running funman**

There are multiple ways to run Funman on a model and request pair. We recommend running funman in Docker, due to some complex dependencies on dReal (and its dependency on ibex).  Funman has a Makefile that supports building three Docker use cases: 1) run a development container that mounts the source code, 2) run a container with a jupyter server, and 3) run a container with uvicorn serving a REST API. Examples of running each of these cases are illustrated by the tests (`test/test_api.py`, and `test/test_terarium.py`).  It is also possible to pull a pre-generated image that will run the API, as described in `terarium/README.md`.

## **Use cases**
### **Compare Bilayer Model to Simulator**:

This use case involves the simulator and FUNMAN reasoning about the CHIME
SIR bilayer model.  See test `test_use_case_bilayer_consistency` in `test/test_use_cases.py`.

It first uses a SimulationScenario to execute the input simulator
function and evaluate the input query function using the simulation results.
In the example below this results in the run_CHIME_SIR simulator function and
evaluating whether or not the number of infected crosses the provided threshold with a custom QueryFunction referencing the `does_not_cross_threshold` function.

It then constructs an instance of the ConsistencyScenario class to evaluate whether a BilayerModel will satisfy the given query. The query asks whether the
number of infected at any time point exceeds a specified threshold.

Once each of these steps is executed the results are compared. The test will
succeed if the SimulatorScenario and ConsistencyScenario agree on the response to the query.

```python
    def compare_against_CHIME_Sim(
        self, bilayer_path, init_values, infected_threshold
    ):
        # query the simulator
        def does_not_cross_threshold(sim_results):
            i = sim_results[2]
            return all(i_t <= infected_threshold for i_t in i)

        query = QueryLE("I", infected_threshold)

        funman = Funman()

        sim_result: SimulationScenarioResult = funman.solve(
            SimulationScenario(
                model=SimulatorModel(run_CHIME_SIR),
                query=QueryFunction(does_not_cross_threshold),
            )
        )

        consistency_result: ConsistencyScenarioResult = funman.solve(
            ConsistencyScenario(
                model=BilayerModel(
                    BilayerDynamics.from_json(bilayer_path),
                    init_values=init_values,
                ),
                query=query,
            )
        )

        # assert the both queries returned the same result
        return sim_result.query_satisfied == consistency_result.query_satisfied

    def test_use_case_bilayer_consistency(self):
        """
        This test compares a BilayerModel against a SimulatorModel to
        determine whether their response to a query is identical.
        """
        bilayer_path = os.path.join(
            RESOURCES, "bilayer", "CHIME_SIR_dynamics_BiLayer.json"
        )
        infected_threshold = 130
        init_values = {"S": 9998, "I": 1, "R": 1}
        assert self.compare_against_CHIME_Sim(
            bilayer_path, init_values, infected_threshold
        )
```

### **Parameter Synthesis**

See tests `test_use_case_simple_parameter_synthesis` and `test_use_case_bilayer_parameter_synthesis` in  `test/test_use_cases.py`.

The base set of types used during Parameter Synthesis include:

- a list of Parameters representing variables to be assigned
- a Model to be encoded as an SMTLib formula 
- a Scenario container representing a set of parameters and model
- a SearchConfig to configure search behavior
- the Funman interface that runs analysis using scenarios and configuration data

In the following example two parameters, x and y, are constructed. A model is 
also constructed that says 0.0 < x < 5.0 and 10.0 < y < 12.0. These parameters
and model are used to define a scenario that will use BoxSearch to synthesize
the parameters. The Funman interface and a search configuration are also 
defined. All that remains is to have Funman solve the scenario using the defined
configuration.

```python
def test_use_case_simple_parameter_synthesis(self):
        x = Symbol("x", REAL)
        y = Symbol("y", REAL)

        formula = And(
            LE(x, Real(5.0)),
            GE(x, Real(0.0)),
            LE(y, Real(12.0)),
            GE(y, Real(10.0)),
        )

        funman = Funman()
        result: ParameterSynthesisScenarioResult = funman.solve(
            ParameterSynthesisScenario(
                [
                    Parameter(name="x", symbol=x),
                    Parameter(name="y", symbol=y),
                ],
                EncodedModel(formula),
            )
        )
        assert result
```

As an additional parameter synthesis example, the following test case
demonstrates how to perform parameter synthesis for a bilayer model.  The
configuration differs from the example above by introducing bilayer-specific
constraints on the initial conditions (`init_values` assignments), parameter
bounds (`parameter_bounds` intervals) and a model query.

```python
    def test_use_case_bilayer_parameter_synthesis(self):
        bilayer_path = os.path.join(
            RESOURCES, "bilayer", "CHIME_SIR_dynamics_BiLayer.json"
        )
        infected_threshold = 3
        init_values = {"S": 9998, "I": 1, "R": 1}

        lb = 0.000067 * (1 - 0.5)
        ub = 0.000067 * (1 + 0.5)

        funman = Funman()
        result: ParameterSynthesisScenarioResult = funman.solve(
            ParameterSynthesisScenario(
                parameters=[Parameter(name="beta", lb=lb, ub=ub)],
                model=BilayerModel(
                    BilayerDynamics.from_json(bilayer_path),
                    init_values=init_values,
                    parameter_bounds={
                        "beta": [lb, ub],
                        "gamma": [1.0 / 14.0, 1.0 / 14.0],
                    },
                ),
                query=QueryLE("I", infected_threshold),
            ),
            config=SearchConfig(tolerance=1e-8),
        )
        assert len(result.parameter_space.true_boxes) > 0 
        assert len(result.parameter_space.false_boxes) > 0 
```
---

## Development Setup

### Pre-commit hooks
FUNMAN has a set of pre-commit hooks to help with code uniformity. These hooks
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
FUNMAN provides tooling to build a Docker image that can be used as a
development container. The image builds on either arm64 or amd64 architectures.

The dev container itself is meant to be ephemeral. The `launch-dev-container`
command will delete the existing dev container if an newer image has been made
available in the local cache. Any data that is meant to be retained from the
dev-container must be kept in one of the mounted volumes.

The dev container supports editing and rebuilding of dreal4 as well. This
requires that a dreal4 repository is cloned as a sibling to the funman
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


from pathlib import Path
from funman.api.run import Runner
import os

def main():
    # Setup Paths
    
    RESOURCES = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../resources"
    )
    EXAMPLE_DIR = os.path.join(RESOURCES, "amr", "petrinet", "amr-examples")
    MODEL_PATH = os.path.join(EXAMPLE_DIR, "sir.json")
    REQUEST_PATH = os.path.join(EXAMPLE_DIR,  "sir_request1.json")

    request_dict = {
        # "query": {
        #     "variable": "I",
        #     "ub": 300
        # },
    "constraints": [
        # { 
        #     "name": "I_bounds",
        #     "variable" : "I", 
        #     "bounds": {"lb":0.85, "ub":2},
        #     "timepoints": {"lb":1, "ub":2}
        # },
        # { 
        #     "name": "R_bounds",
        #     "variable" : "R", 
        #     "bounds": {"lb":0, "ub":1},
        #     "timepoints": {"lb":0, "ub":2}
        # },
        #   { 
        #   "name": "S_bounds",
        #   "variable" : "S", 
        #   "bounds": {"lb":980, "ub":1000},
        #   "timepoints": {"lb":4, "ub":5}
        #  }
    ],
    "parameters": [
        {
        "name": "beta",
        "lb": 2.6e-7,
        "ub": 2.8e-7,
        "label": "all"
        },
        {
        "name": "gamma",
        "lb": 0.1,
        "ub": 0.18,
        "label": "all"
        },
        {
        "name": "S0",
        "lb": 1000,
        "ub": 1000,
        "label": "any"
        },
        {
        "name": "I0",
        "lb": 1,
        "ub": 1,
        "label": "any"
        },
        {
        "name": "R0",
        "lb": 0,
        "ub": 0,
        "label": "any"
        }
    ],
    "structure_parameters": [
        {
        "name": "num_steps",
        "lb": 10,
        "ub": 20,
        "label": "all"
        },
        {
        "name": "step_size",
        "lb": 1,
        "ub": 1,
        "label": "all"
        }
    ],
    "config": {
    "normalize": False,
    "tolerance": 1e-3,
    # "simplify_query": False,
    "use_compartmental_constraints" : False,
    # "profile": True
    # "save_smtlib" : True,
    # "substitute_subformulas": False
    "taylor_series_order": None
    }
    }
    
    # Use request_dict
    results = Runner().run(MODEL_PATH, request_dict, description="Basic SIR with simple request", case_out_dir="./out")

# Use request file
# results = Runner().run(MODEL_PATH, REQUEST_PATH, description="Basic SIR with simple request", case_out_dir="./out")

if __name__ == "__main__":
    main()
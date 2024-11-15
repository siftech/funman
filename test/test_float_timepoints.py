import logging
import os
import sys
import unittest
from pathlib import Path

import pandas as pd
import sympy
from matplotlib import pyplot as plt

from funman.api.run import Runner
from funman.server.query import FunmanWorkRequest
from funman.utils.sympy_utils import SympyBoundedSubstituter, to_sympy

FILE_DIRECTORY = Path(__file__).resolve().parent
API_BASE_PATH = FILE_DIRECTORY / ".."
RESOURCES = API_BASE_PATH / "resources"
EXAMPLE_DIR = os.path.join(
    RESOURCES, "amr", "petrinet", "monthly-demo", "2024-09"
)
BASE_SIR_REQUEST_PATH = os.path.join(EXAMPLE_DIR, "sir_request1.json")
BASE_SIR_MODEL_PATH = os.path.join(EXAMPLE_DIR, "sir.json")

BASE_SIRHD_REQUEST_PATH = os.path.join(EXAMPLE_DIR, "sirhd_request.json")
BASE_SIRHD_MODEL_PATH = os.path.join(EXAMPLE_DIR, "sirhd.json")


class TestFloatTimepoints(unittest.TestCase):
    l = logging.Logger(__name__)

    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        self.l.level = logging.getLogger().level
        self.l.handlers.append(logging.StreamHandler(sys.stdout))
    
    def test_float_timepoints(self):
    
        with open(BASE_SIR_REQUEST_PATH, "r") as f:
            base_request = FunmanWorkRequest.model_validate_json(f.read())
        base_request.config.verbosity = 
        base_request.structure_parameters[0].schedules[0].timepoints = [
            0.0,
            1e-2,
            0.1,
            1,
            2
        ]

        runner = Runner()
        base_result = runner.run(BASE_SIR_MODEL_PATH, BASE_SIR_REQUEST_PATH)

        assert (
            base_result
        ), f"Could not generate a result for model: [{BASE_SIR_MODEL_PATH}], request: [{BASE_SIR_REQUEST_PATH}]"





   
if __name__ == "__main__":
    unittest.main()

import logging
import os
import sys
import unittest
from pathlib import Path

import pandas as pd
import sympy
from matplotlib import pyplot as plt

from funman.api.run import Runner
from funman.constants import MODE_ODEINT, MODE_SMT
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

PS_SIR_REQUEST_PATH = os.path.join(
    RESOURCES, "amr", "petrinet", "amr-examples", "sir_request1.json"
)
PS_SIR_MODEL_PATH = os.path.join(
    RESOURCES, "amr", "petrinet", "amr-examples", "sir.json"
)


class TestFloatTimepoints(unittest.TestCase):
    l = logging.Logger(__name__)

    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        self.l.level = logging.getLogger().level
        self.l.handlers.append(logging.StreamHandler(sys.stdout))

    def setup_common(self, request_file=BASE_SIR_REQUEST_PATH):
        with open(request_file, "r") as f:
            request = FunmanWorkRequest.model_validate_json(f.read())
        request.config.verbosity = 5
        request.structure_parameters[0].schedules[0].timepoints = [
            0.0,
            1e-2,
            0.1,
            1,
            2,
        ]
        return request

    def setup_odeint(self):
        request = self.setup_common()
        request.config.mode = MODE_ODEINT
        return request

    def setup_smt(self, request_file=BASE_SIR_REQUEST_PATH):
        request = self.setup_common(request_file=request_file)
        request.config.mode = MODE_SMT
        return request

    def test_float_timepoints_odeint(self):
        base_request = self.setup_odeint()
        runner = Runner()
        base_result = runner.run(
            BASE_SIR_MODEL_PATH, base_request.model_dump()
        )
        assert (
            base_result
        ), f"Could not generate a result for model: [{BASE_SIR_MODEL_PATH}], request: [{BASE_SIR_REQUEST_PATH}]"

    def test_float_timepoints_smt(self):
        base_request = self.setup_smt()
        runner = Runner()
        base_result = runner.run(
            BASE_SIR_MODEL_PATH, base_request.model_dump()
        )
        df = base_result.dataframe(base_result.points())
        assert (
            base_result
        ), f"Could not generate a result for model: [{BASE_SIR_MODEL_PATH}], request: [{BASE_SIR_REQUEST_PATH}]"

    def test_float_timepoints_ps_smt(self):
        base_request = self.setup_smt(request_file=PS_SIR_REQUEST_PATH)
        runner = Runner()
        base_result = runner.run(PS_SIR_MODEL_PATH, base_request.model_dump())
        df = base_result.dataframe(base_result.points())
        assert (
            base_result
        ), f"Could not generate a result for model: [{BASE_SIR_MODEL_PATH}], request: [{BASE_SIR_REQUEST_PATH}]"


if __name__ == "__main__":
    unittest.main()

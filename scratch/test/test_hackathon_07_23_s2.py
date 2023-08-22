import json
import os
import unittest
from time import sleep
from typing import Tuple

import matplotlib.pyplot as plt
import pydantic
from funman_demo.parameter_space_plotter import ParameterSpacePlotter

from funman.api.api import _wrap_with_internal_model
from funman.api.settings import Settings
from funman.funman import FUNMANConfig
from funman.model.generated_models.petrinet import Model as GeneratedPetrinet
from funman.model.generated_models.regnet import Model as GeneratedRegnet
from funman.server.query import FunmanWorkRequest, FunmanWorkUnit
from funman.server.storage import Storage
from funman.server.worker import FunmanWorker

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)

out_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "out", "hackathon"
)


models = {GeneratedPetrinet, GeneratedRegnet}

AMR_EXAMPLES_DIR = os.path.join(RESOURCES, "amr")
AMR_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "amr-examples")
AMR_REGNET_DIR = os.path.join(AMR_EXAMPLES_DIR, "regnet", "amr-examples")

SKEMA_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "skema")
SKEMA_REGNET_DIR = os.path.join(AMR_EXAMPLES_DIR, "regnet", "skema")

MIRA_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "mira")


cases = [
    # 2. b. Wild Type variant and no vaccination
    (
        os.path.join(MIRA_PETRI_DIR, "models", "scenario2_a.json"),
        os.path.join(MIRA_PETRI_DIR, "requests", "request2_b.json"),
    ),
    # 1. b. 50 days delay
    # (
    #     os.path.join(MIRA_PETRI_DIR, "models", "scenario1_a.json"),
    #     os.path.join(MIRA_PETRI_DIR, "requests", "request1_a_50_days.json"),
    # ),
    # 1. b. 100 days delay
    # (
    #     os.path.join(MIRA_PETRI_DIR, "models", "scenario1_a.json"),
    #     os.path.join(MIRA_PETRI_DIR, "requests", "request1_a_100_days.json"),
    # ),
    # 1. b. all days delay possible
    # (
    #     os.path.join(MIRA_PETRI_DIR, "models", "scenario1_a.json"),
    #     os.path.join(MIRA_PETRI_DIR, "requests", "request1_a_all_days.json"),
    # ),
]

if not os.path.exists(out_dir):
    os.mkdir(out_dir)


class TestModels(unittest.TestCase):
    def test_models(self):
        self.settings = Settings()
        self.settings.data_path = out_dir
        self._storage = Storage()
        self._worker = FunmanWorker(self._storage)
        self._storage.start(self.settings.data_path)
        self._worker.start()

        for case in cases:
            self.run_instance(case)

        self._worker.stop()
        self._storage.stop()

    def get_model(self, model_file: str):
        for model in models:
            try:
                m = _wrap_with_internal_model(
                    pydantic.parse_file_as(model, model_file)
                )
                return m
            except Exception as e:
                pass
        raise Exception(f"Could not determine the Model type of {model_file}")

    def run_instance(self, case: Tuple[str, str]):
        (model_file, request_file) = case

        model = self.get_model(model_file)
        request = pydantic.parse_file_as(FunmanWorkRequest, request_file)

        work_unit: FunmanWorkUnit = self._worker.enqueue_work(
            model=model, request=request
        )
        sleep(2)  # need to sleep until worker has a chance to start working
        while True:
            if self._worker.is_processing_id(work_unit.id):
                results = self._worker.get_results(work_unit.id)
                with open(f"{out_dir}/{work_unit.id}.json", "w") as f:
                    f.write(json.dumps(results.json()))
                # ParameterSpacePlotter(
                #     results.parameter_space, plot_points=True
                # ).plot(show=False)
                # plt.savefig(f"{out_dir}/{model.__module__}.png")
                # plt.close()
                sleep(10)
            else:
                results = self._worker.get_results(work_unit.id)
                break

        # ParameterSpacePlotter(results.parameter_space, plot_points=True).plot(
        #     show=False
        # )
        # plt.savefig(f"{out_dir}/{model.__module__}.png")
        # plt.close()

        assert results

        assert True


if __name__ == "__main__":
    unittest.main()

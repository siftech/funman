import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep

from fastapi.testclient import TestClient

from funman.api.api import app, settings
from funman.representation.representation import ParameterSpace
from funman.server.query import FunmanResults, FunmanWorkUnit

FILE_DIRECTORY = Path(__file__).resolve().parent
API_BASE_PATH = FILE_DIRECTORY / ".."
RESOURCES = API_BASE_PATH / "resources"

TEST_OUT = FILE_DIRECTORY / "out"
TEST_OUT.mkdir(parents=True, exist_ok=True)

TEST_API_TOKEN = "funman-test-api-token"
settings.funman_api_token = TEST_API_TOKEN


class TestDeterminism(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = TemporaryDirectory(prefix=f"{cls.__name__}_")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def setUp(self):
        self.test_dir = Path(self._tmpdir.name) / self._testMethodName
        self.test_dir.mkdir()
        settings.data_path = str(self.test_dir)

    def wait_for_done(self, client, id, wait_time=1.0, steps=10):
        while True:
            sleep(wait_time)
            response = client.get(
                f"/queries/{id}", headers={"token": f"{TEST_API_TOKEN}"}
            )
            assert response.status_code == 200
            data = FunmanResults.parse_raw(response.content.decode())
            if data.done:
                return data
            steps -= 1
            if steps <= 0:
                response = client.get(
                    f"/queries/{id}/halt",
                    headers={"token": f"{TEST_API_TOKEN}"},
                )
                return data

    def check_consistency_success(self, parameter_space: ParameterSpace):
        assert parameter_space is not None
        assert len(parameter_space.true_boxes) == 0
        assert len(parameter_space.false_boxes) == 0
        assert len(parameter_space.true_points) == 1
        assert len(parameter_space.false_points) == 0

    def check_parameter_synthesis_success(
        self, parameter_space: ParameterSpace
    ):
        assert parameter_space is not None
        assert len(parameter_space.true_boxes) > 0

    def print_debug_header(self, text):
        print("=" * 80)
        print(text)
        print("=" * 80)

    def check_determinism(self, model, request):
        with TestClient(app) as client:
            self.print_debug_header("FIRST")
            response = client.post(
                "/queries",
                json={"model": model, "request": request},
                headers={"token": f"{TEST_API_TOKEN}"},
            )
            assert response.status_code == 200
            work_unit = FunmanWorkUnit.parse_raw(response.content.decode())
            data1 = self.wait_for_done(client, work_unit.id)

            self.print_debug_header("SECOND")
            response = client.post(
                "/queries",
                json={"model": model, "request": request},
                headers={"token": f"{TEST_API_TOKEN}"},
            )
            assert response.status_code == 200
            work_unit = FunmanWorkUnit.parse_raw(response.content.decode())
            data2 = self.wait_for_done(client, work_unit.id)

            self.print_debug_header("THIRD")
            response = client.post(
                "/queries",
                json={"model": model, "request": request},
                headers={"token": f"{TEST_API_TOKEN}"},
            )
            assert response.status_code == 200
            work_unit = FunmanWorkUnit.parse_raw(response.content.decode())
            data3 = self.wait_for_done(client, work_unit.id)

            assert sorted(data1.parameter_space) == sorted(
                data2.parameter_space
            )
            assert sorted(data2.parameter_space) == sorted(
                data3.parameter_space
            )
            assert sorted(data3.parameter_space) == sorted(
                data1.parameter_space
            )

    def get_amr_petrinet_sir(self):
        EXAMPLE_DIR = RESOURCES / "amr" / "petrinet" / "amr-examples"
        MODEL_PATH = EXAMPLE_DIR / "sir.json"
        REQUEST_PATH = EXAMPLE_DIR / "sir_request1b.json"

        model = json.loads(MODEL_PATH.read_bytes())
        request = json.loads(REQUEST_PATH.read_bytes())
        return model, request

    @unittest.skip("tmp")
    def test_without_mcts(self):
        model, request = self.get_amr_petrinet_sir()
        request["config"]["dreal_mcts"] = False
        request["config"]["dreal_log_level"] = "debug"
        self.check_determinism(model, request)

    # @unittest.skip("Disable until mcts has an assignable seed")
    def test_with_mcts(self):
        model, request = self.get_amr_petrinet_sir()
        request["config"]["dreal_mcts"] = True
        request["config"]["dreal_log_level"] = "info"
        self.check_determinism(model, request)

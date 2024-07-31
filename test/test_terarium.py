import json
import os
import unittest
from pathlib import Path
from time import sleep

import httpx
from fastapi.testclient import TestClient

from funman.api.api import app

# Read in the model associated with this example
RESOURCES_PREFIX = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)
TEST_JSON = json.loads(
    Path(f"{RESOURCES_PREFIX}/terarium-tests.json").read_bytes()
)


class TestTerarium(unittest.TestCase):
    def get_test_by_name(self, name):
        try:
            test_case = next(
                iter(t for t in TEST_JSON["tests"] if t["name"] == name)
            )
        except StopIteration as e:
            print(f"Could not find {name} model in tests")
            raise e
        return test_case

    def test01_terarium(self):
        tests = TEST_JSON["tests"]
        for test in tests:
            if not test["regression"]:
                continue

            (
                name,
                model,
                request,
                expected_outcome,
                regression,
            ) = self.get_model_and_request(test)

            if not regression:
                continue

            print(f"Testing: {name}")

            with self.subTest(name):
                with TestClient(app) as client:
                    # run the defined test
                    self.subtest_terarium(
                        client, name, model, request, expected_outcome
                    )

    def get_model_and_request(self, test):
        name = test["name"]

        # Read in the model dict
        model = json.loads(
            Path(f'{RESOURCES_PREFIX}/{test["model-path"]}').read_bytes()
        )

        # model file has model and request pair inside
        if "model" in model and "request" in model:
            request = model["request"]
            if "petrinet" in model["model"]:
                model = model["model"]["petrinet"]
            else:
                model = model["model"]

        # Either read in the request json or default to an empty dict
        elif test["request-path"] is None:
            request = {}
        else:
            request = json.loads(
                Path(f'{RESOURCES_PREFIX}/{test["request-path"]}').read_bytes()
            )

        expected_outcome = test["expected-outcome"]
        regression = test["regression"]
        return name, model, request, expected_outcome, regression

    def subtest_terarium(self, client, name, model, request, expected_outcome):
        results = self.post_query_and_wait_until_done(client, model, request)
        assert (
            "parameter_space" in results
        ), "Results does not contain a 'parameter_space' field"
        ps = results["parameter_space"]
        assert ps is not None, "ParameterSpace is None"

        num_true = len(ps.get("true_boxes", []))
        assert (
            num_true == expected_outcome["true-boxes"]
        ), f"Terarium Test '{name}' should have {expected_outcome['true-boxes']}, but has {num_true}  true boxes"

    def POST(self, client: TestClient, url: str, json: dict, *, expect=200):
        """
        Make a POST request through the TestClient
        """
        response = client.post(url, json=json)
        assert (
            response.status_code == expect
        ), f"Unexpected status code {response.status_code}. Expected {expect}"
        return response

    def GET(self, client: TestClient, url: str, *, expect=200):
        """
        Make a GET request through the TestClient
        """
        response = client.get(url)
        assert (
            response.status_code == expect
        ), f"Unexpected status code {response.status_code}. Expected {expect}"
        return response

    def decode_response_to_dict(self, response: httpx.Response) -> dict:
        """
        Convert the response from the TestClient to a dictionary
        """
        return json.loads(response.content.decode())

    def poll_until_done(
        self, client: TestClient, uuid: str, sleep_step=1.0, max_steps=40
    ):
        """
        Helper function to poll the status of the request associated
        with the provided UUID.
        """
        while True:
            # Sleep for wait_time
            sleep(sleep_step)
            # Check the status of the query
            response = client.get(f"/api/queries/{uuid}")
            # Ensure no error status code
            assert (
                response.status_code == 200
            ), f"Response code was not 200: {response.status_code}"
            # Get the results
            results = self.decode_response_to_dict(response)
            assert (
                results.get("error", False) is False
            ), f"Request {uuid} errored during processing"
            # Return if processing is done
            if results.get("done", False):
                return results
            # Track steps
            max_steps -= 1
            assert max_steps > 0

    def post_query(
        self, client: TestClient, model: dict, request: dict
    ) -> str:
        """
        Make a POST request to /api/queries through the TestClient.
        - model: The model to query
        - request: The FunmanWorkRequest

        The response to a query returns a FunmanWorkUnit with fields:
        - id: The UUID assign to the queued FunmanWorkRequest
        - model: A copy of the submitted model
        - request: A copy of the submitted request
        """
        work_unit = self.decode_response_to_dict(
            self.POST(
                client, "/api/queries", {"model": model, "request": request}
            )
        )
        # Extract the UUID
        return work_unit["id"]

    def get_status(self, client: TestClient, uuid: str) -> dict:
        """
        Make a GET request to /api/queries/{uuid} through the TestClient.

        The response to a query returns a FunmanResults with fields:
        - id: The UUID assign to the FunmanWorkRequest
        - model: A copy of the submitted model
        - request: A copy of the submitted request
        - done: A boolean flag for if the request has finished processing
        - error: A boolean flag for if the request errored
        - parameter_space: The current ParameterSpace if one exists
        """
        return self.decode_response_to_dict(
            self.GET(client, f"/api/queries/{uuid}")
        )

    def post_query_and_wait_until_done(
        self,
        client: TestClient,
        model: dict,
        request: dict,
        *,
        expect_error: bool = False,
    ) -> dict:
        """
        Make a query with a provided model and request and wait until it is done processing
        (by polling the status of the request until it reports it is done).
        - model: One of the supported models
        - request: A request to funman
        """
        uuid = self.post_query(client, model, request)
        self.poll_until_done(client, uuid)
        results = self.get_status(client, uuid)

        progress = results["progress"].get("progress", 0.0)
        is_done_processing = results.get("done", False)
        error_occurred = results.get("error", False)

        if expect_error:
            # The results should indicate an error
            assert (
                error_occurred is True
            ), f"An unexpected success occured while processing request with id '{uuid}'"
        else:
            # The results should not indicate an error
            assert (
                error_occurred is False
            ), f"An unexpected error occured while processing request with id '{uuid}'"
            # Processing should be done
            assert (
                is_done_processing is True
            ), f"Expected work to be done for {uuid}"

        assert progress > 0.999999, "Progress was not at 100%"
        return results

    def test02_stress_test(self):
        # here is an example that starts to return 404s GET /api/queries/:id after successfully submitting through POST /api/queries
        # Earlier the GET request was returning, albeit seemingly never finished. Roughly the steps:
        # Submit this payload
        # Query via the ID, couldn't get result, seems to be running forever
        # Resubmit this payload
        # Start to get 404s
        with TestClient(app) as client:
            test = TEST_JSON["tests"][1]
            (
                name,
                model,
                request,
                expected_outcome,
                regression,
            ) = self.get_model_and_request(test)
            uuid = self.post_query(client, model, request)
            sleep(5)
            response = client.get(f"/api/queries/{uuid}")

            assert (
                response.status_code == 200
            ), f"Response code was not 200: {response.status_code}"

            test1 = TEST_JSON["tests"][2]
            (
                name1,
                model1,
                request1,
                expected_outcome,
                regression,
            ) = self.get_model_and_request(test1)

            uuid1 = self.post_query(client, model1, request1)
            sleep(5)
            response1 = client.get(f"/api/queries/{uuid1}")

            assert (
                response1.status_code == 200
            ), f"Response code was not 200: {response1.status_code}"

            # # Get the results
            # results = self.decode_response_to_dict(response)
            # assert (
            #     results.get("error", False) is False
            # ), f"Request {uuid} errored during processing"
            # # # Return if processing is done
            # # if results.get("done", False):
            # #     return results

            # results = self.get_status(client, uuid)

    def test03_stress_test(self):
        # Send a second request before the first request can finish
        # Motivated by bug where received 404 when work file wasn't created yet

        with TestClient(app) as client:
            test = self.get_test_by_name("sir")

            (
                name,
                model,
                request,
                expected_outcome,
                regression,
            ) = self.get_model_and_request(test)
            uuid = self.post_query(client, model, request)
            # sleep(1)
            response = client.get(f"/api/queries/{uuid}")

            assert (
                response.status_code == 200
            ), f"Response code was not 200: {response.status_code}"

            uuid1 = self.post_query(client, model, request)
            # sleep(1)
            response1 = client.get(f"/api/queries/{uuid1}")

            assert (
                response1.status_code == 200
            ), f"Response code was not 200: {response1.status_code}"

            response2 = client.get(f"/api/queries/{uuid}")

            assert (
                response2.status_code == 200
            ), f"Response code was not 200: {response2.status_code}"


if __name__ == "__main__":
    unittest.main()

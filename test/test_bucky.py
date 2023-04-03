import json
import logging
import os
import unittest

from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter

from funman import Funman
from funman.funman import FUNMANConfig
from funman.model import QueryLE
from funman.model.bilayer import BilayerDynamics, BilayerModel
from funman.representation.representation import Parameter
from funman.scenario import (
    ConsistencyScenario,
    ConsistencyScenarioResult,
    ParameterSynthesisScenario,
    ParameterSynthesisScenarioResult,
)
from funman.utils.handlers import ResultCombinedHandler

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)


class TestBucky(unittest.TestCase):
    def setup_use_case_bilayer_common(self):
        bilayer_path = os.path.join(
            RESOURCES, "bilayer", "Bucky_SEIIIRRD_BiLayer_v3.json"
        )
        with open(bilayer_path, "r") as f:
            bilayer_src = json.load(f)

        infected_threshold = 50
        init_values = {
            "S": 99.43,
            "E": 0.4,
            "I_asym": 0.1,
            "I_mild": 0.05,
            "I_crit": 0.02,
            "R": 0,
            "R_hosp": 0,
            "D": 0,
        }

        identical_parameters = [
            ["beta_1", "beta_2"],
            ["gamma_1", "gamma_2"],
        ]

        lb = 0.0
        ub = 1.0

        gamma = 0.456
        gamma_h = 0.3648
        delta_1 = 0.25
        delta_2 = 0.00969
        delta_3 = 0.00121125
        delta_4 = 0.0912
        sigma = 0.017
        theta = 0.1012

        model = BilayerModel(
            bilayer=BilayerDynamics(json_graph=bilayer_src),
            init_values=init_values,
            identical_parameters=identical_parameters,
            parameter_bounds={
                "beta_1": [lb, ub],
                "beta_2": [lb, ub],
                "gamma_1": [gamma, gamma],
                "gamma_2": [gamma, gamma],
                "gamma_h": [gamma_h, gamma_h],
                "delta_1": [delta_1, delta_1],
                "delta_2": [delta_2, delta_2],
                "delta_3": [delta_3, delta_3],
                "delta_4": [delta_4, delta_4],
                "sigma": [sigma, sigma],
                "theta": [theta, theta],
            },
        )

        query = QueryLE(variable="I_crit", ub=infected_threshold)

        return model, query

    def setup_use_case_bilayer_consistency(self):
        model, query = self.setup_use_case_bilayer_common()

        scenario = ConsistencyScenario(model=model, query=query)
        return scenario

    def setup_use_case_bilayer_parameter_synthesis(self):
        model, query = self.setup_use_case_bilayer_common()

        def make_parameter(name):
            [lb, ub] = model.parameter_bounds[name]
            return Parameter(name=name, lb=lb, ub=ub)

        scenario = ParameterSynthesisScenario(
            parameters=[
                make_parameter("beta_1"),
                make_parameter("beta_2"),
            ],
            model=model,
            query=query,
        )

        return scenario

    def test_use_case_bilayer_parameter_synthesis(self):
        scenario = self.setup_use_case_bilayer_parameter_synthesis()
        funman = Funman()
        config = FUNMANConfig(
            tolerance=1e-8,
            number_of_processes=1,
            log_level=logging.INFO,
        )
        # FIXME arguments with form _* do not get assigned when using pydantic
        config._handler = ResultCombinedHandler(
            [
                ResultCacheWriter(f"bucky_box_search.json"),
                RealtimeResultPlotter(
                    scenario.parameters,
                    plot_points=True,
                    title=f"Feasible Regions (beta)",
                    realtime_save_path=f"bucky_box_search.png",
                ),
            ]
        )

        result: ParameterSynthesisScenarioResult = funman.solve(
            scenario, config=config
        )
        assert len(result.parameter_space.true_boxes) > 0
        assert len(result.parameter_space.false_boxes) > 0

    def test_use_case_bilayer_consistency(self):
        scenario = self.setup_use_case_bilayer_consistency()

        funman = Funman()
        config = FUNMANConfig(
            max_steps=5,
            step_size=1,
            solver="dreal",
            log_level=logging.INFO,
        )

        # Show that region in parameter space is sat (i.e., there exists a true point)
        result_sat: ConsistencyScenarioResult = funman.solve(
            scenario, config=config
        )
        df = result_sat.dataframe()

        parameters = result_sat._parameters()

        # assert abs(df["I"][2] - 2.24) < 0.01
        # beta = result_sat._parameters()["beta"]
        # assert abs(beta - 0.00005) < 0.00001

        # # Show that region in parameter space is unsat/false
        # scenario.model.parameter_bounds["beta"] = [
        #     0.000067 * 1.5,
        #     0.000067 * 1.75,
        # ]
        # result_unsat: ConsistencyScenarioResult = funman.solve(scenario)
        # assert not result_unsat.consistent


if __name__ == "__main__":
    unittest.main()
import unittest

import matplotlib.pyplot as plt
from common import TestUnitTests
from pysmt.shortcuts import (
    GE,
    GT,
    LE,
    LT,
    REAL,
    TRUE,
    And,
    Equals,
    Minus,
    Or,
    Plus,
    Real,
    Symbol,
    Times,
)

from funman.funman import Funman, FUNMANConfig
from funman.model.bilayer import BilayerDynamics
from funman.model.query import QueryTrue


class TestS21BUnitTest(TestUnitTests):
    steps = 100
    step_size = 2
    expected_max_infected = 0.6
    test_threshold = 0.1
    expected_max_day = 47
    test_max_day_threshold = 25

    s2_models = [
        "Mosaphir_petri_to_bilayer",
        "UF_petri_to_bilayer",
        "Morrison_bilayer",
        "Skema_bilayer",
    ]

    def sidarthe_extra_1_1_d_2d(self, steps, init_values, step_size=1):
        return And(
            [
                self.make_bounds(steps, init_values, step_size=step_size),
            ]
        )

    def analyze_model(self, model_name: str):
        initial = self.initial_state_sidarthe()
        scenario = self.make_scenario(
            BilayerDynamics(
                json_graph=self.sidarthe_bilayer(self.models[model_name])
            ),
            initial,
            self.bounds_sidarthe(),
            [],
            self.steps,
            QueryTrue(),
            extra_constraints=self.sidarthe_extra_1_1_d_2d(
                self.steps, initial, self.step_size
            ),
        )
        config = FUNMANConfig(
            max_steps=self.steps,
            step_size=self.step_size,
            solver="dreal",
            initial_state_tolerance=0.0,
        )
        result_sat = Funman().solve(scenario, config=config)
        self.report(result_sat, name=model_name)
        max_infected, max_day = self.analyze_results(result_sat)

        return max_infected, max_day

    def analyze_results(self, result_sat):
        df = result_sat.dataframe()

        df["infected_states"] = df.apply(
            lambda x: sum([x["I"], x["D"], x["A"], x["R"], x["T"]]), axis=1
        )
        max_infected = df["infected_states"].max()
        max_day = df["infected_states"].idxmax()
        ax = df["infected_states"].plot(
            title=f"I+D+A+R+T by day (max: {max_infected}, day: {max_day})",
        )
        ax.set_xlabel("Day")
        ax.set_ylabel("I+D+A+R+T")
        try:
            plt.savefig(f"funman_s2_1_b_i_infected.png")
        except Exception as e:
            pass
        plt.clf()

        return max_infected, max_day

    def common_test_model(self, model_name: str):
        max_infected, max_day = self.analyze_model(model_name)
        assert (
            abs(max_infected - self.expected_max_infected)
            < self.test_threshold
        )
        assert (
            abs(max_day - self.expected_max_day) < self.test_max_day_threshold
        )

    def test_model_0(self):
        self.common_test_model(self.s2_models[0])

    @unittest.expectedFailure
    def test_model_1(self):
        self.common_test_model(self.s2_models[1])

    def test_model_2(self):
        self.common_test_model(self.s2_models[2])

    def test_model_3(self):
        self.common_test_model(self.s2_models[3])


if __name__ == "__main__":
    unittest.main()
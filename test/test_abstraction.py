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


class TestUseCases(unittest.TestCase):
    l = logging.Logger(__name__)

    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        self.l.level = logging.getLogger().level
        self.l.handlers.append(logging.StreamHandler(sys.stdout))

    def test_minimize_expression(self):
        tests = [
            {
                "input": ["S", "- S * I * beta/N"],
                "bound": "lb",
                "expected_output": "-I_ub*S_lb*beta_ub/N_lb",
            },
            {
                "input": ["S", "- S * I * beta/N"],
                "bound": "ub",
                "expected_output": "-I_lb*S_ub*beta_lb/N_ub",
            },
            {
                "input": ["S", "- S * I * beta"],
                "bound": "lb",
                "expected_output": "-I_ub*S_lb*beta_ub",
            },
            {
                "input": ["S", "- S * I * beta"],
                "bound": "ub",
                "expected_output": "-I_lb*S_ub*beta_lb",
            },
            {
                "input": ["I", "S * I * beta - I * gamma"],
                "bound": "lb",
                "expected_output": "I_lb*S_lb*beta_lb - I_lb*gamma_ub",
            },
            {
                "input": ["I", "S * I * beta - I * gamma"],
                "bound": "ub",
                "expected_output": "I_ub*S_ub*beta_ub - I_ub*gamma_lb",
            },
            {
                "input": ["R", "I * gamma"],
                "bound": "lb",
                "expected_output": "I_lb*gamma_lb",
            },
            {
                "input": ["I", "I * gamma"],
                "bound": "ub",
                "expected_output": "I_ub*gamma_ub",
            },
        ]

        str_symbols = ["S", "I", "R", "beta", "gamma", "N"]
        symbols = {s: sympy.Symbol(s) for s in str_symbols}
        bound_symbols = {
            sympy.Symbol(s): {
                "lb": sympy.Symbol(f"{s}_lb"),
                "ub": sympy.Symbol(f"{s}_ub"),
            }
            for s in str_symbols
        }
        substituter = SympyBoundedSubstituter(
            bound_symbols=bound_symbols, str_to_symbol=symbols
        )

        for test in tests:
            with self.subTest(f"{test['bound']}({test['input']})"):
                test_fn = (
                    substituter.minimize
                    if test["bound"] == "lb"
                    else substituter.maximize
                )
                var, expr = test["input"]
                test_output = test_fn(
                    [sympy.Symbol(var)], sympy.sympify(expr, symbols)
                )
                # self.l.debug(f"Minimized: [{infection_rate}], to get expression: [{test_output}]")
                assert (
                    str(test_output) == test["expected_output"]
                ), f"Failed to create the expected expression: [{test['expected_output']}], got [{test_output}]"

    # @unittest.skip(reason="tmp")
    def test_stratify(self):
        epsilon = 0.000001

        runner = Runner()
        base_result = runner.run(BASE_SIR_MODEL_PATH, BASE_SIR_REQUEST_PATH)

        assert (
            base_result
        ), f"Could not generate a result for model: [{BASE_SIR_MODEL_PATH}], request: [{BASE_SIR_REQUEST_PATH}]"

        (base_model, _) = runner.get_model(BASE_SIR_MODEL_PATH)

        # Stratify Base model
        stratified_model = base_model.stratify(
            "S",
            ["1", "2"],
            strata_parameters=["beta"],
            strata_transitions=[],
            self_strata_transition=False,
        )

        stratified_params = stratified_model.petrinet.semantics.ode.parameters
        betas = {p.id: p for p in stratified_params if "beta" in p.id}
        betas["beta_1_2_0"].value -= epsilon
        betas["beta_1_2_1"].value += epsilon

        stratified_result = runner.run(
            stratified_model.petrinet.model_dump(), BASE_SIR_REQUEST_PATH
        )

        assert (
            stratified_result
        ), f"Could not generate a result for stratified version of model: [{BASE_SIR_MODEL_PATH}], request: [{BASE_SIR_REQUEST_PATH}]"

        # Abstract and bound stratified Base model
        abstract_model = stratified_model.abstract({"S_1": "S", "S_2": "S"})
        bounded_abstract_model = abstract_model.formulate_bounds()
        bounded_abstract_result = runner.run(
            bounded_abstract_model.petrinet.model_dump(),
            BASE_SIR_REQUEST_PATH,
        )

        assert (
            bounded_abstract_result
        ), f"Could not generate a result for bounded abstracted stratified version of model: [{BASE_SIR_MODEL_PATH}], request: [{BASE_SIR_REQUEST_PATH}]"

        # Check that abstract bounds actually bound the stratified and original model

        bs = [s for s in base_result.model._symbols() if s != "timer_t"]
        b_df = base_result.dataframe(base_result.points())[bs]

        ss = [s for s in stratified_result.model._symbols() if s != "timer_t"]
        s_df = stratified_result.dataframe(stratified_result.points())[ss]

        bass = [
            s
            for s in bounded_abstract_result.model._symbols()
            if s != "timer_t"
        ]
        ba_df = bounded_abstract_result.dataframe(
            bounded_abstract_result.points()
        )[bass]

        ds_df = pd.DataFrame(s_df)
        ds_df["S"] = ds_df.S_1 + ds_df.S_2

        def check_bounds(bounds_df, values_df, variable, values_model_name):
            failures = []
            lb = f"{variable}_lb"
            ub = f"{variable}_ub"
            if not all(values_df[variable] >= bounds_df[lb]):
                failures.append(
                    f"The bounded abstract model does not lower bound the {values_model_name} model {variable}:\n{pd.DataFrame({lb:bounds_df[lb], variable: values_df[variable], f'{variable}-{lb}':values_df[variable]-bounds_df[lb]})}"
                )
            if not all(values_df[variable] <= bounds_df[ub]):
                failures.append(
                    f"The bounded abstract model does not upper bound the {values_model_name} model {variable}:\n{pd.DataFrame({ub:bounds_df[ub], variable: values_df[variable], f'{ub}-{variable}':bounds_df[ub]-values_df[variable]})}"
                )
            return failures

        all_failures = []
        for values_df, model in [(b_df, "base"), (ds_df, "stratified")]:
            for var in ["S", "I", "R"]:
                all_failures += check_bounds(ba_df, values_df, var, model)

        reasons = "\n".join(map(str, all_failures))
        assert (
            len(all_failures) == 0
        ), f"The bounds failed in the following cases:\n{reasons}"

        # # Modify request parameters
        # request_parameters = stratified_request.parameters
        # req_beta_1 = next(p for p in request_parameters if p.name == "beta_1")
        # req_beta_2 = next(p for p in request_parameters if p.name == "beta_2")
        # req_beta_1.interval = Interval(lb=beta_1.value, ub = beta_1.value, closed_upper_bound = True)
        # req_beta_2.interval = Interval(lb=beta_2.value, ub = beta_2.value, closed_upper_bound = True)

        # # stratified_request = FunmanWorkRequest()
        # setup_common(stratified_request, timepoints, debug=True, mode=MODE, synthesize=False,dreal_precision=1)
        # results = run(stratified_request, stratified_model_str, models)
        # report(results, stratified_model_str, stratified_model._state_var_names() + stratified_model._observable_names(), request_results, request_params)
        # stratified_model.to_dot()

    def test_param_synth(self):
        # use a bounded model to define a box, then check constraints on it
        epsilon = 1e-7
        runner = Runner()

        (base_model, _) = runner.get_model(BASE_SIR_MODEL_PATH)
        with open(BASE_SIR_REQUEST_PATH, "r") as f:
            base_request = FunmanWorkRequest.model_validate_json(f.read())
        base_request.structure_parameters[0].schedules[0].timepoints = list(
            range(0, 101, 10)
        )

        base_result = runner.run(
            base_model.petrinet.model_dump(), base_request
        )
        assert (
            base_result
        ), f"Could not generate a result for model: [{BASE_SIR_MODEL_PATH}], request: [{BASE_SIR_REQUEST_PATH}]"

        base_model.petrinet.metadata["abstraction"] = {
            "parameters": {
                "inf": {
                    "beta": {
                        "lb": 0.000315 - epsilon,
                        "ub": 0.000315 + epsilon,
                    },
                    "gamma": {"lb": 0.1 - epsilon, "ub": 0.1 + epsilon},
                },
                "rec": {"gamma": {"lb": 0.1 - epsilon, "ub": 0.1 + epsilon}},
            }
        }

        bounded_model = base_model.formulate_bounds()
        bounded_result = runner.run(
            bounded_model.petrinet.model_dump(),
            base_request,
        )
        assert (
            bounded_result
        ), f"Could not generate a result for bounded version of model: [{BASE_SIR_MODEL_PATH}], request: [{BASE_SIR_REQUEST_PATH}]"

        b_df = base_result.dataframe(base_result.points())
        bnd_df = bounded_result.dataframe(bounded_result.points())

        for var in ["S", "I", "R"]:
            bnd_df[f"{var}_width"] = bnd_df[f"{var}_ub"] - bnd_df[f"{var}_lb"]
        width_cols = [f"{var}_width" for var in ["S", "I", "R"]]

        print(
            f"When the beta interval has width {2*epsilon} the width of the state variables is:\n {bnd_df[width_cols]}"
        )

        # bound_cols = [f"{var}_{bound}" for var in ["S", "I", "R"] for bound in ["lb", "ub"]]
        # bnd_df[bound_cols].plot()
        # plt.savefig("sir_bounds")

        # FIXME need assert

    @unittest.skip(reason="WIP")
    def test_sirhd_stratify(self):
        epsilon = 0.000001
        timepoints = list(range(0, 2, 1))

        with open(BASE_SIRHD_REQUEST_PATH, "r") as f:
            sirhd_base_request = FunmanWorkRequest.model_validate_json(
                f.read()
            )
        # sirhd_request.config.use_compartmental_constraints = False
        # sirhd_request.config.save_smtlib = "./out"
        sirhd_base_request.config.mode = "mode_odeint"
        sirhd_base_request.structure_parameters[0].schedules[
            0
        ].timepoints = timepoints

        runner = Runner()
        base_result = runner.run(BASE_SIRHD_MODEL_PATH, sirhd_base_request)

        # import matplotlib.pyplot as plt

        # base_result.plot()
        # plt.savefig("sirhd")

        assert (
            base_result
        ), f"Could not generate a result for model: [{BASE_SIRHD_MODEL_PATH}], request: [{BASE_SIRHD_REQUEST_PATH}]"

        (base_model, _) = runner.get_model(BASE_SIRHD_MODEL_PATH)

        # Stratify Base model
        stratified_model_S = base_model.stratify(
            "S",
            ["vac", "unvac"],
            strata_parameters=["beta"],
            self_strata_transition=False,
        )
        stratified_model_S.to_dot().render("sirhd_strat_S")

        stratified_model_SI = stratified_model_S.stratify(
            "I",
            ["vac", "unvac"],
            strata_parameters=[],
            self_strata_transition=False,
        )

        stratified_model_SI.to_dot().render("sirhd_strat_SI")

        stratified_params = (
            stratified_model_SI.petrinet.semantics.ode.parameters
        )
        betas = {p.id: p for p in stratified_params if "beta" in p.id}
        betas["beta_vac_unvac_0"].value -= epsilon
        betas["beta_vac_unvac_1"].value += epsilon

        with open(BASE_SIRHD_REQUEST_PATH, "r") as f:
            sirhd_stratified_request = FunmanWorkRequest.model_validate_json(
                f.read()
            )
        # sirhd_request.config.use_compartmental_constraints = False
        # sirhd_request.config.save_smtlib = "./out"
        sirhd_stratified_request.config.mode = "mode_odeint"
        sirhd_stratified_request.structure_parameters[0].schedules[
            0
        ].timepoints = timepoints

        stratified_result = runner.run(
            stratified_model_SI.petrinet.model_dump(), sirhd_stratified_request
        )

        assert (
            stratified_result
        ), f"Could not generate a result for stratified version of model: [{BASE_SIRHD_MODEL_PATH}], request: [{BASE_SIRHD_REQUEST_PATH}]"

        # Abstract and bound stratified Base model
        abstract_model = stratified_model_SI.abstract(
            {
                "S_vac": "S",
                "S_unvac": "S",
                "beta_vac_unvac_0": "agg_beta",
                "beta_vac_unvac_1": "agg_beta",
            }
        )
        # print(abstract_model._state_var_names())
        # print(abstract_model._parameter_names())
        abstract_model.to_dot().render("sirhd_strat_SI_abstract_S")

        bounded_abstract_model = abstract_model.formulate_bounds()
        bounded_abstract_model.to_dot().render(
            "sirhd_strat_SI_bounded_abstract_S"
        )

        # Setup request by removing compartmental constraint that won't be correct
        # for a bounded model
        with open(BASE_SIRHD_REQUEST_PATH, "r") as f:
            sirhd_request = FunmanWorkRequest.model_validate_json(f.read())
        # sirhd_request.config.use_compartmental_constraints = False
        # sirhd_request.config.save_smtlib = "./out"
        sirhd_request.config.mode = "mode_odeint"
        sirhd_request.structure_parameters[0].schedules[
            0
        ].timepoints = timepoints

        bounded_abstract_result = runner.run(
            bounded_abstract_model.petrinet.model_dump(),
            sirhd_request,
        )

        assert (
            bounded_abstract_result
        ), f"Could not generate a result for bounded abstracted stratified version of model: [{BASE_SIRHD_MODEL_PATH}], request: [{BASE_SIRHD_REQUEST_PATH}]"

        # Check that abstract bounds actually bound the stratified and original model

        bs = [s for s in base_result.model._symbols() if s != "timer_t"]
        base_df = base_result.dataframe(base_result.points())[bs]

        ss = [s for s in stratified_result.model._symbols() if s != "timer_t"]
        stratified_df = stratified_result.dataframe(
            stratified_result.points()
        )[ss]

        bass = [
            s
            for s in bounded_abstract_result.model._symbols()
            if s != "timer_t"
        ]
        bounded_abstract_df = bounded_abstract_result.dataframe(
            bounded_abstract_result.points()
        )[bass]
        bounded_abstract_df["I_lb"] = (
            bounded_abstract_df.I_vac_lb + bounded_abstract_df.I_unvac_lb
        )
        bounded_abstract_df["I_ub"] = (
            bounded_abstract_df.I_vac_ub + bounded_abstract_df.I_unvac_ub
        )

        destratified_df = pd.DataFrame(stratified_df)
        destratified_df["S"] = destratified_df.S_vac + destratified_df.S_unvac
        destratified_df["I"] = destratified_df.I_vac + destratified_df.I_unvac

        def check_bounds(bounds_df, values_df, variable, values_model_name):
            failures = []
            lb = f"{variable}_lb"
            ub = f"{variable}_ub"
            if not all(values_df[variable] >= bounds_df[lb]):
                failures.append(
                    f"The bounded abstract model does not lower bound the {values_model_name} model {variable}:\n{pd.DataFrame({lb:bounds_df[lb], variable: values_df[variable], f'{variable}-{lb}':values_df[variable]-bounds_df[lb]})}"
                )
            if not all(values_df[variable] <= bounds_df[ub]):
                failures.append(
                    f"The bounded abstract model does not upper bound the {values_model_name} model {variable}:\n{pd.DataFrame({ub:bounds_df[ub], variable: values_df[variable], f'{ub}-{variable}':bounds_df[ub]-values_df[variable]})}"
                )
            return failures

        all_failures = []
        for values_df, model in [
            (base_df, "base"),
            (destratified_df, "stratified"),
        ]:
            for var in ["S", "I", "H", "R", "D"]:
                all_failures += check_bounds(
                    bounded_abstract_df, values_df, var, model
                )

        reasons = "\n".join(map(str, all_failures))
        assert (
            len(all_failures) == 0
        ), f"The bounds failed in the following cases:\n{reasons}"


# S_ub: 1.492174e+08  S_strat: 1.492175e+08, diff -98.77

# S_ub' = S_ub - IvlbSubBlbp0lb/Nub - IulbSubBlbp1lb/Nub - IvlbSubBlbp0lb/Nub - IulbSubBlbp1lb/Nub
#       =      - 44.76501510409002 - 44.76501510409002 - 44.76501510409002 - 44.76501510409002
#   because S = Sv+Su
# Stratified:
# Sv' = Sv - IvSvB0p0/N - IuSvB0p1/N
#     =  74608773.0 - (22.38250755204501) - (22.38250755204501)
#     = 74608728.2349849
#     = 74608727.79499725
# Su' = Su - IvSuB1p0/N - IuSuB1p1/N
#     = 74608728.23448753
#     = 74608727.79449497
# Su' + Sv' = 149217456.46947244
#           = 149217455.58949223
# I_vac               5.000000e+02
# I_unvac             5.000000e+02
# S_vac               7.460877e+07
# S_unvac             7.460877e+07
# R                   0.000000e+00
# H                   0.000000e+00
# D                   7.814540e+05
# N                   1.500000e+08
# pir                 9.000000e-01
# pih                 1.000000e-01
# rih                 7.000000e-02
# phd                 1.300000e-01
# rhd                 3.000000e-01
# phr                 8.700000e-01
# rhr                 7.000000e-02
# rir                 7.000000e-02
# beta_vac_unvac_0    1.799990e-01
# beta_vac_unvac_1    1.800010e-01
# p_I_vac_unvac_0     5.000000e-01
# p_I_vac_unvac_1     5.000000e-01

# Rate(target='t1_vac_unvac_0_vac_unvac_0', expression='p_I_vac_unvac_0*I_vac*S_vac*beta_vac_unvac_0/N', expression_mathml=None)
# Rate(target='t1_vac_unvac_0_vac_unvac_1', expression='p_I_vac_unvac_1*I_unvac*S_vac*beta_vac_unvac_0/N', expression_mathml=None)
# Rate(target='t1_vac_unvac_1_vac_unvac_0', expression='p_I_vac_unvac_0*I_vac*S_unvac*beta_vac_unvac_1/N', expression_mathml=None)
# Rate(target='t1_vac_unvac_1_vac_unvac_1', expression='p_I_vac_unvac_1*I_unvac*S_unvac*beta_vac_unvac_1/N', expression_mathml=None)

# Rate(target='t1_vac_unvac_0_vac_unvac_0', expression='I_vac*S*agg_beta*p_I_vac_unvac_0/N', expression_mathml=None)
# Rate(target='t1_vac_unvac_0_vac_unvac_1', expression='I_unvac*S*agg_beta*p_I_vac_unvac_1/N', expression_mathml=None)
# Rate(target='t1_vac_unvac_1_vac_unvac_0', expression='I_vac*S*agg_beta*p_I_vac_unvac_0/N', expression_mathml=None)
# Rate(target='t1_vac_unvac_1_vac_unvac_1', expression='I_unvac*S*agg_beta*p_I_vac_unvac_1/N', expression_mathml=None)

# Abstract Bounded:

#   = 149217356.8112546
#   = 149217366.93993962
#  diff = 10.128685027360916
# I_vac_lb              5.000000e+02
# I_vac_ub              5.000000e+02
# I_unvac_lb            5.000000e+02
# I_unvac_ub            5.000000e+02
# R_lb                  0.000000e+00
# R_ub                  0.000000e+00
# H_lb                  0.000000e+00
# H_ub                  0.000000e+00
# D_lb                  7.814540e+05
# D_ub                  7.814540e+05
# S_lb                  1.492175e+08
# S_ub                  1.492175e+08
# N_lb                  1.500000e+08
# N_ub                  1.500000e+08
# pir_lb                9.000000e-01
# pir_ub                9.000000e-01
# pih_lb                1.000000e-01
# pih_ub                1.000000e-01
# rih_lb                7.000000e-02
# rih_ub                7.000000e-02
# phd_lb                1.300000e-01
# phd_ub                1.300000e-01
# rhd_lb                3.000000e-01
# rhd_ub                3.000000e-01
# phr_lb                8.700000e-01
# phr_ub                8.700000e-01
# rhr_lb                7.000000e-02
# rhr_ub                7.000000e-02
# rir_lb                7.000000e-02
# rir_ub                7.000000e-02
# p_I_vac_unvac_0_lb    5.000000e-01
# p_I_vac_unvac_0_ub    5.000000e-01
# p_I_vac_unvac_1_lb    5.000000e-01
# p_I_vac_unvac_1_ub    5.000000e-01
# agg_beta_lb           1.799990e-01
# agg_beta_ub           1.800010e-01
# I_lb                  1.000000e+03
# I_ub                  1.000000e+03

if __name__ == "__main__":
    unittest.main()

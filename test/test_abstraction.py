import logging
import os
import sys
import unittest
from itertools import accumulate
from pathlib import Path

import pandas as pd
import sympy
from matplotlib import pyplot as plt

from funman import (
    Abstraction,
    Interval,
    LinearConstraint,
    Observable,
    StateVariableConstraint,
    StrataTransition,
    Stratification,
    Stratum,
    StratumAttribute,
    StratumAttributeValue,
    StratumAttributeValueSet,
    StratumPartition,
    StratumValuation,
)
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

        st_1 = StratumAttributeValue(name="1")
        st_2 = StratumAttributeValue(name="2")
        stratum_attr = StratumAttribute(name="st", values={st_1, st_2})
        stratum = Stratum(
            values={
                stratum_attr: {
                    StratumAttributeValueSet(values={st_1}),
                    StratumAttributeValueSet(values={st_2}),
                }
            }
        )

        # Stratify Base model
        stratification = Stratification(
            base_state="S",
            base_parameters=["beta"],
            stratum=stratum,
            strata_transitions=False,
        )
        stratified_model = base_model.stratify(stratification)

        stratified_params = stratified_model.petrinet.semantics.ode.parameters
        betas = [p for p in stratified_params if "beta" in p.id]
        for i, b in enumerate(betas):
            if i == 0:
                b.value -= epsilon
            else:
                b.value += epsilon

        stratified_result = runner.run(
            stratified_model.petrinet, BASE_SIR_REQUEST_PATH
        )

        assert (
            stratified_result
        ), f"Could not generate a result for stratified version of model: [{BASE_SIR_MODEL_PATH}], request: [{BASE_SIR_REQUEST_PATH}]"

        # Abstract and bound stratified Base model
        abstract_model = stratified_model.abstract(
            Abstraction(
                abstraction={
                    "S_st_1": "S",
                    "S_st_2": "S",
                    "beta___to_____S_st_1_to__": "beta",
                    "beta___to_____S_st_2_to__": "beta",
                }
            )
        )
        bounded_abstract_model = abstract_model.formulate_bounds()
        bounded_abstract_result = runner.run(
            bounded_abstract_model.petrinet,
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
        ds_df["S"] = ds_df.S_st_1 + ds_df.S_st_2

        # def check_bounds(bounds_df, values_df, variable, values_model_name):
        #     failures = []
        #     lb = f"{variable}_lb"
        #     ub = f"{variable}_ub"
        #     if not all(values_df[variable] >= bounds_df[lb]):
        #         failures.append(
        #             f"The bounded abstract model does not lower bound the {values_model_name} model {variable}:\n{pd.DataFrame({lb:bounds_df[lb], variable: values_df[variable], f'{variable}-{lb}':values_df[variable]-bounds_df[lb]})}"
        #         )
        #     if not all(values_df[variable] <= bounds_df[ub]):
        #         failures.append(
        #             f"The bounded abstract model does not upper bound the {values_model_name} model {variable}:\n{pd.DataFrame({ub:bounds_df[ub], variable: values_df[variable], f'{ub}-{variable}':bounds_df[ub]-values_df[variable]})}"
        #         )
        #     return failures

        all_failures = []
        for values_df, model in [(b_df, "base"), (ds_df, "stratified")]:
            for var in ["S", "I", "R"]:
                all_failures += self.check_bounds(ba_df, values_df, var, model)

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

    def check_bounds(
        self, bounds_df, values_df, variable, values_model_name, tolerance=1e-7
    ):
        failures = []
        lb = f"{variable}_lb"
        ub = f"{variable}_ub"
        if not all(values_df[variable] >= bounds_df[lb] - tolerance):
            failures.append(
                f"The bounded abstract model does not lower bound the {values_model_name} model {variable}:\n{pd.DataFrame({lb:bounds_df[lb], variable: values_df[variable], f'{variable}-{lb}':values_df[variable]-bounds_df[lb]})}"
            )
        if not all(values_df[variable] <= bounds_df[ub] + tolerance):
            failures.append(
                f"The bounded abstract model does not upper bound the {values_model_name} model {variable}:\n{pd.DataFrame({ub:bounds_df[ub], variable: values_df[variable], f'{ub}-{variable}':bounds_df[ub]-values_df[variable]})}"
            )
        return failures

    def check_bounds_configurations(
        self, configurations, variables, bounded_abstract_df
    ):
        all_failures = []
        for values_df, model in configurations:
            for var in variables:
                all_failures += self.check_bounds(
                    bounded_abstract_df, values_df, var, model
                )
        reasons = "\n".join(map(str, all_failures))
        assert (
            len(all_failures) == 0
        ), f"The bounds failed in the following cases:\n{reasons}"

    # @unittest.skip(reason="WIP")
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

        vac_T = StratumAttributeValue(name="T")
        vac_F = StratumAttributeValue(name="F")
        vac_stratum_attr = StratumAttribute(name="vac", values={vac_T, vac_F})
        vac_stratum = Stratum(
            values={
                vac_stratum_attr: {
                    StratumAttributeValueSet(values={vac_T}),
                    StratumAttributeValueSet(values={vac_F}),
                }
            }
        )

        # Stratify Base model
        stratification = Stratification(
            base_state="S",
            base_parameters=["beta"],
            stratum=vac_stratum,
            strata_transitions=False,
        )
        stratified_model_S = base_model.stratify(stratification)
        stratified_model_S.to_dot().render("sirhd_strat_S")

        # Combines rates for stratified t1 transition
        # S_v beta_v I, S_u beta_u I -> S beta I

        betas = [
            p.id
            for p in stratified_model_S.petrinet.semantics.ode.parameters
            if "beta" in p.id
        ]
        beta_abs = {p: "agg_beta" for p in betas}

        undo_S = stratified_model_S.abstract(
            Abstraction(
                abstraction={"S_vac_T": "S", "S_vac_F": "S", **beta_abs}
            )
        )
        undo_S.to_dot().render("undo_S")

        stratification = Stratification(
            base_state="I",
            stratum=vac_stratum,
            strata_transitions=False,
        )
        stratified_model_SI = stratified_model_S.stratify(stratification)
        stratified_model_SI.to_dot().render("sirhd_strat_SI")

        stratified_params = (
            stratified_model_SI.petrinet.semantics.ode.parameters
        )
        betas = [p for p in stratified_params if "beta" in p.id]
        for i, b in enumerate(betas):
            if i == 0:
                b.value -= epsilon
            else:
                b.value += epsilon

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
            stratified_model_SI.petrinet, sirhd_stratified_request
        )

        assert (
            stratified_result
        ), f"Could not generate a result for stratified version of model: [{BASE_SIRHD_MODEL_PATH}], request: [{BASE_SIRHD_REQUEST_PATH}]"

        # Abstract and bound stratified Base model
        abstract_model = stratified_model_SI.abstract(
            Abstraction(
                abstraction={
                    "S_vac_T": "S",
                    "S_vac_F": "S",
                    **{b.id: "agg_beta" for b in betas},
                }
            )
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
            bounded_abstract_model.petrinet,
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
            bounded_abstract_df.I_vac_T_lb + bounded_abstract_df.I_vac_F_lb
        )
        bounded_abstract_df["I_ub"] = (
            bounded_abstract_df.I_vac_T_ub + bounded_abstract_df.I_vac_F_ub
        )

        destratified_df = pd.DataFrame(stratified_df)
        destratified_df["S"] = (
            destratified_df.S_vac_T + destratified_df.S_vac_F
        )
        destratified_df["I"] = (
            destratified_df.I_vac_T + destratified_df.I_vac_F
        )

        configurations = [
            (base_df, "base"),
            (destratified_df, "stratified"),
        ]
        variables = ["S", "I", "H", "R", "D"]

        self.check_bounds_configurations(
            configurations, variables, bounded_abstract_df
        )

    def model_has_expected_parameters(self, model, expected_parameters):
        params = model._parameter_names()
        assert all(
            [p in params for p in expected_parameters]
        ), f"Stratifying I error: did not get expected parameters, got {params}, expecting new parameters {expected_parameters}"

    # @unittest.skip(reason="WIP")
    def test_sirhd_stratify_transitions(self):
        epsilon = 0.000001
        timepoints = list(range(0, 2, 1))

        with open(BASE_SIRHD_REQUEST_PATH, "r") as f:
            sirhd_base_request = FunmanWorkRequest.model_validate_json(
                f.read()
            )
        sirhd_base_request.config.mode = "mode_odeint"
        sirhd_base_request.structure_parameters[0].schedules[
            0
        ].timepoints = timepoints

        runner = Runner()
        base_result = runner.run(BASE_SIRHD_MODEL_PATH, sirhd_base_request)

        assert (
            base_result
        ), f"Could not generate a result for model: [{BASE_SIRHD_MODEL_PATH}], request: [{BASE_SIRHD_REQUEST_PATH}]"

        (base_model, _) = runner.get_model(BASE_SIRHD_MODEL_PATH)

        vac_T = StratumAttributeValue(name="T")
        vac_F = StratumAttributeValue(name="F")
        vac_stratum_attr = StratumAttribute(name="vac", values={vac_T, vac_F})
        vac_stratum = Stratum(
            values={
                vac_stratum_attr: {
                    StratumAttributeValueSet(values={vac_T}),
                    StratumAttributeValueSet(values={vac_F}),
                }
            }
        )

        # Stratify Base model
        stratification_S = Stratification(
            base_state="S",
            base_parameters=["beta"],
            stratum=vac_stratum,
            self_strata_transitions=0.01,
            cross_strata_transitions=True,
        )
        stratification_I = Stratification(
            base_state="I",
            stratum=vac_stratum,
            cross_strata_transitions=True,
        )

        stratified_model_S = base_model.stratify(stratification_S)
        stratified_model_S.to_dot().render("sirhd_strat_S")
        stratified_model_S_parameters = stratified_model_S._parameter_names()

        # # S stratification stratifies beta, allows cross strata transitions, and self strata transitions
        stratified_model_S_expected_parameters = [
            "N",
            "pir",
            "pih",
            "rih",
            "phd",
            "rhd",
            "phr",
            "rhr",
            "rir",
            "beta___to_____S_vac_F_to__",
            "beta___to_____S_vac_T_to__",
            "p_cross_S_vac_T_to_S_vac_F_",
            "p_cross_S_vac_F_to_S_vac_T_",
        ]

        self.model_has_expected_parameters(
            stratified_model_S, stratified_model_S_expected_parameters
        )

        stratified_model_I = base_model.stratify(stratification_I)
        stratified_model_I.to_dot().render("sirhd_strat_I")
        stratified_model_I_parameters = stratified_model_I._parameter_names()

        # I stratification allows cross strata transitions, but will not stratify beta
        stratified_model_I_expected_parameters = [
            "p_cross_I_vac_T_to_I_vac_T___S_to_I_vac_T_",
            "p_cross_I_vac_T_to_I_vac_T___S_to_I_vac_F_",
            "p_cross_I_vac_F_to_I_vac_F___S_to_I_vac_T_",
            "p_cross_I_vac_F_to_I_vac_F___S_to_I_vac_F_",
            "N",
            "beta",
            "pir",
            "pih",
            "rih",
            "phd",
            "rhd",
            "phr",
            "rhr",
            "rir",
        ]
        self.model_has_expected_parameters(
            stratified_model_I, stratified_model_I_expected_parameters
        )

        stratified_model_SI = stratified_model_S.stratify(stratification_I)
        stratified_model_SI.to_dot().render("sirhd_strat_SI")
        stratified_model_SI_parameters = stratified_model_SI._parameter_names()

        # # S stratification stratifies beta, allows cross strata transitions, and self strata transitions
        stratified_model_SI_expected_parameters = [
            "N",
            "pir",
            "pih",
            "rih",
            "phd",
            "rhd",
            "phr",
            "rhr",
            "rir",
            "beta___to_____S_vac_T_to__",
            "p_cross_I_vac_T_to_I_vac_T___S_vac_T_to_I_vac_T_",
            "beta___to_____S_vac_F_to__",
            "p_cross_I_vac_T_to_I_vac_T___S_vac_F_to_I_vac_T_",
            "p_cross_I_vac_T_to_I_vac_T___S_vac_T_to_I_vac_F_",
            "p_cross_I_vac_T_to_I_vac_T___S_vac_F_to_I_vac_F_",
            "p_cross_I_vac_F_to_I_vac_F___S_vac_T_to_I_vac_T_",
            "p_cross_I_vac_F_to_I_vac_F___S_vac_F_to_I_vac_T_",
            "p_cross_I_vac_F_to_I_vac_F___S_vac_T_to_I_vac_F_",
            "p_cross_I_vac_F_to_I_vac_F___S_vac_F_to_I_vac_F_",
            "p_cross_S_vac_T_to_S_vac_F_",
            "p_cross_S_vac_F_to_S_vac_T_",
        ]
        self.model_has_expected_parameters(
            stratified_model_SI, stratified_model_SI_expected_parameters
        )

        stratified_model_IS = stratified_model_I.stratify(stratification_S)
        stratified_model_IS.to_dot().render("sirhd_strat_IS")
        stratified_model_IS_parameters = stratified_model_IS._parameter_names()

        stratified_model_IS_expected_parameters = [
            "N",
            "pir",
            "pih",
            "rih",
            "phd",
            "rhd",
            "phr",
            "rhr",
            "rir",
            "beta___to_____S_vac_T_to__",
            "p_cross_I_vac_T_to_I_vac_T___S_vac_T_to_I_vac_T_",
            "beta___to_____S_vac_F_to__",
            "p_cross_I_vac_T_to_I_vac_T___S_vac_F_to_I_vac_T_",
            "p_cross_I_vac_T_to_I_vac_T___S_vac_T_to_I_vac_F_",
            "p_cross_I_vac_T_to_I_vac_T___S_vac_F_to_I_vac_F_",
            "p_cross_I_vac_F_to_I_vac_F___S_vac_T_to_I_vac_T_",
            "p_cross_I_vac_F_to_I_vac_F___S_vac_F_to_I_vac_T_",
            "p_cross_I_vac_F_to_I_vac_F___S_vac_T_to_I_vac_F_",
            "p_cross_I_vac_F_to_I_vac_F___S_vac_F_to_I_vac_F_",
            "p_cross_S_vac_T_to_S_vac_F_",
            "p_cross_S_vac_F_to_S_vac_T_",
        ]

        self.model_has_expected_parameters(
            stratified_model_IS, stratified_model_IS_expected_parameters
        )

        assert (
            len(
                set(stratified_model_SI_parameters).symmetric_difference(
                    set(stratified_model_IS_parameters)
                )
            )
            == 0
        )

        for stratified_params in [
            stratified_model_SI.petrinet.semantics.ode.parameters,
            stratified_model_IS.petrinet.semantics.ode.parameters,
        ]:
            betas = [p for p in stratified_params if "beta" in p.id]
            for i, b in enumerate(betas):
                if i == 0:
                    b.value -= epsilon
                else:
                    b.value += epsilon

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

        stratified_model_S_result = runner.run(
            stratified_model_S.petrinet, sirhd_stratified_request
        )
        assert (
            stratified_model_S_result
        ), f"Could not generate a result for stratified version of model: [{BASE_SIRHD_MODEL_PATH}], request: [{BASE_SIRHD_REQUEST_PATH}]"

        stratified_model_I_result = runner.run(
            stratified_model_I.petrinet, sirhd_stratified_request
        )
        assert (
            stratified_model_I_result
        ), f"Could not generate a result for stratified version of model: [{BASE_SIRHD_MODEL_PATH}], request: [{BASE_SIRHD_REQUEST_PATH}]"

        stratified_model_SI_result = runner.run(
            stratified_model_SI.petrinet, sirhd_stratified_request
        )
        assert (
            stratified_model_SI_result
        ), f"Could not generate a result for stratified version of model: [{BASE_SIRHD_MODEL_PATH}], request: [{BASE_SIRHD_REQUEST_PATH}]"

        stratified_model_IS_result = runner.run(
            stratified_model_IS.petrinet, sirhd_stratified_request
        )
        assert (
            stratified_model_IS_result
        ), f"Could not generate a result for stratified version of model: [{BASE_SIRHD_MODEL_PATH}], request: [{BASE_SIRHD_REQUEST_PATH}]"

        # Abstract and bound stratified Base model
        stratified_model_SI_abstract_S = stratified_model_SI.abstract(
            Abstraction(
                abstraction={
                    "S_vac_T": "S",
                    "S_vac_F": "S",
                    **{b.id: "beta" for b in betas},
                    # "p_cross_I_vac_F_to_I_vac_F___S_vac_T_to_I_vac_T_": "p_cross_I_vac_F_to_I_vac_F____to_I_vac_T_",
                    # "p_cross_I_vac_F_to_I_vac_F___S_vac_F_to_I_vac_T_": "p_cross_I_vac_F_to_I_vac_F____to_I_vac_T_",
                    # "p_cross_I_vac_F_to_I_vac_F___S_vac_T_to_I_vac_F_": "p_cross_I_vac_F_to_I_vac_F____to_I_vac_F_",
                    # "p_cross_I_vac_F_to_I_vac_F___S_vac_F_to_I_vac_F_": "p_cross_I_vac_F_to_I_vac_F____to_I_vac_F_",
                    # "p_cross_I_vac_T_to_I_vac_T___S_vac_F_to_I_vac_T_": "p_cross_I_vac_T_to_I_vac_T____to_I_vac_T_",
                    # "p_cross_I_vac_T_to_I_vac_T___S_vac_T_to_I_vac_T_": "p_cross_I_vac_T_to_I_vac_T____to_I_vac_T_",
                    # "p_cross_I_vac_T_to_I_vac_T___S_vac_F_to_I_vac_F_": "p_cross_I_vac_T_to_I_vac_T____to_I_vac_F_",
                    # "p_cross_I_vac_T_to_I_vac_T___S_vac_T_to_I_vac_F_": "p_cross_I_vac_T_to_I_vac_T____to_I_vac_F_",
                }
            )
        )
        stratified_model_SI_abstract_S.to_dot().render(
            "sirhd_strat_SI_abstract_S"
        )

        stratified_model_SI_abstract_I = stratified_model_SI.abstract(
            Abstraction(
                abstraction={
                    "I_vac_T": "I",
                    "I_vac_F": "I",
                    **{b.id: "beta" for b in betas},
                    # "p_cross_I_vac_F_to_I_vac_F___S_vac_T_to_I_vac_T_": "p_cross__to____S_vac_T_to__",
                    # "p_cross_I_vac_F_to_I_vac_F___S_vac_F_to_I_vac_T_": "p_cross__to____S_vac_F_to__",
                    # "p_cross_I_vac_F_to_I_vac_F___S_vac_T_to_I_vac_F_": "p_cross__to____S_vac_T_to__",
                    # "p_cross_I_vac_F_to_I_vac_F___S_vac_F_to_I_vac_F_": "p_cross__to____S_vac_F_to__",
                    # "p_cross_I_vac_T_to_I_vac_T___S_vac_F_to_I_vac_T_": "p_cross__to____S_vac_F_to__",
                    # "p_cross_I_vac_T_to_I_vac_T___S_vac_T_to_I_vac_T_": "p_cross__to____S_vac_T_to__",
                    # "p_cross_I_vac_T_to_I_vac_T___S_vac_F_to_I_vac_F_": "p_cross__to____S_vac_F_to__",
                    # "p_cross_I_vac_T_to_I_vac_T___S_vac_T_to_I_vac_F_": "p_cross__to____S_vac_T_to__",
                }
            )
        )
        stratified_model_SI_abstract_I.to_dot().render(
            "sirhd_strat_SI_abstract_I"
        )

        bounded_abstract_model = (
            stratified_model_SI_abstract_S.formulate_bounds()
        )
        bounded_abstract_model.to_dot().render(
            "sirhd_strat_SI_bounded_abstract_S"
        )

        # Abstract and bound stratified Base model
        abstract_model1 = stratified_model_SI.abstract(
            Abstraction(
                abstraction={
                    "I_vac_T": "I",
                    "I_vac_F": "I",
                    **{b.id: "agg_beta" for b in betas},
                }
            )
        )
        # print(abstract_model._state_var_names())
        # print(abstract_model._parameter_names())
        stratified_model_SI_abstract_S.to_dot().render(
            "sirhd_strat_SI_abstract_S"
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
            bounded_abstract_model.petrinet,
            sirhd_request,
        )

        assert (
            bounded_abstract_result
        ), f"Could not generate a result for bounded abstracted stratified version of model: [{BASE_SIRHD_MODEL_PATH}], request: [{BASE_SIRHD_REQUEST_PATH}]"

        bounded_abstract_df = bounded_abstract_result.dataframe(
            bounded_abstract_result.points()
        )
        bounded_abstract_df["I_lb"] = (
            bounded_abstract_df["I_vac_F_lb"]
            + bounded_abstract_df["I_vac_T_lb"]
        )
        bounded_abstract_df["I_ub"] = (
            bounded_abstract_df["I_vac_F_ub"]
            + bounded_abstract_df["I_vac_T_ub"]
        )

        base_df = base_result.dataframe(base_result.points())

        stratified_S_df = stratified_model_S_result.dataframe(
            stratified_model_S_result.points()
        )
        destratified_S_df = pd.DataFrame(stratified_S_df)
        destratified_S_df["S"] = (
            destratified_S_df.S_vac_T + destratified_S_df.S_vac_F
        )

        stratified_I_df = stratified_model_I_result.dataframe(
            stratified_model_I_result.points()
        )
        destratified_I_df = pd.DataFrame(stratified_I_df)
        destratified_I_df["I"] = (
            destratified_I_df.I_vac_T + destratified_I_df.I_vac_F
        )

        stratified_SI_df = stratified_model_SI_result.dataframe(
            stratified_model_SI_result.points()
        )
        destratified_SI_df = pd.DataFrame(stratified_SI_df)
        destratified_SI_df["S"] = (
            destratified_SI_df.S_vac_T + destratified_SI_df.S_vac_F
        )
        destratified_SI_df["I"] = (
            destratified_SI_df.I_vac_T + destratified_SI_df.I_vac_F
        )

        stratified_IS_df = stratified_model_IS_result.dataframe(
            stratified_model_IS_result.points()
        )
        destratified_IS_df = pd.DataFrame(stratified_IS_df)
        destratified_IS_df["S"] = (
            destratified_IS_df.S_vac_T + destratified_IS_df.S_vac_F
        )
        destratified_IS_df["I"] = (
            destratified_IS_df.I_vac_T + destratified_IS_df.I_vac_F
        )

        configurations = [
            (base_df, "base"),
            (destratified_I_df, "stratified_I"),
            (destratified_S_df, "stratified_S"),
            (destratified_SI_df, "stratified_SI"),
            (destratified_IS_df, "stratified_IS"),
        ]
        variables = ["S", "I", "H", "R", "D"]

        self.check_bounds_configurations(
            configurations, variables, bounded_abstract_df
        )

        # @unittest.skip(reason="WIP")

    def test_sirhd_stratify_analysis(self):

        epsilon = 0.0001
        N = 150000000.0
        max_I = 0.3 * N  # 3.542629e+07
        timepoints = [float(t) for t in list(range(0, 2, 1))]
        num_age_groups = 2

        runner = Runner()
        (base_model, _) = runner.get_model(BASE_SIRHD_MODEL_PATH)
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

        # Add constraint on I
        sirhd_stratified_request.constraints.append(
            StateVariableConstraint(
                name="I upper",
                variable="I",
                interval=Interval(ub=max_I),
                soft=False,
            )
        )

        vac_T = StratumAttributeValue(name="T")
        vac_F = StratumAttributeValue(name="F")
        vac_stratum_attr = StratumAttribute(name="vac", values={vac_T, vac_F})
        vac_stratum = Stratum(
            values={
                vac_stratum_attr: {
                    StratumAttributeValueSet(values={vac_T}),
                    StratumAttributeValueSet(values={vac_F}),
                }
            }
        )

        age_values = [
            StratumAttributeValue(name=str(i)) for i in range(num_age_groups)
        ]
        # age_0 = StratumAttributeValue(name="0")
        # age_1 = StratumAttributeValue(name="1")
        # age_2 = StratumAttributeValue(name="2")
        # age_3 = StratumAttributeValue(name="3")
        # age_4 = StratumAttributeValue(name="4")
        # age_5 = StratumAttributeValue(name="5")
        # age_6 = StratumAttributeValue(name="6")
        # age_7 = StratumAttributeValue(name="7")
        # age_8 = StratumAttributeValue(name="8")
        # age_9 = StratumAttributeValue(name="9")

        age_stratum_attr = StratumAttribute(
            name="age",
            values=age_values,
            # {age_0, age_1, age_2
            # , age_3, age_4, age_5, age_6, age_7, age_8, age_9
            # }
        )
        age_stratum = Stratum(
            values={
                age_stratum_attr: {
                    StratumAttributeValueSet(values={age_value})
                    for age_value in age_values
                    # StratumAttributeValueSet(values={age_0}),
                    # StratumAttributeValueSet(values={age_1}),
                    # StratumAttributeValueSet(values={age_2}),
                    # StratumAttributeValueSet(values={age_3}),
                    # StratumAttributeValueSet(values={age_4}),
                    # StratumAttributeValueSet(values={age_5}),
                    # StratumAttributeValueSet(values={age_6}),
                    # StratumAttributeValueSet(values={age_7}),
                    # StratumAttributeValueSet(values={age_8}),
                    # StratumAttributeValueSet(values={age_9})
                }
            }
        )

        base_params = base_model.petrinet.semantics.ode.parameters
        beta = next(iter([p for p in base_params if "beta" in p.id]))

        vac_stratifications = [
            Stratification(
                description="Stratify S and beta wrt. vaccination status.  Set beta values for strata.",
                base_state="S",
                stratum=vac_stratum,
                self_strata_transitions=0.01,
                base_parameters={
                    "beta": {
                        StrataTransition(
                            input_stratum=StratumValuation(
                                values={
                                    vac_stratum_attr: StratumAttributeValueSet(
                                        values={vac_T}
                                    )
                                }
                            ),
                            output_stratum=StratumValuation(),
                        ): beta.value
                        - epsilon,
                        StrataTransition(
                            input_stratum=StratumValuation(
                                values={
                                    vac_stratum_attr: StratumAttributeValueSet(
                                        values={vac_F}
                                    )
                                }
                            ),
                            output_stratum=StratumValuation(),
                        ): beta.value
                        + epsilon,
                    }
                },
            ),
            # FIXME Has an extra self transition that shouldn't be there
            Stratification(
                description="Stratify S_vac_T wrt. age group",
                base_state="S_vac_T",
                stratum=age_stratum,
                self_strata_transitions=0.01,
                base_parameters={
                    "beta___to_____S_vac_T_to__": {
                        StrataTransition(
                            input_stratum=StratumValuation(
                                values={
                                    age_stratum_attr: StratumAttributeValueSet(
                                        values={age_value}
                                    ),
                                    vac_stratum_attr: StratumAttributeValueSet(
                                        values={vac_T}
                                    ),
                                }
                            ),
                            output_stratum=StratumValuation(),
                        ): beta.value
                        + (
                            epsilon
                            * (float(i) - (float(num_age_groups) * 0.5))
                        )
                        for i, age_value in enumerate(age_values)
                        # StrataTransition(
                        #     input_stratum=StratumValuation(
                        #         values={
                        #             age_stratum_attr: StratumAttributeValueSet(
                        #                 values={age_0}
                        #             ),
                        #             vac_stratum_attr: StratumAttributeValueSet(
                        #                 values={vac_T}
                        #             ),
                        #         }
                        #     ),
                        #     output_stratum=StratumValuation(),
                        # ): beta.value
                        # - 2 * epsilon,
                        # StrataTransition(
                        #     input_stratum=StratumValuation(
                        #         values={
                        #             age_stratum_attr: StratumAttributeValueSet(
                        #                 values={age_1}
                        #             ),
                        #             vac_stratum_attr: StratumAttributeValueSet(
                        #                 values={vac_T}
                        #             ),
                        #         }
                        #     ),
                        #     output_stratum=StratumValuation(),
                        # ): beta.value,
                        # StrataTransition(
                        #     input_stratum=StratumValuation(
                        #         values={
                        #             age_stratum_attr: StratumAttributeValueSet(
                        #                 values={age_2}
                        #             ),
                        #             vac_stratum_attr: StratumAttributeValueSet(
                        #                 values={vac_T}
                        #             ),
                        #         }
                        #     ),
                        #     output_stratum=StratumValuation(),
                        # ): beta.value
                        # + 2 * epsilon,
                    }
                },
            ),
            Stratification(
                description="Stratify S_vac_F wrt. age group",
                base_state="S_vac_F",
                stratum=age_stratum,
                self_strata_transitions=0.01,
                base_parameters={
                    "beta___to_____S_vac_F_to__": {
                        StrataTransition(
                            input_stratum=StratumValuation(
                                values={
                                    age_stratum_attr: StratumAttributeValueSet(
                                        values={age_value}
                                    ),
                                    vac_stratum_attr: StratumAttributeValueSet(
                                        values={vac_F}
                                    ),
                                }
                            ),
                            output_stratum=StratumValuation(),
                        ): beta.value
                        + (
                            epsilon
                            * (float(i) - (float(num_age_groups) * 0.75))
                        )
                        for i, age_value in enumerate(age_values)
                        # StrataTransition(
                        #     input_stratum=StratumValuation(
                        #         values={
                        #             age_stratum_attr: StratumAttributeValueSet(
                        #                 values={age_0}
                        #             ),
                        #             vac_stratum_attr: StratumAttributeValueSet(
                        #                 values={vac_F}
                        #             ),
                        #         }
                        #     ),
                        #     output_stratum=StratumValuation(),
                        # ): beta.value
                        # - 3 * epsilon,
                        # StrataTransition(
                        #     input_stratum=StratumValuation(
                        #         values={
                        #             age_stratum_attr: StratumAttributeValueSet(
                        #                 values={age_1}
                        #             ),
                        #             vac_stratum_attr: StratumAttributeValueSet(
                        #                 values={vac_F}
                        #             ),
                        #         }
                        #     ),
                        #     output_stratum=StratumValuation(),
                        # ): beta.value,
                        # StrataTransition(
                        #     input_stratum=StratumValuation(
                        #         values={
                        #             age_stratum_attr: StratumAttributeValueSet(
                        #                 values={age_2}
                        #             ),
                        #             vac_stratum_attr: StratumAttributeValueSet(
                        #                 values={vac_F}
                        #             ),
                        #         }
                        #     ),
                        #     output_stratum=StratumValuation(),
                        # ): beta.value
                        # + 3 * epsilon,
                    }
                },
            ),
            Stratification(
                description="Stratify I wrt. vaccination status.",
                base_state="I",
                stratum=vac_stratum,
                self_strata_transitions=0.01,
            ),
            Stratification(
                description="Stratify I_vac_T wrt. age.",
                base_state="I_vac_T",
                stratum=age_stratum,
                self_strata_transitions=0.01,
            ),
            Stratification(
                description="Stratify I_vac_F wrt. age.",
                base_state="I_vac_F",
                stratum=age_stratum,
                self_strata_transitions=0.01,
            ),
            Stratification(
                description="Stratify R wrt. vaccination status.",
                base_state="R",
                stratum=vac_stratum,
                self_strata_transitions=0.01,
            ),
            Stratification(
                description="Stratify R_vac_T wrt. age.",
                base_state="R_vac_T",
                stratum=age_stratum,
                self_strata_transitions=0.01,
            ),
            Stratification(
                description="Stratify R_vac_F wrt. age.",
                base_state="R_vac_F",
                stratum=age_stratum,
                self_strata_transitions=0.01,
            ),
            Stratification(
                description="Stratify H wrt. vaccination status.",
                base_state="H",
                stratum=vac_stratum,
                self_strata_transitions=0.01,
            ),
            Stratification(
                description="Stratify H_vac_T wrt. age.",
                base_state="H_vac_T",
                stratum=age_stratum,
                self_strata_transitions=0.01,
            ),
            Stratification(
                description="Stratify H_vac_F wrt. age.",
                base_state="H_vac_F",
                stratum=age_stratum,
                self_strata_transitions=0.01,
            ),
            Stratification(
                description="Stratify D wrt. vaccination status.",
                base_state="D",
                stratum=vac_stratum,
                self_strata_transitions=0.01,
            ),
            Stratification(
                description="Stratify D_vac_T wrt. age.",
                base_state="D_vac_T",
                stratum=age_stratum,
                self_strata_transitions=0.01,
            ),
            Stratification(
                description="Stratify D_vac_F wrt. age.",
                base_state="D_vac_F",
                stratum=age_stratum,
                self_strata_transitions=0.01,
            ),
        ]

        vac_abstractions = [
            Abstraction(
                description="Abstract D_vac_T wrt. age.",
                abstraction={
                    f"D_vac_T_age_{i}": "D_vac_T"
                    for i in range(num_age_groups)
                },
            ),
            Abstraction(
                description="Abstract D_vac_F wrt. age.",
                abstraction={
                    f"D_vac_F_age_{i}": "D_vac_F"
                    for i in range(num_age_groups)
                },
            ),
            Abstraction(
                description="Abstract D wrt. vaccination status.",
                abstraction={"D_vac_F": "D", "D_vac_T": "D"},
            ),
            Abstraction(
                description="Abstract H_vac_T wrt. age.",
                abstraction={
                    f"H_vac_T_age_{i}": "H_vac_T"
                    for i in range(num_age_groups)
                },
            ),
            Abstraction(
                description="Abstract H_vac_F wrt. age.",
                abstraction={
                    f"H_vac_F_age_{i}": "H_vac_F"
                    for i in range(num_age_groups)
                },
            ),
            Abstraction(
                description="Abstract H wrt. vaccination status.",
                abstraction={"H_vac_F": "H", "H_vac_T": "H"},
            ),
            Abstraction(
                description="Abstract R_vac_T wrt. age.",
                abstraction={
                    f"R_vac_T_age_{i}": "R_vac_T"
                    for i in range(num_age_groups)
                },
            ),
            Abstraction(
                description="Abstract R_vac_F wrt. age.",
                abstraction={
                    f"R_vac_F_age_{i}": "R_vac_F"
                    for i in range(num_age_groups)
                },
            ),
            Abstraction(
                description="Abstract R wrt. vaccination status.",
                abstraction={"R_vac_F": "R", "R_vac_T": "R"},
            ),
            Abstraction(
                description="Abstract I_vac_T wrt. age.",
                abstraction={
                    f"I_vac_T_age_{i}": "I_vac_T"
                    for i in range(num_age_groups)
                },
            ),
            Abstraction(
                description="Abstract I_vac_F wrt. age.",
                abstraction={
                    f"I_vac_F_age_{i}": "I_vac_F"
                    for i in range(num_age_groups)
                },
            ),
            Abstraction(
                description="Abstract I wrt. vaccination status.",
                abstraction={"I_vac_F": "I", "I_vac_T": "I"},
            ),
            Abstraction(
                description="Abstract S age groups wrt. unvaccination status.",
                abstraction={
                    **{
                        f"S_vac_F_age_{i}": "S_vac_F"
                        for i in range(num_age_groups)
                    },
                    **{
                        f"beta___to_____S_vac_F_to_____to_____S_vac_F_age_{i}_to__": "beta___to_____S_vac_F_to__"
                        for i in range(num_age_groups)
                    },
                },
            ),
            Abstraction(
                description="Abstract S age groups wrt. vaccination status.",
                abstraction={
                    **{
                        f"S_vac_T_age_{i}": "S_vac_T"
                        for i in range(num_age_groups)
                    },
                    **{
                        f"beta___to_____S_vac_T_to_____to_____S_vac_T_age_{i}_to__": "beta___to_____S_vac_T_to__"
                        for i in range(num_age_groups)
                    },
                },
            ),
            Abstraction(
                description="Abstract S and beta vaccination status.",
                abstraction={
                    "S_vac_F": "S",
                    "S_vac_T": "S",
                    "beta___to_____S_vac_T_to__": "beta",
                    "beta___to_____S_vac_F_to__": "beta",
                },
            ),
        ]

        # transformation_sequence = vac_stratifications + vac_abstractions
        transformation_sequence = vac_stratifications + vac_abstractions
        vac_models = []

        current_model = base_model
        for i, t in enumerate(transformation_sequence):
            next_model = current_model.transform(t)
            vac_models.append(next_model)
            # next_model.to_dot().render(f"vac_model_{i}"),
            current_model = next_model

        # vac_models = list(
        #     accumulate(
        #         transformation_sequence,
        #         lambda x, y: x.transform(y),
        #         initial=base_model,
        #     )
        # )

        for m in vac_models:
            infected_states = [
                s.id for s in m.petrinet.model.states if s.id.startswith("I")
            ]
            if len(infected_states) > 1:
                infected_sum = Observable(
                    id="I", expression="+".join(infected_states)
                )
                m.petrinet.semantics.ode.observables = [infected_sum]
            # sirhd_stratified_request.constraints.append(
            #     LinearConstraint(
            #         name="I bound", variables=["I"], additive_bounds={"ub": 1e5}
            #     )
            # )

        # list(
        #     map(
        #         lambda x, name: x.to_dot().render(f"vac_model_{name}"),
        #         vac_models,
        #         range(len(vac_models)),
        #     )
        # )

        vac_model_params = [
            {p.id: p.value for p in m.petrinet.semantics.ode.parameters}
            for m in vac_models
        ]
        bounded_vac_models = [
            m.formulate_bounds() for m in vac_models[-len(vac_abstractions) :]
        ]

        for m in bounded_vac_models:
            # infected_states_lb = [s.id for s in m.petrinet.model.states if s.id.startswith("I") and s.id.endswith("_lb")]
            infected_states_ub = [
                s.id
                for s in m.petrinet.model.states
                if s.id.startswith("I") and s.id.endswith("_ub")
            ]

            infected_ub_sum = Observable(
                id="I", expression="+".join(infected_states_ub)
            )
            m.petrinet.semantics.ode.observables = [infected_ub_sum]
        # sirhd_stratified_request.constraints.append(
        #     LinearConstraint(
        #         name="I_ub bound",
        #         variables=["I_ub"],
        #         additive_bounds={"ub": 1200},
        #     )
        # )
        sirhd_stratified_request.config.use_compartmental_constraints = False

        results = [runner.run(base_model.petrinet, sirhd_stratified_request)]
        results += [
            runner.run(m.petrinet, sirhd_stratified_request)
            for m in vac_models[0 : len(vac_stratifications)]
        ]
        results += [
            runner.run(m.petrinet, sirhd_stratified_request)
            for m in bounded_vac_models
        ]
        # results = [
        #     runner.run(m.petrinet, sirhd_stratified_request)
        #     for m in bounded_vac_models
        # ]

        m = []
        for i, result in enumerate(results):
            df = result.dataframe()
            df["description"] = (
                result.model.petrinet.metadata["transformation_description"]
                if "transformation_description"
                in result.model.petrinet.metadata
                else ""
            )
            df["model_index"] = i
            df["error"] = result.error
            df["true_points"] = len(result.parameter_space.true_points())
            df["runtime (s)"] = (
                f"{result.timing.total_time.seconds}.{result.timing.total_time.microseconds}"
            )
            df["I_bound"] = len(result.parameter_space.true_points()) > 0
            # df = df.set_index(["model_index", "index"])
            df.index = df.index.rename("time")
            df = df.reset_index().set_index(["model_index", "time"])
            m.append(df)
        dfs = pd.concat(m)

        runtimes = dfs.reset_index(["time"])[
            [
                "runtime (s)",
                "description",
                "I_bound",
                # , "I", *[b for b in dfs.columns if b.startswith("beta")]
            ]
        ].drop_duplicates()
        self.l.info(runtimes)
        runtimes.to_csv(f"stratify_analysis_runtimes_{num_age_groups}.csv")
        dfs.to_csv(f"stratify_analysis_{num_age_groups}.csv")
        # I_df = pd.DataFrame([base_df.I, vac_model_df.I_vac_F, vac_model_df.I_vac_T, bounded_df.I_lb, bounded_df.I_ub]).T
        # S_df = pd.DataFrame([base_df.S, vac_model_df.S_vac_F, vac_model_df.S_vac_T, bounded_df.S_lb, bounded_df.S_ub]).T

        # assert (
        #     bounded_abstract_model_result
        # ), f"Could not generate a result for stratified version of model: [{BASE_SIRHD_MODEL_PATH}], request: [{BASE_SIRHD_REQUEST_PATH}]"

    def test_sirhd_nonuniform_abstraction(self):

        epsilon = 0.000001
        timepoints = list(range(0, 2, 1))

        runner = Runner()
        (base_model, _) = runner.get_model(BASE_SIRHD_MODEL_PATH)
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

        age_0 = StratumAttributeValue(name="0")
        age_1 = StratumAttributeValue(name="1")
        age_2 = StratumAttributeValue(name="2")
        age_stratum_attr = StratumAttribute(
            name="age", values={age_0, age_1, age_2}
        )
        age_stratum = Stratum(
            values={
                age_stratum_attr: {
                    StratumAttributeValueSet(values={age_0}),
                    StratumAttributeValueSet(values={age_1}),
                    StratumAttributeValueSet(values={age_2}),
                }
            }
        )

        base_params = base_model.petrinet.semantics.ode.parameters
        beta = next(iter([p for p in base_params if "beta" in p.id]))

        age_0_age_1_val = StratumValuation(
            values={
                age_stratum_attr: StratumAttributeValueSet(
                    values=[age_0, age_1]
                )
            }
        )
        age_1_age_2_val = StratumValuation(
            values={
                age_stratum_attr: StratumAttributeValueSet(
                    values=[age_1, age_2]
                )
            }
        )
        age_0_val = StratumValuation(
            values={age_stratum_attr: StratumAttributeValueSet(values={age_0})}
        )
        age_1_val = StratumValuation(
            values={age_stratum_attr: StratumAttributeValueSet(values={age_1})}
        )
        age_2_val = StratumValuation(
            values={age_stratum_attr: StratumAttributeValueSet(values={age_2})}
        )
        empty_val = StratumValuation()

        vac_stratifications = [
            Stratification(
                base_state="S",
                stratum=age_stratum,
                self_strata_transitions=True,
                partition=StratumPartition(
                    values=[age_0_val, age_1_age_2_val]
                ),
                base_parameters={
                    "beta": {
                        StrataTransition(
                            input_stratum=age_0_val, output_stratum=empty_val
                        ): beta.value
                        - epsilon,
                        StrataTransition(
                            input_stratum=age_1_age_2_val,
                            output_stratum=empty_val,
                        ): beta.value
                        + epsilon,
                    }
                },
            ),
            Stratification(
                base_state="S_age_1_2",
                stratum=age_stratum,
                self_strata_transitions=True,
                partition=StratumPartition(values=[age_1_val, age_2_val]),
                base_parameters={
                    "beta___to_____S_age_1_2_to__": {
                        StrataTransition(
                            input_stratum=age_1_val,
                            output_stratum=empty_val,
                        ): beta.value
                        - epsilon,
                        StrataTransition(
                            input_stratum=age_2_val,
                            output_stratum=empty_val,
                        ): beta.value
                        + epsilon,
                    }
                },
            ),
        ]

        vac_abstractions = [
            Abstraction(
                abstraction={
                    "S_age_0": "S_age_0_age_1",
                    "S_age_1": "S_age_0_age_1",
                }
            ),
        ]

        transformation_sequence = vac_stratifications  # + vac_abstractions

        vac_models = list(
            accumulate(
                transformation_sequence,
                lambda x, y: x.transform(y),
                initial=base_model,
            )
        )

        list(
            map(
                lambda x, name: x.to_dot().render(f"vac_model_{name}"),
                vac_models,
                range(len(vac_models)),
            )
        )

        vac_model_params = [
            {p.id: p.value for p in m.petrinet.semantics.ode.parameters}
            for m in vac_models
        ]

        bounded_vac_models = [m.formulate_bounds() for m in vac_models]

        results = [
            runner.run(m.petrinet, sirhd_stratified_request)
            for m in vac_models[0 : len(vac_stratifications) + 1]
            + bounded_vac_models
        ]

        m = []
        for i, result in enumerate(results):
            df = result.dataframe()
            df["model_index"] = i
            df["runtime (s)"] = (
                f"{result.timing.total_time.seconds}.{result.timing.total_time.microseconds}"
            )
            df["I_bound"] = len(result.parameter_space.true_points()) > 0
            df = df.set_index(["model_index"])
            m.append(df)
        dfs = pd.concat(m)
        runtimes = dfs[
            [c for c in ["runtime (s)"] if c in dfs.columns]
        ].drop_duplicates()
        self.l.info(runtimes)
        # I_df = pd.DataFrame([base_df.I, vac_model_df.I_vac_F, vac_model_df.I_vac_T, bounded_df.I_lb, bounded_df.I_ub]).T
        # S_df = pd.DataFrame([base_df.S, vac_model_df.S_vac_F, vac_model_df.S_vac_T, bounded_df.S_lb, bounded_df.S_ub]).T

        # assert (
        #     bounded_abstract_model_result
        # ), f"Could not generate a result for stratified version of model: [{BASE_SIRHD_MODEL_PATH}], request: [{BASE_SIRHD_REQUEST_PATH}]"


if __name__ == "__main__":
    unittest.main()

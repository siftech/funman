import logging
import os
import sys
import unittest
from pathlib import Path

import pandas as pd
import sympy
from matplotlib import pyplot as plt

from funman import (
    Abstraction,
    Stratification,
    Stratum,
    StratumAttribute,
    StratumAttributeValue,
    StratumAttributeValueSet,
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
        betas = {p.id: p for p in stratified_params if "beta" in p.id}
        betas["beta___None_to_None___st_1_to_None_"].value -= epsilon
        betas["beta___None_to_None___st_2_to_None_"].value += epsilon

        stratified_result = runner.run(
            stratified_model.petrinet, BASE_SIR_REQUEST_PATH
        )

        assert (
            stratified_result
        ), f"Could not generate a result for stratified version of model: [{BASE_SIR_MODEL_PATH}], request: [{BASE_SIR_REQUEST_PATH}]"

        # Abstract and bound stratified Base model
        abstract_model = stratified_model.abstract(
            Abstraction(abstraction={"S_st_1": "S", "S_st_2": "S"})
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

        def check_bounds(
            bounds_df, values_df, variable, values_model_name, tolerance=1e-7
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
            self_strata_transitions=True,
            cross_strata_transitions=True,
        )
        stratification_I = Stratification(
            base_state="I",
            stratum=vac_stratum,
            cross_strata_transitions=True,
        )

        stratified_model_I = base_model.stratify(stratification_I)
        stratified_model_I.to_dot().render("sirhd_strat_I")
        stratified_model_I_parameters = stratified_model_I._parameter_names()

        # I stratification allows cross strata transitions, but will not stratify beta
        stratified_model_I_expected_parameters = [
            "p_abstract_t1_S",
            "p_cross_I_vac_T_vac_T_to_I_vac_T_vac_T___S__to_I_vac_T_vac_T",
            "p_cross_I_vac_T_vac_T_to_I_vac_T_vac_T___S__to_I_vac_F_vac_F",
            "p_cross_I_vac_F_vac_F_to_I_vac_F_vac_F___S__to_I_vac_T_vac_T",
            "p_cross_I_vac_F_vac_F_to_I_vac_F_vac_F___S__to_I_vac_F_vac_F",
        ]
        self.model_has_expected_parameters(
            stratified_model_I, stratified_model_I_expected_parameters
        )

        stratified_model_S = base_model.stratify(stratification_S)
        stratified_model_S.to_dot().render("sirhd_strat_S")
        stratified_model_S_parameters = stratified_model_S._parameter_names()

        # # S stratification stratifies beta, allows cross strata transitions, and self strata transitions
        stratified_model_S_expected_parameters = [
            "beta_I__to_I____S_vac_T_vac_T_to_I_",
            "beta_I__to_I____S_vac_F_vac_F_to_I_",
            "p_cross_S_vac_T_vac_T_to_S_vac_F_vac_F",
            "p_cross_S_vac_F_vac_F_to_S_vac_T_vac_T",
        ]
        self.model_has_expected_parameters(
            stratified_model_S, stratified_model_S_expected_parameters
        )

        # assert ('beta___None_to_None___vac_T_to_None_' in stratified_model_S_parameters and
        #         'beta___None_to_None___vac_F_to_None_' in stratified_model_S_parameters and
        #         'p_self_S__vac_T_to_vac_F_' in stratified_model_S_parameters and
        #         'p_self_S__vac_F_to_vac_T_' in stratified_model_S_parameters), f"Stratifying I error: did not get expected parameters, got {stratified_model_S_parameters}, expecting new parameters [beta___None_to_None___vac_T_to_None_, beta___None_to_None___vac_F_to_None_, p_self_S__vac_T_to_vac_F_, p_self_S__vac_F_to_vac_T_]"

        stratified_model_SI = stratified_model_S.stratify(stratification_I)
        stratified_model_SI.to_dot().render("sirhd_strat_SI")
        stratified_model_SI_parameters = stratified_model_SI._parameter_names()

        # # S stratification stratifies beta, allows cross strata transitions, and self strata transitions
        stratified_model_SI_expected_parameters = [
            "p_cross_I_vac_F_vac_F_to_I_vac_F_vac_F___S_vac_F_vac_F_to_I_vac_F_vac_F",
            "p_cross_I_vac_F_vac_F_to_I_vac_F_vac_F___S_vac_F_vac_F_to_I_vac_T_vac_T",
            "p_cross_I_vac_T_vac_T_to_I_vac_T_vac_T___S_vac_F_vac_F_to_I_vac_F_vac_F",
            "p_cross_I_vac_T_vac_T_to_I_vac_T_vac_T___S_vac_F_vac_F_to_I_vac_T_vac_T",
            "p_cross_I_vac_F_vac_F_to_I_vac_F_vac_F___S_vac_T_vac_T_to_I_vac_F_vac_F",
            "p_cross_I_vac_F_vac_F_to_I_vac_F_vac_F___S_vac_T_vac_T_to_I_vac_T_vac_T",
            "p_cross_I_vac_T_vac_T_to_I_vac_T_vac_T___S_vac_T_vac_T_to_I_vac_F_vac_F",
            "p_cross_I_vac_T_vac_T_to_I_vac_T_vac_T___S_vac_T_vac_T_to_I_vac_T_vac_T",
            "beta_I__to_I____S_vac_F_vac_F_to_I_",
            "beta_I__to_I____S_vac_T_vac_T_to_I_",
        ]
        self.model_has_expected_parameters(
            stratified_model_SI, stratified_model_SI_expected_parameters
        )

        # SI adds to S, stratified beta unchnaged, doesn't need an abstract S parameter
        # new 'p_cross_t1__None_to_None___vac_T_to_None__S_vac_T_vac_T_to_I_vac_T_vac_T'
        # new 'p_cross_t1__None_to_None___vac_T_to_None__S_vac_T_vac_T_to_I_vac_F_vac_F'
        # new 'p_cross_t1__None_to_None___vac_F_to_None__S_vac_F_vac_F_to_I_vac_F_vac_F'
        # new 'p_cross_t1__None_to_None___vac_F_to_None__S_vac_F_vac_F_to_I_vac_T_vac_T'
        # 'beta___None_to_None___vac_T_to_None_'
        # 'beta___None_to_None___vac_F_to_None_'
        # 'p_self_S__vac_T_to_vac_F_'
        # 'p_self_S__vac_F_to_vac_T_'

        stratified_model_IS = stratified_model_I.stratify(stratification_S)
        stratified_model_IS.to_dot().render("sirhd_strat_IS")
        stratified_model_IS_parameters = stratified_model_IS._parameter_names()

        stratified_model_IS_expected_parameters = [
            # 'p_abstract_t1_S',
            "p_cross_I_vac_T_vac_T_to_I_vac_T_vac_T___S__to_I_vac_T_vac_T",
            "p_cross_I_vac_T_vac_T_to_I_vac_T_vac_T___S__to_I_vac_F_vac_F",
            "p_cross_I_vac_F_vac_F_to_I_vac_F_vac_F___S__to_I_vac_T_vac_T",
            "p_cross_I_vac_F_vac_F_to_I_vac_F_vac_F___S__to_I_vac_F_vac_F",
            "beta_I_vac_F_vac_F_to_I_vac_F_vac_F___S_vac_F_vac_F_to_I_vac_F_vac_F",
            "beta_I_vac_F_vac_F_to_I_vac_F_vac_F___S_vac_T_vac_T_to_I_vac_F_vac_F",
            "beta_I_vac_F_vac_F_to_I_vac_F_vac_F___S_vac_F_vac_F_to_I_vac_T_vac_T",
            "beta_I_vac_F_vac_F_to_I_vac_F_vac_F___S_vac_T_vac_T_to_I_vac_T_vac_T",
            "beta_I_vac_T_vac_T_to_I_vac_T_vac_T___S_vac_F_vac_F_to_I_vac_F_vac_F",
            "beta_I_vac_T_vac_T_to_I_vac_T_vac_T___S_vac_T_vac_T_to_I_vac_F_vac_F",
            "beta_I_vac_T_vac_T_to_I_vac_T_vac_T___S_vac_F_vac_F_to_I_vac_T_vac_T",
            "beta_I_vac_T_vac_T_to_I_vac_T_vac_T___S_vac_T_vac_T_to_I_vac_T_vac_T",
        ]

        self.model_has_expected_parameters(
            stratified_model_IS, stratified_model_IS_expected_parameters
        )

        # assert all(
        #     [
        #         p in stratified_model_IS_parameters
        #         for p in stratified_model_IS_expected_parameters
        #     ]
        # ), f"Stratifying IS error: did not get expected parameters, got {stratified_model_IS_parameters}, expecting new parameters [{stratified_model_IS_expected_parameters}]"

        #  remove 'p_abstract_t1_S'
        #  There are two cross parameters here from stratifying I, after stratifying S, then there should be 4
        #  The fix is to stratify any cross parameters relevant to an abstract input
        # 'p_cross_t1_S__to_I_vac_T_vac_T'
        # 'p_cross_t1_S__to_I_vac_F_vac_F'
        #  I had one beta, used in two transitions, both copies get stratified and each applies to four transitions
        #  Need to fix because beta is coupled with the strata of S, not necessarily transitions involving S
        #  new 'beta___vac_T_to_vac_T___vac_T_to_vac_T_'
        #  new 'beta___vac_T_to_vac_T___vac_F_to_vac_T_'
        #  new 'beta___vac_T_to_vac_T___vac_T_to_vac_F_'
        #  new 'beta___vac_T_to_vac_T___vac_F_to_vac_F_'
        #  new 'beta___vac_F_to_vac_F___vac_T_to_vac_T_'
        #  new 'beta___vac_F_to_vac_F___vac_F_to_vac_T_'
        #  new 'beta___vac_F_to_vac_F___vac_T_to_vac_F_'
        #  new 'beta___vac_F_to_vac_F___vac_F_to_vac_F_'
        #  new 'p_self_S__vac_T_to_vac_F_'
        #  new 'p_self_S__vac_F_to_vac_T_'

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
        stratified_model_SI_abstract_S = stratified_model_SI.abstract(
            Abstraction(
                abstraction={
                    "S_vac_T": "S",
                    "S_vac_F": "S",
                    **{b.id: "agg_beta" for b in betas},
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
                    **{b.id: "agg_beta" for b in betas},
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


if __name__ == "__main__":
    unittest.main()

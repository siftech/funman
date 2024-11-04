# Helper functions to setup FUNMAN for different steps of the scenario


import json
import logging

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from funman import (
    MODE_ODEINT,
    MODE_SMT,
    EncodingSchedule,
    FUNMANConfig,
    FunmanWorkRequest,
)
from funman.api.run import Runner
from funman.representation import Interval
from funman.representation.constraint import (
    LinearConstraint,
    ParameterConstraint,
    StateVariableConstraint,
)
from funman.representation.parameter import Schedules
from funman.server.query import FunmanWorkRequest


def get_request(request_path):
    if request_path is None:
        return FunmanWorkRequest()

    with open(request_path, "r") as request:
        funman_request = FunmanWorkRequest.model_validate(json.load(request))
        return funman_request


def get_model(model):
    return (
        Runner().get_model(model)
        if isinstance(model, dict)
        else Runner().get_model(model)
    )


def set_timepoints(funman_request, timepoints):
    if (
        funman_request.structure_parameters is not None
        and len(funman_request.structure_parameters) > 0
    ):
        funman_request.structure_parameters[0].schedules = [
            EncodingSchedule(timepoints=timepoints)
        ]
    else:
        funman_request.structure_parameters = [
            Schedules(schedules=[EncodingSchedule(timepoints=timepoints)])
        ]


def unset_all_labels(funman_request):
    if funman_request.parameters is not None:
        for p in funman_request.parameters:
            p.label = "any"


def set_config_options(
    funman_request, debug=False, dreal_precision=1e-3, mode=MODE_SMT
):
    if funman_request.config is None:
        funman_request.config = FUNMANConfig()
    # Overrides for configuration
    #
    # funman_request.config.substitute_subformulas = True
    # funman_request.config.use_transition_symbols = True
    # funman_request.config.use_compartmental_constraints=False
    if debug:
        funman_request.config.save_smtlib = "./out"
    funman_request.config.tolerance = 0.01
    funman_request.config.dreal_precision = dreal_precision
    funman_request.config.verbosity = logging.ERROR
    funman_request.config.mode = mode
    funman_request.config.normalize = False
    funman_request.config.random_seed = 3
    # funman_request.config.dreal_log_level = "debug"
    # funman_request.config.dreal_prefer_parameters = ["beta","NPI_mult","r_Sv","r_EI","r_IH_u","r_IH_v","r_HR","r_HD","r_IR_u","r_IR_v"]


def get_synthesized_vars(funman_request):
    return (
        [p.name for p in funman_request.parameters if p.label == "all"]
        if funman_request.parameters is not None
        else []
    )


def run(funman_request, model, models, plot=False, SAVED_RESULTS_DIR="./out"):
    to_synthesize = get_synthesized_vars(funman_request)
    results = Runner().run(
        models[model],
        funman_request,
        description="SIERHD Eval 12mo Scenario 1 q1",
        case_out_dir=SAVED_RESULTS_DIR,
        dump_plot=plot,
        print_last_time=True,
        parameters_to_plot=to_synthesize,
    )
    return results


def setup_common(
    funman_request,
    timepoints,
    synthesize=False,
    debug=False,
    dreal_precision=1e-3,
    mode=MODE_SMT,
):
    set_timepoints(funman_request, timepoints)
    if not synthesize:
        unset_all_labels(funman_request)
    set_config_options(
        funman_request, debug=debug, dreal_precision=dreal_precision, mode=mode
    )


def set_compartment_bounds(
    funman_request, model, upper_bound=9830000.0, error=0.01
):
    # Add bounds to compartments
    for var in states[model]:
        funman_request.constraints.append(
            StateVariableConstraint(
                name=f"{var}_bounds",
                variable=var,
                interval=Interval(
                    lb=0, ub=upper_bound, closed_upper_bound=True
                ),
                soft=False,
            )
        )

    # Add sum of compartments
    funman_request.constraints.append(
        LinearConstraint(
            name=f"compartment_bounds",
            variables=states[model],
            additive_bounds=Interval(
                lb=upper_bound - error,
                ub=upper_bound + error,
                closed_upper_bound=False,
            ),
            soft=True,
        )
    )


def relax_parameter_bounds(funman_request, factor=0.1):
    # Relax parameter bounds
    parameters = funman_request.parameters
    for p in parameters:
        interval = p.interval
        width = float(interval.width())
        interval.lb = interval.lb - (factor / 2 * width)
        interval.ub = interval.ub + (factor / 2 * width)


def plot_last_point(results, states, plot_logscale=False):
    pts = results.parameter_space.points()
    print(f"{len(pts)} points")

    if len(pts) > 0:
        # Get a plot for last point
        df = results.dataframe(points=pts[-1:])
        # pd.options.plotting.backend = "plotly"
        for s in states:
            df[s] = df[s].mask(np.isinf)

        ax = df[states].plot()

        if plot_logscale:
            ax.set_yscale("symlog")

        fig = plt.figure()
        # fig.set_yscale("log")
        # fig.savefig("save_file_name.pdf")
        plt.close()


def get_last_point_parameters(results):
    pts = results.parameter_space.points()
    if len(pts) > 0:
        pt = pts[-1]
        parameters = results.model._parameter_names()
        param_values = {k: v for k, v in pt.values.items() if k in parameters}
        return param_values


def pretty_print_request_params(params):
    # print(json.dump(params, indent=4))
    if len(params) > 0:

        df = pd.DataFrame(params)
        print(df.T)

def runtime_stats_dataframe(request_params):
    df = pd.DataFrame(request_params).T
    df.index.name = "Model"
    df = df[["model_size", "total_time", "time_horizon"]].rename(columns={"model_size": "Model Size", "total_time": "Total Time", "time_horizon": "Time Horizon"}).sort_index()
    return df

def report(results, name, states, request_results, request_params, plot_logscale=False, plot=True):
    request_results[name] = results
    if plot:
        plot_last_point(results, states, plot_logscale=plot_logscale)
    param_values = get_last_point_parameters(results)
    # print(f"Point parameters: {param_values}")
    if param_values is not None:
        param_values["total_time"] = results.timing.total_time
        param_values["model_size"] = results.model.num_elements()
        param_values["time_horizon"] = results.time_horizon()
        request_params[name] = param_values
    pretty_print_request_params(request_params)


def add_unit_test(funman_request, model="sidarthe_observables"):
    if model == "destratified_SEI":
        mstates = states["destratified_SEI"]
        funman_request.constraints.append(
            LinearConstraint(
                name="compartment_lb",
                soft=False,
                variables=[s for s in mstates if s.endswith("_lb")],
                additive_bounds={"ub": 19340000.5},
            )
        )
        funman_request.constraints.append(
            LinearConstraint(
                name="compartment_ub",
                soft=False,
                variables=[s for s in mstates if s.endswith("_ub")],
                additive_bounds={"lb": 0},
            )
        )


def plot_bounds(
    point,
    results,
    timespan=None,
    fig=None,
    axs=None,
    vars=["S", "E", "I", "R", "D", "H"],
    model=None,
    basevar_map={},
    **kwargs,
):

    if point.simulation is not None:
        df = point.simulation.dataframe().T
    else:
        df = results.dataframe([point])

    if timespan is not None:
        df = df.loc[timespan[0] : timespan[1]]

    # print(df)

    # Drop the ub vars because they are paired with the lb vars
    no_ub_vars = [v for v in vars if not v.endswith("_ub")]
    no_strat_vars = [v for v in no_ub_vars if not "_noncompliant" in v]

    if fig is None and axs is None:
        fig, axs = plt.subplots(len(basevar_map))
        fig.set_figheight(3 * len(basevar_map))
        fig.suptitle("Variable Bounds over time")

    for var in no_strat_vars:
        # print(var)
        # Get index of list containing var
        i = next(iter([i for i, bv in enumerate(basevar_map) if var in bv]))
        # print(i)
        if var.endswith("_lb"):
            # var is lower bound
            basevar = var.split("_lb")[0]
            lb = f"{basevar}_lb"
            ub = f"{basevar}_ub"
            labels = [lb, ub]
        elif var.endswith("_ub"):
            # skip, handled as part of lb
            continue
        else:
            # var is not of the form varname_lb
            basevar = var
            labels = basevar

        if "_compliant" in basevar:
            basevar = basevar.split("_")[0]
            if isinstance(labels, list):
                lb = (
                    df[f"{basevar}_compliant_lb"]
                    + df[f"{basevar}_noncompliant_lb"]
                )
                ub = (
                    df[f"{basevar}_compliant_ub"]
                    + df[f"{basevar}_noncompliant_ub"]
                )
                labels = [f"{basevar}_lb", f"{basevar}_ub"]
                data = pd.concat([lb, ub], axis=1, keys=labels)

            else:
                data = (
                    df[f"{basevar}_compliant"] + df[f"{basevar}_noncompliant"]
                )
                labels = f"{basevar}"
        else:
            # print(labels)
            data = df[labels]
            if "_compliant" in basevar:
                basevar = basevar.split("_")[0]
                labels = f"{basevar}"

        legend_labels = labels
        if model is not None:
            legend_labels = (
                [f"{model}_{k.rsplit('_', 1)[0]}" for k in labels[0:1]][0]
                if isinstance(labels, list)
                else f"{model}_{labels}"
            )

        # Fill between lb and ub
        if isinstance(labels, list):
            axs[i].fill_between(
                data.index,
                data[labels[0]],
                data[labels[1]],
                label=legend_labels,
                **kwargs,
            )
        else:
            if "hatch" in kwargs:
                del kwargs["hatch"]
            if "alpha" in kwargs:
                del kwargs["alpha"]
            axs[i].plot(data, label=legend_labels, **kwargs)
        axs[i].set_title(f"{basevar} Bounds")

        # axs[i].set_yscale('logit')

        # axs[i].legend(loc="outer")
        axs[i].legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            ncol=1,
            fancybox=True,
            shadow=True,
            prop={"size": 8},
            markerscale=2,
        )
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # fig.tight_layout()
    return fig, axs

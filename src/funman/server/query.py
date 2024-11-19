import logging
import random
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydantic import BaseModel, ValidationInfo, field_validator

from funman import LABEL_ANY, ModelParameter
from funman.config import FUNMANConfig
from funman.model.bilayer import BilayerModel
from funman.model.decapode import DecapodeModel
from funman.model.encoded import EncodedModel
from funman.model.ensemble import EnsembleModel
from funman.model.generated_models.petrinet import Model as GeneratedPetriNet
from funman.model.model import is_observable, is_state_variable
from funman.model.petrinet import GeneratedPetriNetModel, PetrinetModel
from funman.model.query import QueryAnd, QueryFunction, QueryLE, QueryTrue
from funman.model.regnet import GeneratedRegnetModel, RegnetModel
from funman.representation.constraint import FunmanUserConstraint
from funman.representation.explanation import Explanation
from funman.representation.parameter import (
    ModelParameter,
    NumSteps,
    Parameter,
    Schedules,
    StepSize,
)
from funman.representation.representation import Point
from funman.scenario.consistency import (
    ConsistencyScenario,
    ConsistencyScenarioResult,
)
from funman.scenario.parameter_synthesis import (
    ParameterSynthesisScenario,
    ParameterSynthesisScenarioResult,
)
from funman.scenario.scenario import AnalysisScenario

from ..representation.parameter_space import ParameterSpace

l = logging.getLogger(__name__)


class FunmanWorkRequest(BaseModel):
    query: Optional[Union[QueryAnd, QueryLE, QueryFunction, QueryTrue]] = None
    constraints: Optional[List[FunmanUserConstraint]] = None
    parameters: Optional[List[ModelParameter]] = None
    config: Optional[FUNMANConfig] = None
    structure_parameters: Optional[
        List[Union[Schedules, Union[NumSteps, StepSize]]]
    ] = None

    def parameter(self, name: str) -> ModelParameter:
        return next(iter([x for x in self.parameters if x.name == name]))

    @field_validator("constraints")
    @classmethod
    def check_unique_names(
        cls,
        constraints: Optional[List[FunmanUserConstraint]],
        info: ValidationInfo,
    ) -> Optional[List[FunmanUserConstraint]]:
        if constraints is not None and len(constraints) > 0:
            name_counts = Counter([c.name for c in constraints])
            dups = {k: v for k, v in name_counts.items() if v > 1}
            assert (
                max(name_counts.values()) == 1
            ), f"Constraint names need to be unique, duplicate counts: {dups}"
            assert (None not in name_counts) or name_counts[
                None
            ] == 0, f"Constraints need names, found {name_counts[None]} constraints without names."
        return constraints

    def time_horizon(self):
        try:
            schedules = next(
                iter(
                    [
                        p
                        for p in self.structure_parameters
                        if p.name == "schedules"
                    ]
                )
            )
        except StopIteration:
            l.exception("Could not find a Schedule structure parameter.")
        time_horizon = max([max(s.timepoints) for s in schedules.schedules])
        return time_horizon


class FunmanProgress(BaseModel):
    progress: float = 0.0
    coverage_of_search_space: float = 0.0
    coverage_of_representable_space: float = 0.0

    def __str__(self) -> str:
        return f"progress: {self.progress:.5f}"


class FunmanWorkUnit(BaseModel):
    """
    Fields
    ------
    id : The UUID assigned to the request
    request : A copy of the request associated with this response
    """

    id: str
    progress: FunmanProgress = FunmanProgress()
    model: Union[
        RegnetModel,
        PetrinetModel,
        DecapodeModel,
        BilayerModel,
        GeneratedRegnetModel,
        GeneratedPetriNetModel,
    ]
    request: FunmanWorkRequest

    def to_scenario(
        self,
    ) -> Union[ConsistencyScenario, ParameterSynthesisScenario]:
        query = (
            self.request.query
            if self.request.query is not None
            else QueryTrue()
        )

        parameters = []
        if (
            hasattr(self.request, "parameters")
            and self.request.parameters is not None
        ):
            for data in self.request.parameters:
                parameters.append(
                    ModelParameter(
                        name=data.name,
                        interval=data.interval,
                        label=data.label,
                    )
                )
        if (
            hasattr(self.request, "structure_parameters")
            and self.request.structure_parameters is not None
        ):
            for data in self.request.structure_parameters:
                parameters.append(
                    data
                    # StructureParameter(
                    #     name=data.name,
                    #     ub=data.ub,
                    #     lb=data.lb,
                    #     label=data.label,
                    # )
                )

        if (
            not hasattr(self.request, "parameters")
            or self.request.parameters is None
            or all(p.label == LABEL_ANY for p in self.request.parameters)
            or all(p.width() == 0.0 for p in self.request.parameters)
        ):
            return ConsistencyScenario(
                model=self.model,
                query=query,
                parameters=parameters,
                constraints=self.request.constraints,
            )

        if isinstance(self.model, EnsembleModel):
            raise Exception(
                "TODO handle EnsembleModel for ParameterSynthesisScenario"
            )

        return ParameterSynthesisScenario(
            model=self.model,
            query=query,
            parameters=parameters,
            constraints=self.request.constraints,
        )


class FunmanResultsTiming(BaseModel):
    start_time: datetime = None
    end_time: Optional[datetime] = None
    total_time: Optional[timedelta] = None
    solver_time: Optional[timedelta] = None
    encoding_time: Optional[timedelta] = None
    progress_timeseries: List[Tuple[datetime, float]] = []
    additional_time: Dict[str, timedelta] = {}

    def update_progress(
        self, progress, granularity=timedelta(seconds=1)
    ) -> None:
        last_update = (
            self.progress_timeseries[-1][0]
            if len(self.progress_timeseries) > 0
            else None
        )

        now = datetime.now()

        if last_update is not None:
            time_delta = now - last_update
        else:
            time_delta = now - self.start_time

        if time_delta > granularity:
            self.progress_timeseries.append((now, progress))

    def finalize(self):
        """Calculate total time"""
        try:
            self.total_time = self.end_time - self.start_time
        except Exception as e:
            l.exception(
                f"Exception in FunmanResultsTiming:finalize() start_time: {self.start_time} end_time: {self.end_time}"
            )


class FunmanResults(BaseModel):
    _finalized: bool = False

    id: str
    model: Union[
        GeneratedRegnetModel,
        GeneratedPetriNetModel,
        RegnetModel,
        PetrinetModel,
        DecapodeModel,
        BilayerModel,
        EncodedModel,
    ]
    progress: FunmanProgress = FunmanProgress()
    request: FunmanWorkRequest
    done: bool = False
    error: bool = False
    error_message: Optional[str] = None
    parameter_space: Optional[ParameterSpace] = None
    timing: FunmanResultsTiming = FunmanResultsTiming()
    contracted_model: Optional[GeneratedPetriNet] = None

    def start(self):
        self.timing.start_time = datetime.now()

    def stop(self):
        self.timing.end_time = datetime.now()
        self.timing.finalize()

    def is_final(self):
        return self._finalized

    def time_horizon(self):
        return self.request.time_horizon()

    def contract_model(self):
        """
        Use the parameter_space to contract the model parameter bounds and set self.contracted_model

        """
        if not isinstance(self.model, GeneratedPetriNetModel):
            raise NotImplementedError(
                f"Cannot contract model of type {type(self.model)}"
            )

        # Get new bounds for each parameter
        amr_parameters = self.model._parameter_names()
        last_step = self.parameter_space.last_step(true_only=True)
        parameter_bounds = {
            param: self.parameter_space.outer_interval(
                param, true_only=True, steps=[last_step]
            )
            for param in amr_parameters
        }
        self.contracted_model = self.model.contract_parameters(
            parameter_bounds
        )
        return last_step

    def update_parameter_space(
        self, scenario: AnalysisScenario, results: ParameterSpace
    ) -> FunmanProgress:
        # TODO handle copy?
        self.parameter_space = results
        # compute volumes
        labeled_volume = results.labeled_volume(scenario=scenario)
        # TODO precompute and cache?
        search_volume = scenario.search_space_volume(normalize=True)
        # TODO precompute and cache?
        repr_volume = scenario.representable_space_volume()
        # compute ratios
        if search_volume == 0.0:
            # TODO handle point volume?
            coverage_of_search_space = 0.0
        else:
            coverage_of_search_space = float(labeled_volume / search_volume)
            coverage_of_search_space = round(coverage_of_search_space, 15)

        if repr_volume == 0.0:
            # TODO handle point volume?
            coverage_of_repr_space = 0.0
        else:
            coverage_of_repr_space = float(search_volume / repr_volume)

        self.progress.progress = coverage_of_search_space
        self.progress.coverage_of_search_space = coverage_of_search_space
        self.progress.coverage_of_representable_space = coverage_of_repr_space

        try:
            self.timing.update_progress(self.progress.coverage_of_search_space)
        except Exception as e:
            l.exception(f"Unable to update progress due to exception: {e}")

        try:

            before_params = {}
            if self.contracted_model:
                before_params = {
                    p.id: (
                        p.distribution.parameters["minimum"],
                        p.distribution.parameters["maximum"],
                    )
                    for p in self.contracted_model.semantics.ode.parameters
                    if p.distribution
                }
                # l.info(f"Before { before_params }")
            last_step = self.contract_model()
            after_params = {
                p.id: (
                    p.distribution.parameters["minimum"],
                    p.distribution.parameters["maximum"],
                )
                for p in self.contracted_model.semantics.ode.parameters
                if p.distribution
            }
            if after_params != before_params:
                l.debug(f"Contracted @ {last_step} :  { after_params }")
        except NotImplementedError as e:
            l.info(
                f"Bypassing output of contracted model because it is not implmented for this model type: {type(self.model)}"
            )

        return self.progress

    def finalize_result(
        self,
        scenario: AnalysisScenario,
        result: Union[
            ConsistencyScenarioResult, ParameterSynthesisScenarioResult
        ],
    ):
        if self._finalized:
            raise Exception("FunmanResults was already finalized")
        self._finalized = True
        ps = None
        if isinstance(result, ConsistencyScenarioResult):
            ps = result.parameter_space
        if isinstance(result, ParameterSynthesisScenarioResult):
            ps = result.parameter_space

        if ps is None:
            raise Exception("No ParameterSpace for result")
        try:
            self.update_parameter_space(scenario, ps)
        except Exception as e:
            l.error(
                f"Could not update the parameter space while finalizing the result because: {e}"
            )
        self.done = True
        self.progress.progress = 1.0

    def finalize_result_as_error(self, message=None):
        if self._finalized:
            raise Exception("FunmanResults was already finalized")
        self._finalized = True
        self.error = True
        self.error_message = message
        self.done = True
        self.progress.progress = 1.0

    def _scenario(self) -> AnalysisScenario:
        scenario = FunmanWorkUnit(
            id=self.id, model=self.model, request=self.request
        ).to_scenario()

        # Needed to extract
        scenario._process_parameters()
        return scenario

    def point_parameters(
        self, point: Point, scenario: AnalysisScenario = None
    ) -> Dict[Parameter, float]:
        if scenario is None:
            scenario = self._scenario()
        parameters = scenario.model_parameters()
        return {
            p: point.values[p.name]
            for p in parameters
            if p.name in point.values
        }

    def dataframe(
        self, points: List[Point], interpolate="linear", max_time=None
    ):
        """
        Extract a timeseries as a Pandas dataframe.

        Parameters
        ----------
        interpolate : str, optional
            interpolate between time points, by default "linear"

        Returns
        -------
        pandas.DataFrame
            the timeseries

        Raises
        ------
        Exception
            fails if scenario is not consistent
        """
        scenario = self._scenario()
        to_plot = (
            scenario.model._state_var_names()
            + scenario.model._observable_names()
        )
        time_var = scenario.model._time_var()
        if time_var:
            to_plot += ["timer_t"]

        all_df = pd.DataFrame()
        for i, point in enumerate(points):
            timeseries = self.symbol_timeseries(point, to_plot)
            df = pd.DataFrame.from_dict(timeseries)

            if interpolate:
                new_index = np.linspace(
                    df["index"].min(),
                    df["index"].max(),
                    num=int(
                        (df["index"].max() - df["index"].min())
                        / df["index"].diff().min()
                    ),
                )
                df = (
                    df.infer_objects(copy=False)
                    .reindex(new_index)
                    .interpolate(method=interpolate)
                )

            df["id"] = i
            parameters = self.point_parameters(point=point, scenario=scenario)
            for p, v in parameters.items():
                if (
                    isinstance(v, int)
                    or isinstance(v, float)
                    or isinstance(v, bool)
                ):
                    df[p.name] = v
            df["label"] = point.label
            df["label"] = df["label"].astype(str)
            # if max_time:
            # if time_var:
            #     df = df.at[max_time, :] = None
            # df = df.reindex(range(max_time+1), fill_value=None)

            if time_var and any("timer_t" in x for x in df.columns):
                df = (
                    df.rename(columns={"timer_t": "time"})
                    .set_index("time", drop=True)
                    .drop(columns=["index"])
                )

            df = df.reindex(sorted(df.columns), axis=1)

            all_df = pd.concat([all_df, df])

        return all_df

    def symbol_timeseries(
        self, point: Point, variables: List[str]
    ) -> Dict[str, List[Union[float, None]]]:
        """
        Generate a symbol (str) to timeseries (list) of values

        Parameters
        ----------
        pysmtModel : pysmt.solvers.solver.Model
            variable assignment
        """
        series = self.symbol_values(point, variables)
        a_series = {}  # timeseries as array/list
        timestep = point.timestep()
        max_t = point.schedule.timepoints[timestep]

        a_series["index"] = point.schedule.timepoints

        for var, tps in series.items():

            if isinstance(tps, dict):
                vals = [None] * len(a_series["index"])
                for t, v in tps.items():
                    i = point.schedule.timepoints.index(t)
                    vals[i] = v
                    # if (not isinstance(t, float) or t.isdigit()) and int(
                    #     t
                    # ) <= int(max_t):
                    #     vals[int(t)] = v
                a_series[var] = vals
            else:
                a_series[var] = [tps] * len(a_series["index"])
        return a_series

    def symbol_values(
        self, point: Point, variables: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
         Get the value assigned to each symbol in the pysmtModel.

        Parameters
        ----------
        model_encoding : Encoding
            encoding using the symbols
        pysmtModel : pysmt.solvers.solver.Model
            assignment to symbols

        Returns
        -------
        Dict[str, Dict[str, float]]
            mapping from symbol and timepoint to value
        """

        vars = self._symbols(point, variables)
        vals = {}
        for var in vars:
            vals[var] = {}
            if isinstance(vars[var], dict):
                for t in vars[var]:
                    try:
                        value = point.values[vars[var][t]]
                        vals[var][t] = float(value)
                    except OverflowError as e:
                        l.warning(e)
            else:
                vals[var] = point.values[vars[var]]
        return vals

    def _symbols(
        self, point: Point, variables: List[str]
    ) -> Dict[str, Dict[str, str]]:
        symbols = {}
        for var in point.values:
            if is_state_variable(var, self.model) or is_observable(
                var, self.model
            ):
                var_name, timepoint = self._split_symbol(var)
                if timepoint is not None:
                    if var_name not in symbols:
                        symbols[var_name] = {}
                    symbols[var_name][timepoint] = var
                elif timepoint is None:
                    # Could be an observable with not time index
                    if var_name not in symbols:
                        symbols[var_name] = {}
                    symbols[var_name] = var
        return symbols

    def _split_symbol(self, symbol: str) -> Tuple[str, str]:
        try:
            s, t = symbol.rsplit("_", 1)
        except ValueError:
            s = symbol
            t = None
        try:
            t = int(t)
        except Exception:
            try:
                t = float(t)
            except Exception:
                t = None
        return s, t

    def plot_trajectories(self, variable: str, num: int = 200):
        fig, ax = plt.subplots()
        len_tps = len(self.parameter_space.true_points())
        len_fps = len(self.parameter_space.false_points())
        num_tp_samples = min(len_tps, num)
        num_fp_samples = min(len_fps, num)

        tps = random.sample(self.parameter_space.true_points(), num_tp_samples)
        fps = random.sample(
            self.parameter_space.false_points(), num_fp_samples
        )
        if len(tps) > 0:
            tps_df = self.dataframe(tps)
            # tps_df = tps_df[tps_df[variable] != 0.0]
            tps_df.groupby("id")[variable].plot(c="green", alpha=0.2, ax=ax)
        if len(fps) > 0:
            fps_df = self.dataframe(fps)
            # fps_df = fps_df[fps_df[variable] != 0.0]
            fps_df.groupby("id")[variable].plot(c="red", alpha=0.2, ax=ax)

        return ax

    def points(self) -> List[Point]:
        return self.parameter_space.points()

    def plot(
        self,
        points: Optional[List[Point]] = None,
        variables=None,
        log_y=False,
        max_time=None,
        title="Point Trajectories",
        xlabel="Time",
        ylabel="Population",
        label_marker={"true": "+", "false": "o"},
        label_color={"true": "g", "false": "r"},
        legend=None,
        dpi=100,
        **kwargs,
    ):
        """
        Plot the results in a matplotlib plot.

        Raises
        ------
        Exception
            failure if scenario is not consistent.
        """

        import logging

        # remove matplotlib debugging
        logging.getLogger("matplotlib.font_manager").disabled = True
        logging.getLogger("matplotlib.pyplot").disabled = True

        if points is None:
            points = self.points()

        df = self.dataframe(points, max_time=max_time)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
        groups = df.groupby("label")
        for label, group in groups:
            for id, g in group.groupby("id"):
                state_vars = g[self.model._state_var_names()]
                if variables is not None:
                    ax = plt.plot(
                        state_vars[variables],
                        label=label,
                        marker=label_marker[label],
                        c=label_color[label],
                        **kwargs,
                    )
                else:
                    # data = [state_vars[c].tolist() for c in state_vars.columns]
                    ax = plt.plot(
                        # data, #
                        state_vars,
                        label=label,
                        marker=label_marker[label],
                        c=label_color[label],
                        **kwargs,
                    )
            # ax = df.plot(label=label, marker=label_marker[label], **kwargs)
        if legend:
            plt.legend(legend)
        if log_y:
            ax.set_yscale("symlog")
            plt.ylim(bottom=0)
        # plt.show(block=False)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        return ax

    def explain(self) -> Explanation:
        return self.parameter_space.explain()

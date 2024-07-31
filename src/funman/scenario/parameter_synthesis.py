"""
This module defines the Parameter Synthesis scenario.
"""

import threading
from typing import Callable, List, Optional, Union

from pandas import DataFrame
from pydantic import BaseModel, ConfigDict

from funman.representation.parameter import Schedules
from funman.representation.representation import Point
from funman.scenario import (
    AnalysisScenario,
    AnalysisScenarioResult,
    ConsistencyScenario,
)
from funman.utils.math_utils import minus

from ..representation.parameter_space import ParameterSpace


class ParameterSynthesisScenario(AnalysisScenario, BaseModel):
    """
    Parameter synthesis problem description that identifies the parameters to
    synthesize for a particular model.  The general problem is to identify
    multi-dimensional (one dimension per parameter) regions where either all
    points in the region are valid (true) parameters for the model or invalid
    (false) parameters.
    """

    _search: str = "BoxSearch"

    # _assume_model: Optional[FNode] = None
    # _assume_query: Optional[FNode] = None

    @classmethod
    def get_kind(cls) -> str:
        return "parameter_synthesis"

    def get_search(self, config: "FUNMANConfig") -> "Search":
        if config._search is None:
            from funman.search.box_search import BoxSearch

            search = BoxSearch()
        else:
            search = config._search()
        return search

    def solve(
        self,
        config: "FUNMANConfig",
        haltEvent: Optional[threading.Event] = None,
        resultsCallback: Optional[Callable[["ParameterSpace"], None]] = None,
    ) -> "ParameterSynthesisScenarioResult":
        """
        Synthesize parameters for a model.  Use the BoxSearch algorithm to
        decompose the parameter space and identify the feasible and infeasible
        parameter values.

        Parameters
        ----------
        config : SearchConfig
            Options for the Search algorithm.

        Returns
        -------
        ParameterSpace
            The parameter space.
        """
        search = self.initialize(config)

        # schedules = self.parameters_of_type(Schedules)
        # assert len(schedules) <= 1, "Cannot have more than one Schedules parameter."
        # if len(schedules) == 1:
        #     self._original_parameter_widths[schedules[0]] = [len(s.timepoints) for s in schedules[0].schedules]

        parameter_space: ParameterSpace = search.search(
            self,
            config,
            haltEvent=haltEvent,
            resultsCallback=resultsCallback,
        )

        parameter_space.num_dimensions = len(self.parameters)
        return ParameterSynthesisScenarioResult(
            parameter_space=parameter_space, scenario=self
        )


class ParameterSynthesisScenarioResult(AnalysisScenarioResult, BaseModel):
    """
    ParameterSynthesisScenario result, which includes the parameter space and
    search statistics.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # episode: SearchEpisode
    scenario: ParameterSynthesisScenario
    parameter_space: ParameterSpace

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameter_space._reassign_point_labels()
        self.parameter_space._denormalize()
        self.parameter_space._compact()

    def plot(self, **kwargs):
        """
        Plot the results

        Raises
        ------
        NotImplementedError
            TODO
        """
        raise NotImplementedError(
            "ParameterSynthesisScenario.plot() is not implemented"
        )

    # points are of the form (see Point)
    # [
    #     {"values": {"beta": 0.1}}
    # ]
    # Or
    # List[Point]
    def true_point_timeseries(
        self, points: Union[List[Point], List[dict]] = List[DataFrame]
    ):
        """
        Get a timeseries for each of the points, assuming that the points are in the parameter space.

        Parameters
        ----------
        points : Union[List[Point], List[dict]], optional
            points to use to genrate a timeseries, by default None

        Returns
        -------
        List[Dataframe]
            a list of dataframes that list a timeseries of values for each model variable.

        Raises
        ------
        Exception
            malformed points
        """
        # for each true box
        from funman.config import FUNMANConfig

        dfs = []
        for point in points:
            if isinstance(point, dict):
                point = Point.model_validate(point)
            if not isinstance(point, Point):
                raise Exception("Provided point is not of type Point")
            # update the model with the
            for p, v in point.values.items():
                # assign that parameter to the value of the picked point
                self.scenario.model.parameter_bounds[p.name] = [v, v]

            # check the consistency
            scenario = ConsistencyScenario(
                self.scenario.model,
                self.scenario.query,
                _smt_encoder=self.scenario._smt_encoder,
            )
            result = scenario.solve(
                config=FUNMANConfig(solver="dreal", _search="SMTCheck")
            )
            assert result
            # plot the results
            # result.plot(logy=True)
            # print(result.dataframe())
            dfs.append(result.dataframe())
        return dfs

    def __repr__(self) -> str:
        return str(self.model_dump())

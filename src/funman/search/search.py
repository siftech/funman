import datetime
import logging
import threading
from abc import ABC, abstractmethod
from decimal import Decimal
from multiprocessing import Array, Queue, Value
from queue import Queue as SQueue
from typing import Callable, List, Optional, Union

import pysmt
from pydantic import BaseModel, ConfigDict
from pysmt.shortcuts import TRUE, Solver
from pysmt.solvers.solver import Model as pysmtModel

from funman import Box, Interval, ModelParameter
from funman.translate.translate import EncodingSchedule

from ..config import FUNMANConfig
from ..representation.explanation import BoxExplanation, TimeoutExplanation
from ..scenario.scenario import AnalysisScenario

l = logging.getLogger(__name__)

import queue
from multiprocessing import JoinableQueue, Process


def run_with_limited_time(func, args, kwargs, time):
    p = Process(target=func, args=args, kwargs=kwargs)
    l.debug(f"start: {datetime.datetime.now()}, with timout: {time}")
    p.start()
    # p.join(timeout=time)
    try:
        result = args[-1].get(timeout=time)

    except queue.Empty:
        l.debug("empty queue")
        result = None
        # l.debug("get timedout or done")
        # try:
        #     args[-1].task_done()
        #     l.debug("signaled q task done")
        # except ValueError:
        #     pass

    try:
        args[-1].task_done()
        l.debug("signaled q task done")
    except ValueError:
        pass

    l.debug("Awaiting join ...")
    p.join(timeout=time)
    l.debug("joined process")

    # print(f"exitcode: {p.exitcode}, return: {r}")
    # print(args)
    if p.is_alive():
        l.debug(f"kill: {datetime.datetime.now()}")
        p.terminate()
        # sys.exit(1)
        return None
    # p.join()
    l.debug(f"end: {datetime.datetime.now()}")
    # print(args[-1].get())
    # sys.exit(1)
    return result
    # return True


class SearchStatistics(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    _multiprocessing: bool = False
    _num_true: int = 0
    _num_false: int = 0
    _num_unknown: int = 0
    _residuals: SQueue = None
    _current_residual: float = 0.0
    _last_time: List[datetime.datetime] = []
    _iteration_time: SQueue = None
    _iteration_operation: SQueue = None

    def __init__(self, **kw):
        super().__init__(**kw)
        self._residuals = SQueue()
        self._iteration_time = SQueue()
        self._iteration_operation = SQueue()


class SearchStaticsMP(SearchStatistics):
    _multiprocessing: bool = True
    _num_true: Value = 0
    _num_false: Value = 0
    _num_unknown: Value = 0
    _residuals: Queue = None
    _current_residual: Value = 0.0
    _last_time: Array = None
    _iteration_time: Queue = None
    _iteration_operation: Queue = None

    def __init__(self, **kw):
        super().__init__(**kw)

        manager = kw["manager"]
        self._multiprocessing = manager is not None
        self._num_true = manager.Value("i", 0)
        self._num_false = manager.Value("i", 0)
        self._num_unknown = manager.Value("i", 0)
        self._residuals = manager.Queue()
        self._current_residual = manager.Value("d", 0.0)
        self._last_time = manager.Array("u", "")
        self._iteration_time = manager.Queue()
        self._iteration_operation = manager.Queue()


class SearchEpisode(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    schedule: EncodingSchedule
    problem: AnalysisScenario
    config: "FUNMANConfig"
    statistics: SearchStatistics = None
    _model: pysmt.solvers.solver.Model = None

    def num_parameters(self):
        return len(self.problem.parameters)

    def _initial_box(self, schedule: EncodingSchedule) -> Box:
        box = Box(
            bounds={
                p.name: p.interval.model_copy(deep=True)
                for p in self.problem.parameters
                if (isinstance(p, ModelParameter))
            },
            schedule=schedule,
        )
        box.bounds["timestep"] = Interval(
            lb=0, ub=len(schedule.timepoints) - 1, closed_upper_bound=True
        )
        box.bounds["timestep"].original_width = Decimal(
            len(schedule.timepoints) - 1
        )

        return box


class Search(ABC):
    def __init__(self) -> None:
        self.episodes = []

    @abstractmethod
    def search(
        self,
        problem: "AnalysisScenario",
        config: Optional["FUNMANConfig"] = None,
        haltEvent: Optional[threading.Event] = None,
        resultsCallback: Optional[Callable[["ParameterSpace"], None]] = None,
    ) -> SearchEpisode:
        pass

    def invoke_solver(
        self, s: Solver, timeout: int = None
    ) -> Union[pysmtModel, BoxExplanation]:
        l.debug("Invoking solver ...")
        q = JoinableQueue()
        if timeout is not None:

            result = run_with_limited_time(
                Search._internal_invoke_solver, (self, s, q), {}, timeout
            )
            # print(f"get from q, empty?: {q.empty()} ")

            if result is None:
                result = TimeoutExplanation()
                result._expression = TRUE()
        else:
            self._internal_invoke_solver(s, q)
            result = q.get()
        # print(f"invoke_solver, result: [{result}]")
        return result

    # @timeout_decorator.timeout(1)
    def _internal_invoke_solver(self, s: Solver, q):
        l.debug("Solver started")
        result = s.solve()
        l.trace(f"Solver result = {result}")
        try:
            if result:
                # print(f"put: {s.get_model()}")
                q.put(s.get_model())
            else:
                result = BoxExplanation()
                result._expression = s.get_unsat_core()
                # print(f"put: {result}")
                q.put(result)
        except RecursionError:
            l.error("Recursion error pickling solver result in queue.")
            pass
        l.debug("Solver completed")
        # print(result)
        # return result

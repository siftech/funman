"""
This module defines the BoxSearch class and supporting classes.

"""

import glob
import logging
import multiprocessing as mp
import os
import threading
import traceback
from datetime import datetime
from functools import partial
from multiprocessing import Queue, Value
from multiprocessing.synchronize import Condition, Event, Lock
from queue import Empty
from queue import PriorityQueue as PQueueSP
from queue import Queue as QueueSP
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict
from pysmt.formula import FNode
from pysmt.logics import QF_NRA
from pysmt.shortcuts import (
    BOOL,
    REAL,
    And,
    Bool,
    Equals,
    Implies,
    Not,
    Or,
    Real,
    Solver,
    Symbol,
)
from pysmt.solvers.solver import Model as pysmtModel

from funman import (
    LABEL_DROPPED,
    LABEL_FALSE,
    LABEL_TRUE,
    LABEL_UNKNOWN,
    ModelParameter,
)
from funman.config import FUNMANConfig
from funman.representation.assumption import Assumption
from funman.representation.constraint import ParameterConstraint
from funman.representation.explanation import (
    BoxExplanation,
    Explanation,
    TimeoutExplanation,
)
from funman.search import Box, ParameterSpace, Point, Search, SearchEpisode
from funman.search.search import SearchStaticsMP, SearchStatistics
from funman.translate.translate import EncodingOptions, EncodingSchedule
from funman.utils.smtlib_utils import smtlibscript_from_formula_list

l = logging.getLogger(__name__)


class FormulaStackFrame(BaseModel):
    _formulas: List[FNode] = []
    _simplified_formulas: List[FNode] = []

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def add_assertion(self, formula, is_simplified=False):
        if is_simplified:
            self._simplified_formulas.append(formula)
        else:
            self._formulas.append(formula)

    # def __str__(self) -> str:
    #     s_formulas = [f.serialize() for f in self.formulas]
    #     s_simplified_formulas = [f.serialize() for f in self.simplified_formulas]
    #     s_dict = str({"formulas": s_formulas, "simplified_formulas": s_simplified_formulas})
    #     return s_dict


class FormulaStack(BaseModel):
    formula_stack: List[FormulaStackFrame] = []
    time: int = -2
    _solver: Optional[Solver] = None
    _substitutions: Dict[FNode, FNode] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.push(1, push_solver=False)

    def pop(self, levels: int = 1, pop_solver: bool = True):
        for i in range(levels):
            if pop_solver:
                self._solver.pop(1)
            self.formula_stack.pop()
            self.time -= 1

    def push(self, levels: int = 1, push_solver: bool = True):
        for i in range(levels):
            self.formula_stack.append(FormulaStackFrame())
            if push_solver:
                self._solver.push(1)
            self.time += 1

    # def __str__(self)-> str:
    #     s_stack = str([str(frame) for frame in self.formula_stack])
    #     return str({"formula_stack": s_stack, "time": str(self.time)})

    def add_assertion(self, formula):
        self.formula_stack[self.time + 1].add_assertion(
            formula, is_simplified=False
        )

        if self._substitutions is not None:
            simplified_formula = formula.substitute(
                self._substitutions
            ).simplify()
            self.formula_stack[self.time + 1].add_assertion(
                simplified_formula, is_simplified=True
            )
            self._solver.add_assertion(simplified_formula)
        else:
            self._solver.add_assertion(formula)

    def to_list(self, simplified=False) -> List[FNode]:
        if simplified:
            return [
                f for sf in self.formula_stack for f in sf._simplified_formulas
            ]
        else:
            return [f for sf in self.formula_stack for f in sf._formulas]

    def compute_assignment(
        self, episode: SearchEpisode, _smtlib_save_fn: Callable = None
    ) -> pysmtModel:
        self.push()
        original_formulas = [
            formula
            for frame in self.formula_stack
            for formula in frame._formulas
        ]
        for formula in original_formulas:
            self._solver.add_assertion(formula)
            self.formula_stack[-1].add_assertion(
                formula, is_simplified=True
            )  # formula is not simplified but want to layer on top of simplified formulas to compute state variables
        if _smtlib_save_fn:
            _smtlib_save_fn(filename=f"box_search_{episode._iteration}")

        if not self._solver.solve():
            raise Exception(
                f"Could not compute Assignment from simplified formulas"
            )
        result = self._solver.get_model()
        self.pop()
        return result


class BoxSearchEpisode(SearchEpisode):
    """
    A BoxSearchEpisode stores the data required to organize a BoxSearch, including intermediate data and results. It takes as input:

    * config: a SearchConfig object that defines the configuration

    * problem: a ParameterSynthesisScenario to solve

    * manager: a SyncManager for coordinating multiple search processes that is used to declare several shared data

    A BoxSearchEpisode mainly tracks the true, false, and unknown boxes generated by the BoxSearch, along with statistics on the search.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # problem: ParameterSynthesisScenario
    statistics: SearchStatistics = None

    _true_boxes: List[Box] = []
    _false_boxes: List[Box] = []
    _true_points: Set[Point] = set({})
    _false_points: Set[Point] = set({})
    _unknown_boxes: PQueueSP
    _iteration: int = 0
    _formula_stack: FormulaStack = FormulaStack()
    schedule: EncodingSchedule

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._unknown_boxes = PQueueSP()
        self.statistics = SearchStatistics()
        if self.config.substitute_subformulas and self.config.simplify_query:
            self._formula_stack._substitutions = self.problem._encodings[
                self.schedule
            ]._encoder.substitutions(self.schedule)

    def get_candiate_point(self, box: Box) -> Point:
        return None

    def get_candidate_boxes_for_point(self, point: Point) -> List[Box]:
        return []

    def _initialize_boxes(self, expander_count, schedule: EncodingSchedule):
        # initial_box = self._initial_box()
        # if not self.add_unknown(initial_box):
        #     l.error(
        #         f"Did not add an initial box (of width {initial_box.width()}), try reducing config.tolerance, currently {self.config.tolerance}"
        #     )
        initial_boxes = QueueSP()
        initial_box = self._initial_box(schedule)

        initial_boxes.put(initial_box)

        num_boxes = 1
        while num_boxes < expander_count:
            b1, b2 = initial_boxes.get().split()
            initial_boxes.put(b1)
            initial_boxes.put(b2)
            num_boxes += 1
        for i in range(num_boxes):
            b = initial_boxes.get()
            if not self._add_unknown(b):
                l.error(
                    f"Did not find add an initial box (box had width {b.normalized_width()}), try reducing config.tolerance, currently {self.config.tolerance}"
                )
            # l.debug(f"Initial box: {b}")

    def _on_start(self):
        if self.config.number_of_processes > 1:
            self.statistics._last_time.value = str(datetime.now())
        else:
            self.statistics._last_time = str(datetime.now())

    # def close(self):
    #     if self.multiprocessing:
    #         self.unknown_boxes.close()
    #         self.statistics.close()
    #         self.boxes_to_plot.close()

    def _on_iteration(self):
        if self.config.number_of_processes > 1:
            self._iteration.value = self._iteration.value + 1
        else:
            self._iteration = self._iteration + 1

    def _add_unknown_box(self, box: Box) -> bool:
        if (
            box.width(
                parameters=self.problem.synthesized_model_parameters(),
                normalize=True,
            )
            > self.config.tolerance
        ):
            box.label = LABEL_UNKNOWN
            self._unknown_boxes.put(box)
            if self.config.number_of_processes > 1:
                self.statistics._num_unknown.value += 1
            else:
                self.statistics._num_unknown += 1
            return True
        else:
            box.label = LABEL_DROPPED
        return False

    def _add_unknown(self, box: Union[Box, List[Box]]):
        did_add = False
        if isinstance(box, list):
            for b in box:
                did_add |= self._add_unknown_box(b)
        else:
            did_add = self._add_unknown_box(box)
        return did_add

    def _add_false(self, box: Box, explanation: Explanation = None):
        box.label = LABEL_FALSE
        box.explanation = explanation
        self._false_boxes.append(box)
        # with self.statistics.num_false.get_lock():
        #     self.statistics.num_false.value += 1
        # self.statistics.iteration_operation.put("f")

    def _add_false_point(
        self, box: Box, point: Point, explanation: Explanation = None
    ):
        l.trace(f"Adding false point: {point}")
        if point in self._true_points:
            l.trace(
                f"Point: {point} is marked false, but already marked true."
            )
        point.label = LABEL_FALSE

    def _add_true(self, box: Box, explanation: Explanation = None):
        box.label = LABEL_TRUE
        box.explanation = explanation
        self._true_boxes.append(box)
        # with self.statistics.num_true.get_lock():
        #     self.statistics.num_true.value += 1
        # self.statistics.iteration_operation.put("t")

    def _add_true_point(self, box: Box, point: Point):
        l.trace(f"Adding true point: {point}")
        if point in self._false_points:
            l.trace(
                f"Point: {point} is marked true, but already marked false."
            )
        point.label = LABEL_TRUE

    def _get_unknown(self):
        box = self._unknown_boxes.get(timeout=self.config.queue_timeout)
        if self.config.number_of_processes > 1:
            self.statistics._num_unknown.value = (
                self.statistics._num_unknown.value - 1
            )
            self.statistics._current_residual.value = box.normalized_width()
        else:
            self.statistics._num_unknown += 1
            self.statistics._current_residual = box.normalized_width()
        self.statistics._residuals.put(box.normalized_width())
        this_time = datetime.now()
        # FIXME self.statistics.iteration_time.put(this_time - self.statistics.last_time.value)
        # FIXME self.statistics.last_time[:] = str(this_time)
        return box

    def _get_box_to_plot(self):
        return self.boxes_to_plot.get(timeout=self.config.queue_timeout)

    def _extract_point(self, model, box: Box):
        point = Point(
            values={
                p[0].symbol_name(): (
                    float(p[1].constant_value())
                    if p[1].is_real_constant()
                    else p[1].constant_value()
                )
                for p in model
            },
            schedule=box.schedule,
        )
        # Timestep is not in the model (implicit)
        point.values["timestep"] = box.timestep().lb
        point.remove_irrelevant_steps(
            self.problem._smt_encoder._untimed_symbols
        )
        return point


class BoxSearchEpisodeMP(BoxSearchEpisode):
    model_config = ConfigDict()

    statistics: SearchStaticsMP = None
    _unknown_boxes: Queue
    _iteration: Value = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        manager = kwargs["manager"]
        self._unknown_boxes = manager.Queue()
        self._iteration = manager.Value("i", 0)
        self.statistics = SearchStaticsMP(manager=kwargs["manager"])


class BoxSearch(Search):
    """
    Box search algorithm.
    """

    def _split(self, box: Box, episode: BoxSearchEpisode, points=None):
        normalize = episode.problem._original_parameter_widths
        split_points = (
            points if not episode.config.uniform_box_splits else None
        )
        b1, b2 = box.split(
            points=split_points,
            normalize=normalize,
            parameters=episode.problem.model_parameters(),
        )
        episode.statistics._iteration_operation.put("s")
        bw = box.volume(
            normalize=normalize, parameters=episode.problem.model_parameters()
        )
        b1w = b1.volume(
            normalize=normalize, parameters=episode.problem.model_parameters()
        )
        b2w = b2.volume(
            normalize=normalize, parameters=episode.problem.model_parameters()
        )
        l.trace(
            f"Split box with volume = {bw:.5f} into boxes with volumes = [{b1w:.5f}, {b2w:.5f}]"
        )
        return episode._add_unknown([b1, b2])

    def _logger(self, config, process_name=None):
        if config.number_of_processes > 1:
            l = mp.log_to_stderr()
            if process_name:
                l.name = process_name
            l.setLevel(config.verbosity)
        else:
            if not process_name:
                process_name = "BoxSearch"
            l = logging.Logger(process_name)
        return l

    def _handle_empty_queue(
        self, process_name, episode, more_work, idle_mutex, idle_flags
    ):
        if episode.config.number_of_processes > 1:
            # set this processes idle flag and check all the other process idle flags
            # doing so under the idle_mutex to ensure the flags are not under active change
            with idle_mutex:
                idle_flags[id].set()
                should_exit = all(f.is_set() for f in idle_flags)

            # all worker processes appear to be idle
            if should_exit:
                # one last check to see if there is work to be done
                # which would be an error at this point in the code
                if episode._unknown_boxes.qsize() != 0:
                    l.error(
                        f"{process_name} found more work while preparing to exit"
                    )
                    return False
                l.info(f"{process_name} is exiting")
                # tell other worker processing to check again with the expectation
                # that they will also resolve to exit
                with more_work:
                    more_work.notify()
                # break of the while True and allow the process to exit
                return True

            # wait for notification of more work
            l.info(f"{process_name} is awaiting work")
            with more_work:
                more_work.wait()

            # clear the current processes idle flag under the idle_mutex
            with idle_mutex:
                idle_flags[id].clear()

            return False
        else:
            return True

    def _simplify_formula(
        self,
        formula: FNode,
        episode: BoxSearchEpisode,
        options: EncodingOptions,
    ) -> FNode:
        substitutions = episode.problem._encodings[
            options.step_size
        ]._encoder.substitutions(options.step_size)
        return formula.substitute(substitutions).simplify()

    def _initialize_model_encoding(
        self,
        solver: Solver,
        episode: BoxSearchEpisode,
        options: EncodingOptions,
        box: Box,
    ):
        """
        The formula encoding the model M is of the form:

        AM <=> M

        where AM is a symbol denoting whether we assume M is true.  With this formula we can push/pop AM or Not(AM) to assert M or Not(M) without popping M.  Similarly we also assert the query as:

        AQ <==> Q

        Parameters
        ----------
        solver : Solver
            pysmt solver object
        episode : episode
            data for the current search
        """

        if box is None:
            # Signal to pop the formula stack to the start
            time_difference = -episode._formula_stack.time
        else:
            time_difference = box.timestep().lb - episode._formula_stack.time

        # if time_difference < 0:
        #     # Prepare the formula stack by popping irrelevant layers
        #     for i in range(abs(int(time_difference - 1))):
        #         episode._formula_stack.pop()

        # el
        if time_difference > 0:
            # Prepare the formulas for each added layer
            layer_formulas = []
            encoding = episode.problem._encodings[box.schedule]
            # model_encoding = episode.problem._model_encoding[step_size]
            # model_encoder = model_encoding._encoder
            # query_encoding = episode.problem._query_encoding[step_size]
            # query_encoder = query_encoding._encoder
            # step_size_idx = model_encoder._timed_model_elements[
            #     "step_sizes"
            # ].index(step_size)

            for t in range(
                episode._formula_stack.time + 1, int(box.timestep().lb) + 1
            ):
                timepoint = box.schedule.time_at_step(t)
                encoded_constraints = []
                for constraint in episode.problem.constraints:
                    if constraint.encodable() and constraint.relevant_at_time(
                        timepoint
                    ):
                        encoded_constraints.append(
                            encoding.construct_encoding(
                                episode.problem,
                                constraint,
                                options,
                                layers=[t],
                                box=box,
                                assumptions=episode.problem._assumptions,
                            )
                        )
                formula_encoded_constraints = And(encoded_constraints)
                formula = Implies(
                    self._solve_at_step_symbol(t), formula_encoded_constraints
                )

                # neg_formula = Implies(
                #     Not(self._solve_at_step_symbol(t)),
                #     And(
                #         [
                #             Equals(
                #                 encoding._encoder._encode_state_var(
                #                     s, time=timepoint
                #                 ),
                #                 Real(0.0),
                #             )
                #             for s in episode.problem.model._state_var_names()
                #         ]  # + [Not(s) for s in symbols if s.symbol_type() == BOOL]
                #     ),
                # )

                layer_formulas.append(
                    # And(
                    formula
                # , neg_formula)
                )

            for layer, formula in enumerate(layer_formulas):
                episode._formula_stack.push(1)
                episode._formula_stack.add_assertion(formula)

    def _solve_at_step_symbol(self, t: int) -> FNode:
        return Symbol(f"solve_step_{t}", BOOL)

    def _initialize_model_for_box(
        self,
        solver,
        box: Box,
        episode: BoxSearchEpisode,
        options: EncodingOptions,
    ):
        # Setup the model transitions to evaluate the box
        self._initialize_model_encoding(solver, episode, options, box)

        formula = self._initialize_box_encoding(box, episode, options)
        episode._formula_stack.push(1)
        episode._formula_stack.add_assertion(formula)

    def _initialize_box_encoding(
        self, box: Box, episode: SearchEpisode, options: EncodingOptions
    ) -> FNode:
        # Add constraints for boundaries of the box
        projected_box = box.project(
            episode.problem.model_parameters()
        ).project(episode.problem.model_parameters())

        parameter_formulas = []
        for parameter_name, interval in projected_box.bounds.items():
            parameter_formulas.append(
                episode.problem._smt_encoder.encode_constraint(
                    episode.problem,
                    ParameterConstraint(
                        name=parameter_name,
                        parameter=ModelParameter(
                            name=parameter_name, interval=interval
                        ),
                    ),
                    options=options,
                )[0]
            )
        formula = And(parameter_formulas)
        return formula

    def _setup_false_query(self, solver, episode, box, options):
        """
        Setup the assumptions so that satisfying the formulas requires that  either the model or the query is false

        Parameters
        ----------
        solver : Solver
            pysmt solver object
        episode : episode
            data for the current search
        """
        episode._formula_stack.push(1)
        timestep = int(box.timestep().lb)
        timepoint = box.schedule.time_at_step(timestep)
        encoder = episode.problem._encodings[options.schedule]._encoder
        assumptions = {
            k: v
            for k, v in encoder.encode_assumptions(
                episode.problem._assumptions, options
            ).items()
            if k.relevant_at_time(timepoint)
        }

        # Not all assumptions hold
        formula = Not(And([v for k, v in assumptions.items()]))

        # An assumption must hold at all times to hold overall
        formula1 = And(
            [
                Implies(
                    And(
                        [
                            encoder.encode_assumption(k, options, layer_idx=i)
                            for i in range(timestep + 1)
                        ]
                    ),
                    v,
                )
                for k, v in assumptions.items()
                if k.constraint.time_dependent()
            ]
        )

        # Each Assumption has held at all times previously, so check that it does not hold currently.
        # Need at least one to not hold currently.
        formula2 = Or(
            [
                And(
                    [
                        encoder.encode_assumption(k, options, layer_idx=i)
                        for i in range(timestep)
                    ]
                    + [
                        Not(
                            encoder.encode_assumption(
                                k, options, layer_idx=timestep
                            )
                        )
                    ]
                )
                for k, v in assumptions.items()
                if k.constraint.time_dependent()
            ]
            + [
                Not(encoder.encode_assumption(k, options))
                for k, v in assumptions.items()
                if not k.constraint.time_dependent()
            ]
        )
        activate_steps = self.encoding_step_activation_formula(box)
        formulas = And(
            [formula, formula1, formula2, activate_steps]
        ).simplify()

        episode._formula_stack.add_assertion(formulas)

    def encoding_step_activation_formula(self, box: Box) -> FNode:
        # Activate all steps up to and inclusive of box.timestep.lb
        # Deactivate all steps from box.timestep.lb + 1 to box.timestep.ub
        t = int(box.timestep().lb)
        tmax = len(box.schedule.timepoints) - 1  # int(box.timestep().ub)
        return And(
            [self._solve_at_step_symbol(step) for step in range(t + 1)]
            + [
                Not(self._solve_at_step_symbol(step))
                for step in range(t + 1, tmax + 1)
            ]
        )

    def store_smtlib(self, episode, box, filename="dbg"):
        iteration_files = glob.glob(filename + "*")
        last_index = (
            max(
                [
                    int(f.rsplit("_", 1)[1].split(".")[0])
                    for f in iteration_files
                ]
            )
            + 1
            if len(iteration_files) > 0
            else "0"
        )
        filename = f"{filename}_{last_index}.smt2"

        # if os.path.exists(tmp_name + ".smt2"):
        #     filename, count = tmp_name.rsplit("_", 1)
        #     count = int(count) + 1
        #     filename = f"{filename}_{count}"
        # else:
        #     filename = tmp_name
        # filename = filename + ".smt2"
        with open(filename, "w") as f:
            print(f"Saving smtlib file: {filename}")
            smtlibscript_from_formula_list(
                episode._formula_stack.to_list(
                    simplified=episode.config.simplify_query
                ),
                logic=QF_NRA,
            ).serialize(f, daggify=False)

    def _setup_true_query(self, solver, episode, box, options):
        """
        Setup the assumptions so that satisfying the formulas requires that both the model and the query are true

        Parameters
        ----------
        solver : Solver
            pysmt solver object
        episode : episode
            data for the current search
        """
        episode._formula_stack.push(1)
        timestep = int(box.timestep().lb)
        timepoint = box.schedule.time_at_step(timestep)
        encoder = episode.problem._encodings[options.schedule]._encoder
        assumptions = {
            k: v
            for k, v in encoder.encode_assumptions(
                episode.problem._assumptions, options
            ).items()
            if k.relevant_at_time(timepoint)
        }
        formula = And([v for k, v in assumptions.items()])

        formula1 = And(
            [
                Implies(
                    And(
                        [
                            encoder.encode_assumption(k, options, layer_idx=i)
                            for i in range(timestep + 1)
                        ]
                    ),
                    v,
                )
                for k, v in assumptions.items()
                if k.constraint.time_dependent()
            ]
        )

        formula2 = And(
            [
                And(
                    [
                        encoder.encode_assumption(k, options, layer_idx=i)
                        for i in range(timestep + 1)
                    ]
                )
                for k, v in assumptions.items()
                if k.constraint.time_dependent()
            ]
        )
        activate_steps = self.encoding_step_activation_formula(box)
        formulas = And(
            [formula, formula1, formula2, activate_steps]
        ).simplify()

        episode._formula_stack.add_assertion(formulas)

    def _get_points(
        self,
        solver: Solver,
        box: Box,
        existing_points: List[Point],
        episode: SearchEpisode,
        rval,
        _encoding_fn: Callable,
        _point_handler_fn: Callable,
        my_solver: Callable,
        options: EncodingOptions,
        _smtlib_save_fn: Callable = None,
    ):
        explanation = None

        if len(existing_points) == 0:
            # If no cached point, then attempt to generate one
            # print("Checking false query")
            _encoding_fn()
            if _smtlib_save_fn:
                _smtlib_save_fn(
                    filename=os.path.join(
                        episode.config.save_smtlib,
                        f"box_search_{episode._iteration}",
                    )
                )
            result = self.invoke_solver(
                solver, timeout=episode.config.solver_timeout
            )
            if result is not None and isinstance(result, pysmtModel):
                # If substituted formulas are on the stack, then add the original formulas to compute the values of all variables
                if (
                    episode.config.substitute_subformulas
                    and episode.config.simplify_query
                ):
                    result = episode._formula_stack.compute_assignment(
                        episode, _smtlib_save_fn=_smtlib_save_fn
                    )

                # Record the false point
                points = [episode._extract_point(result, box)]
                for point in points:
                    if options.normalize:
                        point = point.denormalize(episode.problem)
                    _point_handler_fn(box, point)
                    # rval.put(point.model_dump())
                    box.add_point(point)
            else:  # unsat
                explanation = result
                explanation.check_assumptions(episode, my_solver, options)
            episode._formula_stack.pop()
        return existing_points, explanation

    def _get_false_points(
        self, solver, episode, box, rval, options, my_solver
    ) -> Optional[Union[List[Point], Explanation]]:
        points, explanation = self._get_points(
            solver,
            box,
            box.false_points(step=box.timestep().lb),
            episode,
            rval,
            partial(self._setup_false_query, solver, episode, box, options),
            episode._add_false_point,
            my_solver,
            options,
            _smtlib_save_fn=(
                partial(
                    self.store_smtlib,
                    episode,
                    box,
                )
                if episode.config.save_smtlib
                else None
            ),
        )

        return box.false_points(step=box.timestep().lb), explanation

    def _point_assumptions(
        self,
        point: Point,
        assumptions: List[Assumption],
        episode: SearchEpisode,
        options: EncodingOptions,
    ) -> List[Assumption]:
        symbols = episode.problem._encodings[step_size].symbols()

        true_assumptions = [
            a
            for a in episode.problem._assumptions
            if point.values.get(str(a)) == 1.0
        ]

    def _negate_assumptions(
        self,
        assumptions: List[Assumption],
        episode: SearchEpisode,
        options: EncodingOptions,
    ) -> FNode:
        formula = Not(
            And(
                [
                    episode.problem._encodings[1]._encoder.encode_assumption(
                        a, options
                    )
                    for a in assumptions
                ]
            )
        )
        return formula

    def _get_witness_assumptions(
        self,
        point: Point,
        assumptions: List[Assumption],
        solver: Solver,
        episode: SearchEpisode,
        options: EncodingOptions,
    ) -> List[Point]:
        assumption_points: List[Point] = []
        point_assumptions = self._point_assumptions(
            point, assumptions, episode, options
        )

        # Check if point is a minimal assumption
        negated_assumptions = self._negate_assumptions(
            point_assumptions, episode, options
        )
        episode._formula_stack.push()
        episode._formula_stack._add_assertion(negated_assumptions)
        result = self.invoke_solver(
            solver, timeout=episode.config.solver_timeout
        )
        if result is not None and isinstance(result, pysmtModel):
            # The assumptions are not minimal
            pass
        else:
            # unsat, and assumptions are minimal
            pass

        # FIXME
        assumption_points.append(point)
        return point

    def _find_witness_points(
        self,
        solver: Solver,
        episode: SearchEpisode,
        box: Box,
        rval,
        options: EncodingOptions,
    ) -> Tuple[Point, Explanation]:
        witnesses = [p for p in episode._true_points if box.contains_point(p)]
        explanation = None
        if len(witnesses) == 0:
            # Generate a witness
            if episode.config.save_smtlib:
                self.store_smtlib(
                    episode,
                    box,
                    filename=os.path.join(
                        episode.config.save_smtlib,
                        f"wp_{episode._iteration}.smt2",
                    ),
                )
            result = self.invoke_solver(
                solver, timeout=episode.config.solver_timeout
            )
            if result is not None and isinstance(result, pysmtModel):
                # Record the false point
                point = episode._extract_point(result, box)
                points = self._get_witness_assumptions(
                    point,
                    episode.problem._assumptions,
                    solver,
                    episode,
                    options,
                )
                for assumption_point in points:
                    assumption_point = assumption_point.denormalize(
                        episode.problem
                    )
                    episode._add_true_point(assumption_point)
                    rval.put(assumption_point.model_dump())
                    witnesses.append(assumption_point)

            else:  # unsat
                explanation = result
        return witnesses, explanation

    def _get_true_points(
        self, solver, episode, box, rval, options, my_solver
    ) -> Optional[Union[List[Point], Explanation]]:
        # At start, the episode._formula_stack will have model constraints up to box.timestep.lb and box constraints
        # Each call to self._get_points() will add the "true" assumptions and tries to find a point
        #
        # While able to find a point, pop the box constraints and add to the model constraints.
        original_box_timestep_lb = box.timestep().lb
        found_point = True
        explanation = None

        while found_point and box.timestep().lb <= box.timestep().ub:
            points, explanation = self._get_points(
                solver,
                box,
                box.true_points(step=box.timestep().lb),
                episode,
                rval,
                partial(self._setup_true_query, solver, episode, box, options),
                episode._add_true_point,
                my_solver,
                options,
                _smtlib_save_fn=(
                    partial(self.store_smtlib, episode, box)
                    if episode.config.save_smtlib
                    else None
                ),
            )
            if len(box.true_points(step=box.timestep().lb)) == 0 or isinstance(
                explanation, TimeoutExplanation
            ):
                # Could not find a point at the current step, so there won't be any at subsequent steps
                # fall out of loop, after setting the upper bound on the box timestep
                # if couldn't find a point, then remove all points from box
                found_point = False

                # Cannot find a point at this time, so box.timestep().ub is previous step
                # box.timestep().ub = max(box.timestep().lb - 1, 0)
            elif box.timestep().lb < box.timestep().ub:
                # Found a true point, and there are more steps to consider
                episode._formula_stack.pop()  # pop the box constraints
                box.timestep().lb += 1
                self._initialize_model_for_box(solver, box, episode, options)
                explanation = None
            else:
                # lb == ub and have a point, so break
                break

            # if (
            #     len(box.false_points(step=box.timestep().lb)) > 0
            #     and len(box.true_points(step=box.timestep().lb)) > 0
            # ):
            #     # Do not continue if there is a true and a false point.  It means we already know we need to split this box.
            #     break

            # if box.timestep().lb == 0:
            #     # Don't check for later timepoints when looking at the initial time step.  This helps evaluate parameter constraints that would only apply to the initial time step.
            #     break

        # reinstate the original lower bound on timestep so that we will check
        # whether no false points exist in the main loop of the box search
        box.timestep().lb = original_box_timestep_lb
        return box.true_points(step=box.timestep().lb), explanation

    def get_box_corners(
        self, solver, episode, box, rval, options, my_solver
    ) -> List[Point]:
        points: List[Point] = box.corners(
            parameters=episode.problem.model_parameters()
        )
        corner_points: List[Point] = []
        for point in points:
            f = episode.problem._encodings[
                options.schedule
            ]._encoder.point_to_smt(point)
            episode._formula_stack.push()
            episode._formula_stack.add_assertion(f)
            result = self.invoke_solver(
                solver, timeout=episode.config.solver_timeout
            )
            if result is not None and isinstance(result, pysmtModel):
                corner_point = episode._extract_point(result, box)
                box.corner_points.append(corner_point)
                corner_point.label = box.label
                # rval.put(corner_point.model_dump())
                corner_points.append(corner_point)
            episode._formula_stack.pop()

        return corner_points

    def _expand(
        self,
        rval,
        episode: BoxSearchEpisode,
        options: EncodingOptions,
        idx: Optional[int] = None,
        more_work: Optional[Condition] = None,
        idle_mutex: Optional[Lock] = None,
        idle_flags: Optional[List[Event]] = None,
        handler: Optional["ResultHandler"] = None,
        all_results=None,
        haltEvent: Optional[threading.Event] = None,
    ):
        """
        A single search process will evaluate and expand the boxes in the
        episode.unknown_boxes queue.  The processes exit when the queue is
        empty.  For each box, the algorithm checks whether the box contains a
        false (infeasible) point.  If it contains a false point, then it checks
        if the box contains a true point.  If a box contains both a false and
        true point, then the box is split into two boxes and both are added to
        the unknown_boxes queue.  If a box contains no false points, then it is
        a true_box (all points are feasible).  If a box contains no true points,
        then it is a false_box (all points are infeasible).

        The return value is pushed onto the rval queue to end the process's work
        within the method.  The return value is a Dict[str, List[Box]] type that
        maps the "true_boxes" and "false_boxes" to a list of boxes in each set.
        Each box in these sets is unique by design.

        Parameters
        ----------
        rval : Queue
            Return value shared queue
        episode : BoxSearchEpisode
            Shared search data and statistics.
        """
        process_name = f"Expander_{(idx if idx else 'S')}_p{os.getpid()}"
        # l = self._logger(episode.config, process_name=process_name)
        last_progress = -1.0
        try:
            if episode.config.solver == "dreal":
                opts = {
                    "dreal_precision": episode.config.dreal_precision,
                    "dreal_log_level": episode.config.dreal_log_level,
                    "dreal_mcts": episode.config.dreal_mcts,
                    "preferred": [p.name for p in episode.problem.parameters],
                }
            else:
                opts = {}
            my_solver = partial(
                Solver,
                name=episode.config.solver,
                logic=QF_NRA,
                solver_options=opts,
            )

            with my_solver() as solver:
                episode._formula_stack._solver = solver
                l.debug(f"{process_name} entering process loop")
                # print("Starting initializing dynamics of model")
                # self._initialize_encoding(solver, episode, [0])
                # print("Initialized dynamics of model")
                while True:
                    if haltEvent is not None and haltEvent.is_set():
                        break
                    try:
                        box: Box = episode._get_unknown()
                        rval.put(box.model_dump())
                        l.trace(f"{process_name} claimed work")
                    except Empty:
                        exit = self._handle_empty_queue(
                            process_name,
                            episode,
                            more_work,
                            idle_mutex,
                            idle_flags,
                        )
                        if exit:
                            break
                        else:
                            continue
                    else:
                        l.debug(f"Expanding box: {box}")
                        l.debug(
                            f"Evaluating box: +: {len(box.true_points())}, -: {len(box.false_points())}, H: {box.point_entropy()}"
                        )
                        # Setup the model constraints up to the box.timestep.lb and add box constraints
                        self._initialize_model_for_box(
                            solver, box, episode, options
                        )

                        l.debug(
                            "\n"
                            + all_results["parameter_space"].__str__(
                                dropped_boxes=all_results["dropped_boxes"]
                            )
                        )
                        # (point, no_witness_explanation) = self._find_witness_points(
                        #     solver, episode, box, rval, options
                        # )

                        # Check whether box intersects t (true region)
                        # First see if a cached false point exists in the box
                        (
                            true_points,
                            not_true_explanation,
                        ) = self._get_true_points(
                            solver, episode, box, rval, options, my_solver
                        )

                        if len(true_points) > 0:
                            # box intersects f (true region)

                            # Check whether box intersects f (false region)
                            # First see if a cached false point exists in the box
                            (
                                false_points,
                                not_false_explanation,
                            ) = self._get_false_points(
                                solver, episode, box, rval, options, my_solver
                            )

                            if len(false_points) > 0:
                                # box intersects f (false region)

                                # box intersects both t and f, so it must be split
                                # use the true and false points to compute a midpoint
                                if self._split(
                                    box,
                                    episode,
                                    points=[true_points, false_points],
                                ):
                                    l.trace(f"{process_name} produced work")
                                else:
                                    rval.put(box.model_dump())
                                if episode.config.number_of_processes > 1:
                                    # FIXME This would only be none when
                                    # the number of processes is 1. This
                                    # can be done more cleanly.
                                    if more_work:
                                        with more_work:
                                            more_work.notify_all()
                                l.debug(
                                    f"Split @ {box.timestep().lb}, (width: {box.width():.5f} (raw) {box.normalized_width():.5f} (norm))"
                                )
                                l.trace(f"XXX Split:\n{box}")
                            elif isinstance(
                                not_false_explanation, BoxExplanation
                            ):
                                # box does not intersect f, so it is in t (true region)
                                curr_step_box = box.current_step()
                                episode._add_true(
                                    curr_step_box,
                                    explanation=not_false_explanation,
                                )
                                rval.put(curr_step_box.model_dump())
                                l.debug(f"True @ {box.timestep().lb}")
                                l.trace(f"+++ True:\n{box}")

                                if episode.config.corner_points:
                                    corner_points: List[Point] = (
                                        self.get_box_corners(
                                            solver,
                                            episode,
                                            curr_step_box,
                                            rval,
                                            options,
                                            my_solver,
                                        )
                                    )

                                # Advance a true box to be considered for later timesteps
                                next_box = box.advance()
                                if next_box:
                                    episode._add_unknown(next_box)
                            else:  # Timeout FIXME copy of split code
                                if self._split(
                                    box,
                                    episode,
                                    points=[true_points, false_points],
                                ):
                                    l.trace(f"{process_name} produced work")
                                else:
                                    rval.put(box.model_dump())
                                if episode.config.number_of_processes > 1:
                                    # FIXME This would only be none when
                                    # the number of processes is 1. This
                                    # can be done more cleanly.
                                    if more_work:
                                        with more_work:
                                            more_work.notify_all()
                                l.debug(
                                    f"Split @ {box.timestep().lb}, (width: {box.width():.5f} (raw) {box.normalized_width():.5f} (norm))"
                                )
                                l.trace(f"XXX Split:\n{box}")
                        elif isinstance(not_true_explanation, BoxExplanation):
                            if len(box.points) == 0:
                                # If we cannot find a true point, the box is false and we may have not computed any false points, so ensure we have at least one.
                                self._get_false_points(
                                    solver,
                                    episode,
                                    box,
                                    rval,
                                    options,
                                    my_solver,
                                )
                            # box is a subset of f (intersects f but not t)
                            episode._add_false(
                                box, explanation=not_true_explanation
                            )  # TODO consider merging lists of boxes

                            l.debug(f"False @ {box.timestep().lb}")
                            l.trace(f"--- False:\n{box}")
                            if episode.config.corner_points:
                                corner_points: List[Point] = (
                                    self.get_box_corners(
                                        solver,
                                        episode,
                                        box,
                                        rval,
                                        options,
                                        my_solver,
                                    )
                                )
                            rval.put(box.model_dump())
                        else:  # Timeout FIXME copy of split code
                            if self._split(
                                box,
                                episode,
                                points=[true_points],
                            ):
                                l.trace(f"{process_name} produced work")
                            else:
                                rval.put(box.model_dump())
                            if episode.config.number_of_processes > 1:
                                # FIXME This would only be none when
                                # the number of processes is 1. This
                                # can be done more cleanly.
                                if more_work:
                                    with more_work:
                                        more_work.notify_all()
                            l.debug(
                                f"Split @ {box.timestep().lb}, (width: {box.width():.5f} (raw) {box.normalized_width():.5f} (norm))"
                            )
                            l.trace(f"XXX Split:\n{box}")
                        episode._formula_stack.pop()  # Remove box constraints from solver
                        episode._on_iteration()
                        if handler:
                            handler(rval, episode.config, all_results)
                            if (
                                "progress" in all_results
                                and all_results["progress"].progress
                                > last_progress
                            ):
                                last_progress = all_results[
                                    "progress"
                                ].progress
                                l.info(all_results["progress"])
                        l.trace(f"{process_name} finished work")
                self._initialize_model_encoding(
                    solver, episode, options, None
                )  # Reset solver stack to empty
        except KeyboardInterrupt:
            l.info(f"{process_name} Keyboard Interrupt")
        except Exception:
            l.error(traceback.format_exc())

    def _run_handler(self, rval, config: "FUNMANConfig"):
        """
        Execute the process that does final processing of the results of expand()
        """
        l = self._logger(config, process_name=f"search_process_result_handler")

        handler: ResultHandler = config._handler
        ps = ParameterSpace()
        dropped_boxes = []
        break_on_interrupt = False
        try:
            handler.open()
            while True:
                try:
                    result: dict = rval.get(timeout=config.queue_timeout)
                except Empty:
                    continue
                except KeyboardInterrupt:
                    if break_on_interrupt:
                        break
                    break_on_interrupt = True
                else:
                    if result is None:
                        break

                    # TODO this is a bit of a mess and can likely be cleaned up
                    inst = ParameterSpace.decode_labeled_object(result)
                    label = inst.label
                    if isinstance(inst, Box):
                        if label == "true":
                            ps.true_boxes.append(inst)
                        elif label == "false":
                            ps.false_boxes.append(inst)
                        elif label == "dropped":
                            dropped_boxes.append(inst)
                        else:
                            l.warning(f"Skipping Box with label: {label}")
                    elif isinstance(inst, Point):
                        if label == "true":
                            ps.true_points.append(inst)
                        elif label == "false":
                            ps.false_points.append(inst)
                        else:
                            l.warning(f"Skipping Point with label: {label}")
                    else:
                        l.error(f"Skipping invalid object type: {type}")

                    try:
                        handler.process(result)
                    except Exception:
                        l.error(traceback.format_exc())

        except Exception as error:
            l.error(error)
        finally:
            handler.close()
        return {
            "parameter_space": ps,
            "dropped_boxes": dropped_boxes,
        }

    def _run_handler_step(
        self, rval, config: "FUNMANConfig", all_results
    ) -> Dict[str, Any]:
        """
        Execute one step of processing the results of expand()
        """
        l = self._logger(config, process_name=f"search_process_result_handler")

        handler: ResultHandler = config._handler
        ps = all_results.get("parameter_space")
        break_on_interrupt = False
        try:
            # handler.open()
            while True:
                try:
                    result = None
                    if not rval.empty():
                        result: dict = rval.get(timeout=config.queue_timeout)
                except Empty:
                    break
                except KeyboardInterrupt:
                    if break_on_interrupt:
                        break
                    break_on_interrupt = True
                else:
                    if result is None:
                        break

                    # TODO this is a bit of a mess and can likely be cleaned up
                    try:
                        inst = ParameterSpace.decode_labeled_object(result)
                    except:
                        l.error(f"Skipping invalid object")
                        continue

                    label = inst.label
                    if isinstance(inst, Box):
                        if label == "true":
                            ps.true_boxes.append(inst)
                        elif label == "false":
                            ps.false_boxes.append(inst)
                        elif label == "dropped":
                            all_results["dropped_boxes"].append(inst)
                        elif label == "unknown":
                            pass  # Allow unknown boxes for plotting
                        else:
                            l.warning(f"Skipping Box with label: {label}")
                    elif isinstance(inst, Point):
                        if label == "true":
                            ps.true_points.append(inst)
                        elif label == "false":
                            ps.false_points.append(inst)
                        else:
                            l.warning(f"Skipping Point with label: {label}")
                    else:
                        l.error(f"Skipping invalid object type: {type(inst)}")
                        continue

                    try:
                        handler.process(result)
                    except Exception:
                        l.error(traceback.format_exc())

        except Exception as error:
            l.error(error)
        finally:
            if config._wait_action is not None:
                config._wait_action.run()
            # handler.close()
        return all_results

    def search(
        self,
        problem: "AnalysisScenario",
        config: "FUNMANConfig",
        haltEvent: Optional[threading.Event] = None,
        resultsCallback: Optional[Callable[[ParameterSpace], None]] = None,
    ) -> ParameterSpace:
        """
        The BoxSearch.search() creates a BoxSearchEpisode object that stores the
        search progress.  This method is the entry point to the search that
        spawns several processes to parallelize the evaluation of boxes in the
        BoxSearch.expand() method.  It treats the zeroth process as a special
        process that is allowed to initialize the search and plot the progress
        of the search.

        Parameters
        ----------
        problem : ParameterSynthesisScenario
            Model and parameters to synthesize
        config : SearchConfig, optional
            BoxSearch configuration, by default SearchConfig()

        Returns
        -------
        BoxSearchEpisode
            Final search results (parameter space) and statistics.
        """

        # problem.encode()

        if config.number_of_processes > 1:
            return self._search_mp(
                problem,
                config,
                haltEvent=haltEvent,
            )
        else:
            return self._search_sp(
                problem,
                config,
                haltEvent=haltEvent,
                resultsCallback=resultsCallback,
            )

    def _search_sp(
        self,
        problem,
        config: "FUNMANConfig",
        haltEvent: Optional[threading.Event],
        resultsCallback: Optional[Callable[[ParameterSpace], None]] = None,
    ) -> ParameterSpace:
        all_results = {
            "parameter_space": ParameterSpace(
                num_dimensions=problem.num_dimensions()
            ),
            "dropped_boxes": [],
        }
        rval = QueueSP()

        def handler(rval, config: "FUNMANConfig", results) -> Dict[str, Any]:
            self._run_handler_step(rval, config, results)
            if resultsCallback is not None:
                progress = resultsCallback(results.get("parameter_space"))
                results["progress"] = progress
            return results

        config._handler.open()

        if problem._smt_encoder._timed_model_elements:
            schedules = problem._smt_encoder._timed_model_elements[
                "schedules"
            ].schedules

            # initialize empty encoding
            for schedule in schedules:
                episode = BoxSearchEpisode(
                    config=config, problem=problem, schedule=schedule
                )
                episode._initialize_boxes(config.num_initial_boxes, schedule)
                options = EncodingOptions(
                    schedule=schedule,
                    normalize=config.normalize,
                    normalization_constant=config.normalization_constant,
                )
                self._expand(
                    rval,
                    episode,
                    options,
                    handler=handler,
                    all_results=all_results,
                    haltEvent=haltEvent,
                )
        else:
            problem._encode_timed(
                1,
                1,
                config,
            )
            structural_configuration = {
                "step_size": 1,
                "num_steps": 1,
            }
            episode = BoxSearchEpisode(
                config=config,
                problem=problem,
                structural_configuration=structural_configuration,
            )
            episode._initialize_boxes(config.num_initial_boxes)
            options = EncodingOptions(
                num_steps=structural_configuration["num_steps"],
                step_size=structural_configuration["step_size"],
                normalize=config.normalize,
                normalization_constant=config.normalization_constant,
            )
            self._expand(
                rval,
                episode,
                options,
                handler=handler,
                all_results=all_results,
                haltEvent=haltEvent,
            )

        config._handler.close()
        return all_results["parameter_space"]

    def _search_mp(
        self,
        problem,
        config: "FUNMANConfig",
        haltEvent: Optional[threading.Event],
    ) -> ParameterSpace:
        l = mp.get_logger()
        l.setLevel(config.verbosity)
        processes = config.number_of_processes
        with mp.Manager() as manager:
            rval = manager.Queue()
            episode = BoxSearchEpisodeMP(
                config=config,
                problem=problem,
                manager=manager,
                structural_configuration=structural_configuration,
            )
            options = EncodingOptions(
                num_steps=structural_configuration["num_steps"],
                step_size=structural_configuration["step_size"],
                normalize=config.normalize,
                normalization_constant=config.normalization_constant,
            )
            expand_count = processes - 1
            episode._initialize_boxes(expand_count)
            idle_mutex = manager.Lock()
            idle_flags = [manager.Event() for _ in range(expand_count)]

            with mp.Pool(processes=processes) as pool:
                more_work_condition = manager.Condition()

                # start the result handler process
                l.info("Starting result handler process")
                rval_handler_process = pool.apply_async(
                    self._run_handler, args=(rval, config)
                )
                # blocking exec of the expansion processes
                l.info(f"Starting {expand_count} expand processes")

                starmap_result = pool.starmap_async(
                    self._expand,
                    [
                        (
                            rval,
                            episode,
                            options,
                            {
                                "idx": idx,
                                "more_work": more_work_condition,
                                "idle_mutex": idle_mutex,
                                "idle_flags": idle_flags,
                            },
                        )
                        for idx in range(expand_count)
                    ],
                )

                # tell the result handler process we are done with the expansion processes
                try:
                    if config._wait_action is not None:
                        while not starmap_result.ready():
                            config._wait_action.run()
                    if haltEvent is None:
                        starmap_result.wait()
                    else:
                        while not starmap_result.wait(0.5):
                            if haltEvent.is_set():
                                break

                except KeyboardInterrupt:
                    l.warning("--- Received Keyboard Interrupt ---")

                rval.put(None)
                l.info("Waiting for result handler process")
                # wait for the result handler to finish
                rval_handler_process.wait(timeout=config.wait_timeout)

                if not rval_handler_process.successful():
                    l.error("Result handler failed to exit")
                all_results = rval_handler_process.get()

                parameter_space = ParameterSpace(
                    true_boxes=all_results.get("true_boxes"),
                    false_boxes=all_results.get("false_boxes"),
                    true_points=all_results.get("true_points"),
                    false_points=all_results.get("false_points"),
                )
                return parameter_space

import json
import logging
import sys
import threading
from typing import Callable, Optional, Tuple, Union

from pysmt.formula import FNode
from pysmt.logics import QF_NRA
from pysmt.shortcuts import BOOL, REAL, And, Bool, Equals, Real, Solver, Symbol
from pysmt.solvers.solver import Model as pysmtModel

from funman.config import FUNMANConfig
from funman.constants import LABEL_FALSE, LABEL_TRUE
from funman.representation.encoding_schedule import EncodingSchedule
from funman.representation.explanation import Explanation
from funman.translate.translate import EncodingOptions
from funman.utils.smtlib_utils import smtlibscript_from_formula_list

from ..representation import Interval, Point
from ..representation.box import Box
from ..representation.parameter_space import ParameterSpace

# import funman.search as search
from .search import Search, SearchEpisode

# from funman.utils.sympy_utils import sympy_to_pysmt, to_sympy


l = logging.getLogger(__file__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class SMTCheck(Search):
    def search(
        self,
        problem,
        config: Optional["FUNMANConfig"] = None,
        haltEvent: Optional[threading.Event] = None,
        resultsCallback: Optional[Callable[["ParameterSpace"], None]] = None,
    ) -> "SearchEpisode":
        parameter_space = ParameterSpace(
            num_dimensions=problem.num_dimensions()
        )
        models = {}
        consistent = {}
        # problem._initialize_encodings(config)
        for schedule in problem._smt_encoder._timed_model_elements[
            "schedules"
        ].schedules:
            l.debug(f"Solving schedule: {schedule}")
            schedule_length = len(schedule.timepoints)
            episode = SearchEpisode(
                config=config, problem=problem, schedule=schedule
            )
            options = EncodingOptions(schedule=schedule)

            # self._initialize_encoding(solver, episode, box_timepoint, box)
            result = self.expand(
                problem,
                episode,
                options,
                parameter_space,
                schedule,
            )
            timestep = len(schedule.timepoints)
            if result is not None and isinstance(result, pysmtModel):
                result_dict = result.to_dict() if result else None
                l.debug(f"Result: {json.dumps(result_dict, indent=4)}")
                if result_dict is not None:
                    parameter_values = {
                        k: v
                        for k, v in result_dict.items()
                        # if k in [p.name for p in problem.parameters]
                    }
                    # for k, v in structural_configuration.items():
                    #     parameter_values[k] = v
                    point = Point(
                        values=parameter_values,
                        label=LABEL_TRUE,
                        schedule=schedule,
                    )

                    point.values["timestep"] = Interval(
                        lb=timestep, ub=timestep, closed_upper_bound=True
                    )
                    if config.normalize:
                        denormalized_point = point.denormalize(problem)
                        point = denormalized_point
                    models[point] = result
                    consistent[point] = result_dict
                    parameter_space.true_boxes.append(Box.from_point(point))
            elif result is not None and isinstance(result, Explanation):
                box = Box(
                    bounds={
                        p.name: p.interval.model_copy()
                        for p in problem.parameters
                    },
                    label=LABEL_FALSE,
                    explanation=result,
                )
                box.bounds["timestep"] = Interval(
                    lb=timestep, ub=timestep, closed_upper_bound=True
                )
                parameter_space.false_boxes.append(box)
            if resultsCallback:
                resultsCallback(parameter_space)

        return parameter_space, models, consistent

    def build_formula(
        self,
        episode: SearchEpisode,
        schedule: EncodingSchedule,
        options: EncodingOptions,
    ) -> Tuple[FNode, FNode]:
        encoding = episode.problem._encodings[schedule]
        layer_formulas = []
        simplified_layer_formulas = []
        assumption_formulas = []
        for a in episode.problem._assumptions:
            assumption_formulas.append(
                encoding._encoder.encode_assumption(a, options)
            )

        for timestep, timepoint in enumerate(schedule.timepoints):
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
                            layers=[timestep],
                            assumptions=episode.problem._assumptions,
                        )
                    )
            formula = And(encoded_constraints)
            layer_formulas.append(formula)

            # Simplify formulas if needed
            if (
                episode.config.simplify_query
                and episode.config.substitute_subformulas
            ):
                substitutions = encoding._encoder.substitutions(schedule)
                simplified_layer_formulas = [
                    x.substitute(substitutions).simplify()
                    for x in layer_formulas
                ]

        all_layers_formula = And(
            And(assumption_formulas), And([f for f in layer_formulas])
        )
        all_simplified_layers_formula = (
            And(
                And(assumption_formulas),
                And([f for f in simplified_layer_formulas]),
            )
            if len(simplified_layer_formulas) > 0
            else None
        )
        return all_layers_formula, all_simplified_layers_formula

    def solve_formula(
        self, s: Solver, formula: FNode, episode
    ) -> Union[pysmtModel, Explanation]:
        s.push(1)
        s.add_assertion(formula)
        if episode.config.save_smtlib:
            self.store_smtlib(
                formula,
                filename=f"dbg_steps.smt2",
            )
        l.trace(f"Solving: {formula.serialize()}")
        result = self.invoke_solver(s)
        s.pop(1)
        l.trace(f"Result: {type(result)}")
        return result

    def expand(
        self,
        problem,
        episode,
        options: EncodingOptions,
        parameter_space,
        schedule: EncodingSchedule,
    ):
        if episode.config.solver == "dreal":
            opts = {
                "dreal_precision": episode.config.dreal_precision,
                "dreal_log_level": episode.config.dreal_log_level,
                "dreal_mcts": episode.config.dreal_mcts,
            }
        else:
            opts = {}
        # s1 = Solver(
        #     name=episode.config.solver,
        #     logic=QF_NRA,
        #     solver_options=opts,
        # )
        with Solver(
            name=episode.config.solver,
            logic=QF_NRA,
            solver_options=opts,
        ) as s:
            formula, simplified_formula = self.build_formula(
                episode, schedule, options
            )

            if simplified_formula is not None:
                # If using a simplified formula, we need to solve it and use its values in the original formula to get the values of all variables
                result = self.solve_formula(s, simplified_formula, episode)
                if result is not None and isinstance(result, pysmtModel):
                    assigned_vars = result.to_dict()
                    substitution = {
                        Symbol(p, (REAL if isinstance(v, float) else BOOL)): (
                            Real(v) if isinstance(v, float) else Bool(v)
                        )
                        for p, v in assigned_vars.items()
                    }
                    result_assignment = And(
                        [
                            (
                                Equals(Symbol(p, REAL), Real(v))
                                if isinstance(v, float)
                                else (
                                    Symbol(p, BOOL)
                                    if v
                                    else Not(Symbol(p, BOOL))
                                )
                            )
                            for p, v in assigned_vars.items()
                        ]
                        + [
                            Equals(Symbol(p.name, REAL), Real(0.0))
                            for p in episode.problem.model_parameters()
                            if p.is_unbound() and p.name not in assigned_vars
                        ]
                    )
                    formula_w_params = And(
                        formula.substitute(substitution), result_assignment
                    )
                    result = self.solve_formula(s, formula_w_params, episode)
                elif result is not None and isinstance(result, str):
                    # Unsat core
                    pass
            else:
                result = self.solve_formula(s, formula, episode)
                if isinstance(result, Explanation):
                    result.check_assumptions(episode, s, options)

        return result

    def store_smtlib(self, formula, filename="dbg.smt2"):
        with open(filename, "w") as f:
            smtlibscript_from_formula_list(
                [formula],
                logic=QF_NRA,
            ).serialize(f, daggify=False)

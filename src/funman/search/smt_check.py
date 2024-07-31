import logging
import os
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
            options = EncodingOptions(
                schedule=schedule,
                normalize=config.normalize,
                normalization_constant=config.normalization_constant,
            )

            # self._initialize_encoding(solver, episode, box_timepoint, box)
            model_result, explanation_result = self.expand(
                problem,
                episode,
                options,
                parameter_space,
                schedule,
            )
            point = None
            timestep = len(schedule.timepoints) - 1
            if model_result is not None and isinstance(
                model_result, pysmtModel
            ):
                # result_dict = model_result.to_dict() if model_result else None
                # l.debug(f"Result: {json.dumps(result_dict, indent=4)}")
                # if result_dict is not None:
                #     parameter_values = {
                #         k: v
                #         for k, v in result_dict.items()
                #         # if k in [p.name for p in problem.parameters]
                #     }
                #     # for k, v in structural_configuration.items():
                #     #     parameter_values[k] = v
                point_label = (
                    LABEL_TRUE if explanation_result is None else LABEL_FALSE
                )
                results_dict = model_result.to_dict()
                point = Point(
                    values=results_dict,
                    label=point_label,
                    schedule=schedule,
                )
                point.values["timestep"] = timestep

                if config.normalize:
                    denormalized_point = point.denormalize(problem)
                    point = denormalized_point

                if point_label == LABEL_TRUE:
                    models[point] = model_result
                    consistent[point] = results_dict

                box = Box.from_point(point)
                # parameter_space.true_boxes.append(Box.from_point(point))
            else:
                box = Box(
                    bounds={
                        p.name: p.interval.model_copy()
                        for p in problem.model_parameters()
                    },
                    label=LABEL_FALSE,  # lack of a point means this must be a false box
                    points=[],
                )
                box.bounds["timestep"] = Interval(
                    lb=timestep, ub=timestep, closed_upper_bound=True
                )
            box.schedule = schedule

            if explanation_result is not None and isinstance(
                explanation_result, Explanation
            ):
                box.explanation = explanation_result

            if box.label == LABEL_TRUE:
                parameter_space.true_boxes.append(box)
            else:
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

        model_formula = And([f for f in layer_formulas])

        all_layers_formula = And(And(assumption_formulas), model_formula)

        all_simplified_layers_formula = (
            And(
                And(assumption_formulas),
                And([f for f in simplified_layer_formulas]),
            )
            if len(simplified_layer_formulas) > 0
            else None
        )
        return all_layers_formula, all_simplified_layers_formula, model_formula

    def solve_formula(
        self, s: Solver, formula: FNode, episode
    ) -> Union[pysmtModel, Explanation]:
        s.push(1)
        s.add_assertion(formula)
        if episode.config.save_smtlib:
            filename = os.path.join(
                episode.config.save_smtlib, "dbg_steps.smt2"
            )
            l.trace(f"Saving smt file: {filename}")
            self.store_smtlib(
                formula,
                filename=filename,
            )
        l.trace(f"Solving: {formula.serialize()}")
        result = self.invoke_solver(s, timeout=episode.config.solver_timeout)
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
        model_result = None
        explanation_result = None
        if episode.config.solver == "dreal":
            opts = {
                "dreal_precision": episode.config.dreal_precision,
                "dreal_log_level": episode.config.dreal_log_level,
                "dreal_mcts": episode.config.dreal_mcts,
                "preferred": episode.config.dreal_prefer_parameters,  # [p.name for p in problem.model_parameters()]if episode.config.dreal_prefer_parameters else [],
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
            formula, simplified_formula, model_formula = self.build_formula(
                episode, schedule, options
            )

            if simplified_formula is not None:
                # If using a simplified formula, we need to solve it and use its values in the original formula to get the values of all variables
                result = self.solve_formula(s, simplified_formula, episode)
                if result is not None and isinstance(result, pysmtModel):
                    model_result = result
                    assigned_vars = model_result.to_dict()
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
                    model_result = self.solve_formula(
                        s, formula_w_params, episode
                    )
                elif result is not None and isinstance(result, str):
                    explanation_result = result
                    # Unsat core
            else:
                model_result = self.solve_formula(s, formula, episode)
                if isinstance(model_result, Explanation):
                    explanation_result = model_result
                    model_result.check_assumptions(episode, s, options)

                    # If formula with assumptions is unsat, then we need to generate a trace of the model by giving up on the assumptions.
                    model_result = self.solve_formula(
                        s, model_formula, episode
                    )

        return model_result, explanation_result

    def store_smtlib(self, formula, filename="dbg.smt2"):
        with open(filename, "w") as f:
            smtlibscript_from_formula_list(
                [formula],
                logic=QF_NRA,
            ).serialize(f, daggify=False)

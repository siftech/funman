from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel
from pysmt.formula import FNode

from .assumption import Assumption


class Explanation(BaseModel):
    expression: Optional[str] = None
    symbols: Optional[List[str]] = None

    def explain(self) -> Dict[str, Any]:
        return {
            "description": "The expression is implied by this scenario and is unsatisfiable",
            "expression": self.expression,
        }

    def set_expression(self, e: FNode):
        self.expression = e.serialize()
        self.symbols = [str(s) for s in e.get_free_variables()]

    def check_assumptions(
        self,
        episode: "BoxSearchEpisode",
        my_solver: Callable,
        options: "EncodingOptions",
    ) -> List[Assumption]:
        return []


class TimeoutExplanation(Explanation):
    pass


class BoxExplanation(Explanation):
    relevant_assumptions: List[Assumption] = []
    

    def check_assumptions(
        self,
        episode: "BoxSearchEpisode",
        my_solver: Callable,
        options: "EncodingOptions",
    ) -> List[Assumption]:
        """
        Find the assumptions that are unit clauses in the expression (unsat core).

        Parameters
        ----------
        episode : BoxSearchEpisode
            _description_
        my_solver : Callable
            _description_

        Returns
        -------
        List[Assumption]
            _description_
        """

        # FIXME use step size from options
        assumption_symbols: Dict[str, Assumption] = {
            str(a): a for a in episode.problem._assumptions
        }
    
        self.relevant_assumptions = [
            a
            for symbol, a in assumption_symbols.items()
            if any(symbol in es for es in self.symbols)
        ]
        return self.relevant_assumptions

    # def satisfies_assumption(self, assumption: FNode)-> bool:
    #     # solver.push(1)
    #     # solver.add_assertion(assumption)
    #     # is_sat = solver.solve()
    #     # solver.pop(1)
    #     is_sat =
    #     return is_sat

    def explain(self) -> Dict[str, Any]:
        relevant_constraints = [
            a.constraint.model_dump() for a in self.relevant_assumptions
        ]
        expl = {"relevant_constraints": relevant_constraints}
        if self.expression is not None:
            expl["expression"] = self.expression
        return expl

    def __str__(self) -> str:
        return self.explain()


class ParameterSpaceExplanation(Explanation):
    true_explanations: List[BoxExplanation] = []
    false_explanations: List[BoxExplanation] = []

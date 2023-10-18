import logging
from typing import Dict, List, Union

from pydantic import BaseModel, ConfigDict
from pysmt.constants import Numeral
from pysmt.formula import FNode
from pysmt.shortcuts import (  # type: ignore
    BOOL,
    GE,
    LE,
    LT,
    REAL,
    TRUE,
    And,
    Div,
    Equals,
    Iff,
    Implies,
    Plus,
    Real,
    Symbol,
    Times,
    get_env,
)
from pysmt.solvers.solver import Model as pysmtModel

from funman.config import FUNMANConfig
from funman.constants import NEG_INFINITY, POS_INFINITY
from funman.model.model import Model
from funman.model.query import (
    QueryAnd,
    QueryEncoded,
    QueryGE,
    QueryLE,
    QueryTrue,
)
from funman.representation.constraint import (
    Constraint,
    LinearConstraint,
    ModelConstraint,
    ParameterConstraint,
    QueryConstraint,
    StateVariableConstraint,
)
from funman.translate.simplifier import FUNMANSimplifier
from funman.utils import math_utils
from funman.utils.sympy_utils import (
    FUNMANFormulaManager,
    sympy_to_pysmt,
    to_sympy,
)

from ..representation.assumption import Assumption
from ..representation.parameter import ModelParameter
from ..representation.representation import EncodingSchedule
from ..representation.symbol import ModelSymbol

l = logging.getLogger(__name__)
l.setLevel(logging.DEBUG)


class EncodingOptions(BaseModel):
    """
    EncodingOptions
    """

    schedule: EncodingSchedule
    normalize: bool = False
    normalization_constant: float = 1.0


class Encoding(BaseModel):
    _substitutions: Dict[FNode, FNode] = {}


class FlatEncoding(BaseModel):
    """
    An encoding comprises a formula over a set of symbols.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    _formula: FNode = None
    _symbols: Union[List[FNode], Dict[str, Dict[str, FNode]]] = None

    def encoding(self):
        return _formula

    def assume(self, assumption: FNode):
        _formula = Iff(assumption, _formula)

    def symbols(self):
        return self._symbols

    # @validator("formula")
    # def set_symbols(cls, v: FNode):
    #     cls.symbols = Symbol(v, REAL)

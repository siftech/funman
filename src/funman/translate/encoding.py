import logging
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict
from pysmt.formula import FNode
from pysmt.shortcuts import Iff  # type: ignore

from ..representation.representation import EncodingSchedule

l = logging.getLogger(__name__)


class EncodingOptions(BaseModel):
    """
    EncodingOptions
    """

    schedule: EncodingSchedule
    normalize: bool = False
    normalization_constant: Optional[float] = 1.0


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

"""
This module defines the Funman class, the primary entry point for FUNMAN
analysis.
"""

import logging
import os
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from funman.utils.handlers import (
    NoopResultHandler,
    ResultCombinedHandler,
    ResultHandler,
    WaitAction,
)

l = logging.getLogger(__file__)


class FUNMANConfig(BaseModel):
    """
    Base definition of a configuration object
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
    )

    tolerance: float = 1e-3
    """Algorithm-specific tolerance for approximation, used by BoxSearch"""

    queue_timeout: int = 1
    """Multiprocessing queue timeout, used by BoxSearch"""

    number_of_processes: int = 1  # mp.cpu_count()
    """Number of BoxSearch processes"""

    _handler: Union[
        ResultCombinedHandler, NoopResultHandler, ResultHandler
    ] = NoopResultHandler()
    wait_timeout: Optional[int] = None
    """Timeout for BoxSearch procesess to wait for boxes to evaluate"""

    _wait_action: WaitAction = None
    wait_action_timeout: float = 0.05
    """Time to sleep proceses waiting for work"""

    _read_cache: ResultHandler = None

    _search: str = None
    """Name of search algorithm to use"""

    solver: str = "dreal"  # "z3"
    """Name of pysmt solver to use"""

    num_steps: int = 2
    """Number of timesteps to encode"""

    step_size: int = 1
    """Step size for encoding"""

    num_initial_boxes: int = 1
    """Number of initial boxes for BoxSearch"""

    solver_timeout: Optional[int] = None
    """Number of seconds to allow each call to SMT solver"""

    initial_state_tolerance: float = 0.0
    """Factor used to relax initial state values bounds"""

    save_smtlib: Optional[str] = None
    """Whether to save each smt invocation as an SMTLib file"""

    dreal_precision: float = 1e-1
    """Precision delta for dreal solver"""

    dreal_log_level: str = "off"
    """Constraint noise term to relax constraints"""

    constraint_noise: float = 0.0
    """Use MCTS in dreal"""

    dreal_mcts: bool = True
    """Substitute subformulas to simplify overall encoding"""

    substitute_subformulas: bool = False
    """Enforce compartmental variable constraints"""

    normalization_constant: Optional[float] = None
    """ Simplify query by propagating substutions """

    use_compartmental_constraints: bool = False
    """Normalize scenarios prior to solving"""

    compartmental_constraint_noise: float = 0.01
    """Additional factor used to relax compartmental constraint (needed due to floating point imprecision)"""

    normalize: bool = True
    """Normalization constant to use for normalization (attempt to compute if None)"""

    simplify_query: bool = False
    """ Convert query object to a single formula based only on the parameters (deprecated) """

    series_approximation_threshold: Optional[float] = None
    """ Series approximation threshold for dropping series terms """

    profile: bool = False
    """ Generate profiling output"""

    taylor_series_order: Optional[int] = None
    """ Use Taylor series of given order to approximate transition function, if None, then do not compute series """

    corner_points: bool = False
    """ Compute Corner points of each box """

    verbosity: int = logging.INFO
    """ Verbosity (INFO, DEBUG, WARN, ERROR)"""

    use_transition_symbols: bool = False
    """ Use transition symbols in encoding transition functions """

    uniform_box_splits: bool = False
    """ Uniformly split boxes in box search, instead of separating points in boxes """

    dreal_prefer_parameters: List[str] = []
    """ Prefer to split the listed parameters in dreal """

    point_based_evaluation: bool = False
    """ Evaluate parameters using point-based simulation over interval-based SMT encoding """

    prioritize_box_entropy: bool = True
    """ When comparing boxes, prefer those with low entropy """

    @field_validator("solver")
    @classmethod
    def import_dreal(cls, v: str) -> str:
        if v == "dreal":
            try:
                import funman_dreal
            except:
                raise Exception(
                    "The funman_dreal package failed to import. Do you have it installed?"
                )
            else:
                funman_dreal.ensure_dreal_in_pysmt()
        return v

    @model_validator(mode="after")
    def check_use_compartmental_constraints(self) -> "FUNMANConfig":
        if self.use_compartmental_constraints:
            assert (
                self.normalization_constant
            ), "Need to set normalization_constant in configuration to enforce compartmental constraints.  The normalization_constant provides the population size used in the constraint upper bound."

        if self.save_smtlib:
            assert os.path.exists(
                os.path.dirname(self.save_smtlib)
            ), "save_smtlib option must be an existing path"

        return self

from typing import Dict, Set

from pysmt.formula import FNode
from pysmt.shortcuts import REAL, TRUE, And, Symbol, Times, substitute

from funman.model.model import FunmanModel

from .translate import Encoder, Encoding

from pysmt.solvers.solver import Model as pysmtModel

class EnsembleEncoder(Encoder):
    def encode_model(self, scenario: "AnalysisScenario") -> Encoding:
        """
        Encode a model into an SMTLib formula.

        Parameters
        ----------
        model : FunmanModel
            model to encode

        Returns
        -------
        Encoding
            formula and symbols for the encoding
        """
        return Encoding(formula=TRUE(), symbols={})

    def _encode_next_step(
        self, model: "FunmanModel", step: int, next_step: int
    ) -> FNode:
        model_steps = {
            model.name: substitute(
                model.default_encoder(self.config)._encode_next_step(
                    model, step, next_step
                ),
                self._submodel_substitution_map(
                    model, step=step, next_step=next_step
                ),
            )
            for model in model.models
        }

        return And(list(model_steps.values()))

    def _submodel_substitution_map(
        self, model: FunmanModel, step=None, next_step=None
    ) -> Dict[Symbol, Symbol]:
        curr_var_sub_map: Dict[Symbol, Symbol] = {
            Symbol(f"{variable}_{step}", REAL): Symbol(
                f"model_{model.name}_{variable}_{step}", REAL
            )
            for variable in model._state_var_names()
        }
        next_var_sub_map: Dict[Symbol, Symbol] = {
            Symbol(f"{variable}_{next_step}", REAL): Symbol(
                f"model_{model.name}_{variable}_{next_step}", REAL
            )
            for variable in model._state_var_names()
        }
        parameter_sub_map: Dict[Symbol, Symbol] = {
            Symbol(f"{variable}", REAL): Symbol(
                f"model_{model.name}_{variable}", REAL
            )
            for variable in model._parameter_names()
        }
        return {**curr_var_sub_map, **next_var_sub_map, **parameter_sub_map}

    def _encode_transition_term(
        self,
        t_index,
        transition,
        current_state,
        next_state,
        input_edges,
        output_edges,
    ):
        ins = [
            current_state[edge["is"] - 1]
            for edge in input_edges
            if edge["it"] == t_index + 1
        ]
        param_symbol = self._encode_state_var(
            transition["tprop"]["parameter_name"]
        )

        return Times([param_symbol] + ins)

    def _get_timed_symbols(self, model: FunmanModel) -> Set[str]:
        """
        Get the names of the state (i.e., timed) variables of the model.

        Parameters
        ----------
        model : FunmanModel
            The petrinet model

        Returns
        -------
        List[str]
            state variable names
        """
        return set(model._state_var_names())

    def symbol_values(
        self, model_encoding: Encoding, pysmtModel: pysmtModel
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

        vars = self._symbols(model_encoding.symbols())
        vals = {}
        for var, model_vars in vars.items():
            for model_idx, model_var in model_vars.items():    
                vals[model_var] = {}
                for t in vars[var]:
                    try:
                        symbol = vars[var][t]
                        value = pysmtModel.get_py_value(symbol)
                        # value = pysmtModel.completed_assignment[symbol]
                        if isinstance(value, Numeral):
                            value = 0.0
                        vals[model_var][t] = float(value)
                    except OverflowError as e:
                        l.warning(e)
        return vals
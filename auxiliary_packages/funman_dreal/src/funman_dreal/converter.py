import functools
import logging
import re
from fractions import Fraction
from typing import List

import dreal
from pysmt.decorators import catch_conversion_error
from pysmt.exceptions import UndefinedSymbolError
from pysmt.formula import FNode
from pysmt.parsing import (
    ClosePar,
    Constant,
    GrammarSymbol,
    HRLexer,
    Identifier,
    InfixOpAdapter,
    PrattParser,
    Rule,
    UnaryOpAdapter,
)
from pysmt.shortcuts import FALSE, Symbol
from pysmt.smtlib.parser import SmtLibParser, Tokenizer
from pysmt.solvers.solver import (
    Converter,
    IncrementalTrackingSolver,
    Model,
    SolverOptions,
    UnsatCoreSolver,
)
from pysmt.walkers import DagWalker

l = logging.getLogger(__name__)


def CoreParser(env=None):
    return PrattParser(CoreLexer)


class BinaryLiteralExpr(GrammarSymbol):
    """Adapter for unary operator."""

    def __init__(self, operator, lbp):
        GrammarSymbol.__init__(self)
        self.operator = operator
        self.lbp = lbp

    def nud(self, parser):
        # parser.advance()  # OpenPar
        parser.advance()  # OpenPar
        if type(parser.token) != ClosePar:
            r = parser.expression()
        if type(parser.token) != ClosePar:
            raise SyntaxError("Expected ')'")
        parser.advance()  # ClosePar
        if type(parser.token) != ClosePar:
            raise SyntaxError("Expected ')'")
        parser.advance()  # ClosePar
        return r

    def __repr__(self):
        return repr(self.operator)


class PowOpAdapter(GrammarSymbol):
    """Adapter for infix operator."""

    def __init__(self, operator, lbp):
        GrammarSymbol.__init__(self)
        self.operator = operator
        self.lbp = lbp

    def nud(self, parser):
        parser.advance()  # OpenPar
        left = parser.expression()  # base
        parser.advance()  # ExprComma
        right = parser.expression()  # exponent
        parser.advance()  # ClosePar
        return self.operator(left, right)

    def __repr__(self):
        return repr(self.operator)


class CoreLexer(HRLexer):
    def __init__(self, env=None):
        super().__init__(env=env)

        self.identifier_map = {"and": "&", "or": "|", "==": "="}

        self.rules = (
            [
                Rule(r"(pow)", PowOpAdapter(self.mgr.Pow, 80), False),  # pow
                Rule(
                    r"(and)", InfixOpAdapter(self.AndOrBVAnd, 40), False
                ),  # conjunction
                Rule(
                    r"(or)", InfixOpAdapter(self.OrOrBVOr, 30), False
                ),  # disjunction
                Rule(
                    r"(abs)", UnaryOpAdapter(self.mgr.Abs, 50), False
                ),  # absolute value
                Rule(
                    r"(b\()", BinaryLiteralExpr(self.BinaryLiteral, 50), False
                ),  # b()
                Rule(
                    r"(==)", InfixOpAdapter(self.mgr.Equals, 60), False
                ),  # eq
                Rule(
                    r"(-?\d+e\+?\d+)", self.real_constant, True
                ),  # decimals scientific
                Rule(
                    r"(-?\d+?e-?\d+)", self.real_constant, True
                ),  # decimals scientific
                Rule(
                    r"(-?\d+\.\d+e\+?\d+)", self.real_constant, True
                ),  # decimals scientific
                Rule(
                    r"(-?\d+\.\d+e-?\d+)", self.real_constant, True
                ),  # decimals scientific
            ]
            + self.rules[0:-1]
            + [
                Rule(
                    r"(([A-Za-z_]|[^\u0000-\u007F])([A-Za-z_]|[^\u0000-\u007F])*)",
                    self.identifier,
                    True,
                ),  # unicode identifiers
            ]
            + self.rules[-1:]
        )
        self.compile()

    def BinaryLiteral(self, x):
        return x

    def int_constant(self, read):
        return Constant(self.mgr.Real(int(read)))

    def identifier(self, read):
        r = self.identifier_map.get(read, read)
        try:
            return Identifier(r, env=self.env)
        except UndefinedSymbolError as e:
            l.exception(
                f"Could not resolve parsed identifier: {r}, because:\n{e}"
            )


class DRealConverter(Converter, DagWalker):
    def __init__(self, environment):
        # self.__setattr__("walk_abs", walk_abs)
        DagWalker.__init__(self, environment)
        self.backconversion = {}
        self.mgr = environment.formula_manager
        self._get_type = environment.stc.get_type

        # Maps a Symbol into the corresponding internal yices instance
        self.symbol_to_decl = {}
        # Maps an internal yices instance into the corresponding symbol
        self.decl_to_symbol = {}

    def rewrite_dreal_formula(self, formula: dreal.Formula) -> str:
        # Convert and, or, and "b" markers
        str_formula = (
            str(formula)
            .replace(" and ", " & ")
            .replace(" or ", " | ")
            .replace("b(", "(")
            .replace("==", "=")
        )

        # Remove scientific notation
        str_formula = re.sub(
            r"(?<![.d_0-9a-z])[0-9]+[.]*[0-9]*e(-|\+|)[0-9]+(?![.d/])",
            lambda x: str(Fraction(x.group())),
            str_formula,
        )

        # Replace integers with floats
        str_formula = re.sub(
            r"(?<![.d_0-9a-zA-Z])[0-9]+(?!([.de-]|[0-9]))",
            r"\g<0>.0",
            str_formula,
        )

        # Remove "pow" and add "^"
        # str_formula = str_formula.replace("pow(gamma, 2.0)", "gamma^2.0")
        # str_formula = str_formula.replace("pow(beta, 2.0)", "beta^2.0")

        str_formula = re.sub(
            r"pow\([\(\)\-a-z0-9\_\+\*\.\^ ]+\, [\-0-9a-z.]+\)",
            lambda x: "("
            + x.group().split(",")[0].split("(", 1)[1]
            + ")"
            + "^"
            + "("
            + x.group().split(",")[1].split(")")[0].strip()
            + ")",
            str_formula,
        )

        return str_formula

    def create_dreal_symbols(self, rewritten_formula: str) -> List[Symbol]:
        patterns = ["(disj[0-9]+)", "(conj[0-9]+)", "neg", "ITE[0-9]*"]
        symbol_names = [
            q for p in patterns for q in list(re.findall(p, rewritten_formula))
        ]
        symbols = [Symbol(s) for s in symbol_names]
        return symbols

    def back(self, dreal_formula: dreal.Formula) -> FNode:
        try:
            # rewritten_formula = self.rewrite_dreal_formula(dreal_formula)
            l.debug(
                f"Extracting dreal unsat core expression: {str(dreal_formula)}"
            )
            new_symbols = self.create_dreal_symbols(str(dreal_formula))
            formula = CoreParser().parse(str(dreal_formula))
        except Exception as e:
            raise e
        return formula

    @catch_conversion_error
    def convert(self, formula):
        return self.walk(formula)

    def walk_and(self, formula, args, **kwargs):
        res = functools.reduce(lambda a, b: dreal.And(a, b), args)
        # self._check_term_result(res)
        return res

    def walk_iff(self, formula, args, **kwargs):
        res = args[0] == args[1]  # dreal.Equals(args[0], args[1])
        # self._check_term_result(res)
        return res

    def walk_implies(self, formula, args, **kwargs):
        res = dreal.Implies(args[0], args[1])
        # self._check_term_result(res)
        return res

    def walk_ite(self, formula, args, **kwargs):
        converted_args = []
        for arg in args:
            if isinstance(arg, Fraction):
                converted_args.append(float(arg))
            elif isinstance(arg, dreal.Variable):
                converted_args.append(dreal.Expression(arg))
            else:
                converted_args.append(arg)

        res = dreal.if_then_else(
            converted_args[0], converted_args[1], converted_args[2]
        )
        # self._check_term_result(res)
        return res

    def walk_or(self, formula, args, **kwargs):
        res = functools.reduce(lambda a, b: dreal.Or(a, b), args)
        # self._check_term_result(res)
        return res

    def walk_not(self, formula, args, **kwargs):
        res = dreal.Not(args[0])
        return res

    def walk_forall(self, formula, args, **kwargs):
        symbols = [self.walk(s) for s in formula._content.payload]
        res = dreal.forall(symbols, args[0])

        return res

    def walk_symbol(self, formula, **kwargs):
        symbol_type = formula.symbol_type()
        # var_type = self._type_to_dreal(symbol_type)
        if symbol_type.is_real_type():
            term = dreal.Variable(formula.symbol_name(), dreal.Variable.Real)
        elif symbol_type.is_bool_type():
            term = dreal.Variable(formula.symbol_name(), dreal.Variable.Bool)
        self.symbol_to_decl[formula] = term
        self.decl_to_symbol[term] = formula
        # yicespy.yices_set_term_name(term, formula.symbol_name())
        # self._check_term_result(term)
        return term

    def walk_le(self, formula, args, **kwargs):
        return self.bool_to_formula(args[0] <= args[1])

    def walk_lt(self, formula, args, **kwargs):
        return self.bool_to_formula(args[0] < args[1])

    def walk_plus(self, formula, args, **kwargs):
        return functools.reduce(lambda a, b: a + b, args)

    def walk_minus(self, formula, args, **kwargs):
        return args[0] - args[1]

    def walk_times(self, formula, args, **kwargs):
        try:
            res = functools.reduce(lambda a, b: a * b, args)
        except OverflowError as e:
            pass

        return res

    def walk_pow(self, formula, args, **kwargs):
        exponent = float(args[1]) if isinstance(args[1], Fraction) else args[1]
        base = float(args[0]) if isinstance(args[0], Fraction) else args[0]
        res = dreal.pow(base, exponent)
        return res

    def walk_abs(self, formula, args, **kwargs):
        res = abs(args[0])
        return res

    def bool_to_formula(self, value):
        if isinstance(value, bool):
            value = dreal.Formula.TRUE() if value else dreal.Formula.FALSE()
        return value

    def walk_equals(self, formula, args, **kwargs):
        res = args[0] == args[1]
        # tp = self._get_type(formula.arg(0))
        # res = None
        # if tp.is_bv_type():
        #     res = yicespy.yices_bveq_atom(args[0], args[1])
        # elif tp.is_int_type() or tp.is_real_type():
        #     res = yicespy.yices_arith_eq_atom(args[0], args[1])
        # else:
        #     assert tp.is_custom_type()
        #     res = yicespy.yices_eq(args[0], args[1])
        # self._check_term_result(res)

        return self.bool_to_formula(res)

    def walk_bool_constant(self, formula, **kwargs):
        if formula.constant_value():
            return dreal.Formula.TRUE()
        else:
            return dreal.Formula.FALSE()

    def walk_real_constant(self, formula, **kwargs):
        frac = formula.constant_value()
        return frac
        # n, d = frac.numerator, frac.denominator
        # # print(f"n = {n}, d = {d}")
        # try:
        #     res = float(n) / float(d)
        # except OverflowError as e:
        #     # int cannot be coverted to float
        #     try:
        #         f = Fraction(n, d).limit_denominator(1e309)
        #         res = float(f)
        #     except OverflowError as e1:
        #         res = frac

        # # self._check_term_result(res)
        # return res

    def _type_to_dreal(self, tp):
        if tp.is_bool_type():
            return dreal._dreal_py.Type.Bool
        elif tp.is_real_type():
            return dreal._dreal_py.Type.Real
        elif tp.is_int_type():
            return dreal._dreal_py.Type.Int
        elif tp.is_function_type():
            stps = [self._type_to_dreal(x) for x in tp.param_types]
            rtp = self._type_to_dreal(tp.return_type)
            raise NotImplementedError(tp)
            # arr = (yicespy.type_t * len(stps))(*stps)
            # return yicespy.yices_function_type(len(stps),
            #                                   stps,
            #                                   rtp)
        # elif tp.is_bv_type():
        #     return yicespy.yices_bv_type(tp.width)
        # elif tp.is_custom_type():
        #     return self.yicesSort(str(tp))
        else:
            raise NotImplementedError(tp)

    def get_dreal_ref(self, formula):
        # if formula.node_type in op.QUANTIFIERS:
        #     return z3.QuantifierRef
        # elif formula.node_type() in BOOLREF_SET:
        #     return z3.BoolRef
        # elif formula.node_type() in ARITHREF_SET:
        #     return z3.ArithRef
        # elif formula.node_type() in BITVECREF_SET:
        #     return z3.BitVecRef
        # el
        if formula.is_symbol() or formula.is_function_application():
            if formula.is_function_application():
                type_ = formula.function_name().symbol_type()
                type_ = type_.return_type
            else:
                type_ = formula.symbol_type()

            if type_.is_bool_type():
                return dreal._dreal_py.Type.Bool
            elif type_.is_real_type():
                return dreal._dreal_py.Type.Real
            elif type_.is_int_type():
                return dreal._dreal_py.Type.Int
            else:
                raise NotImplementedError(formula)
        elif formula.is_ite():
            child = formula.arg(1)
            return self.get_dreal_ref(child)
        else:
            assert formula.is_constant(), formula
            type_ = formula.constant_type()
            if type_.is_bool_type():
                return dreal._dreal_py.Type.Bool
            elif type_.is_real_type():
                return dreal._dreal_py.Type.Real
            else:
                raise NotImplementedError(formula)

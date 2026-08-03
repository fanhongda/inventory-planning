"""
Safe expression evaluator for contract derivations and assertions.

Both `derivable_from` (contract) and `derivations` (adapter) declare business logic
as short expression strings, e.g.::

    open_qty: "order_qty - coalesce(delivered_qty, 0)"

Assertions use the same grammar and evaluate to a boolean mask::

    "open_qty >= 0"
    "open_qty <= order_qty"

The evaluator walks a whitelisted Python AST and evaluates node-by-node against a
DataFrame, so an expression can only reference columns and the functions in
``FUNCTIONS``. There is no ``eval()`` / ``exec()`` anywhere on this path — adapter
YAML is a data artefact that may be LLM-generated, so it must not be able to reach
the interpreter.

Vectorised throughout: every leaf is either a pandas Series (column) or a scalar,
and every operator is a pandas/numpy operation over the full column.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd


class ExpressionError(ValueError):
    """Raised when an expression is malformed, unsafe, or references a missing column."""


# ── Whitelisted operators ────────────────────────────────────────────────────

def _safe_div(left: Any, right: Any) -> Any:
    """
    Division that yields NaN rather than inf when the denominator is zero.

    Every division in a contract is a unit-rate derivation — cost per unit, days per
    order — and a zero denominator means the rate is *undefined*, not unbounded. An
    inf here is worse than a null because it survives arithmetic and silently poisons
    any average computed downstream.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = operator.truediv(left, right)
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan)
    if np.isscalar(result) and not np.isfinite(result):
        return np.nan
    return result


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: _safe_div,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

def _isin(left: Any, right: Any) -> Any:
    """`col in [a, b]` — membership, element-wise over a Series."""
    values = list(right) if isinstance(right, (list, tuple, set)) else [right]
    if isinstance(left, pd.Series):
        return left.isin(values)
    return left in values


def _not_isin(left: Any, right: Any) -> Any:
    result = _isin(left, right)
    return ~result if isinstance(result, pd.Series) else not result


_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: _isin,
    ast.NotIn: _not_isin,
}

_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Not: operator.invert,   # pandas boolean Series negation
}


# ── Whitelisted functions ────────────────────────────────────────────────────

def _coalesce(*args: Any) -> Any:
    """First non-null value, element-wise. Mirrors SQL COALESCE."""
    if not args:
        raise ExpressionError("coalesce() requires at least one argument")
    result = args[0]
    for nxt in args[1:]:
        if isinstance(result, pd.Series):
            result = result.fillna(nxt)
        elif result is None or (np.isscalar(result) and pd.isna(result)):
            result = nxt
    return result


def _greatest(*args: Any) -> Any:
    """Element-wise maximum across arguments."""
    return _elementwise_reduce(np.maximum, args, "greatest")


def _least(*args: Any) -> Any:
    """Element-wise minimum across arguments."""
    return _elementwise_reduce(np.minimum, args, "least")


def _elementwise_reduce(ufunc, args, name: str):
    if not args:
        raise ExpressionError(f"{name}() requires at least one argument")
    result = args[0]
    for nxt in args[1:]:
        result = ufunc(result, nxt)
    return result


def _abs(x: Any) -> Any:
    return np.abs(x)


def _if_null(value: Any, fallback: Any) -> Any:
    """Two-argument coalesce; kept as a separate name because it reads better in YAML."""
    return _coalesce(value, fallback)


def _to_number(x: Any) -> Any:
    """Coerce to numeric, non-parseable becomes NaN. Useful for dirty ERP exports."""
    if isinstance(x, pd.Series):
        return pd.to_numeric(
            x.astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )
    return pd.to_numeric(x, errors="coerce")


def _is_null(x: Any) -> Any:
    return pd.isna(x)


def _not_null(x: Any) -> Any:
    return ~pd.isna(x)


def _days_between(later: Any, earlier: Any) -> Any:
    """Whole days between two datetime columns. Returns float so NaT becomes NaN."""
    delta = pd.to_datetime(later, errors="coerce") - pd.to_datetime(earlier, errors="coerce")
    if isinstance(delta, pd.Series):
        return delta.dt.total_seconds() / 86400.0
    return delta.total_seconds() / 86400.0


def _startswith(text: Any, prefix: str) -> Any:
    """Case-insensitive prefix match. Nulls are False, never NaN."""
    if isinstance(text, pd.Series):
        return text.astype(str).str.lower().str.startswith(str(prefix).lower(), na=False)
    return str(text).lower().startswith(str(prefix).lower())


def _contains(text: Any, needle: str) -> Any:
    """Case-insensitive substring match. Nulls are False, never NaN."""
    if isinstance(text, pd.Series):
        return text.astype(str).str.lower().str.contains(
            str(needle).lower(), na=False, regex=False
        )
    return str(needle).lower() in str(text).lower()


def _lower(text: Any) -> Any:
    return text.astype(str).str.lower() if isinstance(text, pd.Series) else str(text).lower()


FUNCTIONS = {
    "coalesce": _coalesce,
    "ifnull": _if_null,
    "greatest": _greatest,
    "least": _least,
    "abs": _abs,
    "to_number": _to_number,
    "is_null": _is_null,
    "not_null": _not_null,
    "days_between": _days_between,
    "startswith": _startswith,
    "contains": _contains,
    "lower": _lower,
}


# ── Evaluator ────────────────────────────────────────────────────────────────

class Expression:
    """
    A parsed, validated expression that can be evaluated against a DataFrame.

    Parsing happens once at load time so a malformed adapter fails when the adapter
    is loaded, not midway through a production run.
    """

    def __init__(self, source: str):
        self.source = source
        try:
            self._tree = ast.parse(source, mode="eval").body
        except SyntaxError as exc:
            raise ExpressionError(f"Cannot parse expression {source!r}: {exc}") from exc
        self.columns: Set[str] = set()
        self._collect_names(self._tree)

    def _collect_names(self, node: ast.AST) -> None:
        """Walk the tree once up front: validate node types and record column refs."""
        if isinstance(node, ast.Name):
            if node.id not in FUNCTIONS:
                self.columns.add(node.id)
            return
        if isinstance(node, ast.Constant):
            return
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BIN_OPS:
                raise ExpressionError(f"Operator {type(node.op).__name__} not allowed in {self.source!r}")
            self._collect_names(node.left)
            self._collect_names(node.right)
            return
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY_OPS:
                raise ExpressionError(f"Unary operator {type(node.op).__name__} not allowed in {self.source!r}")
            self._collect_names(node.operand)
            return
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if type(op) not in _CMP_OPS:
                    raise ExpressionError(f"Comparison {type(op).__name__} not allowed in {self.source!r}")
            self._collect_names(node.left)
            for comp in node.comparators:
                self._collect_names(comp)
            return
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                self._collect_names(value)
            return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                fname = getattr(node.func, "id", type(node.func).__name__)
                raise ExpressionError(
                    f"Function {fname!r} not allowed in {self.source!r}. "
                    f"Available: {sorted(FUNCTIONS)}"
                )
            if node.keywords:
                raise ExpressionError(f"Keyword arguments not supported in {self.source!r}")
            for arg in node.args:
                self._collect_names(arg)
            return
        if isinstance(node, (ast.List, ast.Tuple)):
            for elt in node.elts:
                self._collect_names(elt)
            return
        raise ExpressionError(
            f"Expression node {type(node).__name__} not allowed in {self.source!r}"
        )

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame, extra: Dict[str, Any] = None) -> Any:
        """
        Evaluate against a DataFrame. `extra` supplies scalars/Series not in df
        (e.g. a config-supplied default lead time).

        Raises ExpressionError if a referenced column is absent — callers that want
        "skip when not satisfiable" should check `can_evaluate()` first.
        """
        scope: Dict[str, Any] = dict(extra or {})
        missing = self.missing_columns(df, extra)
        if missing:
            raise ExpressionError(
                f"Expression {self.source!r} needs column(s) {sorted(missing)} "
                f"which are not present"
            )
        for col in self.columns:
            if col in df.columns:
                scope[col] = df[col]
        return self._eval(self._tree, scope)

    def missing_columns(self, df: pd.DataFrame, extra: Dict[str, Any] = None) -> Set[str]:
        available = set(df.columns) | set((extra or {}).keys())
        return self.columns - available

    def can_evaluate(self, df: pd.DataFrame, extra: Dict[str, Any] = None) -> bool:
        """True when every referenced column is available — drives derivation fallback."""
        return not self.missing_columns(df, extra)

    def _eval(self, node: ast.AST, scope: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id in scope:
                return scope[node.id]
            raise ExpressionError(f"Unknown name {node.id!r} in {self.source!r}")

        if isinstance(node, ast.BinOp):
            return _BIN_OPS[type(node.op)](
                self._eval(node.left, scope), self._eval(node.right, scope)
            )

        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand, scope)
            if isinstance(node.op, ast.Not):
                # `not` on a Series must be element-wise negation, not truthiness
                return ~operand.astype(bool) if isinstance(operand, pd.Series) else (not operand)
            return _UNARY_OPS[type(node.op)](operand)

        if isinstance(node, ast.Compare):
            # Chained comparison (a < b < c) folds to element-wise AND
            result = None
            left = self._eval(node.left, scope)
            for op, comp_node in zip(node.ops, node.comparators):
                right = self._eval(comp_node, scope)
                part = _CMP_OPS[type(op)](left, right)
                result = part if result is None else (result & part)
                left = right
            return result

        if isinstance(node, ast.BoolOp):
            values = [self._eval(v, scope) for v in node.values]
            combine = operator.and_ if isinstance(node.op, ast.And) else operator.or_
            result = values[0]
            for val in values[1:]:
                result = combine(result, val)
            return result

        if isinstance(node, ast.Call):
            args = [self._eval(a, scope) for a in node.args]
            return FUNCTIONS[node.func.id](*args)

        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval(e, scope) for e in node.elts]

        raise ExpressionError(f"Cannot evaluate {type(node).__name__} in {self.source!r}")

    def __repr__(self) -> str:
        return f"Expression({self.source!r})"


def evaluate_first_satisfiable(
    candidates: List[str],
    df: pd.DataFrame,
    extra: Dict[str, Any] = None,
) -> tuple:
    """
    Try each candidate expression in order; return (value, source_expression) for the
    first one whose columns are all present. Returns (None, None) if none apply.

    This is the mechanism behind `derivable_from`: the contract lists the ways a field
    can be reconstructed, ordered by preference, and whichever one this particular ERP
    export supports is the one that fires.
    """
    for source in candidates:
        expr = Expression(source)
        if expr.can_evaluate(df, extra):
            return expr.evaluate(df, extra), source
    return None, None

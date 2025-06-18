import functools
import math
import operator
from collections.abc import Callable
from typing import Any, Union

import polars as pl
from polars import DataFrame, Expr, Series

__all__ = [
    "AttrComparison",
    "EdgeAttr",
    "ExprInput",
    "NodeAttr",
    "attr_comps_to_strs",
    "polars_reduce_attr_comps",
    "split_attr_comps",
]

Scalar = int | float | str | bool
ExprInput = Union[str, Scalar, "AttrExpr", Expr]


class AttrComparison:
    def __init__(self, attr: "AttrExpr", op: Callable, other: Any) -> None:
        self.attr = attr
        self.op = op
        self.other = other

    def __repr__(self) -> str:
        return f"Attr({self.attr._attr_name}) '{self.op.__name__}' {self.other}"


class AttrExpr:
    """
    A class to compose an attribute expression for graph attributes.
    Can be used both as a simple attribute reference and as a complex expression.

    Parameters
    ----------
    value : ExprInput
        The value to compose the attribute expression from.

    Examples
    --------
    >>> `AttrExpr("iou").log()`
    >>> `AttrExpr(1.0)`
    >>> `AttrExpr((1 - AttrExpr("iou")) * AttrExpr("distance"))`
    >>> `(AttrExpr("iou") > 0.5).to_comparison()`  # Creates an AttrComparison
    """

    expr: Expr

    def __init__(self, value: ExprInput) -> None:
        self._inf_exprs = []  # expressions multiplied by +inf
        self._neg_inf_exprs = []  # expressions multiplied by -inf

        if isinstance(value, str):
            self.expr = pl.col(value)
            self._attr_name = value  # Store the attribute name for comparison operations
        elif isinstance(value, AttrExpr):
            self.expr = value.expr
            self._attr_name = value._attr_name
            # Copy infinity tracking from the other AttrExpr
            self._inf_exprs = value.inf_exprs
            self._neg_inf_exprs = value.neg_inf_exprs
        elif isinstance(value, Expr):
            self.expr = value
            self._attr_name = None
        else:
            self.expr = pl.lit(value)
            self._attr_name = None

    def to_comparison(self) -> AttrComparison:
        """
        Create an AttrComparison object from this AttrExpr.
        Only works on simple comparison expressions (e.g. iou > 0.5).

        Returns
        -------
        AttrComparison
            A new AttrComparison object

        Raises
        ------
        ValueError
            If the expression is not a simple comparison
        """
        # Check if this is a simple comparison expression
        if not isinstance(self.expr, pl.Expr):
            raise ValueError("Cannot create comparison from non-expression")

        # Get the operation and operands from the expression
        op = self.expr.meta.root_names()
        if len(op) != 1:
            raise ValueError("Cannot create comparison from complex expression")

        # Extract the operation and operands
        op_name = op[1]
        if op_name not in {"eq", "ne", "lt", "le", "gt", "ge"}:
            raise ValueError(f"Expression must be a simple comparison. Found {op_name}")

        # Get the operands
        operands = self.expr.meta.arguments()
        if len(operands) != 2:
            raise ValueError("Comparison must have exactly two operands")

        # First operand should be a column reference
        if not isinstance(operands[1], pl.Expr) or not operands[1].meta.is_column():
            raise ValueError("Left operand must be a column reference")

        # Get the operation function
        op_func = getattr(operator, op_name)

        # Get the right operand value
        if isinstance(operands[0], pl.Expr) and operands[0].meta.is_literal():
            other = operands[0].meta.literal()
        else:
            raise ValueError("Right operand must be a literal value")

        return AttrComparison(self, op_func, other)

    def _wrap(self, expr: Expr | Any) -> Union["AttrExpr", Any]:
        if isinstance(expr, Expr):
            # Preserve the type of self (NodeAttr or EdgeAttr)
            result = type(self)(expr)
            # Propagate infinity tracking
            result._inf_exprs = self._inf_exprs.copy()
            result._neg_inf_exprs = self._neg_inf_exprs.copy()
            return result
        return expr

    def _delegate_operator(
        self, other: ExprInput, op: Callable[[Expr, Expr], Expr], reverse: bool = False
    ) -> "AttrExpr":
        # Special handling for multiplication with infinity
        if op == operator.mul:
            # Check if we're multiplying with infinity scalar
            # In both reverse and non-reverse cases, 'other' is the infinity value
            # and 'self' is the AttrExpr we want to track
            if isinstance(other, int | float) and math.isinf(other):
                result = type(self)(pl.lit(0))  # Clean expression is zero (infinity term removed)

                # Copy existing infinity tracking
                result._inf_exprs = self._inf_exprs.copy()
                result._neg_inf_exprs = self._neg_inf_exprs.copy()

                # Add the expression to appropriate infinity list
                if other > 0:
                    result._inf_exprs.append(self)
                else:
                    result._neg_inf_exprs.append(self)

                return result

        # Regular operation - no infinity involved
        left = AttrExpr(other).expr if reverse else self.expr
        right = self.expr if reverse else AttrExpr(other).expr
        result = type(self)(op(left, right))

        # Combine infinity tracking from both operands
        if isinstance(other, AttrExpr):
            result._inf_exprs = self._inf_exprs + other._inf_exprs
            result._neg_inf_exprs = self._neg_inf_exprs + other._neg_inf_exprs

            # Special handling for subtraction: flip signs of the second operand's infinity terms
            if op == operator.sub and not reverse:
                # self - other: other's positive infinity becomes negative, negative becomes positive
                result._inf_exprs = self._inf_exprs + other._neg_inf_exprs
                result._neg_inf_exprs = self._neg_inf_exprs + other._inf_exprs
            elif op == operator.sub and reverse:
                # other - self: self's positive infinity becomes negative, negative becomes positive
                result._inf_exprs = other._inf_exprs + self._neg_inf_exprs
                result._neg_inf_exprs = other._neg_inf_exprs + self._inf_exprs
        else:
            result._inf_exprs = self._inf_exprs.copy()
            result._neg_inf_exprs = self._neg_inf_exprs.copy()

        return result

    def alias(self, name: str) -> "AttrExpr":
        result = AttrExpr(self.expr.alias(name))
        result._inf_exprs = self._inf_exprs.copy()
        result._neg_inf_exprs = self._neg_inf_exprs.copy()
        return result

    def evaluate(self, df: DataFrame) -> Series:
        return df.select(self.expr).to_series()

    @property
    def columns(self) -> list[str]:
        return list(dict.fromkeys(self.expr_columns + self.inf_columns + self.neg_inf_columns))

    @property
    def inf_exprs(self) -> list["AttrExpr"]:
        """Get the expressions multiplied by positive infinity."""
        return self._inf_exprs.copy()

    @property
    def neg_inf_exprs(self) -> list["AttrExpr"]:
        """Get the expressions multiplied by negative infinity."""
        return self._neg_inf_exprs.copy()

    @property
    def expr_columns(self) -> list[str]:
        """Get the names of columns in the expression."""
        return list(dict.fromkeys(self.expr.meta.root_names()))

    @property
    def inf_columns(self) -> list[str]:
        """Get the names of columns multiplied by positive infinity."""
        columns = []
        for attr_expr in self._inf_exprs:
            columns.extend(attr_expr.columns)
        return list(dict.fromkeys(columns))

    @property
    def neg_inf_columns(self) -> list[str]:
        """Get the names of columns multiplied by negative infinity."""
        columns = []
        for attr_expr in self._neg_inf_exprs:
            columns.extend(attr_expr.columns)
        return list(dict.fromkeys(columns))

    def has_inf(self) -> bool:
        """
        Check if any column in the expression is multiplied by infinity or negative infinity.

        Returns
        -------
        bool
            True if any column is multiplied by infinity, False otherwise.
        """
        return self.has_pos_inf() or self.has_neg_inf()

    def has_pos_inf(self) -> bool:
        """
        Check if any column in the expression is multiplied by positive infinity.
        """
        return len(self._inf_exprs) > 0

    def has_neg_inf(self) -> bool:
        """
        Check if any column in the expression is multiplied by negative infinity.
        """
        return len(self._neg_inf_exprs) > 0

    def __invert__(self) -> "AttrExpr":
        return AttrExpr(~self.expr)

    def __neg__(self) -> "AttrExpr":
        return AttrExpr(-self.expr)

    def __pos__(self) -> "AttrExpr":
        return AttrExpr(+self.expr)

    def __abs__(self) -> "AttrExpr":
        return AttrExpr(abs(self.expr))

    def __getattr__(self, attr: str) -> Any:
        # Don't delegate our internal attributes to the expr
        if attr.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

        # To auto generate operator methods such as `.log()``
        expr_attr = getattr(self.expr, attr)
        if callable(expr_attr):

            @functools.wraps(expr_attr)
            def _wrapped(*args, **kwargs):
                return self._wrap(expr_attr(*args, **kwargs))

            return _wrapped
        return expr_attr

    def __repr__(self) -> str:
        prefix = type(self).__name__
        if self._attr_name is not None:
            return f"{prefix}({self._attr_name})"
        return f"{prefix}({_pretty_polars_expr(self.expr)})"

    @property
    def name(self) -> str:
        """Return the attribute name if available, else raise an error."""
        if hasattr(self, "_attr_name") and self._attr_name is not None:
            return self._attr_name
        raise AttributeError(f"{type(self).__name__} does not have a name (not a simple attribute reference)")


class NodeAttr(AttrExpr):
    """A class to represent a node attribute."""

    pass


class EdgeAttr(AttrExpr):
    """A class to represent an edge attribute."""

    pass


# Auto-generate operator methods using functools.partialmethod


def _add_operator(name: str, op: Callable, reverse: bool = False) -> None:
    method = functools.partialmethod(AttrExpr._delegate_operator, op=op, reverse=reverse)
    setattr(AttrExpr, name, method)


def _setup_ops() -> None:
    """
    Setup the operator methods for the AttrExpr class.
    """
    bin_ops = {
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "truediv": operator.truediv,
        "floordiv": operator.floordiv,
        "mod": operator.mod,
        "pow": operator.pow,
        "and": operator.and_,
        "or": operator.or_,
        "xor": operator.xor,
        "eq": operator.eq,
        "ne": operator.ne,
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
    }

    for op_name, op_func in bin_ops.items():
        _add_operator(f"__{op_name}__", op_func, reverse=False)
        _add_operator(f"__r{op_name}__", op_func, reverse=True)


_setup_ops()


def _pretty_polars_expr(expr: pl.Expr) -> str:
    """Recursively pretty-print a polars expression as a human-readable string."""
    # Handle literal
    if expr.meta.is_literal():
        # Try to extract the value from the string representation
        s = str(expr)
        # e.g. 'dyn float: 1' or 'dyn int: 0'
        if s.startswith("dyn "):
            return s.split(": ", 1)[1]
        return s
    # Handle column
    if expr.meta.is_column():
        return expr.meta.root_names()[0]
    # Handle binary operations
    args = expr.meta.arguments()
    if len(args) == 2:
        left = _pretty_polars_expr(args[0])
        right = _pretty_polars_expr(args[1])
        expr_str = str(expr)
        # Find the operator in the string
        for op in ["+", "-", "*", "/", "==", "!=", "<", "<=", ">", ">="]:
            if op in expr_str:
                return f"{left} {op} {right}"
        # Fallback
        return f"({left}, {right})"
    # Fallback to str
    return str(expr)


def as_attr_comparison_list(attr_comps: list[Any]) -> list[AttrComparison]:
    """
    Convert a list of mixed expressions to a list of valid AttrComparison objects.
    Uses .to_comparison() if possible.
    Skips items that cannot be converted.
    """
    result = []
    for item in attr_comps:
        if isinstance(item, AttrComparison):
            result.append(item)
        elif isinstance(item, (NodeAttr, EdgeAttr)):
            try:
                comp = item.to_comparison()
                result.append(comp)
            except Exception:
                raise ValueError(f"Cannot convert {item} to AttrComparison")
    return result


def split_attr_comps(attr_comps: list[Any]) -> tuple[list[AttrComparison], list[AttrComparison]]:
    """
    Split a list of attribute comparisons into node and edge attribute comparisons.
    Accepts mixed input and converts to AttrComparison where possible.
    Only includes AttrComparison objects whose .attr is a NodeAttr or EdgeAttr.
    """
    node_attr_comps = []
    edge_attr_comps = []
    attr_comps = as_attr_comparison_list(attr_comps)
    for attr_comp in attr_comps:
        if isinstance(attr_comp.attr, NodeAttr):
            node_attr_comps.append(attr_comp)
        elif isinstance(attr_comp.attr, EdgeAttr):
            edge_attr_comps.append(attr_comp)
        # else: skip
    return node_attr_comps, edge_attr_comps


def attr_comps_to_strs(attr_comps: list[AttrComparison]) -> list[str]:
    """
    Convert a list of attribute comparisons to a list of strings.
    """
    return [attr_comp.attr.name for attr_comp in attr_comps]


def polars_reduce_attr_comps(df: pl.DataFrame, attr_comps: list[AttrComparison]) -> pl.Expr:
    """
    Reduce a list of attribute comparisons to a single polars expression.

    Parameters
    ----------
    df : pl.DataFrame
        The dataframe to reduce the attribute comparisons on.
    attr_comps : List[AttrComparison]
        The attribute comparisons to reduce.

    Returns
    -------
    pl.Expr
        The reduced polars expression.
    """
    return pl.reduce(
        lambda x, y: x & y, [attr_comp.op(df[attr_comp.attr.name], attr_comp.other) for attr_comp in attr_comps]
    )

"""Formulas: a small expression language, parsed here, never handed to ``eval``.

A parameter can be a formula of other parameters (``x_h = W - 0.15``) and a region's
coefficient can be a formula of position (``45 + 955 * exp(-((x - 0.3)^2 + (y - 0.15)^2) / 0.002)``),
as the plan's section 3.4 asks.  What the language has, and nothing more:

* numbers (``3``, ``0.5``, ``2e-3``), ``pi`` and ``e``;
* names: the parameters, and ``x``, ``y`` (and ``T``) where a field is evaluated;
* ``+ - * / ^`` (also ``**``, and the page's ``×``, ``÷``, ``−``), unary minus, brackets;
* the functions ``exp log sqrt abs sin cos tan tanh min max``.

`parse` builds a tree of tuples; `evaluate` walks it, with NumPy, so a field is
evaluated at every cell centre at once.  A name that is not defined, a function that
is not in the list, or anything else is refused with its reason (`FormulaError`).
The page has the same language in ``static/js/geo/formula.js`` for its previews; a
test checks the two agree.
"""

from __future__ import annotations

import math
import re

import numpy as np

FUNCTIONS = {
    "exp": (1, np.exp), "log": (1, np.log), "sqrt": (1, np.sqrt), "abs": (1, np.abs),
    "sin": (1, np.sin), "cos": (1, np.cos), "tan": (1, np.tan), "tanh": (1, np.tanh),
    "min": (2, np.minimum), "max": (2, np.maximum),
}
CONSTANTS = {"pi": math.pi, "e": math.e}

_TOKEN = re.compile(r"\s*(?:(\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?)"
                    r"|([A-Za-z_][A-Za-z0-9_]*)|(\*\*|[-+*/^(),]))")


class FormulaError(ValueError):
    """A formula that cannot be read or evaluated; the message says why."""


def _normalise(text: str) -> str:
    return (text.replace("−", "-").replace("×", "*").replace("÷", "/")
            .replace("·", "*"))


def tokens(text: str) -> list[tuple[str, str]]:
    text = _normalise(text)
    out, pos = [], 0
    while pos < len(text):
        if text[pos:].strip() == "":
            break
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise FormulaError(f"cannot read {text[pos:].strip()[:12]!r}")
        num, name, op = m.groups()
        out.append(("num", num) if num else ("name", name) if name else ("op", "^" if op == "**" else op))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else ("end", "")

    def take(self, kind=None, value=None):
        tok = self.peek()
        if (kind and tok[0] != kind) or (value and tok[1] != value):
            want = value or kind
            raise FormulaError(f"expected {want!r}, found {tok[1] or 'the end'!r}")
        self.i += 1
        return tok

    def expr(self):
        node = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            node = (op, node, self.term())
        return node

    def term(self):
        node = self.unary()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            node = (op, node, self.unary())
        return node

    def unary(self):
        if self.peek() == ("op", "-"):
            self.take()
            return ("neg", self.unary())
        if self.peek() == ("op", "+"):
            self.take()
            return self.unary()
        return self.power()

    def power(self):
        base = self.atom()
        if self.peek() == ("op", "^"):
            self.take()
            return ("^", base, self.unary())            # right-associative
        return base

    def atom(self):
        kind, val = self.peek()
        if kind == "num":
            self.take()
            return ("num", float(val))
        if kind == "name":
            self.take()
            if self.peek() == ("op", "("):
                if val not in FUNCTIONS:
                    raise FormulaError(f"no function {val!r} (there are "
                                       f"{', '.join(sorted(FUNCTIONS))})")
                self.take("op", "(")
                args = [self.expr()]
                while self.peek() == ("op", ","):
                    self.take()
                    args.append(self.expr())
                self.take("op", ")")
                n = FUNCTIONS[val][0]
                if len(args) != n:
                    raise FormulaError(f"{val} takes {n} argument{'s' * (n != 1)}")
                return ("call", val, args)
            return ("name", val)
        if (kind, val) == ("op", "("):
            self.take()
            node = self.expr()
            self.take("op", ")")
            return node
        raise FormulaError(f"unexpected {val or 'end'!r}")


def parse(text: str):
    if not isinstance(text, str) or not text.strip():
        raise FormulaError("an empty formula")
    if len(text) > 2000:
        raise FormulaError("a formula of at most 2000 characters")
    p = _Parser(tokens(text))
    node = p.expr()
    if p.peek()[0] != "end":
        raise FormulaError(f"unexpected {p.peek()[1]!r}")
    return node


def names(node) -> set[str]:
    """Every name a formula reads."""
    if node[0] == "name":
        return {node[1]}
    if node[0] == "call":
        return set().union(*(names(a) for a in node[2]))
    if node[0] in ("neg",):
        return names(node[1])
    if node[0] == "num":
        return set()
    return names(node[1]) | names(node[2])


def evaluate(node, env: dict):
    k = node[0]
    if k == "num":
        return node[1]
    if k == "name":
        n = node[1]
        if n in env:
            return env[n]
        if n in CONSTANTS:
            return CONSTANTS[n]
        raise FormulaError(f"{n!r} is not defined here")
    if k == "neg":
        return -evaluate(node[1], env)
    if k == "call":
        return FUNCTIONS[node[1]][1](*(evaluate(a, env) for a in node[2]))
    a, b = evaluate(node[1], env), evaluate(node[2], env)
    with np.errstate(all="ignore"):
        if k == "+":
            return a + b
        if k == "-":
            return a - b
        if k == "*":
            return a * b
        if k == "/":
            return a / b
        if k == "^":
            return np.power(a, b)
    raise FormulaError(f"unknown operation {k!r}")


def value(text: str, env: dict) -> float:
    v = evaluate(parse(text), env)
    v = float(np.asarray(v))
    if not math.isfinite(v):
        raise FormulaError(f"{text!r} is not a finite number here")
    return v


def resolve_parameters(params: list) -> dict[str, float]:
    """Each parameter's value, formulas evaluated in the order they depend on each
    other; a cycle is refused with the names in it."""
    defs = {p.name: p for p in params}
    out: dict[str, float] = {}
    visiting: list[str] = []

    def get(n: str) -> float:
        if n in out:
            return out[n]
        if n in visiting:
            raise FormulaError("the parameters depend on each other in a circle: "
                               + " -> ".join(visiting[visiting.index(n):] + [n]))
        p = defs[n]
        if not p.expr:
            out[n] = float(p.value)
            return out[n]
        visiting.append(n)
        tree = parse(p.expr)
        env = {m: get(m) for m in names(tree) if m in defs}
        out[n] = value(p.expr, env)
        visiting.pop()
        return out[n]
    for n in defs:
        get(n)
    return out


__all__ = ["CONSTANTS", "FUNCTIONS", "FormulaError", "evaluate", "names", "parse",
           "resolve_parameters", "tokens", "value"]

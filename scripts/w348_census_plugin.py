"""W348's census: every graph the test suite compiles, and every non-admit decision.

A pytest plugin, loaded through ``PYTEST_PLUGINS=w348_census_plugin`` with this
folder on ``PYTHONPATH`` and ``W348_CENSUS`` naming an output folder (see
`scripts/w348_census.py`, which runs it and diffs two censuses).  It wraps
`atlas.compiler.compile_scheme` BEFORE any test module is imported, so every name
the package binds to it -- ``atlas.compile_scheme``, the case modules' own imports --
reaches the wrapper, and writes one JSON line per compile: the test that ran it,
the graph's name, the verdict, and every decision that is not ``admit`` (layer,
rule, verdict, subject).  The census is what W348 is diffed on: "the refusal set of
every graph the suite compiles, before and after; only the intended rows move".
"""

from __future__ import annotations

import json
import os
import sys

_OUT = os.environ.get("W348_CENSUS")


def _install() -> None:
    import atlas.compiler as C
    original = C.compile_scheme
    if getattr(original, "_w348_census", False):
        return

    def compile_scheme(graph, *args, **kwargs):
        result = original(graph, *args, **kwargs)
        try:
            _write(graph, result)
        except Exception as exc:                          # the census never breaks a test
            sys.stderr.write(f"w348 census: {type(exc).__name__}: {exc}\n")
        return result
    compile_scheme._w348_census = True
    compile_scheme.__wrapped__ = original
    C.compile_scheme = compile_scheme
    for mod in list(sys.modules.values()):
        if mod is not None and getattr(mod, "compile_scheme", None) is original:
            setattr(mod, "compile_scheme", compile_scheme)


def _write(graph, result) -> None:
    if not _OUT:
        return
    os.makedirs(_OUT, exist_ok=True)
    rows = []
    for d in result.decisions:
        v = d.verdict.value
        if v != "admit":
            rows.append([d.layer, d.rule, v, d.subject or ""])
    line = {"test": os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" ", 1)[0],
            "graph": getattr(graph, "name", "?"),
            "agents": len(getattr(graph, "agents", []) or []),
            "verdict": result.verdict.value, "decisions": rows}
    path = os.path.join(_OUT, f"census-{os.getpid()}.jsonl")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, sort_keys=True) + "\n")


_install()

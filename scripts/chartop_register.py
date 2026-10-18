"""Stage 0 (W355): write the chart operator's registration, BEFORE any training run.

    python scripts/chartop_register.py

Writes ``out/chartop/registered.txt``: the gates and comparison bars of
`atlas/chartop/gates.py` with that module's sha256 (line endings normalised), the
splits and their seeds (the test split is drawn only at the one evaluation), the
training envelope, the held-out benchmarks, and the git HEAD.
``tests/test_chartop_gates.py`` asserts that the file and the module still agree, so
a bar changed after registration fails the suite rather than passing silently.

It refuses to overwrite an existing registration: re-registering is a decision, made
by deleting the file in a commit that says why.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out", "chartop", "registered.txt")
MODULE = os.path.join(ROOT, "atlas", "chartop", "gates.py")


def module_hash(path: str = MODULE) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def main() -> int:
    from atlas.chartop import data, gates as G
    if os.path.exists(OUT):
        raise SystemExit("%s exists; a registration is not overwritten" % OUT)
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    env = data.ENVELOPE
    lines = [
        "THE CHART OPERATOR, STAGE 0 (CO-D, W355): REGISTRATION",
        "",
        "written            %s" % dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "before             any training run (no network has been trained)",
        "repository at      %s (the commit adding this file follows it)" % head,
        "gate module        atlas/chartop/gates.py  sha256 %s" % module_hash(),
        "registered as      %s" % G.REGISTERED,
        "",
        "SPLITS (example i of a split has seed base + i)",
    ]
    for k, v in G.SEED_BASE.items():
        lines.append("  %-16s seed base %d" % (k, v))
    lines += [
        "  validation       %d examples (picks the weights)" % G.N_VAL,
        "  test             %d examples, seeds %d..%d, drawn only at the one evaluation"
        % (G.N_TEST, G.TEST_SEEDS[0], G.TEST_SEEDS[1]),
        "  held out         the benchmarks %s, problems %s, at n = %d"
        % (", ".join(G.HELD_OUT_BENCHMARKS), " and ".join(G.BENCHMARK_PROBLEMS), G.BENCHMARK_N),
        "",
        "THE TRAINING ENVELOPE (data.ENVELOPE; the card declares it)",
        "  sup|mu| <= %g, max log J - min log J <= %.4g, |grad log J| + |grad mu| <= %g,"
        % (env.k_mu, env.l_J, env.s_grad),
        "  |log kappa_hat| <= %g, Fo in [%g, %g] or steady" % (env.l_kappa, env.fo_min, env.fo_max),
        "",
        G.text(),
        "",
        "the measures are defined in the module's docstring; the judgements are its judge_*",
        "functions, applied once by scripts/chartop_evaluate.py.  The arms' budgets are fixed",
        "after pricing, before training, in out/chartop/plan.json.",
        "",
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(lines))
    print("registered ->", OUT)
    print("gate module sha256", module_hash())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

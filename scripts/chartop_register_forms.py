"""The registration after Stage 0 (W356-W358): pin `atlas/chartop/gates_forms.py`, BEFORE
any network in these forms is trained.

    python scripts/chartop_register_forms.py

Writes ``out/chartop/registered-forms.txt``, as `scripts/chartop_register.py` wrote
Stage 0's ``registered.txt``.  It records the bars, comparison and arms of
`atlas/chartop/gates_forms.py`, with that module's sha256 and the sha256 of
`atlas/chartop/gates.py`, whose bars it reads (line endings normalised).  It also
records the splits and their seeds, the training envelope, the held-out benchmarks,
and the git HEAD.  ``tests/test_chartop_gates_forms.py`` asserts that the file and both
modules still agree, so a bar changed after registration fails the suite rather than
passing silently.

It refuses to overwrite an existing registration: re-registering is a decision, made by
deleting the file in a commit that says why.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out", "chartop", "registered-forms.txt")
MODULE = os.path.join(ROOT, "atlas", "chartop", "gates_forms.py")
STAGE0 = os.path.join(ROOT, "atlas", "chartop", "gates.py")


def module_hash(path: str = MODULE) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", default="any network in these forms was trained (no training "
                                        "run, no rental; E5 not yet priced)",
                    help="what this registration precedes, stated exactly")
    args = ap.parse_args()
    from atlas.chartop import data, gates_forms as F
    if os.path.exists(OUT):
        raise SystemExit("%s exists; a registration is not overwritten" % OUT)
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    env = data.ENVELOPE
    lines = [
        "THE CHART OPERATOR AFTER STAGE 0 (CO-D, W356-W358): REGISTRATION",
        "",
        "written            %s" % dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "before             %s" % args.before,
        "repository at      %s (the commit adding this file follows it)" % head,
        "gate module        atlas/chartop/gates_forms.py  sha256 %s" % module_hash(MODULE),
        "reads the bars of  atlas/chartop/gates.py  sha256 %s" % module_hash(STAGE0),
        "registered as      %s" % F.REGISTERED,
        "",
        "SPLITS (example i of a split has seed base + i), as Stage 0",
    ]
    for k, v in F.SEED_BASE.items():
        lines.append("  %-16s seed base %d" % (k, v))
    lines += [
        "  validation       %d examples (picks the weights)" % F.N_VAL,
        "  test             %d examples, seeds %d..%d, drawn only at the one evaluation"
        % (F.N_TEST, F.TEST_SEEDS[0], F.TEST_SEEDS[1]),
        "  held out         the benchmarks %s, problems %s, at n = %d"
        % (", ".join(F.HELD_OUT_BENCHMARKS), " and ".join(F.BENCHMARK_PROBLEMS), F.BENCHMARK_N),
        "",
        "THE TRAINING ENVELOPE (data.ENVELOPE), as Stage 0",
        "  sup|mu| <= %g, max log J - min log J <= %.4g, |grad log J| + |grad mu| <= %g,"
        % (env.k_mu, env.l_J, env.s_grad),
        "  |log kappa_hat| <= %g, Fo in [%g, %g] or steady" % (env.l_kappa, env.fo_min, env.fo_max),
        "",
        F.text(),
        "",
        "the measures are defined in the module's docstring; the judgements are its functions",
        "(e_h_learned, judge_*, cw_measure, compare_pairings), applied once at the evaluation.",
        "The arms' steps and seeds are fixed here; their batches, learning rate and validation",
        "cadence are Stage 0's plan's, written for these arms after E5's pricing, before training.",
        "",
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(lines))
    print("registered ->", OUT)
    print("gate module sha256", module_hash(MODULE))
    print("gates.py    sha256", module_hash(STAGE0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

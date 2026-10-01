"""W348: the refusal set of every graph the suite compiles, before and after R10 moves.

    python scripts/w348_census.py run before     # the full suite, every compile recorded
    python scripts/w348_census.py run after
    python scripts/w348_census.py diff before after [--relaxed]

``run`` runs `scripts/run_suite.py` (the whole suite, its logs in ``out/suite/``) with
`w348_census_plugin` loaded into every pytest process, which writes one JSON line per
`compile_scheme` call to ``out/w348/<label>/``.  ``diff`` keys each compile by (the
test, the graph's name, which call of that graph in that test it was) and prints
every graph whose verdict moved and every non-admit decision that appeared or went,
then a summary (``out/workbench/records/w348/census/diff-<before>-<after>.json``; the
raw census stays in the ignored ``out/w348/``).  W348's gate, from the gap worklist:
"only the intended rows move".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "w348")                       # the raw census, ignored
RECORDS = os.path.join(ROOT, "out", "workbench", "records", "w348", "census")


def run(label: str) -> int:
    folder = os.path.join(OUT, label)
    os.makedirs(folder, exist_ok=True)
    for f in os.listdir(folder):
        if f.startswith("census-"):
            os.remove(os.path.join(folder, f))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [os.path.join(ROOT, "scripts"),
                                                      env.get("PYTHONPATH", "")]))
    env["PYTEST_PLUGINS"] = "w348_census_plugin"
    env["W348_CENSUS"] = folder
    started = time.time()
    with open(os.path.join(folder, "started.txt"), "w", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S\n", time.localtime(started)))
    code = subprocess.call([sys.executable, os.path.join(ROOT, "scripts", "run_suite.py")],
                           cwd=ROOT, env=env)
    with open(os.path.join(folder, "finished.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} exit {code} "
                 f"({time.time() - started:.0f} s)\n")
    return code


def load(label: str, relaxed: bool = False) -> dict[tuple, dict]:
    """``relaxed`` strips a parametrized test's ``[...]`` id, so a test whose
    parameters were renamed (a pinned verdict in the id) still pairs with itself:
    its compiles run in the same order on both sides, so the occurrence index
    aligns them."""
    folder = os.path.join(OUT, label)
    rows = []
    for f in sorted(os.listdir(folder)):
        if f.startswith("census-") and f.endswith(".jsonl"):
            with open(os.path.join(folder, f), encoding="utf-8") as fh:
                rows += [json.loads(line) for line in fh if line.strip()]
    seen: Counter = Counter()
    out = {}
    for r in rows:
        test = r["test"].split("[", 1)[0] if relaxed else r["test"]
        base = (test, r["graph"])
        out[base + (seen[base],)] = r
        seen[base] += 1
    return out


def diff(before: str, after: str, relaxed: bool = False) -> int:
    a, b = load(before, relaxed), load(after, relaxed)
    keys = sorted(set(a) | set(b))
    moved, only_a, only_b = [], [], []
    rule_went: Counter = Counter()
    rule_came: Counter = Counter()
    for k in keys:
        if k not in b:
            only_a.append(k)
            continue
        if k not in a:
            only_b.append(k)
            continue
        ra = Counter(tuple(x) for x in a[k]["decisions"])
        rb = Counter(tuple(x) for x in b[k]["decisions"])
        went, came = ra - rb, rb - ra
        if went or came or a[k]["verdict"] != b[k]["verdict"]:
            moved.append({"test": k[0], "graph": k[1], "call": k[2],
                          "verdict": [a[k]["verdict"], b[k]["verdict"]],
                          "went": sorted(went.elements()), "came": sorted(came.elements())})
            for x in went.elements():
                rule_went[(x[0], x[1], x[2])] += 1
            for x in came.elements():
                rule_came[(x[0], x[1], x[2])] += 1
    graphs = defaultdict(set)
    for m in moved:
        graphs[m["graph"]].add(tuple(m["verdict"]))
    summary = {
        "before": before, "after": after, "relaxed": relaxed,
        "compiles": [len(a), len(b)],
        "refusing": [sum(1 for r in a.values() if r["verdict"] == "refuse"),
                     sum(1 for r in b.values() if r["verdict"] == "refuse")],
        "moved": len(moved), "only_before": len(only_a), "only_after": len(only_b),
        "graphs_moved": {g: sorted(list(v)) for g, v in sorted(graphs.items())},
        "decisions_went": {"/".join(k): n for k, n in sorted(rule_went.items())},
        "decisions_came": {"/".join(k): n for k, n in sorted(rule_came.items())},
    }
    os.makedirs(RECORDS, exist_ok=True)
    for label in (before, after):
        for f in ("started.txt", "finished.txt"):
            src = os.path.join(OUT, label, f)
            if os.path.exists(src):
                with open(src, encoding="utf-8") as fh:
                    summary[f"{label}_{f[:-4]}"] = fh.read().strip()
    path = os.path.join(RECORDS, f"diff-{before}-{after}{'-relaxed' if relaxed else ''}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "moved": moved,
                   "only_before": [list(k) for k in only_a],
                   "only_after": [list(k) for k in only_b]}, fh, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"written {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        sys.exit(run(sys.argv[2]))
    if len(sys.argv) >= 4 and sys.argv[1] == "diff":
        sys.exit(diff(sys.argv[2], sys.argv[3], relaxed="--relaxed" in sys.argv))
    print(__doc__)
    sys.exit(2)

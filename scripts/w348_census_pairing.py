"""W348's census: the front wing's four unpaired rows, paired by graph instead of by test.

    python scripts/w348_census_pairing.py > out/workbench/records/w348/census/front-wing-pairing.txt

`w348_census.py diff before after --relaxed` left two front-wing compiles seen only
before and two seen only after, in `tests/test_tier30_front_wing.py`: the same graphs,
credited to different tests.  A compile shared by several tests (cached within a
process) is credited to whichever test ran it first, and the new test file moved the
suite's parallel groups.  This prints, per graph, whether the multiset of
(verdict, non-admit decisions) over every compile of it is the same before and
after, which is what "nothing moved" means for these rows.
"""
import json
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rows(label):
    out = []
    folder = os.path.join(ROOT, "out", "w348", label)
    for f in sorted(os.listdir(folder)):
        if f.startswith("census-") and f.endswith(".jsonl"):
            for line in open(os.path.join(folder, f), encoding="utf-8"):
                out.append(json.loads(line))
    return out


for graph in ("front-wing-3x2-riding", "front-wing-3x2-fixed-shape"):
    for label in ("before", "after"):
        rs = [r for r in rows(label) if r["graph"] == graph
              and r["test"].startswith("tests/test_tier30_front_wing.py")]
        sigs = Counter((r["verdict"], tuple(map(tuple, r["decisions"]))) for r in rs)
        print(f"{graph} {label}: {len(rs)} compiles in tier30, "
              f"{len(sigs)} distinct (verdict, decisions)")
        print("   tests:", sorted({r['test'].split('::')[1][:60] for r in rs}))
    a = Counter((r["verdict"], tuple(map(tuple, r["decisions"]))) for r in rows("before")
                if r["graph"] == graph)
    b = Counter((r["verdict"], tuple(map(tuple, r["decisions"]))) for r in rows("after")
                if r["graph"] == graph)
    print(f"  {graph}, every test file: the multiset of (verdict, decisions) is "
          f"{'IDENTICAL' if a == b else 'DIFFERENT'} before and after "
          f"({sum(a.values())} and {sum(b.values())} compiles)")

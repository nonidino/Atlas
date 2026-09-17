"""Tier 74 -- the bundle carrying the column it claims to, and a commit refused
before it costs anything.

    python scripts/tier74_showable.py --out out/racelab23 --stages commit_check,bundle,summary

Two section 12 criteria, not polish:

  * **criterion 1** -- one clone, one command, a car simulating, on Windows AND
    on macOS or Linux.  The bundle's file lists stopped at Tier 57, so it shipped
    the porous column and **none** of Tiers 58-73: no body-fitted car, no
    certified mode, no second dashboard.
  * **criterion 2, for the six geometry knobs** -- a viewer could move one,
    commit it, and watch ninety seconds of re-cutting end in a column that
    declines to march (W293).  The refusal now happens before the rebuild.

  ``commit_check``  the refusal, its control, and the car it names
  ``bundle``        the bundle built and audited, against what the body-fitted
                    demo actually reads from disk
  ``summary``       every registered prediction judged in code

The two-operating-system verification is NOT in this script: it runs the
bundle's own `run.sh` / `run.py --check` on each machine and is recorded in
[[poc3-racelab-showable]], because a script here cannot claim what another
operating system did.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse                                                         # noqa: E402
import datetime as dt                                                   # noqa: E402
import json                                                             # noqa: E402
import re                                                               # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402
import traceback                                                        # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (HERE, os.path.join(HERE, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from atlas.demo_racelab import bodyfitted as BF                         # noqa: E402
import build_racelab_bundle as B                                        # noqa: E402
import tier60_body_fitted_grids as T60                                  # noqa: E402

READ_BEFORE_THIS_RUN = (
    "The bundle's file lists (SCRIPTS, TESTS, WIKI_PAGES, ARTIFACTS) stopped at Tier 57 and its "
    "last verification was commit 5a84ec8, so it shipped the porous column and none of Tiers "
    "58-73.",
    "`BodyFittedColumn.build` imports `tier62_car_solids` and `tier63_duct_openings` from "
    "scripts/ at build time, and `_sizing` reads out/racelab13/racelab13.json -- so the "
    "body-fitted page does not start without them, and nothing else in the bundle would notice.",
    "W293: a geometry commit re-cuts the car, the new car has no settled field, and `step()` "
    "raises. Until this tier the column re-cut FIRST -- ninety seconds -- and refused after.",
    "Explored before these predictions were written: a scratch build (--allow-dirty) came to "
    "112.4 MB with 4 allowed binaries and 0 unexpected; `python run.py --check` in it passed "
    "both columns and both socket checks; the body-fitted page served frames from the bundle at "
    "1.72 steps/s; and moving the rake slider on that page turned the Commit button into "
    "'Commit refused - this car has no spun-up field' while the header still read MARCHING THE "
    "CAR BEFORE THESE CHANGES and the status stayed `marching`.",
    "Also explored: the Linux interpreter (~/mm-py310, 3.10.21) had NO packages at all, so the "
    "Linux half needed an install, which the user approved.",
    "NOT measured before these predictions: whether the built bundle's lists are closed -- that "
    "every script, test, page and record it names exists and that nothing the body-fitted demo "
    "reads is missing from it.",
)

PREDICTION = (
    {"id": "R1", "claim": "the commit refusal costs no rebuild: `commit_check` names the car a "
                          "commit would build, and refuses, without constructing a composite",
     "why": "`geometry_fingerprint` is a function of the PARAMETERS, so the new car can be named "
            "from the knobs alone and whether it has a settled field is one `isfile`"},
    {"id": "R2", "claim": "the refusal tracks the geometry rather than latching: the nominal car "
                          "is committable, a moved geometry knob is refused, and moving it back "
                          "is committable again",
     "why": "it is a function of the knob state with no memory. **The first clause is the "
            "control**: a check that refused everything would pass a test that only moved a knob"},
    {"id": "R3", "claim": "the bundle's lists are CLOSED -- every script, test, wiki page and "
                          "record it names exists in this repository",
     "why": "the lists are hand-written and sixteen of each were added at once; a name that does "
            "not resolve fails the build, but only when someone runs it"},
    {"id": "R4", "claim": "the bundle carries everything the body-fitted demo READS from disk: "
                          "the two tier scripts it imports while building, Tier 64's record for "
                          "the machine sizing, and the settled field named for this car",
     "why": "these are not test fixtures -- they are what `build`, `_devices` and `_sizing` open. "
            "A bundle missing them ships a page that never draws, which is the failure this "
            "project has already shipped once (W251)"},
    {"id": "R5", "claim": "the built bundle contains no unexpected binary, no trace of the "
                          "unlicensed structural checkpoint, and no vendored scOT",
     "why": "the scans already existed; what is new is 24 MB of recorded runs and a 15 MB field, "
            "and BINARY_ALLOWED had to be widened to admit them -- which is exactly the kind of "
            "widening that lets something else through"},
    {"id": "R6", "claim": "both columns are reachable by one command each, on different ports",
     "why": "`python -m atlas.demo_racelab --column body-fitted` is new; two engines on one port "
            "would be one engine"},
)

OUT = NAME = None


def configure(out: str) -> None:
    global OUT, NAME
    OUT = os.path.abspath(out)
    NAME = os.path.basename(os.path.normpath(OUT))


def persist(obj) -> str:
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, NAME + ".json")
    tmp = p + ".tmp"

    def write():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(T60.clean(obj), fh, indent=1)

    T60._retry(write)
    T60._retry(lambda: os.replace(tmp, p))
    return p


def load() -> dict:
    p = os.path.join(OUT, NAME + ".json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def say(*a) -> None:
    print(*a, flush=True)


def stage_commit_check(res) -> dict:
    """R1 and R2: refused before the rebuild, and not latching."""
    t0 = time.perf_counter()
    col = BF.BodyFittedColumn()
    idle = col.commit_check(HERE)

    col.pending_regrid = ["rake"]
    nominal = col.commit_check(HERE)

    st = col.knob_state()
    st.set("rake", 0.10)
    moved = col.commit_check(HERE)

    st.set("rake", 0.0)
    back = col.commit_check(HERE)
    wall = time.perf_counter() - t0

    out = {
        "wall_s": wall,
        "built_a_composite": col.ov is not None,
        "idle": idle, "nominal": nominal, "moved": moved, "back": back,
        "names_a_different_car": (nominal.get("settled") != moved.get("settled")),
        "restores": bool(back.get("ok") and back.get("settled") == nominal.get("settled")),
    }
    res["commit_check"] = out
    say("  four checks in %.3f s, composite built: %s"
        % (wall, out["built_a_composite"]))
    say("    idle    %-18s %s" % (idle["kind"], idle["ok"]))
    say("    nominal %-18s %s  %s" % (nominal["kind"], nominal["ok"],
                                      nominal.get("settled")))
    say("    moved   %-18s %s  %s" % (moved["kind"], moved["ok"],
                                      moved.get("settled")))
    say("    back    %-18s %s  %s" % (back["kind"], back["ok"],
                                      back.get("settled")))
    persist(res)
    return out


def stage_bundle(res) -> dict:
    """R3, R4, R5, R6: the lists are closed and carry what the demo reads."""
    out = {}

    # R3 -- every name resolves
    missing = {"scripts": [s for s in B.SCRIPTS
                           if not os.path.isfile(os.path.join(HERE, "scripts", s))],
               "tests": [t for t in B.TESTS
                         if not os.path.isfile(os.path.join(HERE, "tests", t))],
               "artifacts": [a for a, _w in B.ARTIFACTS
                             if not os.path.isfile(os.path.join(HERE, *a.split("/")))]}
    have_pages = {f for _dp, _dn, fn in os.walk(os.path.join(HERE, "wiki")) for f in fn}
    missing["pages"] = [p for p in B.WIKI_PAGES if p not in have_pages]
    out["missing"] = missing
    out["closed"] = not any(missing.values())
    out["counts"] = {"scripts": len(B.SCRIPTS), "tests": len(B.TESTS),
                     "pages": len(B.WIKI_PAGES), "artifacts": len(B.ARTIFACTS)}

    # R4 -- what the body-fitted demo opens
    named_scripts = set(B.SCRIPTS)
    reads = {
        "tier62_car_solids.py": "tier62_car_solids.py" in named_scripts,
        "tier63_duct_openings.py": "tier63_duct_openings.py" in named_scripts,
        "out/racelab13/racelab13.json": any(
            a == "out/racelab13/racelab13.json" for a, _w in B.ARTIFACTS),
    }
    rel, name = B._bodyfitted_settled()
    reads["settled_field_present"] = os.path.isfile(os.path.join(HERE, *rel.split("/")))
    reads["settled_field_allowed"] = any(p.match(rel) for p in B.BINARY_ALLOWED)
    reads["settled_field"] = name
    out["body_fitted_reads"] = reads
    out["carries_what_it_reads"] = all(v for k, v in reads.items()
                                       if k != "settled_field")

    # R5 -- the scans, on the build this tier's record is about
    built = res.get("built_bundle")
    if built and os.path.isdir(built):
        out["scan_on"] = built
        nb = B.scan_unlicensed_checkpoint(built)
        data = B.scan_data(built)
        out["scan"] = {
            "unlicensed_files": nb["files_or_dirs_named_for_it"],
            "unlicensed_uses": nb["uses"],
            "prose_mentions": nb["prose_mentions_total"],
            "binaries_allowed": len(data["allowed"]),
            "binaries_unexpected": data["unexpected"],
            "scot_dirs": B.scan_scot(built),
        }
        out["clean"] = bool(not nb["files_or_dirs_named_for_it"] and not nb["uses"]
                            and not data["unexpected"] and not B.scan_scot(built))
    else:
        out["scan_on"] = None
        out["clean"] = None          # not measured, and not False

    # R6 -- one command per column
    from atlas.demo_racelab.__main__ import COLUMNS, PORTS
    out["columns"] = {"names": list(COLUMNS), "ports": dict(PORTS),
                      "distinct_ports": len(set(PORTS.values())) == len(PORTS)}

    res["bundle"] = out
    say("  lists closed: %s  %s" % (out["closed"], out["counts"]))
    if not out["closed"]:
        say("    MISSING: %s" % {k: v for k, v in missing.items() if v})
    say("  body-fitted reads carried: %s (%s)"
        % (out["carries_what_it_reads"], reads["settled_field"]))
    say("  scans clean: %s (on %s)" % (out["clean"], out["scan_on"]))
    say("  columns: %s on %s" % (out["columns"]["names"], out["columns"]["ports"]))
    persist(res)
    return out


def judge(res) -> dict:
    v = {}
    C, D = res.get("commit_check"), res.get("bundle")
    if C:
        v["R1"] = bool(not C["built_a_composite"] and C["wall_s"] < 5.0
                       and C["moved"]["kind"] == "no_spun_up_field"
                       and "W293" in C["moved"]["reason"])
        v["R2"] = bool(C["nominal"]["ok"] and not C["moved"]["ok"]
                       and C["names_a_different_car"] and C["restores"])
    if D:
        v["R3"] = bool(D["closed"])
        v["R4"] = bool(D["carries_what_it_reads"])
        if D.get("clean") is not None:
            v["R5"] = bool(D["clean"])
        v["R6"] = bool(D["columns"]["distinct_ports"]
                       and set(D["columns"]["names"]) == {"porous", "body-fitted"})
    return v


def stage_summary(res) -> dict:
    verdicts = judge(res)
    res["verdicts"] = verdicts
    res["verdicts_missing"] = sorted({p["id"] for p in PREDICTION} - set(verdicts))
    persist(res)
    for p in PREDICTION:
        say("%-3s %-6s %s" % (p["id"], verdicts.get(p["id"]), p["claim"][:86]))
    if res["verdicts_missing"]:
        say("NOT JUDGED (None, not False):", res["verdicts_missing"])
    say("%d of %d judged predictions held"
        % (sum(1 for x in verdicts.values() if x), len(verdicts)))
    return verdicts


STAGES = ("commit_check", "bundle", "summary")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--built-bundle", default=None,
                    help="a directory the bundle was built into, to scan (R5)")
    args = ap.parse_args(argv)
    configure(args.out)
    res = load()
    if "prediction" not in res:
        res.update(tier=74, prediction=list(PREDICTION),
                   prediction_recorded_at=dt.datetime.now().isoformat(timespec="seconds"),
                   read_before_this_run=list(READ_BEFORE_THIS_RUN),
                   machine_state=T60.machine_state())
        persist(res)
    elif [p["id"] for p in res["prediction"]] != [p["id"] for p in PREDICTION]:
        raise SystemExit("the record's prediction differs from this file's")
    if args.built_bundle:
        res["built_bundle"] = os.path.abspath(args.built_bundle)
        persist(res)
    for st in [s.strip() for s in args.stages.split(",") if s.strip()]:
        if st not in STAGES:
            raise SystemExit(f"unknown stage {st!r}; stages are {STAGES}")
        say("stage", st, "...")
        t0 = time.perf_counter()
        try:
            {"commit_check": stage_commit_check, "bundle": stage_bundle,
             "summary": stage_summary}[st](res)
        except Exception:
            err = load()
            err.setdefault("stage_errors", {})[st] = traceback.format_exc()[-3000:]
            persist(err)
            say(f"stage {st} FAILED")
            raise
        res.setdefault("stage_wall_s", {})[st] = time.perf_counter() - t0
        res.get("stage_errors", {}).pop(st, None)
        if res.get("stage_errors") == {}:
            res.pop("stage_errors", None)
        persist(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The website's claims pipeline: every number of Atlas's own that the site shows.

Reads the run records and writes ``site/data/claims.json``. Each claim carries its
value, how it is displayed, its units, the sentence it appears in, the record it is
read from (and the path inside that record), the record's public copy on the site,
the date, the machine's state, its caveat and the vault page that discusses it.

It also publishes the records the claims cite: each is copied to ``site/records/``
after its private details are removed (the computer's name, process ids, local
folder paths, the installer's scans), so that every number on the public site links
to a record a visitor can open. The test (``tests/test_site_claims.py``) re-reads
every record and fails if a claim's value differs from it, or if a number on the
site is in neither this file nor ``site/data/literature.json``.

No number on the site is typed by hand: ``scripts/site_build.py`` writes each one
into the pages from this file.

Run:  python scripts/site_claims.py
"""
from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
import re
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "site")
REC = "out/workbench/records"

TIMES = "×"
MINUS = "−"
EN = "–"

#: the machine every timed workbench claim was measured on, from the records' own
#: `machine` blocks (checked by the test: Windows 11, 22 logical processors, on AC)
LAPTOP = "the owner's Windows 11 laptop, 22 logical processors, on mains power, no other Python process"

#: keys removed from published copies, wherever they occur
DROP_KEYS = {"host", "pid", "log", "build_dir", "argv", "logs", "verifier_python", "venv_python",
             "clone", "bundle", "identity", "tail"}
DROP_PREFIXES = ("scan_",)
PRIVATE = [
    (re.compile(r"[A-Za-z]:\\+Users\\+[^\"\s]*", re.I), "<local path>"),
    (re.compile(r"/mnt/c/Users/[^\"\s]+", re.I), "<local path>"),
    (re.compile(r"/home/[^/\"\s]+(/[^\"\s]*)?"), "<local path>"),
    (re.compile(r"OneDrive[^\"\s]*"), "<local path>"),
    (re.compile(r"nauni", re.I), "<user>"),
    (re.compile(r"\bchamp\b"), "<host>"),
]


# -- formatting: the one place a number becomes text --------------------------------

def sup(n: int) -> str:
    return str(n).replace("-", MINUS)


def fmt(kind: str, v: Any, nd: int = 2) -> tuple[str, str]:
    """(text, html) for a value. The text is what a reader sees; the test compares it."""
    if kind == "ratio":
        t = f"{v:.{nd}f}{TIMES}"
        return t, t
    if kind == "pct":
        t = f"{100 * v:.{nd}f}%"
        return t, t
    if kind == "sci":
        if v == 0:
            return "0", "0"
        e = int(f"{v:.{nd}e}".split("e")[1])
        m = f"{v / 10 ** e:.{nd}f}"
        return f"{m}{TIMES}10{sup(e)}", f"{m}{TIMES}10<sup>{sup(e)}</sup>"
    if kind == "int":
        t = f"{int(v):,}"
        return t, t
    if kind == "sec":
        t = f"{v:.{nd}f} s"
        return t, t
    if kind == "ms":
        t = f"{v:.{nd}f} ms"
        return t, t
    if kind == "num":
        t = f"{v:.{nd}f}"
        return t, t
    if kind == "text":
        return str(v), str(v)
    raise ValueError(kind)


# -- records ---------------------------------------------------------------------------

def load(rel: str) -> Any:
    p = os.path.join(ROOT, rel)
    if rel.endswith(".jsonl"):
        with open(p, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def resolve(obj: Any, pointer: str) -> Any:
    """`a.b[0].c` style paths into a record."""
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", pointer):
        if part.startswith("["):
            obj = obj[int(part[1:-1])]
        else:
            obj = obj[part]
    return obj


def clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()
                if k not in DROP_KEYS and not k.startswith(DROP_PREFIXES)}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, str):
        s = obj
        for pat, rep in PRIVATE:
            s = pat.sub(rep, s)
        return s
    return obj


def public_path(rel: str) -> str:
    """Where a record's cleaned copy lives under site/."""
    rel = rel.replace("\\", "/")
    for a, b in (("out/workbench/records/", "records/workbench/"), ("out/learned-case/", "records/learned-case/"),
                 ("out/lean/", "records/lean/"), ("out/w346/", "records/w346/"), ("out/arch/", "records/arch/"),
                 ("out/site/", "records/site/")):
        if rel.startswith(a):
            return b + rel[len(a):]
    raise ValueError(rel)


#: False while the test rebuilds the claims in memory: nothing is written
WRITE = True


def published_copy(rel: str) -> dict:
    """The cleaned copy of a record, exactly as it is published."""
    with open(os.path.join(ROOT, rel), "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {"_published": {"source": rel, "source_sha256": sha,
                           "removed": "the computer's name, process ids, local folder paths, log excerpts and "
                                      "the installer's file scans; every other value is the record's own"},
            "record": clean(copy.deepcopy(load(rel)))}


def publish(rel: str) -> dict:
    wrapped = published_copy(rel)
    pub = public_path(rel).replace(".jsonl", ".json")
    if WRITE:
        dest = os.path.join(SITE, pub)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(wrapped, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
    return {"source": rel, "public": pub, "sha256": wrapped["_published"]["source_sha256"]}


# -- the claims ------------------------------------------------------------------------

CLAIMS: list[dict] = []
PUBLISHED: dict[str, dict] = {}


def site_config() -> dict:
    with open(os.path.join(SITE, "data", "site.json"), encoding="utf-8") as fh:
        return json.load(fh)


def cite(rel: str) -> str:
    if rel.startswith("lean/"):
        return "proofs/" + rel                       # the Lean sources, copied whole (site_build.py)
    if rel.startswith("atlas/workbench/bundle/"):      # the launchers' paperwork, in the public repository
        c = site_config()
        return f"{c['public_repo']}/blob/{c['source_branch']}/{rel}"
    if rel not in PUBLISHED:
        PUBLISHED[rel] = publish(rel)
    return PUBLISHED[rel]["public"]


def add(cid: str, value: Any, kind: str, *, sentence: str, source: str | list[str], pointer: str | None = None,
        nd: int = 2, units: str = "", date: str = "", machine: str = "", caveat: str = "", vault: str = "",
        derived: str = "") -> None:
    text, html = fmt(kind, value, nd)
    sources = source if isinstance(source, list) else [source]
    CLAIMS.append({
        "id": cid, "value": value, "kind": kind, "nd": nd, "display": text, "display_html": html,
        "units": units, "sentence": sentence, "source": sources, "pointer": pointer,
        "derived": derived, "public": [cite(s) for s in sources], "date": date, "machine": machine,
        "caveat": caveat, "vault": vault,
    })


FAST = {  # kind -> (example, confirmation record): the gallery's §8 table
    "farm": ("fast-farm", "confirm-fast-farm-20260930-232234.json", "wind farm"),
    "heat": ("fast-heat", "confirm-fast-heat-20260930-232435.json", "heat conduction"),
    "river": ("fast-river", "confirm-fast-river-20260930-233019.json", "river plume"),
    "sound": ("fast-sound", "confirm-fast-sound-20260930-232408.json", "sound"),
    "structure": ("plate-hole", "confirm-plate-hole-20260930-230850.json", "loaded structure"),
    "circuit": ("plate-circuit", "confirm-plate-circuit-20260930-230910.json", "current in a plate"),
    "bimetal": ("bimetal-arc", "confirm-bimetal-arc-20260930-230912.json", "heated structure"),
    "cooled": ("cooled-block", "confirm-cooled-block-20260930-230915.json", "cooled block"),
}
LIMITS = {
    "structure": "The whole is one direct solve, factored once; the pieces iterate until they agree. Substructuring could win, and it is not built.",
    "circuit": "The film and its circuit are one sparse solve, factored once; the pieces iterate until they agree.",
    "bimetal": "It is split by physics, not by space, so at most it can halve the time.",
    "cooled": "The block and its coolant are one system, factored once; the pieces iterate until they agree.",
}
MECH = {
    "farm": "parallel pieces, each small enough to stay in a core's cache",
    "heat": "multirate: the copper spreader takes its own short time steps",
    "river": "parallel pieces in cache, 24 steps between exchanges",
    "sound": "parallel pieces in cache, 16 steps between exchanges",
}


def fast_claims() -> None:
    met = []
    for k, (ex, fname, label) in FAST.items():
        rel = f"{REC}/fast/{fname}"
        rec = load(rel)
        run = rec["runs"][0]
        assert run["example"] == ex and rec["machine"]["power"]["ac"], fname
        arm = run["fastest_arm"]
        s = run["speedup"][arm]
        bar = run["bars"][0]
        when = rec["when"]
        add(f"fast.{k}.s", s, "ratio", nd=3 if s < 1 else 2, units=f"{TIMES} the full domain's speed",
            sentence=f"{label}: the pieces run at {s:.3g}{TIMES} the speed of the undivided solve",
            source=rel, pointer=f"runs[0].speedup.{arm}", date=when, machine=LAPTOP,
            caveat="a 2-D case on one laptop; the ratio belongs to this case and machine",
            vault="showcase-gallery §8")
        add(f"fast.{k}.cells", run["cells"], "int", units="grid cells", sentence=f"{label}: grid size",
            source=rel, pointer="runs[0].cells", date=when)
        if run.get("threads", 1) > 1:
            add(f"fast.{k}.threads", run["threads"], "int", units="threads", sentence=f"{label}: threads used",
                source=rel, pointer="runs[0].threads", date=when)
        add(f"fast.{k}.windows", run["windows"], "int", units="pieces", sentence=f"{label}: pieces",
            source=rel, pointer="runs[0].windows", date=when)
        agree = bar.get("agreement")
        if k == "farm":
            add("fast.farm.agree", agree, "pct", units="of farm power", sentence="farm power differs from the full domain's by",
                source=rel, pointer="runs[0].bars[0].agreement", date=when)
        elif k == "sound":
            add("fast.sound.agree", "bit for bit", "text", sentence="the pieces equal the full domain bit for bit",
                source=rel, date=when, derived="runs[0].checks[key=bitwise].passed is true")
        elif agree is not None:
            add(f"fast.{k}.agree", agree, "sci", nd=1, units="of the field's scale",
                sentence=f"{label}: largest difference from the full domain", source=rel,
                pointer="runs[0].bars[0].agreement", date=when)
        if bar["met"]:
            met.append((k, s))
    m = load(f"{REC}/fast/{FAST['farm'][1]}")["machine"]
    os_name = "Windows " + re.match(r"Windows-(\d+)", m["platform"]).group(1)
    add("machine.os", os_name, "text", sentence="the operating system of the laptop every timed workbench claim ran on",
        source=f"{REC}/fast/{FAST['farm'][1]}", derived="machine.platform, its first two fields")
    add("machine.cpus", m["cpu_count"], "int", units="logical processors", sentence="the laptop's logical processors",
        source=f"{REC}/fast/{FAST['farm'][1]}", pointer="machine.cpu_count")
    lo, hi = min(s for _, s in met), max(s for _, s in met)
    add("fast.met", len(met), "int", sentence="kinds of physics whose Fast example meets both bars",
        source=[f"{REC}/fast/{FAST[k][1]}" for k in FAST], derived="count of runs[0].bars[0].met", units="of 8")
    add("fast.kinds", len(FAST), "int", sentence="kinds of physics in the workbench, one Fast example each",
        source=[f"{REC}/fast/{FAST[k][1]}" for k in FAST], derived="count of distinct runs[0].family")
    add("fast.lo", lo, "ratio", machine=LAPTOP, sentence="the slowest of the four that meet the bar", source=f"{REC}/fast/{FAST['farm'][1]}",
        pointer="runs[0].speedup.parallel")
    add("fast.hi", hi, "ratio", machine=LAPTOP, sentence="the fastest of the four that meet the bar", source=f"{REC}/fast/{FAST['river'][1]}",
        pointer="runs[0].speedup.parallel")


ITERATED_STYLES = ("B", "C")


def gallery_claims() -> None:
    files = sorted(glob.glob(os.path.join(ROOT, REC, "step-f", "*.json")) +
                   glob.glob(os.path.join(ROOT, REC, "step-g", "*.json")) +
                   glob.glob(os.path.join(ROOT, REC, "step-j", "*", "[0-9]*.json")) +
                   glob.glob(os.path.join(ROOT, REC, "demo-fork", "river-fork-2*.json")))
    files = [os.path.relpath(f, ROOT).replace("\\", "/") for f in files if "compile" not in os.path.basename(f)]
    files += [f"{REC}/fast/{FAST[k][1]}" for k in FAST]
    worst, worst_src, n_bal, n_pass = 0.0, None, 0, 0
    for rel in files:
        rec = load(rel)
        runs = rec["runs"] if "runs" in rec else [rec]
        for run in runs:
            for c in run.get("checks", []):
                if c["kind"] == "balance" and c.get("passed") is not None:
                    n_bal += 1
                    n_pass += bool(c["passed"])
            if "runs" not in rec and run.get("style") in ITERATED_STYLES:
                for c in run.get("checks", []):
                    if c["key"] == "reference" and c.get("value") is not None and c["value"] > worst:
                        worst, worst_src = c["value"], rel
    add("agree.worst", worst, "sci", nd=1, units="of the field's scale",
        sentence="in every example whose pieces iterate to agreement, the assembled answer is within this of the undivided solve",
        source=worst_src, pointer="checks[1].value", date=load(worst_src)["started"], machine=LAPTOP,
        caveat="against the same discretization solved whole; the scale is the temperature span or the largest displacement",
        vault="showcase-gallery §2")
    add("balance.total", n_bal, "int", sentence="balance checks (mass, energy, charge, force) measured across the gallery and the Fast examples",
        source=files, derived="count of checks of kind 'balance' with a verdict")
    add("balance.passed", n_pass, "int", sentence="of them passed", source=files,
        derived="count of those checks that passed")


def schannel_claims() -> None:
    rel = "out/site/s-channel.json"
    d = load(rel)
    add("schan.windows", len(d["windows"]), "int", sentence="the S-channel's windows, generated from its own shape",
        source=rel, derived="len(windows)")
    add("schan.seams", len(d["seams"]), "int", sentence="the seams where they overlap", source=rel, derived="len(seams)")
    add("schan.agree", d["agreement"]["relative_to_span"], "sci", nd=1, units="of the 100 K span",
        sentence="the assembled field differs from the undivided solve by at most", source=rel,
        pointer="agreement.relative_to_span", caveat="steady conduction, windows iterated to 1e-10, same grid")
    run = f"{REC}/step-j/s-channel/20260929-213316.json"
    add("schan.agree.record", load(run)["checks"][1]["value"], "sci", nd=1,
        sentence="the same agreement in the workbench's own run record", source=run, pointer="checks[1].value",
        date="2026-09-29")


def w346_claims() -> None:
    rel = "out/w346/w346.json"
    d = load(rel)
    vals, power, rotors = [], [], []
    for k, r in d["rungs"].items():
        if 5 <= r["n_rotors"] <= 21:
            rotors.append(r["n_rotors"])
            vals.append(max(1 / v for a, v in r["cost"]["ratio"].items() if a.startswith("Ep")))
            vals.append(max(1 / v for a, v in r["cost_replicate"]["ratio_min"].items() if a.startswith("Ep")))
            power.append(abs(r["accuracy"]["E"]["power_rel_diff"]))
    add("w346.lo", min(vals), "ratio", nd=1, sentence="decomposition's speed over the undivided solve, 5 to 21 rotors, lowest",
        source=rel, derived="min over rungs of 5-21 rotors and two timing draws of the best thread count",
        machine=LAPTOP.replace(", no other Python process", ""), vault="decomposition-speed-by-rotor-count")
    add("w346.hi", max(vals), "ratio", nd=1, sentence="the same, highest", source=rel, derived="max of the same")
    add("w346.plo", min(power), "pct", nd=1, sentence="farm power differs from the undivided solve's by, at least", source=rel,
        derived="min |accuracy.E.power_rel_diff| over the same rungs")
    add("w346.phi", max(power), "pct", nd=1, sentence="and at most", source=rel, derived="max of the same")
    add("w346.rmin", min(rotors), "int", sentence="the smallest farm in the range, in rotors", source=rel,
        derived="min rungs[].n_rotors at or above the first rung where decomposition wins")
    add("w346.rmax", max(rotors), "int", sentence="the largest farm in the range, in rotors", source=rel,
        derived="max rungs[].n_rotors with an accuracy record")
    # every rung, for the chart of speed against farm size: the best thread count of each timing draw
    for k, r in d["rungs"].items():
        n = r["n_rotors"]
        add(f"w346.r{n}.a", max(1 / v for a, v in r["cost"]["ratio"].items() if a.startswith("Ep")), "ratio", nd=1,
            sentence=f"{n} rotor{'s' * (n != 1)}: decomposition's speed over the undivided solve, first timing",
            source=rel, derived=f"max over the Ep arms of 1 / rungs.{k}.cost.ratio",
            machine=LAPTOP.replace(", no other Python process", ""))
        rep = (r.get("cost_replicate") or {}).get("ratio_min") or {}
        if any(a.startswith("Ep") for a in rep):
            add(f"w346.r{n}.b", max(1 / v for a, v in rep.items() if a.startswith("Ep")), "ratio", nd=1,
                sentence=f"{n} rotors: the same, repeat timing", source=rel,
                derived=f"max over the Ep arms of 1 / rungs.{k}.cost_replicate.ratio_min",
                machine=LAPTOP.replace(", no other Python process", ""))
        add(f"w346.r{n}.n", n, "int", sentence=f"rotors in the farm at rung {k}", source=rel,
            derived=f"rungs.{k}.n_rotors")


def learned_claims() -> None:
    rel = "out/learned-case/gate-20261001-154054.json"
    g = load(rel)
    v = g["verdict"]
    add("lc.ep_over_l", v["G1"]["Ep_over_L"], "ratio", sentence="the learned pieces against the classical pieces, speed",
        source=rel, pointer="verdict.G1.Ep_over_L", date=g["at"], machine=LAPTOP, vault="showcase-gallery §9")
    add("lc.f_over_l", v["G1"]["F_over_L"], "ratio", sentence="the learned pieces against the full domain, speed",
        source=rel, pointer="verdict.G1.F_over_L", date=g["at"], machine=LAPTOP)
    for arm in ("F", "Ep", "L", "Cc"):
        add(f"lc.{arm}.t", g["means_s"][arm], "sec", nd=3 if g["means_s"][arm] < 1 else 2,
            sentence=f"arm {arm}: mean time per macro-step", source=rel, pointer=f"means_s.{arm}", date=g["at"])
        add(f"lc.{arm}.P", g["arms"][arm]["errors"]["12"]["P"], "pct", sentence=f"arm {arm}: farm power vs the truth at macro-step 12",
            source=rel, pointer=f"arms.{arm}.errors.12.P", date=g["at"])
        add(f"lc.{arm}.V", g["arms"][arm]["errors"]["12"]["V"], "pct", sentence=f"arm {arm}: velocity vs the truth at macro-step 12",
            source=rel, pointer=f"arms.{arm}.errors.12.V", date=g["at"])
    passed = sum(1 for k in ("G1", "G2", "G3", "G4", "G5", "G6") if v[k]["passed"])
    add("lc.passed", passed, "int", sentence="bars of the registered gate passed", source=rel, derived="count of verdict.G*.passed", units="of 6")
    add("lc.steps", g["steps"], "int", sentence="macro-steps evaluated", source=rel, pointer="steps")
    add("lc.h", v["G2"]["comparisons"][0]["h"], "int", sentence="the macro-step at which the errors shown are read",
        source=rel, pointer="verdict.G2.comparisons[0].h")
    add("lc.g3", v["G3"]["max"], "sci", nd=1, sentence="the largest divergence (mass conservation) of the learned arm", source=rel,
        pointer="verdict.G3.max")
    tl = "out/learned-case/training-log.json"
    t = load(tl)
    add("lc.params", t["parameters"], "int", sentence="the network's parameters", source=tl, pointer="parameters")
    add("lc.layouts.train", len(t["train_seeds"]), "int", sentence="random farm layouts the network was trained on",
        source=tl, derived="len(train_seeds)")


def lean_claims() -> None:
    ax = "out/lean/axioms.json"
    a = load(ax)
    add("lean.decls", a["declarations"], "int", sentence="declarations whose axioms the Lean kernel listed", source=ax,
        pointer="declarations", date=a["when"], vault="formal-proofs-record §7.12")
    add("lean.failing", a["failing"], "int", sentence="of them failing", source=ax, pointer="failing")
    add("lean.sorry", a["with_sorry"], "int", sentence="proofs left as sorry", source=ax, pointer="with_sorry")
    axioms = sorted({x for r in a["rows"] for x in r["axioms"]})
    add("lean.axioms", len(axioms), "int", sentence="axioms used, all standard: " + ", ".join(axioms), source=ax,
        derived="distinct rows[].axioms")
    bl = "out/lean/builds.jsonl"
    b = [r for r in load(bl) if r.get("label") == "verify-integration-clean"][-1]
    add("lean.build", b["seconds"], "sec", nd=0, sentence="a clean re-check of the whole project", source=bl,
        derived="the verify-integration-clean row's seconds", date=b["when"],
        machine="the owner's laptop, on battery", caveat="on battery; on mains it is roughly half")
    files = sorted(glob.glob(os.path.join(ROOT, "lean", "AtlasProofs", "*.lean")))
    n_thm, n_lines = 0, 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            txt = fh.read()
        n_lines += txt.count("\n")
        n_thm += len(re.findall(r"^\s*(?:private\s+)?(?:theorem|lemma)\s", txt, flags=re.M))
    add("lean.files", len(files), "int", sentence="Lean files", source="lean/AtlasProofs", derived="count of *.lean")
    add("lean.theorems", n_thm, "int", sentence="theorem and lemma declarations in the source", source="lean/AtlasProofs",
        derived="regex count of theorem/lemma at line start")
    add("lean.lines", n_lines, "int", sentence="lines of Lean", source="lean/AtlasProofs", derived="newline count")
    # (the 105 approved theorems are counted in a vault page, which is not public, so the
    # site shows the counts a visitor can re-derive from the published sources instead)
    t3 = "out/lean/t3_counterexample.json"
    load(t3)
    cite(t3)


def arch_claims() -> None:
    bt = "out/arch/backbone_timing.json"
    rows = load(bt)["rows"]
    r = [x for x in rows if x["size"] == "medium" and x["n"] == 64 and x["batch"] == 1][0]
    add("arch.ms", r["ms_per_piece"], "ms", nd=1, sentence="forward cost of the proposed (untrained) network per 64x64 piece",
        source=bt, derived="rows[size=medium, n=64, batch=1].ms_per_piece",
        machine="4 threads of a cloud container's CPU", caveat="an untrained network: a cost, not a result")
    add("arch.params", r["params"] / 1e6, "num", nd=2, units="million parameters", sentence="the trunk's size", source=bt,
        derived="rows[size=medium].params / 1e6")
    sg = "out/arch/superelement_gram.json"
    res = load(sg)["results"][0]
    row = [x for x in res["rows"] if abs(x["eps"] - 0.1) < 1e-12][0]
    add("arch.eps", 0.1, "pct", nd=0, sentence="the error put into every mode by hand", source=sg, derived="results[0].rows[eps=0.1].eps")
    add("arch.gram_err", row["gram_port_rel_err_mean"], "pct", nd=1, sentence="the port matrix's error when formed as the energy Gram",
        source=sg, derived="results[0].rows[eps=0.1].gram_port_rel_err_mean",
        caveat="synthetic perturbations of exact modes, not a trained network")
    add("arch.direct_err", row["direct_port_rel_err_mean"], "pct", nd=0, sentence="the error when the matrix is predicted directly",
        source=sg, derived="results[0].rows[eps=0.1].direct_port_rel_err_mean")
    add("arch.indef", row["direct_pieces_indefinite"], "int", sentence="pieces that turn indefinite when predicted directly",
        source=sg, derived="results[0].rows[eps=0.1].direct_pieces_indefinite")
    add("arch.pieces", res["pieces"][0] * res["pieces"][1], "int", sentence="pieces in the test", source=sg, derived="results[0].pieces product")


def installer_claims() -> None:
    w = f"{REC}/installer/verify_windows-amd64_release4.json"
    lx = f"{REC}/installer/verify_linux-x86_64_release4.json"
    rw, rl = load(w), load(lx)
    assert rw["launch_check"]["exit_code"] == 0
    add("inst.win.s", rw["launch_check"]["wall_s"], "sec", nd=0,
        sentence="a fresh clone's install and self-test on Windows", source=w, pointer="launch_check.wall_s", date=rw["at"],
        machine=LAPTOP, caveat="pip's download cache was already warm, so a first download takes longer; measured on the "
                "demo's first install branch, which carried the workbench alone")
    add("inst.lin.s", rl["launch_check"]["wall_s"], "sec", nd=0, sentence="the same on Linux (WSL on the same laptop)",
        source=lx, pointer="launch_check.wall_s", date=rl["at"],
        caveat="exit code 3: everything ran, and the optional Gmsh would not load (libGLU)")
    add("inst.examples", len(rw["self_test_examples"]), "int", sentence="kinds the self-test runs", source=w,
        derived="len(self_test_examples)")
    readme = "atlas/workbench/bundle/README.md"
    with open(os.path.join(ROOT, readme), encoding="utf-8") as fh:
        txt = fh.read()
    py = re.search(r"\*\*Python (3\.\d+, 3\.\d+ or 3\.\d+)\.\*\*", txt).group(1)
    add("inst.python", "Python " + py, "text", sentence="the Pythons the launcher accepts", source=readme,
        derived="the README's Requirements, first item")
    disk = re.search(r"`\.venv` took ([\d.]+ GB) on Windows", txt).group(1)
    add("inst.disk", disk, "text", sentence="the private environment's size on Windows with every optional part",
        source=readme, derived="the README's Requirements, 'Disk'", caveat="measured by the demo's chat on the owner's laptop")


def build() -> dict:
    CLAIMS.clear()
    PUBLISHED.clear()
    fast_claims()
    gallery_claims()
    schannel_claims()
    w346_claims()
    learned_claims()
    lean_claims()
    arch_claims()
    installer_claims()
    ids = [c["id"] for c in CLAIMS]
    assert len(ids) == len(set(ids)), "duplicate claim ids"
    return {"schema": "atlas-site/claims@1", "made_by": "scripts/site_claims.py",
            "claims": CLAIMS, "records": sorted(PUBLISHED.values(), key=lambda r: r["public"])}


def main() -> int:
    data = build()
    os.makedirs(os.path.join(SITE, "data"), exist_ok=True)
    with open(os.path.join(SITE, "data", "claims.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"{len(data['claims'])} claims, {len(data['records'])} records published to site/records/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

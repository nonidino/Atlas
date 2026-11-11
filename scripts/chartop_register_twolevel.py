"""The registration after the Stage 0 rework (`atlas.chartop.gates_twolevel`; the rework page's
section 18): what it reads with the classical solver and the generator only, and its pin.

Accepted by the owner as drafted on 2026-11-10 ("Accept as drafted (Recommended)").  No
learned expert reads anything here.

    python scripts/chartop_register_twolevel.py              # the classical record
    python scripts/chartop_register_twolevel.py --pin        # only with the owner's approval

**The classical record** (``out/chartop/registration-twolevel.json``):
- the held-out benchmark problems (`benchmarks`' ``variant="held-out"``), built and measured with
  the classical superelement and the full-domain solve only: per problem and frame the 16-mode
  floor, its maximum error, the refusals and a fingerprint (`chartop_register_prior.measure`);
- the fresh test split, seeds 6,000,000 to 6,002,399, drawn by the generator and never solved:
  its content sha256 (`test_split_sha256`: each example's stored arrays, `train.KEYS` in order,
  in seed order), so that the one evaluation can show it reads exactly these pieces.  Floating
  point differs between platforms in the last bit, so the hash is the laptop's, where the
  evaluation's accuracy runs.

**The pin** (``--pin``, with the owner's approval, after the price protocol's dry run passed):
``out/chartop/registered-twolevel.txt`` holds the sha256 (line endings normalised) of the scope
(`PINNED`), of what earlier registrations pinned (`ALREADY_PINNED`, each agreeing with every pin
file that lists it), of what is recorded beside (`RECORDED_BESIDE`, the passing dry run among
them), and by value the arms, the gates, the problems and the test split's hash.  It refuses to
overwrite a registration, and to pin before the dry run passed on the pricing CPU with the code
pinned here.  `check_pin` is what the one evaluation runs first.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

import chartop_evaluate as ev0  # noqa: E402
import chartop_register_prior as RP  # noqa: E402
import chartop_registry as registry  # noqa: E402
from atlas.chartop import gates_squares as GS, gates_twolevel as GT  # noqa: E402

RECORD = os.path.join(ROOT, "out", "chartop", "registration-twolevel.json")
REGISTRATION = os.path.join(ROOT, "out", "chartop", "registered-twolevel.txt")
#: the passing dry run (`chartop_price_twolevel.py --dry-run`, 2026-11-10, a rented whole machine
#: with the pricing CPU, commit 345055c): six runs, every measurement counted, the A/A check within
#: 0.004; its box's logs beside it in ``out/chartop/dryrun-twolevel-20261110-201853-box/``
DRY_RUN = "out/chartop/dryrun-twolevel-20261110-201853.json"

#: the scope: what TL is, how it trains, what feeds it, how it is priced
PINNED = {"gate module": "atlas/chartop/gates_twolevel.py",
          "the coarse level": "atlas/chartop/coarse.py",
          "its torch path": "atlas/chartop/coarse_patches.py",
          "the expert": "atlas/chartop/twolevel.py",
          "training": "atlas/chartop/train.py",
          "training's runner": "scripts/chartop_train.py",
          "the data's flags": "scripts/chartop_data.py",
          "price protocol": "scripts/chartop_price_twolevel.py",
          "price payload": "scripts/chartop_price_twolevel_payload.py",
          "price box runner": "scripts/chartop_price_twolevel_box_run.sh",
          "training payload": "scripts/chartop_box_payload.py",
          "training box": "scripts/chartop_box_run.sh"}
#: what a price run executes (`chartop_price_twolevel.py`'s imports, the expert it builds and the
#: loader of S's weights): the pin compares these, and only these, with the passing dry run's
#: commit.  The training side (the data's flags and light labels, the training runner and box)
#: is pinned by hash but no price run reads it.
PRICE_CODE = ("gate module", "the coarse level", "its torch path", "the expert", "training",
              "price protocol", "price payload", "price box runner")
#: pinned already by earlier registrations; each hash must agree with every pin file that lists it
ALREADY_PINNED = {"reads the bars of": "atlas/chartop/gates_forms.py",
                  "and through it": "atlas/chartop/gates.py",
                  "the held-out problems": "atlas/chartop/gates_prior.py",
                  "R2's protocol": "atlas/chartop/gates_squares.py",
                  "the price function": "atlas/chartop/gates_hybrid.py",
                  "layout helpers": "atlas/chartop/benchmarks.py",
                  "R2's problems": "atlas/chartop/benchmarks_squares.py",
                  "dry-run problems": "atlas/chartop/benchmarks_workbench.py",
                  "S's form": "atlas/chartop/forms.py",
                  "the host": "atlas/chartop/host.py",
                  "H2r's expert": "atlas/chartop/reuse.py",
                  "R2's price runs": "scripts/chartop_price_squares.py"}
#: recorded beside, by hash: what the arms are built with that no registration owns
RECORDED_BESIDE = {"the adapter": "atlas/chartop/expert.py",
                   "the generator": "atlas/chartop/data.py",
                   "U's builder": "scripts/chartop_evaluate_hybrid.py",
                   "the stand-in": "scripts/chartop_forms_diagnose.py",
                   "utilities": "scripts/chartop_evaluate.py"}
BEFORE = "any network of TL was trained, and before any expert read the test split"


def file_hash(path: str) -> str:
    return RP.file_hash(path)


# ---------------------------------------------------------------------------
# the classical record
# ---------------------------------------------------------------------------


def example_digest(seed: int) -> tuple:
    """``(n, sha256 of the example's stored arrays)``: `data.arrays` of the generator's draw of
    ``seed``, the keys of `train.KEYS` in order, each array's dtype, shape and bytes."""
    from atlas.chartop import data, train
    ex = data.draw(int(seed))
    a = data.arrays(ex)
    h = hashlib.sha256()
    for k in train.KEYS:
        v = np.ascontiguousarray(np.asarray(a[k]))
        h.update(f"{k}|{v.dtype.str}|{v.shape}|".encode())
        h.update(v.tobytes())
    return ex.n, h.hexdigest()


def test_split_sha256(seeds, workers: int = 1) -> dict:
    """The test split's content sha256: each example's digest (`example_digest`) in seed order,
    hashed together; with the resolutions' counts."""
    seeds = [int(s) for s in seeds]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            got = pool.map(example_digest, seeds, chunksize=8)
    else:
        got = [example_digest(s) for s in seeds]
    h = hashlib.sha256()
    counts = {}
    for s, (n, d) in zip(seeds, got):
        h.update(f"{s}:{n}:{d}\n".encode())
        counts[n] = counts.get(n, 0) + 1
    return {"seeds": [seeds[0], seeds[-1]] if seeds else [], "count": len(seeds),
            "by_n": {str(k): v for k, v in sorted(counts.items())}, "sha256": h.hexdigest()}


def classical_record(workers: int) -> dict:
    t0 = time.perf_counter()
    rec = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "experts_run": ["classical"],
           "registered": GT.REGISTERED, "held_out": [RP.measure(name, problem, GT.HELD_OUT_VARIANT)
                                                     for name, problem in GT.HELD_OUT]}
    lo, hi = GT.TEST_SEEDS
    rec["test_split"] = test_split_sha256(range(lo, hi), workers)
    import chartop_data
    rec["generator_sha256"] = chartop_data.source_hash()
    rec["platform"] = {"system": sys.platform, "numpy": np.__version__}
    rec["elapsed_s"] = time.perf_counter() - t0
    return rec


# ---------------------------------------------------------------------------
# the pin
# ---------------------------------------------------------------------------


def every_path() -> list:
    beside = list(RECORDED_BESIDE.values()) + ([DRY_RUN] if DRY_RUN else [])
    return [*PINNED.values(), "out/chartop/registration-twolevel.json", *ALREADY_PINNED.values(), *beside]


def check_pin(path: str = REGISTRATION, root: str = ROOT, paths=None) -> dict:
    """{file: sha256} for every file the pin records, or SystemExit if one is missing or changed.
    The one evaluation runs this first."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: the registration is not pinned")
    text = open(path, encoding="ascii").read()
    out = {}
    for rel in (paths or every_path()):
        m = re.search(re.escape(rel) + r"\s+sha256 ([0-9a-f]{64})", text)
        if not m:
            raise SystemExit(f"{path} does not pin {rel}")
        h = file_hash(os.path.join(root, *rel.split("/")))
        if h != m.group(1):
            raise SystemExit(f"{rel} is not the registered file: sha256 {h[:16]}... against {m.group(1)[:16]}...")
        out[rel] = h
    return out


def same_as_run(rel: str, commit: str) -> bool:
    """Whether ``rel`` is the file ``commit`` held, line endings normalised (the gate module
    apart from its docstring and `REGISTERED`, which record the pin)."""
    import chartop_register_squares as RS
    then = subprocess.run(["git", "-C", ROOT, "-c", "core.autocrlf=false", "show", f"{commit}:{rel}"],
                          capture_output=True).stdout.replace(b"\r\n", b"\n")
    now = open(os.path.join(ROOT, *rel.split("/")), "rb").read().replace(b"\r\n", b"\n")
    gate = rel == PINNED["gate module"]
    return bool(then) and RS._code_of(then, gate) == RS._code_of(now, gate)


def refusals_before_pin() -> list:
    """Why the scope may not be pinned; empty when it may."""
    why = []
    if not GT.REGISTERED or "accepted by the owner" not in GT.REGISTERED:
        why.append("the owner has not accepted the registration")
    if not os.path.exists(RECORD):
        why.append("no classical record: run this script without --pin first")
    else:
        rec = json.load(open(RECORD, encoding="utf-8"))
        if rec.get("experts_run") != ["classical"] or len(rec.get("held_out", [])) != len(GT.HELD_OUT):
            why.append("the classical record ran another expert, or misses a held-out problem")
        if rec.get("test_split", {}).get("count") != GT.N_TEST:
            why.append("the classical record does not hash the whole test split")
    pins = {f: open(os.path.join(ROOT, "out", "chartop", f), encoding="utf-8").read()
            for f in sorted(os.listdir(os.path.join(ROOT, "out", "chartop"))) if re.fullmatch(r"registered.*\.txt", f)}
    for rel in ALREADY_PINNED.values():
        found = [re.search(re.escape(rel) + r"\s+sha256 ([0-9a-f]{64})", t) for t in pins.values()]
        found = [m.group(1) for m in found if m]
        if not found or any(h != file_hash(os.path.join(ROOT, rel)) for h in found):
            why.append(f"{rel} is not as the earlier registrations pinned it")
    if DRY_RUN is None:
        why.append("no passing dry run of the price protocol is recorded (DRY_RUN)")
    else:
        run = json.load(open(os.path.join(ROOT, *DRY_RUN.split("/")), encoding="utf-8"))
        if not (run["passes"] is True and run["on_price_cpu"] and run["machine"]["cpu"] == GS.PRICE_CPU):
            why.append("the dry run did not pass on the pricing CPU")
        code = str(run.get("code") or "")
        ran = [PINNED[k] for k in PRICE_CODE] + [*ALREADY_PINNED.values(), *RECORDED_BESIDE.values()]
        for rel in ran:
            if not re.fullmatch(r"[0-9a-f]{40}", code) or not same_as_run(rel, code):
                why.append(f"{rel} is not the file the passing dry run ran ({code[:7]})")
    return why


def write_pin(path: str, before: str = BEFORE) -> None:
    """The registration's text (no refusals checked: `pin` checks them)."""
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; a registration is not overwritten")
    rec = json.load(open(RECORD, encoding="utf-8")) if os.path.exists(RECORD) else {}
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    h = lambda rel: file_hash(os.path.join(ROOT, *rel.split("/")))  # noqa: E731
    ts = rec.get("test_split", {})
    lines = ["THE CHART OPERATOR, THE REGISTRATION AFTER THE STAGE 0 REWORK (CO-D, THE TWO-LEVEL EXPERT TL)",
             "",
             "written            %s" % dt.datetime.now().astimezone().isoformat(timespec="seconds"),
             "before             %s" % before,
             "repository at      %s (the commit adding this file follows it)" % head]
    lines += ["%-18s %s  sha256 %s" % (label, rel, h(rel)) for label, rel in PINNED.items()]
    lines.append("classical record   out/chartop/registration-twolevel.json  sha256 %s"
                 % h("out/chartop/registration-twolevel.json"))
    lines += ["%-18s %s  sha256 %s  (as the earlier registrations pinned it)" % (label, rel, h(rel))
              for label, rel in ALREADY_PINNED.items()]
    beside = dict(RECORDED_BESIDE, **({"the dry run": DRY_RUN} if DRY_RUN else {}))
    lines += ["%-18s %s  sha256 %s" % (label, rel, h(rel)) for label, rel in beside.items()]
    lines += ["registered as      %s" % GT.REGISTERED,
              "S's weights        %s (TL's start, sha256 as downloaded)" % GT.INIT_SHA256,
              "the test split     seeds %s to %s, %s pieces (%s), content sha256 %s" % (
                  f"{GT.TEST_SEEDS[0]:,}", f"{GT.TEST_SEEDS[1] - 1:,}", ts.get("count"), ts.get("by_n"), ts.get("sha256")),
              "",
              GT.text(),
              ""]
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(lines))


def pin(before: str = BEFORE, path: str = REGISTRATION) -> int:
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; a registration is not overwritten")
    why = refusals_before_pin()
    if why:
        raise SystemExit("not pinning: " + "; ".join(why[:5]))
    write_pin(path, before)
    print("registered ->", os.path.relpath(path, ROOT))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pin", action="store_true",
                    help="write out/chartop/registered-twolevel.txt (only with the owner's approval)")
    import chartop_cpus
    ap.add_argument("--workers", type=int, default=chartop_cpus.default_workers(2),
                    help="the test split's drawing pool (the CPUs this process may use, less 2)")
    args = ap.parse_args(argv)
    if args.pin:
        return pin()
    rec = classical_record(args.workers)
    registry.write_atomic(RECORD, json.dumps(ev0.strip(rec), indent=1).encode("utf-8"))
    print("TWO-LEVEL  held-out problem        floor E% registered / lowered   refusals")
    for r in rec["held_out"]:
        f = r["frames"]
        print(f"           {r['name']:12s} {r['problem']:6s}   {100 * f['registered']['truncation_energy_rel']:8.4f} / "
              f"{100 * f['lowered']['truncation_energy_rel']:8.4f}   {sum(len(f[k]['refusals']) for k in f)}")
    ts = rec["test_split"]
    print(f"           the test split: {ts['count']} pieces {ts['by_n']}, sha256 {ts['sha256'][:16]}...; "
          f"wrote {os.path.relpath(RECORD, ROOT)} ({rec['elapsed_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

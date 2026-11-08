"""R2's one evaluation (`scripts/chartop_evaluate_squares.py`), tested without reading a
registered layout: every test that runs H2, H2r, U or P-s0 runs them on seeds the registration
never draws (from `TEST_SEED`) or on section 11.17's problems, which have been read; the
registered layouts are only refused, or judged from numbers made up here."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import chartop_evaluate_squares as E  # noqa: E402
import chartop_register_squares as RS  # noqa: E402
from atlas.chartop import gates_squares as GS  # noqa: E402

TEST_SEED = 9_000_000
P_S0 = os.path.join(ROOT, "out", "chartop", "runs", "prior", "P-s0", "best.pt")
needs_p_s0 = pytest.mark.skipif(not os.path.exists(P_S0), reason="P-s0's weights are not on this machine")


@pytest.fixture(scope="module")
def experts():
    if not os.path.exists(P_S0):
        pytest.skip("P-s0's weights are not on this machine")
    import torch
    torch.set_num_threads(GS.PRICE_THREADS)
    return E.make_experts()


def _args(*argv):
    return E.build_parser().parse_args(list(argv))


def test_the_registered_defaults():
    args = _args()
    assert E.specs_of(args) == E.REGISTERED_SPECS and len(E.REGISTERED_SPECS) == 16 and not E.test_flags(args)
    assert E.REGISTERED_SPECS[0] == ("squares", "7000000", "steady")
    assert {int(s[1]) for s in E.REGISTERED_SPECS} == set(GS.SEEDS)
    assert len(E.CONTROL) == 12 and ("heated-strip", "step") in E.CONTROL
    assert os.path.exists(os.path.join(ROOT, *E.CONTROL_RECORD.split("/")))
    assert E.ROUND_FEATURES == ("hole", "round-inclusion") and args.registration == RS.REGISTRATION


def test_no_test_flag_touches_a_registered_layout(tmp_path):
    for argv in (["--n", "16"], ["--reps", "1"], ["--any-cpu"], ["--control", "wall-2:steady"],
                 ["--problems", "squares:7000003:step"], ["--problems", f"squares:{TEST_SEED}:steady", "squares:7000000:step"]):
        args = _args(*argv)
        with pytest.raises(SystemExit, match="read only by the registered run"):
            E.refuse_tests_on_registered(args, E.specs_of(args))
    E.refuse_tests_on_registered(_args(), E.specs_of(_args()))            # the registered run itself
    args = _args("--problems", f"squares:{TEST_SEED}:steady", "--n", "16")
    E.refuse_tests_on_registered(args, E.specs_of(args))
    for argv in (["--n", "16"], ["--price", "--stamp", "x", "--any-cpu"]):  # refused before anything is built
        with pytest.raises(SystemExit, match="read only by the registered run"):
            E.main([*argv, "--out", str(tmp_path)])
    assert os.listdir(tmp_path) == []


def test_the_one_evaluation_runs_once(tmp_path):
    stamp, d = E.begin(str(tmp_path), resume=False)
    assert os.path.isdir(d) and d.endswith(f"{E.STAMP_PREFIX}{stamp}-checkpoints")
    with pytest.raises(SystemExit, match="--resume"):
        E.begin(str(tmp_path), resume=False)
    assert E.begin(str(tmp_path), resume=True) == (stamp, d)
    open(E.path_of(str(tmp_path), stamp, ".json"), "w").write("{}")
    with pytest.raises(SystemExit, match="has run"):
        E.begin(str(tmp_path), resume=True)
    with pytest.raises(SystemExit, match="nothing to resume"):
        E.begin(str(tmp_path / "empty"), resume=True)


@needs_p_s0
def test_the_control_reproduces_section_11_17_bit_for_bit(experts):
    """wall-2, where R2 hands the classical solve every piece, and heated-strip, where it hands U
    every piece: the floor, U and P-s0 as section 11.17 recorded them, and H2 as U or the floor."""
    ctl = E.control(experts, [("wall-2", "steady"), ("heated-strip", "step")])
    assert ctl["wall-2 steady"]["classical pieces"] == ctl["wall-2 steady"]["pieces"]
    assert ctl["heated-strip step"]["classical pieces"] == 0
    assert all(v["equals section 11.17's"] and v["H2 equals U or the floor"] for v in ctl.values())


@needs_p_s0
def test_the_accuracy_on_a_layout_never_drawn(experts):
    spec = ("squares", str(TEST_SEED + 3), "steady")                   # a hole: ring pieces to the classical solve
    res = E.accuracy(spec, experts, 16)
    assert res["H2 answers U's bits"] and res["picks"] and res["pieces"] > len(res["picks"])
    assert set(res["verdicts"]) == {"H2", "H2r", "U", "P-s0"}
    for f in E.FRAMES:
        fr = res["frames"][f]
        assert set(fr["arms"]) == {"H2", "H2r", "U", "P-s0"}
        e_r, floor = fr["arms"]["H2r"]["energy_rel"], fr["floor"]
        assert abs(e_r - floor) <= 1e-10 + 1e-8 * floor                 # reuse is the classical answer


def test_a_pick_that_differs_is_refused(monkeypatch):
    spec = ("squares", str(TEST_SEED), "steady")
    b = E.PS.build_problem(spec, 16)
    monkeypatch.setattr(E, "expected_picks", lambda s: [999])
    with pytest.raises(SystemExit, match="rule R2 picks"):
        E.picks_of(spec, b)


def test_the_price_stage_refuses_off_the_pricing_cpu(tmp_path):
    if E.GS.on_price_cpu(E.PS.machine()):
        pytest.skip("this machine is the pricing CPU")
    with pytest.raises(SystemExit, match="measured on"):
        E.main(["--price", "--stamp", "s", "--problems", f"squares:{TEST_SEED}:steady", "--n", "16", "--reps", "1",
                "--out", str(tmp_path)])


@needs_p_s0
def test_the_whole_run_on_problems_never_drawn(tmp_path, capsys):
    """Stage A (the control on wall-2, the accuracy on a hole never drawn), stage B off the pricing
    CPU (--any-cpu) and stage C, which may not judge PR2 there."""
    out = str(tmp_path)
    test = ["--problems", f"squares:{TEST_SEED + 3}:step", "--n", "16", "--out", out]
    assert E.main([*test, "--control", "wall-2:steady"]) == 0
    stamp = os.path.basename(next(p for p in os.listdir(out) if p.endswith("-checkpoints")))[len(E.STAMP_PREFIX):-12]
    A = json.load(open(E.path_of(out, stamp, "-accuracy.json"), encoding="utf-8"))
    assert A["registered_run"] is False and A["test_flags"] and A["control"]["wall-2 steady"]["equals section 11.17's"]
    assert E.main(["--price", "--stamp", stamp, "--any-cpu", "--reps", "1", *test]) == 0
    P = json.load(open(E.path_of(out, stamp, "-price.json"), encoding="utf-8"))
    assert P["controls"] == [E.spec_key(c) for c in GS.AA_CONTROLS] and len(P["runs"]) >= GS.PRICE_RUNS
    assert set(P["prices_H2"]) == {f"squares {TEST_SEED + 3} step", *P["controls"]}
    with pytest.raises(SystemExit, match="has run"):
        E.main(["--price", "--stamp", stamp, "--any-cpu", "--reps", "1", *test])
    assert E.main(["--judge", "--stamp", stamp, "--out", out]) == 0
    rec = json.load(open(E.path_of(out, stamp, ".json"), encoding="utf-8"))
    assert rec["PR2"]["met"] is None and any("not measured on" in w for w in rec["PR2"]["why"])
    assert rec["predictions"] == {} and rec["outcome"].startswith("E4-H2")
    lines = open(E.path_of(out, stamp, ".txt"), encoding="utf-8").read().splitlines()
    assert 2 <= len(lines) <= 10
    with pytest.raises(SystemExit, match="judged"):
        E.main(["--judge", "--stamp", stamp, "--out", out])


def _made_up(stamp="20990101-000000", code="c" * 40, h2=0.80, h2r=0.40, aa=True, excess=1e-6, u_ring_met=False):
    """Records for the registered problems made up here: nothing is built or run."""
    feats = E.layout_features()
    acc = {}
    for seed, problem in GS.new_problems():
        ring = feats[str(seed)] in E.ROUND_FEATURES
        fr = {f: {"floor": 0.002, "arms": {"H2": {"energy_rel": 0.002 + excess}, "H2r": {"energy_rel": 0.002},
                                           "U": {"energy_rel": 0.03 if ring else 0.002},
                                           "P-s0": {"energy_rel": 0.01}}} for f in E.FRAMES}
        acc[f"squares {seed} {problem}"] = {"frames": fr, "verdicts": {
            "H2": {"met": True}, "H2r": {"met": True}, "U": {"met": u_ring_met or not ring}, "P-s0": {"met": True}}}
    keys = list(acc)
    A = {"stamp": stamp, "code": code, "registered_run": True, "accuracy": acc,
         "E4-H2": {"met": True, "count": "16 of 16", "not_met": []}}
    P = {"stamp": stamp, "code": code, "registered_run": True, "on_price_cpu": True, "machine": {"cpu": GS.PRICE_CPU},
         "runs": [{"thread_pools": [{"user_api": "blas", "num_threads": 4}]}] * 3, "tally": {k: 3 for k in keys},
         "suffice": True, "aa": {"holds": aa, "worst": 0.004 if aa else 0.05, "problems": {}},
         "prices_H2": {k: [h2] * 3 for k in keys}, "prices_H2r": {k: [h2r] * 3 for k in keys}}
    return A, P


def _judge(tmp_path, A, P):
    out = str(tmp_path)
    json.dump(A, open(E.path_of(out, A["stamp"], "-accuracy.json"), "w", encoding="utf-8"))
    json.dump(P, open(E.path_of(out, P["stamp"], "-price.json"), "w", encoding="utf-8"))
    assert E.main(["--judge", "--stamp", A["stamp"], "--out", out]) == 0
    rec = json.load(open(E.path_of(out, A["stamp"], ".json"), encoding="utf-8"))
    return rec, open(E.path_of(out, A["stamp"], ".txt"), encoding="utf-8").read().splitlines()


def test_the_verdicts_on_made_up_numbers(tmp_path):
    (tmp_path / "a").mkdir()
    rec, lines = _judge(tmp_path / "a", *_made_up())
    assert rec["outcome"] == "both gates met" and rec["PR2"]["met"] and abs(rec["PR2"]["geometric_mean"] - 0.8) < 1e-12
    assert abs(rec["H2r"]["ratio_to_H2"] - 0.5) < 1e-12
    assert {k: v["held"] for k, v in rec["predictions"].items()} == {"1": True, "2": True, "3": True, "4": True, "5": True}
    assert len(lines) == 10 and lines[1].split()[0] == "7000000"                # a header, eight layouts, the verdicts
    (tmp_path / "b").mkdir()
    rec, _ = _judge(tmp_path / "b", *_made_up(h2=0.95, excess=2e-4, u_ring_met=True))
    assert rec["outcome"] == "E4-H2 met, PR2 not met"
    assert {k: v["held"] for k, v in rec["predictions"].items()} == {"1": False, "2": True, "3": False, "4": False, "5": True}


def test_the_price_is_not_judged_when_its_checks_fail(tmp_path):
    for i, (A, P) in enumerate([_made_up(aa=False), _made_up()]):
        if i == 1:
            P["code"] = "d" * 40                                                 # another commit
        (tmp_path / str(i)).mkdir()
        rec, _ = _judge(tmp_path / str(i), A, P)
        assert rec["PR2"]["met"] is None and rec["outcome"] == "E4-H2 met, PR2 not judged"
    A, P = _made_up()
    P["runs"] = [{"thread_pools": [{"user_api": "blas", "num_threads": 24}]}]
    assert any("BLAS" in w for w in E.price_checks(A, P))
    P = _made_up()[1]
    P["machine"] = {"cpu": "Intel(R) Core(TM) Ultra 7 155H"}
    assert any("not measured on" in w for w in E.price_checks(A, P))


def test_the_outcome():
    assert E.outcome({"met": False}, {"met": True}) == "E4-H2 not met"
    assert E.outcome({"met": True}, {"met": None}) == "E4-H2 met, PR2 not judged"
    assert E.outcome({"met": True}, {"met": True}) == "both gates met"
    assert E.outcome({"met": True}, {"met": False}) == "E4-H2 met, PR2 not met"


def _has_this_script_at_head() -> bool:
    try:
        subprocess.run(["git", "-C", ROOT, "cat-file", "-e", "HEAD:scripts/chartop_evaluate_squares.py"],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", ROOT, "diff", "--quiet", "HEAD", "--", "scripts/chartop_evaluate_squares.py"],
                       check=True, capture_output=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


@pytest.mark.skipif(not _has_this_script_at_head(), reason="needs this script committed, unchanged, at HEAD")
def test_the_payload_is_checked_against_the_pin(tmp_path):
    import chartop_price_box_payload as PB
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    built = PB.build("HEAD", str(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    json.dump({"code": head}, open(E.path_of(str(out), "s", "-accuracy.json"), "w", encoding="utf-8"))
    v = E.verify_payload(built["path"], "s", str(out))
    assert v["ok"] and v["this script"] and v["the box driver"] and all(v["files"].values()), v
    bad = tmp_path / "bad.tar.gz"                                    # the failing control: one byte in a pinned file
    with tarfile.open(built["path"]) as src, tarfile.open(bad, "w:gz") as dst:
        for m in src.getmembers():
            data = src.extractfile(m).read() if m.isfile() else None
            if m.name == "chartop/atlas/chartop/gates_squares.py":
                data += b"#"
                m.size = len(data)
            dst.addfile(m, io.BytesIO(data) if data is not None else None)
    v = E.verify_payload(str(bad), "s", str(out))
    assert not v["ok"] and v["files"]["atlas/chartop/gates_squares.py"] is False
    json.dump({"code": "e" * 40}, open(E.path_of(str(out), "s", "-accuracy.json"), "w", encoding="utf-8"))
    assert not E.verify_payload(built["path"], "s", str(out))["ok"]          # another commit's payload

"""TL's one evaluation (`scripts/chartop_evaluate_twolevel.py`): how it reaches the registered test
seeds with the pinned generator, the split's content hash against the registration's, the refusals
of test flags, the weights' checks, the host adapter's kept modes, and the readings of PR and of
the five predictions.  Validation seeds (read already), seeded random weights and synthetic
records only: no registered piece or problem is read."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import types

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("psutil")

import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import chartop_evaluate_twolevel as ET  # noqa: E402
import chartop_price_twolevel as CPT  # noqa: E402
import chartop_register_twolevel as RT  # noqa: E402
from atlas.chartop import gates_squares as GS, gates_twolevel as GT, host as hs, twolevel as tl  # noqa: E402


def _args(**kw):
    a = ET.build_parser().parse_args([])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# ---------------------------------------------------------------------------
# the test split
# ---------------------------------------------------------------------------


def test_the_generator_reaches_the_registered_seeds():
    a = ET.draw_args(GT.TEST_SEED_BASE, GT.N_TEST)
    split, start = a[a.index("--split") + 1], int(a[a.index("--start") + 1])
    import chartop_data as CD
    assert CD.SPLIT_SEED[split] + start == 6_000_000 and split != "train-conformal"
    assert a[a.index("--count") + 1] == str(GT.N_TEST) and "--corners" in a and a[a.index("--name") + 1] == "test"
    v = ET.draw_args(2_000_100, 12)
    assert v[v.index("--split") + 1] == "val" and v[v.index("--start") + 1] == "100"


def test_the_split_digest_is_the_registrations(tmp_path):
    """Drawn by the pinned generator from three validation seeds, the stored arrays hash as
    `chartop_register_twolevel.example_digest` hashes the generator's own draw, seed by seed."""
    seeds = [2_000_100, 2_000_101, 2_000_102]
    out = str(tmp_path / "split")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "chartop_data.py"), *ET.draw_args(seeds[0], len(seeds)),
           "--workers", "1", "--out", out]
    subprocess.run(cmd, check=True, capture_output=True)
    assert ET.split_complete(out, seeds[0], len(seeds))
    assert not ET.split_complete(out, seeds[0], len(seeds) + 1)
    got = ET.split_digest(out)
    want = RT.test_split_sha256(seeds)
    assert got == want
    assert len(ET.split_files(out)) == 3 * len(ET.ev0.split_shards(out))


# ---------------------------------------------------------------------------
# the refusals
# ---------------------------------------------------------------------------


def test_the_registered_run_has_no_test_flag():
    assert ET.test_flags(_args()) == []
    ET.refuse_tests_on_registered(_args(), "A")
    ET.refuse_tests_on_registered(_args(price=True), "B")


def test_test_flags_never_read_the_registered_pieces_or_problems():
    with pytest.raises(SystemExit, match="registered test seeds"):
        ET.refuse_tests_on_registered(_args(test_count=8, bench_variant="seen"), "A")
    with pytest.raises(SystemExit, match="registered test seeds"):
        ET.refuse_tests_on_registered(_args(test_seed_base=5_999_990, test_count=8, bench_variant="seen"), "A")
    with pytest.raises(SystemExit, match="held-out problems"):
        ET.refuse_tests_on_registered(_args(test_seed_base=2_000_100, test_count=8), "A")
    ET.refuse_tests_on_registered(_args(test_seed_base=2_000_100, test_count=8, l_test=2, bench_variant="seen"), "A")
    with pytest.raises(SystemExit, match="--problems"):
        ET.refuse_tests_on_registered(_args(any_cpu=True), "B")
    with pytest.raises(SystemExit, match="--problems"):
        ET.refuse_tests_on_registered(_args(any_cpu=True, problems=["held-out:bend-3:steady"]), "B")
    ET.refuse_tests_on_registered(_args(any_cpu=True, problems=["workbench:plate-hole:steady"]), "B")


# ---------------------------------------------------------------------------
# the weights
# ---------------------------------------------------------------------------


def _fixture_run(runs: str, name: str, seed: int, config: dict | None = None, arm: dict | None = None):
    torch.manual_seed(seed)
    e = tl.TwoLevelExpert(**CPT._s_form(), n_c=GT.N_C, corners=GT.CORNERS)
    d = os.path.join(runs, name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "best.pt")
    torch.save({"model": e.state_dict(), "config": dict(e.config(), **(config or {})), "step": 2500,
                "val_e_H_median": 0.1}, path)
    a = dict(GT.arm_config("TL"), name=name, seed=seed, batch={"32": 32, "64": 32, "128": 8}, val_every=2500)
    a.pop("m")
    a.update(arm or {})
    json.dump({"arm": a}, open(os.path.join(d, "arm.json"), "w", encoding="utf-8"))
    return path


def _manifest(runs: str, names):
    import chartop_registry as registry
    with open(os.path.join(runs, "weights-sha256.txt"), "w", encoding="utf-8") as fh:
        for n in names:
            fh.write(f"{registry.sha256_file(os.path.join(runs, n, 'best.pt'))} {n}/best.pt\n")


def _plan():
    return {"registration": "gates_twolevel",
            "arms": [{"name": "TL-s0", "registered": "TL", "seed": 0}, {"name": "TL-s1", "registered": "TL", "seed": 1}]}


def test_the_weights_are_checked(tmp_path):
    runs = str(tmp_path / "runs")
    _fixture_run(runs, "TL-s0", 0)
    _fixture_run(runs, "TL-s1", 1)
    _manifest(runs, ["TL-s0", "TL-s1"])
    got = ET.registered_runs(_plan(), runs)
    assert sorted(got) == ["TL-s0", "TL-s1"] and got["TL-s0"]["seed"] == 0 and got["TL-s1"]["best_step"] == 2500
    # a byte changed after download
    with open(os.path.join(runs, "TL-s1", "best.pt"), "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(SystemExit, match="recorded at download"):
        ET.registered_runs(_plan(), runs)


@pytest.mark.parametrize("config,arm,match", [
    ({"corners": None}, None, "configuration"),
    ({"n_c": 32}, None, "configuration"),
    (None, {"objective": "label-free"}, "not trained as TL"),
    (None, {"steps": 150_000}, "not trained as TL"),
    (None, {"seed": 1}, "not trained as TL"),
])
def test_runs_that_are_not_tl_are_refused(tmp_path, config, arm, match):
    runs = str(tmp_path / "runs")
    _fixture_run(runs, "TL-s0", 0, config, arm)
    _fixture_run(runs, "TL-s1", 1)
    _manifest(runs, ["TL-s0", "TL-s1"])
    with pytest.raises(SystemExit, match=match):
        ET.registered_runs(_plan(), runs)


# ---------------------------------------------------------------------------
# the host adapter: the rollout's kept modes
# ---------------------------------------------------------------------------


class _FakeCC:
    kind, m, envelope = "sine-hat", 16, None

    def __init__(self, answers):
        self.answers, self.calls, self.coarse_seconds = list(answers), 0, 0.0

    def respond(self, problems, pb_of):
        self.calls += len(problems)
        return self.answers.pop(0)


def test_kept_modes_are_checked_bit_for_bit():
    H, lev = np.arange(6.0).reshape(3, 2), np.ones(3)
    first = [hs.Response(H, np.zeros(3), lev)]
    same = [hs.Response(H.copy(), np.full(3, 2.0), lev.copy())]
    moved = [hs.Response(H + 1e-15, np.full(3, 3.0), lev.copy())]
    arm = ET.HostArm(_FakeCC([same, moved, first]))
    out = arm.respond([None], None, modes_from=first)
    assert out[0] is same[0] and arm.kept == [True]
    out = arm.respond([None], None, modes_from=first)
    assert np.array_equal(out[0].H, H) and np.array_equal(out[0].up, np.full(3, 3.0)) and arm.kept == [True, False]
    assert arm.forward([None])[0][0] is H and arm.last_levels == [lev] and arm.calls == 3


# ---------------------------------------------------------------------------
# the ledger's coupled part
# ---------------------------------------------------------------------------


def _held(curve, e_tl, e_u=0.05, floor=0.001):
    runs = {k: {"energy_rel": v, "respond_s_per_piece": 0.01 * (1 + i)} for i, (k, v) in enumerate(zip(ET.TL0_KEYS, curve))}
    runs.update({"U": {"energy_rel": e_u}, "TL-s0": {"energy_rel": e_tl, "respond_s_per_piece": 0.05}})
    fr = {"runs": runs, "truncation_energy_rel": floor}
    return [{"name": "bend-3", "problem": "step", "frames": {"registered": fr, "lowered": fr}}]


def test_n_c_star_is_formed_only_on_a_falling_curve():
    out = ET.coupled_ledger(_held([0.04, 0.02, 0.01, 0.005], 0.0075), "TL-s0")["bend-3 step registered"]
    assert 16 < out["n_c_star"]["n_c_star"] < 32 and not out["n_c_star"]["extrapolated"]
    assert 0.0 < out["attribution"]["network_share_of_left"] < 1.0
    assert out["n_c_star"]["laptop_tl_s_per_piece"] == 0.05
    flat = ET.coupled_ledger(_held([3.27e-6, 3.27e-6 * (1 - 1e-12), 3.27e-6, 3.27e-6], 0.0075, e_u=3.27e-6,
                                   floor=2e-12), "TL-s0")["bend-3 step lowered"]
    assert "no n_c* is formed" in flat["n_c_star"]["refused"] and flat["attribution"]["at_floor"]


# ---------------------------------------------------------------------------
# the readings
# ---------------------------------------------------------------------------


def _price(tl_=1.2, tl0=0.8, h2r=0.4, aa_holds=True, suffice=True, sha="w0", code="abc"):
    keys = [f"held-out {b} {p}" for b, p in GT.HELD_OUT] + [f"squares {s} {p}" for s, p in GS.new_problems()]
    first = {"TL": {k: [tl_] * GS.PRICE_RUNS for k in keys}, "TL0": {k: [tl0] * GS.PRICE_RUNS for k in keys},
             "H2r": {k: [h2r] * GS.PRICE_RUNS for k in keys}, "AA": {k: [1.0] * GS.PRICE_RUNS for k in keys}}
    pools = [{"user_api": "blas", "num_threads": GS.BLAS_THREADS}]
    return {"code": code, "on_price_cpu": True, "registered_run": True, "suffice": suffice,
            "aa": {"holds": aa_holds, "worst": 0.01}, "problems": keys, "first_counted": first,
            "runs": [{"thread_pools": pools, "weights_sha256": sha}] * 3}


def test_pr_is_read_only_when_measured_so():
    acc = {"main": "TL-s0", "weights": {"TL-s0": "w0", "TL-s1": "w1"}, "code": "abc"}
    pr = ET.judge_pr(_price(), acc)
    assert pr["measured"] and abs(pr["geometric_mean"]["TL"]["all"] - 1.2) < 1e-12
    assert pr["faster_than"] == {"the classical solve": False, "H2r as registered (0.306, on R2's sixteen)": False,
                                 "H2r measured beside (R2's sixteen)": False, "TL0": False}
    fast = ET.judge_pr(_price(tl_=0.25), acc)["faster_than"]
    assert all(fast.values())
    for bad in (_price(aa_holds=False), _price(suffice=False), _price(sha="other"), _price(code="def")):
        r = ET.judge_pr(bad, acc)
        assert not r["measured"] and "faster_than" not in r
    assert ET.judge_pr(None, acc) == {"measured": False, "why": "no price record"}


def _gates(cx=True, k=True, d=True, l_=True, nc=True, e4=True, pr=None):
    per = {f"{b} {p}": e4 for b, p in GT.HELD_OUT}
    return {"gates": {"CX": {"met": cx}, "K": {"met": k}, "D": {"met": d}, "L": {"met": l_},
                      "NC": {"met": nc, "median": 0.4 if nc else 0.2, "p10": 0.1},
                      "E4": {"met": e4, "per_problem": per}, "PR": pr or {"measured": False, "why": "no price record"}}}


def _acc(h=0.1, a=0.3, b=0.5):
    return {"ledger": {"test": {"groups": {"hats": {"median": h}, "j1-3": {"median": a}, "j4-8": {"median": b}}}}}


def test_the_predictions_are_read_as_written():
    p = ET.judge_predictions(_gates(), _acc())
    assert p["words"] == list(GT.PREDICTIONS)
    assert [p[k]["held"] for k in "1234"] == [True, True, True, True] and p["5"]["held"] is None
    p = ET.judge_predictions(_gates(cx=False, nc=False, e4=False), _acc(h=0.4))
    assert [p[k]["held"] for k in "1234"] == [False, False, False, False]
    assert ET.judge_predictions(_gates(l_=False), _acc())["1"]["held"] is False
    assert ET.judge_predictions(_gates(), _acc(h=0.3, a=0.3))["4"]["held"] is False      # a tie is not "lowest"
    assert ET.judge_predictions(_gates(), _acc(h=None))["4"]["held"] is False
    acc = {"main": "TL-s0", "weights": {"TL-s0": "w0"}, "code": "abc"}
    slow = ET.judge_predictions(_gates(pr=ET.judge_pr(_price(tl_=1.0), acc)), _acc())
    fast = ET.judge_predictions(_gates(pr=ET.judge_pr(_price(tl_=0.9), acc)), _acc())
    assert slow["5"]["held"] is True and fast["5"]["held"] is False

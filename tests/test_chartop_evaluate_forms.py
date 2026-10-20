"""The one evaluation of the registration after Stage 0 (`scripts/chartop_evaluate_forms.py`),
on small cases: throwaway weights, and a stand-in test split drawn from VALIDATION seeds
(the registered test base is patched for the test), so no registered test piece is drawn.

What is real here:
- E1's rows (``e_H``, ``e_H,L`` against an independent numpy computation, ``e_K``);
- one benchmark problem in both frames at n = 32 (E4, D, K, L, SE2, and E3, SE1 and SE3 for
  the main arm);
- the weights' checks, the registry, the judgements and the record's assembly, CW's
  pairing included.

What is replaced by cheap stand-ins in the end-to-end test: the benchmarks, the dual
bracket, KT and the rollout, which the first two tests and Stage 0's tests cover."""
from __future__ import annotations

import glob
import json
import os
import sys
import types

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from atlas.chartop import data, fe, forms, gates as G, gates_forms as GF, ports, train  # noqa: E402
from atlas.chartop.expert import LearnedExpert  # noqa: E402

N = 32
STAND_IN = G.SEED_BASE["val"] + 950          # validation seeds stand in for the test split


@pytest.fixture(scope="module")
def ef():
    import chartop_evaluate_forms
    return chartop_evaluate_forms


def _randomised(net, std, seed=0):
    torch.manual_seed(seed)
    for p in net.parameters():
        torch.nn.init.normal_(p, 0.0, std)
    return net.eval()


def _expert(bubble, strips, size="0.3M", std=0.02, seed=0):
    return _randomised(forms.KernelExpert("split", "unet", size, p_size=size, bubble=bubble, strips=strips),
                       std, seed)


def _split(folder, count=3):
    """A whole split in the format `chartop_evaluate.split_complete` accepts, both bases labelled."""
    seeds = [STAND_IN + i for i in range(count)]
    exs = [data.draw(s, n=N) for s in seeds]
    arrs = [data.arrays(e) for e in exs]
    out = {k: np.stack([a[k] for a in arrs]) for k in train.KEYS}
    out.update(seed=np.array(seeds), sup_mu=np.array([e.envelope["sup_abs_mu"] for e in exs]),
               freq=np.array([e.freq for e in exs]), fo=np.array([-1.0 if e.fo is None else e.fo for e in exs]),
               family=np.array([e.family for e in exs]))
    np.savez(folder / f"test_n{N}_0000.npz", **out)
    for kind, pre in (("sine-hat", "labels_"), ("cosine", "labels-cosine_")):
        lab = [data.labels_from_arrays(a, ports.basis(kind, N, 16).Q) for a in arrs]
        np.savez(folder / f"{pre}test_n{N}_0000.npz", H=np.stack([x["H"].T.astype(np.float32) for x in lab]),
                 up=np.stack([x["up"].astype(np.float32) for x in lab]),
                 trace_Lambda=np.array([x["trace_Lambda"] for x in lab]),
                 up_energy=np.array([x["up_energy"] for x in lab]), lseed=np.array(seeds))
    return exs


@pytest.fixture()
def stand_in(tmp_path, monkeypatch):
    monkeypatch.setitem(G.SEED_BASE, "test", STAND_IN)
    folder = tmp_path / "test"
    folder.mkdir()
    return folder, _split(folder)


def test_e1_rows_measure_the_learned_columns_against_their_own_energy(ef, stand_in):
    """``e_H,L`` of each piece equals an independent computation from the exact modes of the
    piece's own matrix; ``e_H`` is Stage 0's measure; e_K is zero to rounding; and on these
    pieces the learned columns' error is not the whole-set one."""
    folder, exs = stand_in
    net = _expert("corner", True)
    rows = ef.e1_streamed(net, str(folder))
    assert len(rows["e_H_L"]) == 3 and list(rows["seed"]) == [e.seed for e in exs]
    assert float(np.max(rows["e_K"])) < 1e-12 and float(np.max(rows["e_K_inf"])) < 1e-12
    assert float(np.max(rows["trace_gap"])) < 1e-5
    ex = LearnedExpert(net, envelope=data.ENVELOPE)
    pb = ports.basis("sine-hat", N, 16)
    cols = list(forms.learned_columns("sine-hat", 16))
    for i, e in enumerate(exs):
        p = e.problem
        exact = fe.Exact(p.matrix(), N)
        H = exact.modes(pb.Q)
        Hh = ex.respond([p], lambda n_: pb)[0].H
        E = Hh - H
        err = np.einsum("ik,ik->k", E, np.asarray(exact.A_I @ E))
        lam = np.diag(exact.port_matrix(pb.Q, H))
        want = np.sqrt(err[cols].sum() / lam[cols].sum())
        assert rows["e_H_L"][i] == pytest.approx(want, rel=2e-3)
        assert rows["e_H"][i] == pytest.approx(np.sqrt(err.sum() / lam.sum()), rel=2e-3)
    assert not np.allclose(rows["e_H_L"], rows["e_H"], rtol=1e-2)


def test_one_benchmark_problem_in_both_frames(ef, monkeypatch):
    """bend-3 at n = 32, steady and step, for a main arm with the strips and a run without:
    the kernel-exact split forms meet K, D, L and SE2 whatever the weights; E3, SE1 and SE3 are
    judged for the main arm only; E4 has both frames; the steady problem carries the main
    arm's dual bracket; and every piece has e_H and e_H,L against the exact modes."""
    monkeypatch.setattr(ef, "BENCH_N", N)
    experts = {"S": LearnedExpert(_expert("corner", True), envelope=data.ENVELOPE),
               "W-s0": LearnedExpert(_expert("square", False, seed=1), envelope=data.ENVELOPE)}
    for problem in ("steady", "step"):
        r = ef.benchmark_problem("bend-3", problem, experts, "S")
        assert set(r["frames"]) == {"registered", "lowered"} and r["pieces"] == 15
        for run in experts:
            rr = r["runs"][run]
            assert rr["d"]["met"] and rr["d"]["ratio_judged"], (problem, run, rr["d"])
            assert rr["k"]["met"] and all(rr["l_pieces_met"]) and rr["se2"]["met"]
            assert set(rr["e4"]) == {"met", "as registered", "lowered by 300 K"}
            pieces = r["frames"]["registered"]["runs"][run]["pieces"]
            assert len(pieces["e_H"]) == len(pieces["e_H_L"]) == 15
        reg = r["frames"]["registered"]["runs"]
        assert {"e3", "se1", "se3", "network_term"} <= set(reg["S"]) and "e3" not in reg["W-s0"]
        assert reg["S"]["e3"]["met"] and reg["S"]["se1"]["met"]
        assert ("dual" in r) == (problem == "steady")
        if problem == "steady":
            assert len(r["dual"]) == 15
        else:   # a step's reference energy holds the level: lowering it changes the relative error
            low = r["frames"]["lowered"]["runs"]["S"]["energy_rel"]
            assert low > 2 * reg["S"]["energy_rel"]


def _fake_bench(ef, name, problem, experts, main):
    """A benchmark record with the shape `benchmark_problem` returns: W's energy errors twice
    S''s, seed by seed, so that CW is 2 in both pairings."""
    base = 0.01 * (1 + len(name) % 3) + (0.0 if problem == "steady" else 0.005)
    frames = {}
    for frame in ("registered", "lowered"):
        runs = {}
        for run in experts:
            e = base * (2.0 if run.startswith("W") else 1.0) * (1.0 if frame == "registered" else 3.0)
            rec = {"energy_rel": e, "energy_abs": e, "max_err": 0.001, "calls_per_piece": 1.0,
                   "pieces": {"e_H": [0.2, 0.3] if run.startswith("W") else [0.1, 0.15],
                              "e_H_L": [0.1, 0.1], "e_K": [0.0, 0.0], "e_K_inf": [0.0, 0.0]}}
            if run == main and frame == "registered":
                rec.update(e3={"met": True}, se1={"met": True}, se3={"met": True}, network_term=0.1)
            runs[run] = rec
        frames[frame] = {"span": 100.0, "floor_max": 0.0, "reference_energy": 1.0, "runs": runs}
    out = {"name": name, "problem": problem, "pieces": 2, "frames": frames, "elapsed_s": 0.0,
           "runs": {run: {"e4": {"met": False}, "d": {"met": True}, "k": {"met": True},
                          "l_pieces_met": [True, True], "se2": {"met": True}} for run in experts}}
    if problem == "steady":
        out["dual"] = [{"min_eig_rel": {"lo": 0.5}}, {"min_eig_rel": {"lo": 0.25}}]
    return out


def _weights(runs_dir, plan, corrupt=None):
    """Throwaway weights for every run of the plan, the registered configuration (1 M + 1 M), and
    the manifest of their sha256 as the download wrote it."""
    lines = []
    for d in plan["arms"]:
        a = GF.ARMS[d["registered"]]
        e = _expert(a["bubble"], a["strips"], size="1M", seed=d["seed"])
        cfg = e.config()
        if corrupt == d["name"]:
            cfg = dict(cfg, bubble="corner" if cfg["bubble"] == "square" else "square")
        (runs_dir / d["name"]).mkdir(parents=True, exist_ok=True)
        path = runs_dir / d["name"] / "best.pt"
        torch.save({"model": e.state_dict(), "config": cfg}, path)
        import hashlib
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()} {d['name']}/best.pt")
    (runs_dir / "weights-sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_evaluation_end_to_end_on_throwaway_weights(ef, stand_in, tmp_path, monkeypatch):
    """The real plan's five runs on throwaway weights: the weights are checked against the
    manifest and the registered arms; E1 runs for S alone on the stand-in split; CW pairs W
    with S' seed by seed; the gates are assembled; the record is written; and the same weights
    are refused a second time, as are a tampered file and a configuration that is not its arm's."""
    folder, _exs = stand_in
    plan = json.load(open(ef.PLAN, encoding="utf-8"))
    runs_dir = tmp_path / "runs"
    _weights(runs_dir, plan)
    calls = []
    monkeypatch.setattr(ef, "benchmark_problem",
                        lambda name, problem, experts, main: calls.append((name, problem, main))
                        or _fake_bench(ef, name, problem, experts, main))
    monkeypatch.setattr(ef, "test_pieces", lambda count: calls.append(("test_pieces", count))
                        or [data.draw(STAND_IN + i, n=N).problem for i in range(2)])
    monkeypatch.setattr(ef.ev0, "dual_records",
                        lambda ex, problems, traces: [{"min_eig_rel": {"lo": 0.4}} for _ in problems])
    monkeypatch.setattr(ef.data, "draw", lambda s, **kw: types.SimpleNamespace(problem=("test-piece", s)))
    monkeypatch.setattr(ef.ev0, "kt_record", lambda name, ex: {"kt": {"met": True}})
    monkeypatch.setattr(ef.ev0, "rollout_record",
                        lambda ex: {"records": [{"step": 1, "e4": {"met": True}}, {"step": 20, "e4": {"met": False}}]})
    monkeypatch.setattr(ef, "load_blind", lambda ex, problems, seed=11: [True] * len(problems))
    out = tmp_path / "out"
    argv = ["--runs", str(runs_dir), "--out", str(out), "--test-dir", str(folder), "--test-count", "3",
            "--l-test", "2", "--dual-test", "2", "--threads", "1", "--workers", "1",
            "--untrained", os.path.join(ROOT, "out", "chartop", "forms-controls-corner-strips.json")]
    assert ef.main(argv) == 0
    assert sorted(c for c in calls if c[0] != "test_pieces") == sorted(
        (n_, p_, "S") for n_ in GF.HELD_OUT_BENCHMARKS for p_ in GF.BENCHMARK_PROBLEMS)
    (path,) = glob.glob(str(out / "evaluation-forms-*[0-9].json"))
    rec = json.load(open(path, encoding="utf-8"))
    assert rec["main"] == "S" and set(rec["weights"]) == {"S", "Sp-s0", "Sp-s1", "W-s0", "W-s1"}
    assert rec["E1"]["gate"] == "E1" and "e_H,L" in rec["E1"]["measure"]
    assert rec["E1_beside"]["e_H_all_64"]["count"] == 3 and rec["K_test"]["met"]
    assert rec["L_test"] == {"met": True, "pieces": 2}
    cw = rec["CW"]
    assert cw["verdict"] == "alternative loses" and cw["pairs"] == {"0": ["W-s0", "Sp-s0"], "1": ["W-s1", "Sp-s1"]}
    assert all(p["ratio"] == pytest.approx(2.0) for p in cw["pairings"].values())
    assert cw["CW0"] == pytest.approx(1.0023, abs=1e-4)
    assert cw["e_H_median_benchmark_pieces"]["W-s0"] == pytest.approx(0.25)
    g = rec["gates"]
    assert g["K"]["met"] and g["D"]["met"] and g["L"]["met"] and not g["E4"]["met"]
    assert g["E3"]["met"] and g["SE1"]["met"] and g["SE2"]["met"] and g["SE3"]["met"] and g["KT"]["met"]
    assert g["DUAL"]["met"] and rec["DUAL"]["benchmark_pieces"] == 6 and g["S11"]["e4_met_at"] == [1]
    assert rec["E5"]["records"]
    txt = open(os.path.splitext(path)[0] + ".txt", encoding="utf-8").read()
    assert "CW: alternative loses" in txt and "E4    NOT MET" in txt
    with pytest.raises(SystemExit, match="runs once"):
        ef.main(argv)
    with pytest.raises(SystemExit, match="runs once"):
        ef.main(argv + ["--resume"])
    # a tampered weights file, and a configuration that is not its registered arm's
    other = tmp_path / "runs2"
    _weights(other, plan)
    with open(other / "W-s1" / "best.pt", "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(SystemExit, match="not the one recorded"):
        ef.main(["--runs", str(other), "--out", str(tmp_path / "out2")] + argv[4:])
    third = tmp_path / "runs3"
    _weights(third, plan, corrupt="W-s0")
    with pytest.raises(SystemExit, match="is not registered arm 'W'"):
        ef.main(["--runs", str(third), "--out", str(tmp_path / "out3")] + argv[4:])
    assert not glob.glob(str(tmp_path / "out2" / "evaluation-*")) and not glob.glob(str(tmp_path / "out3" / "evaluation-*"))

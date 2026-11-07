"""The one evaluation of the registration after the forms' evaluation
(`scripts/chartop_evaluate_prior.py`), on small cases: throwaway weights, and a stand-in
test split drawn from VALIDATION seeds (the fresh split's base is patched for the test), so
no registered test piece is drawn and no trained weight is read.

What is real here: one held-out problem in both frames at n = 32 (E4, D, K, L, SE2, and E3,
SE1, SE3 and the dual bracket for the seed-0 runs), the seen problem beside it, the pin's,
the weights' and the reused runs' checks, the registry, E1's rows on the stand-in split,
the comparisons' pairing and the record's assembly.  The benchmarks, the dual bracket's
test pieces, KT and the rollout are cheap stand-ins in the end-to-end test, as in
`tests/test_chartop_evaluate_forms.py`."""
from __future__ import annotations

import glob
import hashlib
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

from atlas.chartop import data, forms, gates as G, gates_prior as GP, ports, train  # noqa: E402
from atlas.chartop.expert import LearnedExpert  # noqa: E402

N = 32
STAND_IN = G.SEED_BASE["val"] + 960          # validation seeds stand in for the fresh split


@pytest.fixture(scope="module")
def epr():
    import chartop_evaluate_prior
    return chartop_evaluate_prior


def _expert(arm, seed=0, std=0.02, size="1M"):
    c = GP.arm_config(arm) if arm in GP.ARMS else dict(GP.COMMON, backbone="unet")
    e = forms.make(c["backbone"], size, c["form"], c["kind"], c["m"], size, c["bubble"], c["strips"], c["hats"])
    torch.manual_seed(seed)
    for p in e.parameters():
        torch.nn.init.normal_(p, 0.0, std)
    return e.eval()


def _split(folder, count=3):
    seeds = [STAND_IN + i for i in range(count)]
    exs = [data.draw(s, n=N) for s in seeds]
    arrs = [data.arrays(e) for e in exs]
    out = {k: np.stack([a[k] for a in arrs]) for k in train.KEYS}
    out.update(seed=np.array(seeds), sup_mu=np.array([e.envelope["sup_abs_mu"] for e in exs]),
               freq=np.array([e.freq for e in exs]), fo=np.array([-1.0 if e.fo is None else e.fo for e in exs]),
               family=np.array([e.family for e in exs]))
    np.savez(folder / f"test_n{N}_0000.npz", **out)
    lab = [data.labels_from_arrays(a, ports.basis("sine-hat", N, 16).Q) for a in arrs]
    np.savez(folder / f"labels_test_n{N}_0000.npz", H=np.stack([x["H"].T.astype(np.float32) for x in lab]),
             up=np.stack([x["up"].astype(np.float32) for x in lab]),
             trace_Lambda=np.array([x["trace_Lambda"] for x in lab]),
             up_energy=np.array([x["up_energy"] for x in lab]), lseed=np.array(seeds))
    return exs


@pytest.fixture()
def stand_in(tmp_path, monkeypatch):
    monkeypatch.setattr(GP, "TEST_SEED_BASE", STAND_IN)
    folder = tmp_path / "test"
    folder.mkdir()
    return folder, _split(folder)


def test_one_held_out_problem_in_both_frames(epr, monkeypatch):
    """bend-3's held-out variant at n = 32, steady and step, for the two arms' seed-0 runs and
    S': the kernel-exact forms meet K, D, L and SE2 whatever the weights; E3, SE1 and SE3 and
    the dual bracket are the seed-0 runs' only; E4 has both frames; the problem is the
    held-out one; and the seen problem beside it carries E4's energy error alone."""
    monkeypatch.setattr(epr, "BENCH_N", N)
    experts = {"P-s0": LearnedExpert(_expert("P", size="0.3M"), envelope=data.ENVELOPE),
               "PG-s0": LearnedExpert(_expert("PG", size="0.3M", seed=1), envelope=data.ENVELOPE),
               "Sp-s0": LearnedExpert(_expert("Sp", size="0.3M", seed=2), envelope=data.ENVELOPE)}
    mains = {"P-s0", "PG-s0"}
    for problem in ("steady", "step"):
        r = epr.benchmark_problem("bend-3", problem, experts, mains)
        assert r["variant"] == "held-out" and r["pieces"] == 15
        for run in experts:
            rr = r["runs"][run]
            assert rr["d"]["met"] and rr["k"]["met"] and all(rr["l_pieces_met"]) and rr["se2"]["met"], (problem, run)
            assert set(rr["e4"]) == {"met", "as registered", "lowered by 300 K"}
        reg = r["frames"]["registered"]["runs"]
        for run in mains:
            assert {"e3", "se1", "se3", "network_term"} <= set(reg[run]) and reg[run]["e3"]["met"]
        assert "e3" not in reg["Sp-s0"]
        assert ("dual" in r) == (problem == "steady")
        if problem == "steady":
            assert set(r["dual"]) == mains and all(len(v) == 15 for v in r["dual"].values())
    s = epr.benchmark_problem("bend-3", "steady", experts, set(), variant=None)
    assert s["variant"] is None and "dual" not in s
    assert all("pieces" not in s["frames"]["registered"]["runs"][run] for run in experts)
    assert set(s["runs"]["P-s0"]) == {"e4"}
    held_floor = epr.benchmark_problem("bend-3", "steady", {}, set())["frames"]["registered"]["truncation_energy_rel"]
    assert s["frames"]["registered"]["truncation_energy_rel"] < 1e-6 < held_floor


def _fake_bench(name, problem, experts, mains, variant="held-out"):
    """A record shaped as `benchmark_problem` returns it: P's energy errors half S''s and PG's
    equal to P's on the held-out problems, so that CP is 0.5 and CG 1 in both pairings; on the
    seen problems every run alike."""
    base = 0.01 * (1 + len(name) % 3) + (0.0 if problem == "steady" else 0.005)

    def scale(run):
        if variant is None:
            return 1.0
        return 1.0 if run.startswith("Sp") else 0.5
    frames = {}
    for frame in ("registered", "lowered"):
        runs = {}
        for run in experts:
            e = base * scale(run) * (1.0 if frame == "registered" else 3.0)
            rec = {"energy_rel": e, "energy_abs": e, "max_err": 0.001, "calls_per_piece": 1.0}
            if variant is not None and frame == "registered":
                rec["pieces"] = {"e_H": [0.1, 0.15], "e_H_L": [0.1, 0.1], "e_K": [0.0, 0.0], "e_K_inf": [0.0, 0.0]}
                if run in mains:
                    rec.update(e3={"met": True}, se1={"met": True}, se3={"met": True}, network_term=0.1)
            runs[run] = rec
        frames[frame] = {"span": 100.0, "floor_max": 0.0, "reference_energy": 1.0, "runs": runs}
    out = {"name": name, "problem": problem, "variant": variant, "pieces": 2, "frames": frames, "elapsed_s": 0.0}
    if variant is None:
        out["runs"] = {run: {"e4": {"met": False}} for run in experts}
        return out
    out["runs"] = {run: {"e4": {"met": run == "P-s0" and problem == "steady"}, "d": {"met": True},
                         "k": {"met": True}, "l_pieces_met": [True, True], "se2": {"met": True}} for run in experts}
    if problem == "steady":
        out["dual"] = {run: [{"min_eig_rel": {"lo": 0.5}}, {"min_eig_rel": {"lo": 0.25}}] for run in mains}
    return out


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _weights(runs_dir, plan, corrupt=None):
    """Throwaway weights for every run of the plan, the registered configurations, each run's
    arm.json as training wrote it, and the manifest."""
    lines = []
    for d in plan["arms"]:
        e = _expert(d["registered"], seed=d["seed"])
        cfg = e.config()
        if corrupt == d["name"]:
            cfg = dict(cfg, backbone="unet" if cfg["backbone"] == "unet-spectral" else "unet-spectral")
        (runs_dir / d["name"]).mkdir(parents=True, exist_ok=True)
        path = runs_dir / d["name"] / "best.pt"
        torch.save({"model": e.state_dict(), "config": cfg}, path)
        arm = {k: v for k, v in d.items() if k != "registered"}
        (runs_dir / d["name"] / "arm.json").write_text(json.dumps({"arm": arm, "config": cfg}), encoding="utf-8")
        lines.append(f"{_sha(path)} {d['name']}/best.pt")
    (runs_dir / "weights-sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _reused(forms_dir):
    lines, reused = [], {}
    for seed in (0, 1):
        name = f"Sp-s{seed}"
        (forms_dir / name).mkdir(parents=True, exist_ok=True)
        path = forms_dir / name / "best.pt"
        e = _expert("Sp", seed=10 + seed)
        torch.save({"model": e.state_dict(), "config": e.config()}, path)
        lines.append(f"{_sha(path)} {name}/best.pt")
        reused[seed] = (name, _sha(path))
    (forms_dir / "weights-sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return reused


def test_the_evaluation_end_to_end_on_throwaway_weights(epr, stand_in, tmp_path, monkeypatch):
    """The real plan's four runs and two reused S' runs, on throwaway weights: the pin, the
    weights, their configurations and training options, and the reused runs are checked;
    E1 runs for each arm's seed-0 run on the stand-in split; CP pairs P with S' and CG PG
    with P, seed by seed; the gates are assembled per arm; the record is written; and the same
    weights are refused a second time, as are a tampered file, a configuration that is not
    its arm's, reused weights the registration does not name, and a changed pinned file."""
    folder, _exs = stand_in
    plan = json.load(open(epr.PLAN, encoding="utf-8"))
    runs_dir, forms_dir = tmp_path / "runs", tmp_path / "forms-runs"
    _weights(runs_dir, plan)
    named = _reused(forms_dir)
    real_reused = GP.REUSED
    monkeypatch.setattr(GP, "REUSED", named)
    calls = []
    monkeypatch.setattr(epr, "benchmark_problem",
                        lambda name, problem, experts, mains, variant="held-out":
                        calls.append((name, problem, variant, tuple(sorted(mains))))
                        or _fake_bench(name, problem, experts, mains, variant))
    monkeypatch.setattr(epr, "test_pieces", lambda count: [data.draw(STAND_IN + i, n=N).problem for i in range(2)])
    monkeypatch.setattr(epr.ef, "load_blind", lambda ex, problems, seed=11: [True] * len(problems))
    monkeypatch.setattr(epr.ef, "dual_test_records", lambda ex, seeds: [{"min_eig_rel": {"lo": 0.4}} for _ in seeds])
    monkeypatch.setattr(epr.ev0, "kt_record", lambda name, ex: {"kt": {"met": True}})
    monkeypatch.setattr(epr.ev0, "rollout_record",
                        lambda ex: {"records": [{"step": 1, "e4": {"met": True}}, {"step": 20, "e4": {"met": False}}]})
    out = tmp_path / "out"
    argv = ["--runs", str(runs_dir), "--forms-runs", str(forms_dir), "--out", str(out), "--test-dir", str(folder),
            "--test-count", "3", "--l-test", "2", "--dual-test", "2", "--threads", "1", "--workers", "1"]
    assert epr.main(argv) == 0
    held = [c for c in calls if c[2] == "held-out"]
    assert sorted((c[0], c[1]) for c in held) == sorted(GP.HELD_OUT)
    assert all(c[3] == ("P-s0", "PG-s0") for c in held)
    assert sorted((c[0], c[1]) for c in calls if c[2] is None) == sorted(GP.SEEN)
    (path,) = glob.glob(str(out / "evaluation-prior-*[0-9].json"))
    rec = json.load(open(path, encoding="utf-8"))
    assert rec["mains"] == {"P": "P-s0", "PG": "PG-s0"}
    assert set(rec["weights"]) == {"P-s0", "P-s1", "PG-s0", "PG-s1"} and set(rec["reused"]) == {"Sp-s0", "Sp-s1"}
    assert set(rec["E1"]) == {"P-s0", "PG-s0"} and rec["E1"]["P-s0"]["gate"] == "E1"
    assert rec["K_test"]["P-s0"]["met"] and rec["L_test"]["P-s1"] == {"met": True, "pieces": 2}
    assert rec["CP"]["verdict"] == "alternative wins" and rec["CP"]["pairs"] == {"0": ["P-s0", "Sp-s0"], "1": ["P-s1", "Sp-s1"]}
    assert all(p["ratio"] == pytest.approx(0.5) for p in rec["CP"]["pairings"].values())
    assert all(v == pytest.approx(1.0) for v in rec["CP"]["seen_beside"].values())
    assert rec["CG"]["verdict"] == "tie" and rec["CG"]["pairs"] == {"0": ["PG-s0", "P-s0"], "1": ["PG-s1", "P-s1"]}
    g = rec["gates"]
    assert set(g) == {"P", "PG"}
    for arm in ("P", "PG"):
        assert g[arm]["K"]["met"] and g[arm]["D"]["met"] and g[arm]["L"]["met"] and g[arm]["SE2"]["met"]
        assert g[arm]["E3"]["met"] and g[arm]["SE1"]["met"] and g[arm]["SE3"]["met"] and g[arm]["KT"]["met"]
        assert g[arm]["DUAL"]["met"] and g[arm]["S11"]["e4_met_at"] == [1] and not g[arm]["E4"]["met"]
    assert g["P"]["E4"]["per_problem"]["bend-3 steady"] and not g["PG"]["E4"]["per_problem"]["bend-3 steady"]
    assert rec["DUAL"]["P-s0"]["benchmark_pieces"] == 6 and rec["E5"]["records"]
    txt = open(os.path.splitext(path)[0] + ".txt", encoding="utf-8").read()
    assert "CP: alternative wins" in txt and "CG: tie" in txt and "arm PG (PG-s0)" in txt
    with pytest.raises(SystemExit, match="runs once"):
        epr.main(argv)
    with pytest.raises(SystemExit, match="runs once"):
        epr.main(argv + ["--resume"])
    other = tmp_path / "runs2"
    _weights(other, plan)
    with open(other / "PG-s1" / "best.pt", "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(SystemExit, match="not the one recorded"):
        epr.main(["--runs", str(other)] + argv[2:4] + ["--out", str(tmp_path / "out2")] + argv[6:])
    third = tmp_path / "runs3"
    _weights(third, plan, corrupt="PG-s0")
    with pytest.raises(SystemExit, match="is not registered arm 'PG'"):
        epr.main(["--runs", str(third)] + argv[2:4] + ["--out", str(tmp_path / "out3")] + argv[6:])
    monkeypatch.setattr(GP, "REUSED", real_reused)
    with pytest.raises(SystemExit, match="is not the reused run"):
        epr.main(argv[:4] + ["--out", str(tmp_path / "out4")] + argv[6:])
    monkeypatch.setattr(GP, "REUSED", named)
    bad = tmp_path / "registered-prior.txt"
    text = open(epr.REGISTRATION, encoding="ascii").read()
    bad.write_text(text.replace("benchmarks.py  sha256 ", "benchmarks.py  sha256 0"), encoding="ascii")
    with pytest.raises(SystemExit):
        epr.main(argv[:4] + ["--out", str(tmp_path / "out5"), "--registration", str(bad)] + argv[6:])
    for k in (2, 3, 4, 5):
        assert not glob.glob(str(tmp_path / f"out{k}" / "evaluation-*"))

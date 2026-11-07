"""The registration after the forms' one evaluation (`gates_prior`), against its draft
([[chart-operator-kernel-exact-forms]] section 11.7) and the owner's two choices:
every bar is gates_forms', the fresh test split is unused, the arms build as drafted,
S' is reused by its downloaded hashes, and the held-out problems are new materials and
data on the same layouts, off by default.  No network is evaluated here."""
from __future__ import annotations

import hashlib
import json
import os
import re

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from atlas.chartop import benchmarks as bm
from atlas.chartop import data, gates as G, gates_forms as GF, gates_prior as GP, host

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = os.path.join(ROOT, "out", "chartop", "registration-prior-heldout.json")
REG = os.path.join(ROOT, "out", "chartop", "registered-prior.txt")
PINNED = ("atlas/chartop/gates_prior.py", "atlas/chartop/gates_forms.py", "atlas/chartop/gates.py",
          "atlas/chartop/benchmarks.py", "out/chartop/registration-prior-heldout.json")


def _hash(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def test_the_registered_files_are_unchanged():
    """The pin (`scripts/chartop_register_prior.py --pin`, approved by the owner on
    2026-11-07): each pinned file's sha256, line endings normalised, is the one recorded,
    so a bar, a held-out problem or the classical record changed after registration fails
    here; and the registration's own text is the module's."""
    assert os.path.exists(REG), "out/chartop/registered-prior.txt is missing: pin before training"
    text = open(REG, encoding="ascii").read()
    for rel in PINNED:
        m = re.search(re.escape(rel) + r"\s+sha256 ([0-9a-f]{64})", text)
        assert m, rel
        assert _hash(os.path.join(ROOT, rel)) == m.group(1), f"{rel} changed after registration"
    assert GP.REGISTERED in text and GP.text() in text


def test_the_treatments_are_as_registered():
    """The prior and the spectral path by value: the kernel's lengths, the floor, and the
    numbers the prior gives at n = 64 (to 1e-9, not by hash: they pass through LAPACK), and
    the path's modes and channels."""
    from atlas.chartop import forms as fm, model as md
    text = open(REG, encoding="ascii").read()
    m = re.search(r"tr Sigma = ([0-9.eE+-]+); learned block's eigenvalues ([0-9.eE+-]+) to ([0-9.eE+-]+)", text)
    assert m
    S = fm.trace_covariance("generator-field", GP.KIND, 64, GP.M_PORTS)
    L = list(fm.learned_columns(GP.KIND, GP.M_PORTS))
    lam = np.linalg.eigvalsh(S[np.ix_(L, L)])
    for got, want in ((np.trace(S), m.group(1)), (lam.max(), m.group(2)), (lam.min(), m.group(3))):
        assert abs(float(got) - float(want)) <= 1e-9 * abs(float(want))
    assert fm.MODE_PRIORS == ("generator-field",) and fm.MODE_PRIOR_FLOOR == 1e-3
    assert data.FREQUENCIES == {"low": 0.5, "mid": 0.25, "high": 0.125}
    assert (md.SPECTRAL_K, md.SPECTRAL_WIDTH) == (8, 32)
    assert "floored at 0.001" in text and "8 cosine modes per direction on 32 channels" in text

SHARED = ("KIND", "M_PORTS", "N_COLUMNS", "N_LEARNED", "STRIP_SINES", "N_VAL", "N_TEST", "BENCHMARK_N",
          "E1_MODES_MEDIAN", "E1_MODES_P90", "E1_PARTICULAR_MEDIAN", "E1_PARTICULAR_P90", "E1_MIN_BIN",
          "MU_BINS", "FREQ_BINS", "E3_ROUNDING", "E4_ENERGY", "E4_MAX_OF_RANGE", "SE1_IDENTITY",
          "SE1_LAMBDA_MIN", "SE2_CALLS_PER_PIECE", "DUAL_CHAIN_ROUNDING", "KAPPA_T_BETA", "KAPPA_T_TOL",
          "KAPPA_T_MAX_ITERATIONS", "ROLLOUT_BENCHMARK", "ROLLOUT_STEPS", "WIN", "LOSE", "EVAL_PRECISION",
          "K_E_K", "K_E_K_INF", "DATUM_SHIFT_K", "D_RATIO", "D_FLOOR", "D_MAX_OF_RANGE", "L_TEST_PIECES",
          "L_SOURCES", "E5_PRICED", "CEILING_USD", "BOX_MAX_USD_PER_HOUR")


def test_every_bar_is_gates_forms():
    for name in SHARED:
        assert getattr(GP, name) == getattr(GF, name), name
    assert (GP.BENCHMARK_NAMES, GP.BENCHMARK_PROBLEMS) == (GF.HELD_OUT_BENCHMARKS, GF.BENCHMARK_PROBLEMS)
    for f in ("judge_k", "judge_d", "judge_l", "judge_e1", "judge_e4_both_frames", "compare_pairings",
              "e_h_learned"):
        assert getattr(GP, f) is getattr(GF, f), f


def test_the_gates_are_gates_forms_but_for_the_problems_and_the_split():
    """Every gate of gates_forms, in its order, with the same title; the bars differ only
    where they name the problems, the split or the arm; KT, S11, E5, SE2 and DUAL are
    gates_forms' word for word; and a swap that matches nothing is refused."""
    ours, theirs = {g.key: g for g in GP.GATES}, {g.key: g for g in GF.GATES}
    assert [g.key for g in GP.GATES] == [g.key for g in GF.GATES]
    assert all(ours[k].title == theirs[k].title for k in ours)
    for k in ("KT", "S11", "E5", "SE2", "DUAL"):
        assert ours[k].bar == theirs[k].bar, k
    for k in ("K", "D", "L", "E3", "E4", "SE1", "SE3"):
        assert "held-out benchmark" in ours[k].bar and ours[k].bar != theirs[k].bar, k
        assert "every benchmark p" not in ours[k].bar, k
    assert "5,000,000 to 5,002,399" in ours["E1"].bar and "arm S" not in ours["E1"].bar
    assert "fresh test split" in ours["K"].bar and "fresh test split" in ours["L"].bar
    with pytest.raises(ValueError):
        GP._swap("a bar", ("not there", "x"))


def test_the_fresh_split_is_unused():
    assert GP.TEST_SEEDS == (5_000_000, 5_000_000 + GF.N_TEST)
    assert GP.TEST_SEED_BASE not in G.SEED_BASE.values()
    for base in G.SEED_BASE.values():
        assert base + 1_000_000 <= GP.TEST_SEED_BASE or base >= GP.TEST_SEEDS[1]


def test_the_arms_are_the_draft():
    """P and PG: S's form at S''s budget, two seeds, the strips skipped in training, the
    generator's trace prior; P on the U-Net and PG on the spectral path.  Each builds a
    `train.Arm` and an expert whose configuration is the arm's."""
    from atlas.chartop import forms as fm
    from atlas.chartop import train
    assert set(GP.ARMS) == {"P", "PG"}
    for arm, backbone in (("P", "unet"), ("PG", "unet-spectral")):
        c = GP.arm_config(arm)
        assert c["backbone"] == backbone and c["mode_prior"] == "generator-field"
        assert c["bubble"] == "corner" and c["strips"] is True and c["hats"] == "learned"
        assert c["strips_in_training"] is False and c["steps"] == GF.ARMS["Sp"]["steps"] == 37_500
        assert GP.ARMS[arm]["seeds"] == (0, 1)
        fields = train.Arm.__dataclass_fields__
        assert set(c) - set(fields) == {"m"} and c["m"] == 16    # the expert's own default
        a = train.Arm(arm, **{k: v for k, v in c.items() if k in fields})
        e = fm.make(a.backbone, "0.3M", a.form, a.kind, c["m"], "0.3M", a.bubble, a.strips, a.hats)
        cfg = e.config()
        assert (cfg["backbone"], cfg["form"], cfg["bubble"], cfg["strips"], cfg["hats"]) == (
            backbone, "split", "corner", True, "learned")


def test_s_prime_is_reused_by_its_downloaded_hashes():
    with open(os.path.join(ROOT, "out", "chartop", "runs", "forms", "weights-sha256.txt"),
              encoding="utf-8") as fh:
        manifest = {rel.lstrip("*").split("/")[0]: h for h, rel in (ln.split() for ln in fh if ln.strip())}
    assert sorted(GP.REUSED) == list(GF.ARMS["Sp"]["seeds"])
    for seed, (run, sha) in GP.REUSED.items():
        assert run == f"Sp-s{seed}" and manifest[run] == sha


def _dirichlet(b):
    out = []
    for i, p in enumerate(b.problems):
        for s in range(4):
            kind, val = b.bc(i, s, host._side_points(p, s))
            if kind == "dirichlet":
                out.append(np.asarray(val, float))
    return out


@pytest.mark.parametrize("name", bm.NAMES)
def test_the_held_out_problems_are_new_materials_and_data_on_the_same_layouts(name):
    """At n = 16: the same pieces and charts as the default, new materials by role, and the
    ends' new data (hot 400 + 20 cos(pi s), 420 to 380; cold 300 + 10 cos(2 pi s), 310 at
    both walls and 290 at the middle), every other side insulated.  The default is the
    old problem: steel or copper and steel, the ends at 400 and 300 K.  An unknown
    variant is refused."""
    kw = {"envelope": data.ENVELOPE} if name == "s-channel" else {}
    old = bm.build(name, 16, "step", **kw)
    new = bm.build(name, 16, "step", variant="held-out", **kw)
    assert len(old.problems) == len(new.problems) and new.meta["variant"] == "held-out"
    assert "variant" not in old.meta
    for p, q in zip(old.problems, new.problems):
        assert p.label == q.label and np.array_equal(p.chart.X, q.chart.X)
    lib = bm._material
    roles = bm.HELD_OUT_MATERIALS[name]
    assert {k: v["k"] for k, v in new.meta["materials"].items()} == {r: lib(m)["k"] for r, m in roles.items()}
    assert {float(q.kappa_ref) for q in new.problems} == {lib(m)["k"] for m in roles.values()}
    lo, hi = data.FO_RANGE
    assert all(lo <= q.fo <= hi for q in new.problems)
    vals_old, vals_new = _dirichlet(old), _dirichlet(new)
    assert sorted({float(v) for a in vals_old for v in a}) == [bm.T_COLD, bm.T_HOT]
    assert len(vals_new) == len(vals_old)
    hot = np.concatenate([a for a in vals_new if a.max() > 350.0])
    cold = np.concatenate([a for a in vals_new if a.max() < 350.0])
    assert abs(hot.max() - 420.0) < 1e-9 and abs(hot.min() - 380.0) < 1e-9
    assert abs(cold.max() - 310.0) < 1e-9 and 290.0 - 1e-9 <= cold.min() <= 291.0
    with pytest.raises(ValueError):
        bm.build(name, 16, "steady", variant="nope", **kw)


def test_the_held_out_problems_were_measured_classically_and_are_meetable():
    """`scripts/chartop_register_prior.py`'s record: the classical solver only; the seen
    problems' floors equal the one evaluation's bit for bit (the defaults did not move);
    every held-out problem inside the envelope and the generator's Fourier numbers, with a
    classical floor below E4's bar in both frames."""
    rec = json.load(open(RECORD, encoding="utf-8"))
    assert rec["experts_run"] == ["classical"]
    assert rec["seen_floor_equals_evaluation"] is True and rec["held_out_fo_within_training"] is True
    assert rec["held_out_refusals"] == 0
    assert {(r["name"], r["problem"]) for r in rec["held_out"]} == set(GP.HELD_OUT)
    for r in rec["held_out"]:
        assert r["variant"] == GP.HELD_OUT_VARIANT
        for f in ("registered", "lowered"):
            assert r["frames"][f]["truncation_energy_rel"] < GP.E4_ENERGY, (r["name"], r["problem"], f)


def test_the_comparison_measure_and_the_text():
    er = {(b, p, f): float(i) for i, (b, p) in enumerate(GP.HELD_OUT) for f in ("registered", "lowered")}
    assert GP.cw_measure(er) == float(np.mean(range(len(GP.HELD_OUT))))
    t = GP.text()
    assert GP.REGISTERED in t and all(p in t for p in GP.PREDICTIONS)
    assert "unet-spectral" in t and "Sp-s0" in t and "hash-pinned" in GP.REGISTERED

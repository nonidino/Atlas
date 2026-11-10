"""The two-level expert with branch 2c's corner treatment (`twolevel.TwoLevelExpert(corners=8)`,
`coarse_patches`), the training path that carries it (`train.Arm.corners`) and the data's
corner flags (`scripts/chartop_data.py --corners`).  Seeded random weights or zero heads only,
on pieces from seeds no registration draws; a few training steps.

- the option is off by default (no key in the configuration) and refused without a coarse
  size; without the batch's flags the expert refuses to guess;
- the torch path (a batch's arrays and flags) and the numpy path (operators and flags) agree to
  1e-12, and the answers are coarse-exact on each example's own V_e and kernel-exact;
- zero heads are TL0 with B (`coarse.correct_fields` with the patch columns) to 1e-12;
- through the host (`twolevel.host_expert`: the form inside `coarse.CoarseCorrected`, each
  piece's own chart) the answers are the expert's to 1e-10;
- `train.train` with corners starts at exactly 2 (TL0 with B is its twin), loads back with its
  treatment, refuses corners without a coarse size and flags whose seeds disagree;
- `chartop_data.py --corners` writes each example's flags from its chart, and leaves the shards'
  arrays as they were without it."""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas.chartop import coarse as co, coarse_patches as cpt, data, fe, forms, ports, train  # noqa: E402
from atlas.chartop import twolevel as tl  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
torch.set_num_threads(1)

KIND, M = "sine-hat", 16
SEEDS = (9_000_100, 9_000_102, 9_000_112)            # a step, a steady piece, a short step
CONFORMAL = 9_000_104
S_CONFIG = dict(form="split", backbone="unet", size="0.3M", kind=KIND, m=M, p_size="0.3M",
                bubble="corner", strips=True)


def _expert(n_c, corners=8, std=None, seed=0):
    torch.manual_seed(0)
    e = tl.TwoLevelExpert(S_CONFIG["form"], S_CONFIG["backbone"], S_CONFIG["size"], KIND, M,
                          S_CONFIG["p_size"], S_CONFIG["bubble"], S_CONFIG["strips"], "learned", n_c, corners)
    if std is not None:
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for prm in e.parameters():
                prm.copy_(std * torch.randn(prm.shape, generator=g, dtype=prm.dtype))
    return e.eval()


_B: dict = {}


def _problems(n=32):
    return [data.draw(s, n=n).problem for s in SEEDS] + [data.draw(CONFORMAL, family="conformal-map", n=n).problem]


def _batch(n=32):
    """The pieces' own float64 problems (as the host reads them), as a batch with their flags."""
    if n not in _B:
        probs = _problems(n)
        b = {}
        for k in ("a11", "a12", "a22", "sigma", "w"):
            b[k] = torch.tensor(np.stack([p.tensors()[k] for p in probs]), dtype=torch.float64)
        b["g"] = torch.tensor(np.stack([p.g for p in probs]), dtype=torch.float64)
        ins = [p.inputs() for p in probs]
        b["inputs"] = torch.tensor(np.stack([c for c, _s, _r in ins]), dtype=torch.float32)
        b["scalars"] = torch.tensor(np.stack([s for _c, s, _r in ins]), dtype=torch.float32)
        b["s0"] = torch.tensor([p.s0 for p in probs], dtype=torch.float64)
        b["corners"] = torch.tensor(np.stack([cpt.corner_flags(p.chart.X) for p in probs]))
        _B[n] = (b, probs)
    return _B[n]


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def test_the_option_is_off_by_default_and_checked():
    assert "corners" not in _expert(8, corners=None).config()
    assert _expert(8).config()["corners"] == 8
    for bad in (dict(n_c=None, corners=8), dict(n_c=8, corners=6), dict(n_c=8, corners=2)):
        with pytest.raises(ValueError):
            _expert(**bad)
    b, _p = _batch()
    flags = b["corners"].tolist()
    assert flags[:3] == [[True] * 4] * 3 and flags[3] == [False] * 4
    no_flags = {k: v for k, v in b.items() if k != "corners"}
    with pytest.raises(ValueError):
        with torch.no_grad():
            tl.call(_expert(8), no_flags)


@pytest.mark.parametrize("n_c", [4, 8])
def test_torch_and_numpy_paths_agree_and_answer_exactly_on_V_e(n_c):
    b, probs = _batch()
    n = 32
    T = _expert(n_c, std=0.02, seed=3)
    with torch.no_grad():
        Ha, upa = tl.call(T, b)
        la = T.last_level.clone()
        Hb, upb = T(b["inputs"], b["scalars"], b["s0"], w=b["w"], g=b["g"], ops=forms.batch_operators(b),
                    corner_flags=b["corners"])
        lb = T.last_level.clone()
    assert _rel(Hb, Ha) < 1e-12 and _rel(upb, upa) < 1e-12 and _rel(lb, la) < 1e-12
    Q = ports.basis(KIND, n, M).Q
    c1 = forms.kernel_coefficients(KIND, n, M)
    K = Ha.shape[1]
    for k, p in enumerate(probs):
        Mx = p.matrix()
        loop, inner = fe.grid(n)
        cs = co.CoarseSpace.of_matrix(Mx, n, n_c, extra=co.CornerTreatment("B", frac=8).of_problem(p, n_c))
        assert cs.n_extra == (0 if k == 3 else 4 * ((n // 8) ** 2 - (1 if n_c == 8 else 0)))
        BQ = -np.asarray(Mx.tocsr()[inner][:, loop] @ Q)
        Hk = Ha[k].reshape(K, -1).T.numpy()
        assert co.coarse_exactness(cs.P, co.residual(Mx, n, Hk, Q), BQ)["rel"] < 1e-12
        bI = p.load()[inner]
        assert co.coarse_exactness(cs.P, co.residual(Mx, n, upa[k].reshape(-1).numpy(), rhs=bI), bI)["rel"] < 1e-12
        r = Hk @ c1 + (p.s0 * la[k].reshape(-1).numpy() if p.s0 > 0 else 0.0) - 1.0
        assert np.abs(r).max() < 1e-12


def test_zero_heads_are_tl0_with_b():
    b, probs = _batch()
    n, n_c = 32, 8
    twin = tl.zero_heads_twin(_expert(n_c, std=0.02))
    assert twin.corners == 8
    free = _expert(None, corners=None)
    with torch.no_grad():
        Ht, upt = tl.call(twin, b)
        lt = twin.last_level
        H0, up0 = tl.call(free, b)
        l0 = free.last_level
    Q = ports.basis(KIND, n, M).Q
    K = Ht.shape[1]
    for k, p in enumerate(probs):
        _loop, inner = fe.grid(n)
        S = co.CornerTreatment("B", frac=8).of_problem(p, n_c)
        Hn, upn, ln = co.correct_fields(p.matrix(), n, n_c, H0[k].reshape(K, -1).T.numpy(),
                                        up0[k].reshape(-1).numpy(), l0[k].reshape(-1).numpy(), p.s0,
                                        p.load()[inner], Q, extra=S)
        assert _rel(Ht[k].reshape(K, -1).T.numpy(), Hn) < 1e-12
        assert _rel(upt[k].reshape(-1).numpy(), upn) < 1e-12
        if p.s0 > 0:
            assert _rel(lt[k].reshape(-1).numpy(), ln) < 1e-12


def test_through_the_host_the_answers_are_the_experts():
    b, probs = _batch()
    T = _expert(8, std=0.02, seed=7)
    with torch.no_grad():
        H, up = tl.call(T, b)
        lev = T.last_level.clone()
    ad = tl.host_expert(T)
    assert ad.n_c == 8 and ad.corners.name == "B-n/8"
    pb = lambda n: ports.basis(KIND, n, M)            # noqa: E731
    got = ad.respond(probs, pb)
    assert ad.extra_columns == 3 * 4 * ((32 // 8) ** 2 - 1)
    K = H.shape[1]
    for k, r in enumerate(got):
        assert _rel(r.H, H[k].reshape(K, -1).T.numpy()) < 1e-10
        assert _rel(r.up, up[k].reshape(-1).numpy()) < 1e-10
        if probs[k].s0 > 0:
            assert _rel(r.level, lev[k].reshape(-1).numpy()) < 1e-10
    plain = tl.host_expert(_expert(8, corners=None, std=0.02, seed=7))
    assert plain.corners is None


# ---------------------------------------------------------------------------
# training with corners
# ---------------------------------------------------------------------------


def _shards(folder, split, seeds, n=32, corners=True, bad_seed=False):
    exs = [data.draw(s, n=n) for s in seeds]
    arrs = [data.arrays(e) for e in exs]
    out = {k: np.stack([a[k] for a in arrs]) for k in train.KEYS}
    out.update(seed=np.array(seeds), sup_mu=np.array([e.envelope["sup_abs_mu"] for e in exs]),
               logJ_range=np.array([e.envelope["logJ_range"] for e in exs]),
               s_grad=np.array([e.envelope["s_grad"] for e in exs]),
               fo=np.array([-1.0 if e.fo is None else e.fo for e in exs]),
               freq=np.array([e.freq for e in exs]), family=np.array([e.family for e in exs]),
               method=np.array([e.method for e in exs]))
    name = f"{split}_n{n}_0000.npz"
    np.savez(folder / name, **out)
    Q = ports.basis(KIND, n, M).Q
    lab = [data.labels_from_arrays(a, Q) for a in arrs]
    np.savez(folder / f"labels_{name}", H=np.stack([x["H"].T.astype(np.float32) for x in lab]),
             up=np.stack([x["up"].astype(np.float32) for x in lab]),
             trace_Lambda=np.array([x["trace_Lambda"] for x in lab]),
             up_energy=np.array([x["up_energy"] for x in lab]), lseed=np.array(seeds))
    if corners:
        cs = np.array(seeds) + (1 if bad_seed else 0)
        np.savez(folder / f"corners_{name}", corners=np.stack([cpt.corner_flags(e.problem.chart.X) for e in exs]),
                 cseed=cs)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("tlc-data")
    _shards(d, "train", [9_000_100, 9_000_101, 9_000_104, 9_000_105])
    _shards(d, "val", [9_000_106, 9_000_109])
    return d


def _arm(name, **kw):
    base = dict(form="split", size="0.3M", p_size="0.3M", bubble="corner", strips=True, steps=4,
                batch={32: 2}, val_every=2, coarse=8, corners=8, objective="relative", lr=1e-3)
    base.update(kw)
    return train.Arm(name, **base)


def test_a_two_level_arm_with_corners_trains_and_loads(dataset, tmp_path):
    out = tmp_path / "tlc"
    train.train(_arm("tlc"), str(dataset), str(out), device="cpu", log_every=1)
    log = [json.loads(line) for line in open(out / "log.jsonl", encoding="utf-8")]
    first = next(r for r in log if "loss" in r)
    assert abs(first["loss"] - 2.0) < 1e-9                    # its twin is TL0 with B
    assert any("val_rho_median" in r for r in log)
    e = train.load_expert(str(out / "last.pt"))
    assert isinstance(e, tl.TwoLevelExpert) and (e.n_c, e.corners) == (8, 8)
    arm = json.load(open(out / "arm.json", encoding="utf-8"))
    assert arm["config"]["corners"] == 8 and arm["arm"]["corners"] == 8


def test_the_arm_refuses_corners_it_cannot_carry(dataset, tmp_path):
    with pytest.raises(ValueError):
        train.train(_arm("x", coarse=None), str(dataset), str(tmp_path / "x"), device="cpu")
    bad = tmp_path / "bad"
    bad.mkdir()
    _shards(bad, "train", [9_000_100, 9_000_101], bad_seed=True)
    _shards(bad, "val", [9_000_106, 9_000_109])
    with pytest.raises(ValueError):
        train.train(_arm("y"), str(bad), str(tmp_path / "y"), device="cpu")
    none = tmp_path / "none"
    none.mkdir()
    _shards(none, "train", [9_000_100, 9_000_101], corners=False)
    _shards(none, "val", [9_000_106, 9_000_109], corners=False)
    with pytest.raises(FileNotFoundError):
        train.train(_arm("z"), str(none), str(tmp_path / "z"), device="cpu")
    out = tmp_path / "plain"                                   # without corners, no flags are read
    train.train(_arm("plain", corners=None, steps=2), str(none), str(out), device="cpu", log_every=1)
    assert "corners" not in json.load(open(out / "arm.json", encoding="utf-8"))["config"]


@pytest.mark.parametrize("k", range(8))
def test_the_corner_flags_follow_the_squares_symmetries(k):
    """Under symmetry k of the square, the flags move with the arrays (`train.d4_corners`): the
    enlarged correction of the transformed batch, with its flags transformed, is the transform
    of the original's.  Asymmetric flags, so a wrong permutation would show."""
    from atlas.chartop import coarse as co2, model
    b, _probs = _batch()
    n, n_c = 32, 8
    flags = torch.tensor([[True, False, False, False], [False, True, True, False], [True, False, True, True],
                          [False, False, False, True]])
    ta = [b[key] for key in ("a11", "a12", "a22", "sigma")]
    rng = np.random.default_rng(k)
    X = torch.tensor(rng.standard_normal((4, n - 1, n - 1)))
    F = torch.tensor(rng.standard_normal((4, n - 1, n - 1)))

    def corrected(arrays, fl, x, f):
        ts = cpt.TorchEnlargedSpace.of_arrays(*arrays, n_c, cpt.patch_nodes(n, n_c, fl.numpy(), 8))
        return ts.correct(x, co2.residual_torch(*arrays, x[:, None], rhs=f[:, None])[:, 0])
    want = model._spatial(corrected(ta, flags, X, F), k & 1, (k >> 1) & 1, (k >> 2) & 1)
    bk = model.d4(dict(b, corners=flags), k)
    fk = train.d4_corners(flags, k)
    sp = lambda a: model._spatial(a, k & 1, (k >> 1) & 1, (k >> 2) & 1)      # noqa: E731
    got = corrected([bk[key] for key in ("a11", "a12", "a22", "sigma")], fk, sp(X), sp(F))
    assert float((got - want).abs().max() / want.abs().max()) < 1e-12
    assert sorted(train.corner_perm(k)) == [0, 1, 2, 3] and (k == 0) == (train.corner_perm(k) == (0, 1, 2, 3))
    if k:                                                      # unpermuted flags would not commute
        wrong = corrected([bk[key] for key in ("a11", "a12", "a22", "sigma")], flags, sp(X), sp(F))
        if not torch.equal(fk, flags):
            assert float((wrong - want).abs().max() / want.abs().max()) > 1e-6


def test_light_labels_follow_the_symmetries_as_the_modes_do():
    """The light labels' energies permute as the exact modes do under `model.d4_labels`."""
    from atlas.chartop import model
    b, lab, _a = _labelled()
    n = 32
    lam = torch.stack([tl.column_energies({k: v[i:i + 1] for k, v in b.items()}, lab["H"][i:i + 1])[0]
                       for i in range(lab["H"].shape[0])])
    for k in range(8):
        full = model.d4_labels(lab, KIND, M, k)
        bk = model.d4(b, k)
        lam_k = tl.column_energies(bk, full["H"])
        light = train.d4_light_labels({"lam": lam, "trace_Lambda": lab["trace_Lambda"],
                                       "up_energy": lab["up_energy"]}, KIND, M, n, k)
        assert float((light["lam"] - lam_k).abs().max() / lam_k.abs().max()) < 1e-10


def _labelled(n=32):
    from atlas.chartop import ports as pts
    arrs = [data.arrays(data.draw(s, n=n)) for s in SEEDS]
    b = {k: torch.tensor(np.stack([a[k] for a in arrs]), dtype=torch.float64)
         for k in ("a11", "a12", "a22", "sigma", "w", "g")}
    b["inputs"] = torch.tensor(np.stack([a["inputs"] for a in arrs]), dtype=torch.float32)
    b["scalars"] = torch.tensor(np.stack([a["scalars"] for a in arrs]), dtype=torch.float32)
    b["s0"] = torch.tensor(np.array([a["s0"] for a in arrs]), dtype=torch.float64)
    Q = pts.basis(KIND, n, M).Q
    labs = [data.labels_from_arrays(a, Q) for a in arrs]
    K = 4 * M
    lab = {"H": torch.tensor(np.stack([x["H"].T.reshape(K, n - 1, n - 1) for x in labs])),
           "up": torch.tensor(np.stack([x["up"].reshape(n - 1, n - 1) for x in labs])),
           "trace_Lambda": torch.tensor([x["trace_Lambda"] for x in labs], dtype=torch.float64),
           "up_energy": torch.tensor([x["up_energy"] for x in labs], dtype=torch.float64)}
    return b, lab, arrs


def test_a_relative_arm_trains_on_light_labels(tmp_path):
    """The training split with light labels only (as `chartop_data.py --light-labels` writes
    them): a two-level arm with corners starts at exactly 2 and trains; the light labels'
    energies are the full labels' column energies, and their trace is the labels' trace."""
    import chartop_data
    d = tmp_path / "light"
    d.mkdir()
    _shards(d, "train", [9_000_100, 9_000_101, 9_000_104, 9_000_105])
    _shards(d, "val", [9_000_106, 9_000_109])
    name = "train_n32_0000.npz"
    with np.load(d / name) as zf:
        z = {k: zf[k] for k in zf.files}
    with np.load(d / f"labels_{name}") as ff:
        full_trace = ff["trace_Lambda"].copy()
    lights = [chartop_data.light_labels({k: z[k][i] for k in train.KEYS}) for i in range(len(z["seed"]))]
    np.savez(d / f"light_{name}", lam=np.stack([x["lam"] for x in lights]),
             trace_Lambda=np.array([x["trace_Lambda"] for x in lights]),
             up_energy=np.array([x["up_energy"] for x in lights]), lseed=z["seed"])
    assert np.allclose([x["trace_Lambda"] for x in lights], full_trace, rtol=1e-12, atol=0)
    assert np.allclose([x["lam"].sum() for x in lights], full_trace, rtol=1e-10, atol=0)
    os.remove(d / f"labels_{name}")                            # the training split has light labels only
    split = train.load_split(str(d), "train", with_labels=True, with_corners=True)
    assert "lam" in split[32] and "H" not in split[32]
    out = tmp_path / "run"
    train.train(_arm("light", steps=4), str(d), str(out), device="cpu", log_every=1)
    log = [json.loads(line) for line in open(out / "log.jsonl", encoding="utf-8")]
    assert abs(next(r for r in log if "loss" in r)["loss"] - 2.0) < 1e-9
    with pytest.raises(ValueError):                            # the supervised objective needs the modes
        train.train(_arm("sup", objective="supervised", coarse=None, corners=None, steps=2), str(d),
                    str(tmp_path / "sup"), device="cpu")


# ---------------------------------------------------------------------------
# the data's corner flags
# ---------------------------------------------------------------------------


def test_the_data_scripts_corner_flags(tmp_path, monkeypatch):
    """Two training examples drawn twice, with and without ``--corners``: the shards' arrays are
    the same, and the flags are each example's chart's (`coarse_patches.corner_flags`)."""
    import chartop_data
    runs = {}
    for tag, extra in (("plain", []), ("flags", ["--corners"])):
        out = tmp_path / tag
        monkeypatch.setattr(sys, "argv", ["chartop_data.py", "--split", "train", "--count", "2",
                                          "--start", "49998", "--workers", "1", "--out", str(out), *extra])
        assert chartop_data.main() == 0
        runs[tag] = out
    names = sorted(f for f in os.listdir(runs["plain"]) if f.endswith(".npz"))
    assert names and not any(f.startswith("corners_") for f in names)
    for f in names:
        a, b = np.load(runs["plain"] / f), np.load(runs["flags"] / f)
        assert sorted(a.files) == sorted(b.files)
        assert all(np.array_equal(a[k], b[k]) for k in a.files)
        c = np.load(runs["flags"] / f"corners_{f}")
        assert np.array_equal(c["cseed"], a["seed"])
        for seed, flags in zip(a["seed"], c["corners"]):
            assert np.array_equal(flags, cpt.corner_flags(data.draw(int(seed)).problem.chart.X))
    man = json.load(open(runs["flags"] / "manifest-train-49998.json", encoding="utf-8"))
    assert man["corners"] is True
    assert "corners" not in json.load(open(runs["plain"] / "manifest-train-49998.json", encoding="utf-8"))

"""Phase 3 of the Stage 0 rework (`atlas/chartop/twolevel.py`): the two-level expert in torch,
its relative objective and the network ledger, with seeded random weights or zero heads only.
Nothing is trained beyond a few steps on pieces from seeds no registration draws.

- without a coarse size it is S bit for bit; with zero heads it is TL0 bit for bit, and Phase
  2's numpy TL0 (`coarse.CoarseCorrected`) to rounding;
- with random weights its answers are coarse-exact (CX) and kernel-exact (K); its numpy path
  (the host's operators) and its torch path (a batch's arrays) agree;
- through the host: the load never reaches the modes or the level field (L), the uniform
  state is a fixed point (K), and the datum drops out (D), each judged by `gates_forms`;
- the relative objective: 1 per term at TL0, 0 at the exact fields, its gradient the label-free
  gradient over TL0's error, and the guard leaving out a square, where TL0 is exact;
- the ledger on hand-built cases: rho 0, 1 and negative, the guard; the attribution and its
  floor; the equivalent coarse size; kappa_kill at 1 and 0; the dead-network stop;
- `train.train` on a two-level arm: the relative objective reads 2 at the start, rho is logged,
  the dead network stops, and the checkpoint loads as a two-level expert."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas.chartop import benchmarks as bm, coarse as co, data, fe, forms, gates_forms as GF  # noqa: E402
from atlas.chartop import host, model, ports, train  # noqa: E402
from atlas.chartop import twolevel as tl  # noqa: E402
from atlas.chartop.expert import LearnedExpert  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
torch.set_num_threads(1)

KIND, M = "sine-hat", 16
STEP, SHORT_STEP, STEADY = 9_000_100, 9_000_112, 9_000_102       # Fo 0.63, Fo 0.0117, steady
S_CONFIG = dict(form="split", backbone="unet", size="0.3M", kind=KIND, m=M, p_size="0.3M",
                bubble="corner", strips=True)


def _randomise(net, std, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for prm in net.parameters():
            prm.copy_(std * torch.randn(prm.shape, generator=g, dtype=prm.dtype))
    return net.eval()


def _expert(n_c, std=None, seed=0):
    torch.manual_seed(0)
    e = tl.TwoLevelExpert(S_CONFIG["form"], S_CONFIG["backbone"], S_CONFIG["size"], KIND, M,
                          S_CONFIG["p_size"], S_CONFIG["bubble"], S_CONFIG["strips"], "learned", n_c)
    return _randomise(e, std, seed) if std is not None else e.eval()


_BATCHES: dict = {}


def _batch(n=32, seeds=(STEP, STEADY, SHORT_STEP)):
    """A batch (float64 coefficients, float32 inputs) and its labels, from stored arrays."""
    key = (n, tuple(seeds))
    if key not in _BATCHES:
        arrs = [data.arrays(data.draw(s, n=n)) for s in seeds]
        b = {k: torch.tensor(np.stack([a[k] for a in arrs]), dtype=torch.float64)
             for k in ("a11", "a12", "a22", "sigma", "w", "g")}
        b["inputs"] = torch.tensor(np.stack([a["inputs"] for a in arrs]), dtype=torch.float32)
        b["scalars"] = torch.tensor(np.stack([a["scalars"] for a in arrs]), dtype=torch.float32)
        b["s0"] = torch.tensor(np.array([a["s0"] for a in arrs]), dtype=torch.float64)
        Q = ports.basis(KIND, n, M).Q
        lab = [data.labels_from_arrays(a, Q) for a in arrs]
        K = 4 * M
        labels = {"H": torch.tensor(np.stack([x["H"].T.reshape(K, n - 1, n - 1) for x in lab])),
                  "up": torch.tensor(np.stack([x["up"].reshape(n - 1, n - 1) for x in lab])),
                  "trace_Lambda": torch.tensor([x["trace_Lambda"] for x in lab], dtype=torch.float64),
                  "up_energy": torch.tensor([x["up_energy"] for x in lab], dtype=torch.float64)}
        _BATCHES[key] = (b, labels, arrs)
    return _BATCHES[key]


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


# ---------------------------------------------------------------------------
# the expert
# ---------------------------------------------------------------------------


def test_without_a_coarse_size_it_is_S_bit_for_bit():
    b, _lab, _a = _batch()
    S = _randomise(forms.KernelExpert(**{k: v for k, v in S_CONFIG.items()}), 0.02)
    T = tl.from_kernel(S, None).eval()
    with torch.no_grad():
        HS, upS = forms.call(S, b)
        HT, upT = tl.call(T, b)
    assert torch.equal(HS, HT) and torch.equal(upS, upT) and torch.equal(S.last_level, T.last_level)
    assert T.config()["n_c"] is None and not hasattr(S, "n_c")


@pytest.mark.parametrize("n_c", [4, 8])
def test_zero_heads_are_TL0_bit_for_bit_and_phase_2s_TL0_to_rounding(n_c):
    b, _lab, arrs = _batch()
    trained = _expert(n_c, std=0.02)
    twin = tl.zero_heads_twin(trained)
    fresh = _expert(n_c)                                     # zero heads, its own backbone
    with torch.no_grad():
        Ht, upt = tl.call(twin, b)
        lt = twin.last_level
        Hf, upf = tl.call(fresh, b)
        lf = fresh.last_level
        Hr, _ = tl.call(trained, b)
    assert torch.equal(Ht, Hf) and torch.equal(upt, upf) and torch.equal(lt, lf)
    assert not torch.equal(Hr, Ht)                           # the random weights do move it
    assert sum(p.numel() for p in twin.parameters()) == 0    # no backbone runs in the twin
    # Phase 2's TL0: U behind the host's adapter, numpy, on the same stored arrays
    n = 32
    Q = ports.basis(KIND, n, M).Q
    for k, a in enumerate(arrs):
        A = {key: np.asarray(a[key], np.float64) for key in ("a11", "a12", "a22", "sigma", "w", "g")}
        Mx = fe.assemble(n, A["a11"], A["a12"], A["a22"], A["sigma"])
        _l, inner = fe.grid(n)
        bI = (fe.mass(n, A["w"]) @ A["g"].ravel())[inner]
        with torch.no_grad():
            fresh_free = _expert(None)
            H0, up0 = tl.call(fresh_free, {key: v[k:k + 1] for key, v in b.items()})
        K = H0.shape[1]
        Hn, upn, _lev = co.correct_fields(Mx, n, n_c, H0[0].reshape(K, -1).T.numpy(), up0[0].reshape(-1).numpy(),
                                          fresh_free.last_level[0].reshape(-1).numpy(), float(a["s0"]), bI, Q)
        assert _rel(Hf[k].reshape(K, -1).T.numpy(), Hn) < 1e-12
        assert _rel(upf[k].reshape(-1).numpy(), upn) < 1e-12


@pytest.mark.parametrize("n_c", [4, 16])
def test_random_weights_answer_coarse_exact_and_kernel_exact(n_c):
    b, _lab, arrs = _batch()
    T = _expert(n_c, std=0.02, seed=3)
    with torch.no_grad():
        H, up = tl.call(T, b)
        raw_H, raw_up, raw_level = T.last_uncorrected
    n = 32
    Q = ports.basis(KIND, n, M).Q
    c1 = forms.kernel_coefficients(KIND, n, M)
    P = co.prolongation(n, n_c)
    K = H.shape[1]
    for k, a in enumerate(arrs):
        A = {key: np.asarray(a[key], np.float64) for key in ("a11", "a12", "a22", "sigma", "w", "g")}
        Mx = fe.assemble(n, A["a11"], A["a12"], A["a22"], A["sigma"])
        loop, inner = fe.grid(n)
        BQ = -np.asarray(Mx.tocsr()[inner][:, loop] @ Q)
        Hk = H[k].reshape(K, -1).T.numpy()
        assert co.coarse_exactness(P, co.residual(Mx, n, Hk, Q), BQ)["rel"] < 1e-12
        assert co.coarse_exactness(P, co.residual(Mx, n, raw_H[k].reshape(K, -1).T.numpy(), Q), BQ)["rel"] > 1e-6
        bI = (fe.mass(n, A["w"]) @ A["g"].ravel())[inner]
        assert co.coarse_exactness(P, co.residual(Mx, n, up[k].reshape(-1).numpy(), rhs=bI), bI)["rel"] < 1e-12
        s0 = float(a["s0"])
        r = Hk @ c1 + (s0 * T.last_level[k].reshape(-1).numpy() if s0 > 0 else 0.0) - 1.0
        assert np.abs(r).max() < 1e-12, (k, np.abs(r).max())


def test_the_numpy_path_is_the_torch_path():
    b, _lab, _a = _batch()
    T = _expert(8, std=0.02, seed=5)
    with torch.no_grad():
        Ha, upa = tl.call(T, b)
        la = T.last_level.clone()
        Hb, upb = T(b["inputs"], b["scalars"], b["s0"], w=b["w"], g=b["g"], ops=forms.batch_operators(b))
        lb = T.last_level.clone()
    assert _rel(Hb, Ha) < 1e-12 and _rel(upb, upa) < 1e-12 and _rel(lb, la) < 1e-12
    with pytest.raises(ValueError):
        T(b["inputs"], b["scalars"], b["s0"], w=b["w"], g=b["g"], strip_solve=False)


# ---------------------------------------------------------------------------
# through the host: L, K and D as gates_forms judges them
# ---------------------------------------------------------------------------


def test_through_the_host_the_load_never_reaches_the_modes_and_K_holds():
    T = _expert(4, std=0.02, seed=7)
    ex = LearnedExpert(T, envelope=data.ENVELOPE)
    probs = [data.draw(s, n=32).problem for s in (STEP, STEADY, SHORT_STEP)]
    t = np.arange(33) / 32
    XI, ET = np.meshgrid(t, t)
    rnd = [replace(p, g=data.Field(np.random.default_rng(11 + k), 0.25)(XI, ET)) for k, p in enumerate(probs)]
    calls = (probs, [replace(p, g=np.zeros_like(p.g)) for p in probs], rnd)
    modes, levels = [], []
    for ps in calls:
        modes.append([h for h, _up in ex.forward(ps)])
        levels.append(list(ex.last_levels))
    for i in range(len(probs)):
        assert GF.judge_l([m[i] for m in modes], [v[i] for v in levels])["met"]
    pb = lambda n: ports.basis(KIND, n, M)                   # noqa: E731
    for p in probs:
        r = forms.kernel_residual(ex, p, pb(p.n))
        assert np.abs(r).max() < 1e-12


@pytest.mark.parametrize("problem", ["steady", "step"])
def test_through_the_host_the_datum_drops_out(problem):
    T = _expert(4, std=0.02, seed=9)
    ex = LearnedExpert(T, envelope=data.ENVELOPE)
    b = bm.build("bend-3", 16, problem)
    res = {}
    for frame, bb in (("registered", b), ("lowered", bm.datum_shift(b, GF.DATUM_SHIFT_K))):
        L = bb.layout(KIND, M)
        mono = host.monolithic(L)
        cp = L.solve(ex)
        num, den = cp.energy_error(mono.fields)
        mx = max(float(np.max(np.abs(u - v))) for u, v in zip(mono.fields, cp.fields))
        span = (max(float(np.max(u)) for u in mono.fields) - min(float(np.min(u)) for u in mono.fields))
        res[frame] = (num, den, mx, span)
    (a, ref, ma, span), (z, _r2, mz, _s2) = res["registered"], res["lowered"]
    verdict = GF.judge_d(a, z, ref, ma, mz, span)
    assert verdict["met"], verdict


# ---------------------------------------------------------------------------
# the relative objective
# ---------------------------------------------------------------------------


def test_the_relative_objective_reads_one_per_term_at_TL0_and_zero_at_the_exact_fields():
    b, lab, _a = _batch()
    T = _expert(8)                                           # zero heads: TL0
    twin = tl.zero_heads_twin(T)
    ref = tl.reference(twin, b, lab)
    assert not ref["easy_H"].any() and not ref["easy_p"].any()
    with torch.no_grad():
        H, up = tl.call(T, b, strip_solve=False)
    o = tl.relative_objective(b, H, up, ref)
    assert torch.allclose(o["term_H"], torch.ones(3, dtype=torch.float64), rtol=0, atol=1e-12)
    assert torch.allclose(o["term_p"], torch.ones(3, dtype=torch.float64), rtol=0, atol=1e-12)
    ex = tl.relative_objective(b, lab["H"], lab["up"], ref)
    assert float(ex["term_H"].abs().max()) < 1e-9 and float(ex["term_p"].abs().max()) < 1e-9


def test_the_relative_objectives_gradient_is_the_label_free_one_over_TL0s_error():
    b, lab, _a = _batch()
    T = _expert(8, std=0.02, seed=13)
    twin = tl.zero_heads_twin(T)
    ref = tl.reference(twin, b, lab)
    T.train()
    H, up = tl.call(T, b, strip_solve=False)
    loss = tl.relative_objective(b, H, up, ref)["loss"].sum()
    g_rel = torch.autograd.grad(loss, [p for p in T.parameters()], allow_unused=True)
    H, up = tl.call(T, b, strip_solve=False)
    L = sorted(sum(tl.learned_groups().values(), []))
    free = ((tl.column_energies(b, H)[:, L].sum(-1) / ref["eps0"]).sum()
            + (tl.particular_energy(b, up) / ref["eps_p0"]).sum())
    g_free = torch.autograd.grad(free, [p for p in T.parameters()], allow_unused=True)
    worst = 0.0
    for x, y in zip(g_rel, g_free):
        if x is None or y is None:
            assert x is None and y is None
            continue
        worst = max(worst, float((x - y).abs().max() / max(float(y.abs().max()), 1e-300)))
    assert worst < 1e-10
    # the label-free gradient over all 64 columns is the same: the strips and C carry none
    H, up = tl.call(T, b, strip_solve=False)
    all64 = ((tl.column_energies(b, H).sum(-1) / ref["eps0"]).sum()
             + (tl.particular_energy(b, up) / ref["eps_p0"]).sum())
    g_all = torch.autograd.grad(all64, [p for p in T.parameters()], allow_unused=True)
    for x, y in zip(g_free, g_all):
        if x is not None:
            assert float((x - y).abs().max()) <= 1e-10 * max(float(y.abs().max()), 1e-300)


def test_the_guard_leaves_out_a_square_where_TL0_is_exact():
    import chartop_w362_diagnose as w362
    n = 32
    p = w362.rectangle(1.0, n)                               # a square of one material, steady, no source
    t = p.tensors()
    chans, scal, _rms = p.inputs()
    b = {k: torch.tensor(np.asarray(t[k], np.float64)[None]) for k in ("a11", "a12", "a22", "sigma", "w")}
    b["g"] = torch.zeros(1, n + 1, n + 1, dtype=torch.float64)
    b["inputs"] = torch.tensor(chans[None])
    b["scalars"] = torch.tensor(scal[None])
    b["s0"] = torch.tensor([0.0], dtype=torch.float64)
    ex = fe.Exact(p.matrix(), n)
    Q = ports.basis(KIND, n, M).Q
    Hx = ex.modes(Q)
    lab = {"H": torch.tensor(Hx.T.reshape(1, 4 * M, n - 1, n - 1)),
           "up": torch.zeros(1, n - 1, n - 1, dtype=torch.float64),
           "trace_Lambda": torch.tensor([float(np.trace(ex.port_matrix(Q, Hx)))], dtype=torch.float64),
           "up_energy": torch.tensor([0.0], dtype=torch.float64)}
    T = _expert(8, std=0.02)
    ref = tl.reference(tl.zero_heads_twin(T), b, lab)
    assert bool(ref["easy_H"][0]) and bool(ref["easy_p"][0])
    with torch.no_grad():
        H, up = tl.call(T, b, strip_solve=False)
    o = tl.relative_objective(b, H, up, ref)
    assert float(o["loss"][0]) == 0.0
    assert np.isnan(tl.rho([1.0], ref["eps0"].numpy(), lab["trace_Lambda"].numpy())[0])


# ---------------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------------


def test_rho_and_its_guard_on_hand_built_cases():
    e0 = np.array([2.0, 2.0, 2.0, 2.0, 1e-12])
    energy = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    r = tl.rho([2.0, 0.0, 4.0, 1.0, 0.0], e0, energy)
    assert r[0] == 0.0 and r[1] == 1.0 and r[2] == -1.0 and r[3] == 0.5 and np.isnan(r[4])
    s = tl.summarise_rho(r)
    assert s["pieces"] == 4 and s["nothing_to_learn"] == 1 and s["median"] == 0.25
    none = tl.summarise_rho([np.nan, np.nan])
    assert none["median"] is None and none["nothing_to_learn"] == 2


def test_the_attribution_and_its_floor():
    a = tl.attribution(e_U=0.05, e_TL0=0.03, e_TL=0.02, e_floor=0.01)
    assert a["order"].startswith("coarse first")
    assert abs(a["coarse_share"] - 0.5) < 1e-12 and abs(a["network_share"] - 0.25) < 1e-12
    assert abs(a["network_share_of_left"] - 0.5) < 1e-12 and a["at_floor"] is False
    f = tl.attribution(e_U=0.05, e_TL0=0.01 + 5e-5, e_TL=0.01, e_floor=0.01)
    assert f["at_floor"] and f["network_share"] is None and "nothing for the network" in f["note"]
    assert tl.attribution(0.05, 0.03, 0.02, 0.01, rounding=0.03)["at_floor"]


def test_the_equivalent_coarse_size():
    ncs, e = [4, 8, 16, 32], [1 / 4, 1 / 8, 1 / 16, 1 / 32]          # e = 1 / n_c exactly
    r = tl.equivalent_nc(ncs, e, 1 / 12)
    assert abs(r["n_c_star"] - 12.0) < 1e-9 and not r["extrapolated"]
    assert abs(tl.equivalent_nc(ncs, e, 1 / 64)["n_c_star"] - 64.0) < 1e-9
    assert tl.equivalent_nc(ncs, e, 1 / 64)["extrapolated"]
    assert abs(tl.equivalent_nc(ncs, e, 1 / 32)["n_c_star"] - 32.0) < 1e-9
    with pytest.raises(ValueError):
        tl.equivalent_nc(ncs, [0.1, 0.2, 0.05, 0.01], 0.03)


def test_kappa_kill_is_one_on_the_coarse_space_and_zero_off_it():
    b, _lab, _a = _batch()
    T = _expert(8)
    with torch.no_grad():
        tl.call(T, b, strip_solve=False)
    n, n_c = 32, 8
    p = co.prolongation_torch(n, n_c)
    g = torch.Generator().manual_seed(17)
    v = torch.randn(3, 2, n_c - 1, n_c - 1, dtype=torch.float64, generator=g)
    inV = co.prolong(v, p)
    k1 = tl.kappa_kill(T, inV)
    assert torch.allclose(k1, torch.ones(3, dtype=torch.float64), atol=1e-12)
    z = torch.randn(3, 2, n - 1, n - 1, dtype=torch.float64, generator=g)
    a11, a12, a22, sig = T.last_arrays
    Z = torch.zeros(3, 2, n + 1, n + 1, dtype=torch.float64)
    Z[..., 1:-1, 1:-1] = z
    AZ = co.apply_M(a11, a12, a22, sig, Z)[..., 1:-1, 1:-1]
    off = z - T.last_coarse.correction(AZ)                  # Q_c z: A-orthogonal to the coarse space
    assert float(tl.kappa_kill(T, off).abs().max()) < 1e-12
    mix = tl.kappa_kill(T, z)
    assert bool(((mix > 0) & (mix < 1)).all())


def test_the_dead_network_stop():
    assert not tl.dead(199, 1000, 0.0) and tl.dead(200, 1000, 0.049)
    assert not tl.dead(200, 1000, 0.05) and not tl.dead(900, 1000, None)


# ---------------------------------------------------------------------------
# train.train on a two-level arm
# ---------------------------------------------------------------------------


def _shards(folder, split, seeds, n=32):
    exs = [data.draw(s, n=n) for s in seeds]
    arrs = [data.arrays(e) for e in exs]
    out = {k: np.stack([a[k] for a in arrs]) for k in train.KEYS}
    out.update(seed=np.array(seeds), sup_mu=np.array([e.envelope["sup_abs_mu"] for e in exs]),
               logJ_range=np.array([e.envelope["logJ_range"] for e in exs]),
               s_grad=np.array([e.envelope["s_grad"] for e in exs]),
               fo=np.array([-1.0 if e.fo is None else e.fo for e in exs]),
               freq=np.array([e.freq for e in exs]), family=np.array([e.family for e in exs]),
               method=np.array([e.method for e in exs]))
    np.savez(folder / f"{split}_n{n}_0000.npz", **out)
    Q = ports.basis(KIND, n, M).Q
    lab = [data.labels_from_arrays(a, Q) for a in arrs]
    np.savez(folder / f"labels_{split}_n{n}_0000.npz",
             H=np.stack([x["H"].T.astype(np.float32) for x in lab]),
             up=np.stack([x["up"].astype(np.float32) for x in lab]),
             trace_Lambda=np.array([x["trace_Lambda"] for x in lab]),
             up_energy=np.array([x["up_energy"] for x in lab]), lseed=np.array(seeds))


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("tl-data")
    _shards(d, "train", [9_000_100, 9_000_101, 9_000_104, 9_000_105])     # all steps: both terms live
    _shards(d, "val", [9_000_106, 9_000_109])
    return d


def _arm(name, **kw):
    base = dict(form="split", size="0.3M", p_size="0.3M", bubble="corner", strips=True, steps=4,
                batch={32: 2}, val_every=2, coarse=8, objective="relative", lr=1e-3)
    base.update(kw)
    return train.Arm(name, **base)


def _log(out):
    return [json.loads(line) for line in open(out / "log.jsonl", encoding="utf-8")]


def test_a_two_level_arm_trains_logs_rho_and_loads(dataset, tmp_path):
    out = tmp_path / "tl"
    train.train(_arm("tl", steps=20, val_every=2), str(dataset), str(out), device="cpu", log_every=1)
    log = _log(out)
    first = next(r for r in log if "loss" in r)
    assert abs(first["loss"] - 2.0) < 1e-9                   # TL0: one per term, before any update
    vals = [r for r in log if "val_rho_median" in r]
    assert vals[0]["step"] == 2 and all(r["val_rho_pieces"] == 2 for r in vals)
    # the dead-network stop may end a run this short, but only at a validation past 20% of it
    stops = [r for r in log if "stopped" in r]
    assert all(s["step"] >= 4 and "dead network" in s["stopped"] for s in stops)
    if stops:
        assert vals[-1]["step"] == stops[0]["step"] and vals[-1]["val_rho_median"] < tl.DEAD_RHO
    e = train.load_expert(str(out / "best.pt"))
    assert isinstance(e, tl.TwoLevelExpert) and e.n_c == 8
    assert json.load(open(out / "arm.json", encoding="utf-8"))["config"]["n_c"] == 8


def test_the_dead_network_stops_the_run(dataset, tmp_path):
    out = tmp_path / "dead"
    train.train(_arm("dead", lr=0.0, steps=10, val_every=2), str(dataset), str(out), device="cpu", log_every=1)
    log = _log(out)
    stop = [r for r in log if "stopped" in r]
    assert len(stop) == 1 and stop[0]["step"] == 2 and "dead network" in stop[0]["stopped"]
    assert [r["val_rho_median"] for r in log if "val_rho_median" in r] == [0.0]


def test_the_arm_refuses_what_it_cannot_do(dataset, tmp_path):
    with pytest.raises(ValueError):
        train.train(_arm("x", coarse=None), str(dataset), str(tmp_path / "x"), device="cpu")
    with pytest.raises(ValueError):
        train.train(_arm("y", head_init="ones"), str(dataset), str(tmp_path / "y"), device="cpu")


def test_init_from_weights_with_zeroed_heads_starts_at_TL0(dataset, tmp_path):
    src = tmp_path / "src"
    train.train(_arm("src", objective="label-free", coarse=None, steps=2), str(dataset), str(src),
                device="cpu", log_every=1)
    for head_init, at_tl0 in (("zero", True), ("loaded", False)):
        out = tmp_path / head_init
        train.train(_arm(head_init, init_from=str(src / "last.pt"), head_init=head_init, steps=2),
                    str(dataset), str(out), device="cpu", log_every=1)
        first = next(r for r in _log(out) if "loss" in r)
        assert (abs(first["loss"] - 2.0) < 1e-9) is at_tl0, (head_init, first["loss"])

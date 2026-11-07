"""The options built after the W358 registration's one evaluation (the forms page,
section 11), checked untrained with random weights, and in two-step training runs:

- B1, ``hats=``: the kernel columns from the network (the default), the plain square's
  ``H0``, or exact solves of the piece's own operator;
- B2, ``train.Arm.mode_prior``: the objective weights the modes' errors by a prior
  covariance of a piece's traces, the training generator's own source model;
- B3, the ``"unet-spectral"`` backbone: a cosine-basis path at the U-Net's coarsest level.

Each is off by default and leaves the default bit for bit; each keeps the kernel-exact
forms exact."""
from __future__ import annotations

import copy
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import scipy.sparse.linalg as spla

torch = pytest.importorskip("torch")

from atlas.chartop import data, fe, forms, host, model, ports, train
from atlas.chartop.expert import LearnedExpert

torch.set_num_threads(1)

M = 16
FIXED = ("principal", "classical")


def _randomise(net, std, seed=0):
    torch.manual_seed(seed)
    for prm in net.parameters():
        torch.nn.init.normal_(prm, 0.0, std)
    return net.eval()


def _operator_batch(n, seed=0):
    """Three examples, steady and two steps (s0 = 0, 3, 100): random channels, element
    weights, a source with a level, and a positive tensor field with ``sigma = s0 w``."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(3, 5, n + 1, n + 1, generator=gen)
    s = torch.randn(3, 2, generator=gen)
    s0 = torch.tensor([0.0, 3.0, 100.0], dtype=torch.float64)
    w = torch.rand(3, n, n, generator=gen, dtype=torch.float64) + 0.5
    g = torch.randn(3, n + 1, n + 1, generator=gen, dtype=torch.float64) + 0.7
    gen = torch.Generator().manual_seed(seed + 1)
    a11 = torch.rand(3, n, n, generator=gen, dtype=torch.float64) + 0.5
    a22 = torch.rand(3, n, n, generator=gen, dtype=torch.float64) + 0.5
    a12 = 0.8 * (torch.rand(3, n, n, generator=gen, dtype=torch.float64) - 0.5) * torch.sqrt(a11 * a22)
    return {"inputs": x, "scalars": s, "s0": s0, "w": w, "g": g, "a11": a11, "a12": a12,
            "a22": a22, "sigma": w * s0[:, None, None]}


def _kernel_columns(n):
    return [int(k) for k in np.flatnonzero(forms.kernel_coefficients("sine-hat", n, M))]


def _same_state(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same_state(x, y) for x, y in zip(a, b))
    return a == b


# ---------------------------------------------------------------------------
# B1: the hats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strips", [False, True])
@pytest.mark.parametrize("form", forms.FORMS)
def test_the_hats_are_learned_by_default_and_the_default_unchanged(form, strips):
    """``hats="learned"`` is the default: the same parameters from the same seed, and for
    any weights the same answer bit for bit; it needs an operator only for the strips.
    With the same weights, the other options answer bit for bit the same in every column
    but the kernel's, and the same particular field, and they differ in the kernel's."""
    n = 32
    b = _operator_batch(n)
    kw = dict(m=M, bubble="corner", strips=strips)
    torch.manual_seed(0)
    d1 = forms.KernelExpert(form, "unet", "0.3M", **kw)
    torch.manual_seed(0)
    d2 = forms.KernelExpert(form, "unet", "0.3M", hats="learned", **kw)
    assert _same_state(d1.state_dict(), d2.state_dict())
    assert d1.config()["hats"] == "learned" and d1.needs_operator == strips
    e = _randomise(d1, 0.3)
    d2.load_state_dict(e.state_dict())
    d2.eval()
    with torch.no_grad():
        Ha, ua = forms.call(e, b)
        Hb, ub = forms.call(d2, b)
    assert torch.equal(Ha, Hb) and torch.equal(ua, ub)
    kc = _kernel_columns(n)
    other = [k for k in range(4 * M) if k not in kc]
    for hats in FIXED:
        f = forms.KernelExpert(form, "unet", "0.3M", hats=hats, **kw)
        f.load_state_dict(e.state_dict())
        f.eval()
        assert f.needs_operator == (strips or hats == "classical")
        with torch.no_grad():
            Hf, uf = forms.call(f, b)
        assert torch.equal(Hf[:, other], Ha[:, other]) and torch.equal(uf, ua)
        assert float((Hf[:, kc] - Ha[:, kc]).abs().max()) > 1e-6


@pytest.mark.parametrize("hats", FIXED)
@pytest.mark.parametrize("form", forms.FORMS)
def test_the_fixed_hats_keep_the_forms_exact(form, hats):
    """S's configuration (corner bubble, strips) with random weights (std 0.3), steady and
    for two steps.  The uniform state is a fixed point to float64 rounding.  Steady, the
    kernel columns are the plain square's ``H0`` (``"principal"``, to its float32 rounding)
    or the piece's exact hats (``"classical"``, to 1e-10).  The network's kernel rows are
    not read: other weights in them change nothing, bit for bit."""
    n = 32
    b = _operator_batch(n)
    e = _randomise(forms.KernelExpert(form, "unet", "0.3M", m=M, bubble="corner", strips=True, hats=hats), 0.3)
    with torch.no_grad():
        H, up = forms.call(e, b)
    assert torch.isfinite(H).all() and torch.isfinite(up).all()
    ks = forms.kernel_square(n, "sine-hat", M, "cpu")
    pre = float(e.last_kernel_residual.abs().max())
    assert float(forms.residual(H, e.last_level, b["s0"], ks).abs().max()) <= 1e-14 * (1.0 + pre)
    kc = _kernel_columns(n)
    if hats == "principal":
        assert float((H[0, kc] - e.last_principal[0, kc]).abs().max()) <= 1e-5
    else:
        A_I, M_IG = forms.batch_operators(b)[0]
        Q = ports.basis("sine-hat", n, M).Q
        exact = -spla.splu(A_I.tocsc()).solve(np.asarray(M_IG @ Q[:, kc]))
        assert np.abs(H[0, kc].numpy().reshape(len(kc), -1).T - exact).max() <= 1e-10
        assert e.last_hat_seconds > 0.0
    rows = [i for i, k in enumerate(e.learned) if k in kc]
    assert len(rows) == 4
    with torch.no_grad():
        e.net.head.weight[rows] += 1.0
        e.net.head.bias[rows] += 1.0
        H2, up2 = forms.call(e, b)
    assert torch.equal(H2, H) and torch.equal(up2, up)


@pytest.mark.parametrize("hats", FIXED)
def test_the_split_forms_modes_with_fixed_hats_never_see_the_load(hats):
    """Gate L's identity: the split form's modes and level field from calls with the
    batch's own source, none, and another are the same bit for bit; the classical hats
    read the operator, never the load."""
    n = 32
    b = _operator_batch(n)
    e = _randomise(forms.KernelExpert("split", "unet", "0.3M", m=M, bubble="corner", strips=True, hats=hats), 0.3)
    out = []
    for g in (b["g"], torch.zeros_like(b["g"]), 3.0 * b["g"] - 1.0):
        with torch.no_grad():
            H, _up = forms.call(e, dict(b, g=g))
        out.append((H, e.last_level.clone()))
    for H, lev in out[1:]:
        assert torch.equal(H, out[0][0]) and torch.equal(lev, out[0][1])


def test_the_classical_hats_through_the_host():
    """bend-3 steady at n = 32 through `LearnedExpert`, S's configuration with classical hats
    and random weights: one call per piece; each response's kernel columns are the
    classical expert's to 1e-10; the uniform state stays a fixed point through the host;
    lowering every temperature by 300 K lowers the coupled answer by 300 K to rounding
    (gate D's identity); and the hats' time is counted."""
    from atlas.chartop import benchmarks as bm
    bk = bm.bend3(32, "steady")
    net = _randomise(forms.KernelExpert("split", "unet", "0.3M", bubble="corner", strips=True,
                                        hats="classical"), 0.02)
    ex = LearnedExpert(net, envelope=data.ENVELOPE)
    L = bk.layout("sine-hat", M)
    cp = L.solve(ex)
    assert cp.calls == len(bk.problems) and net.last_hat_seconds > 0.0
    cl = L.solve(host.ClassicalExpert())
    kc = _kernel_columns(32)
    for i in range(len(bk.problems)):
        assert np.abs(cp.responses[i].H[:, kc] - cl.responses[i].H[:, kc]).max() <= 1e-10
    p = L.problems[0]
    pb = L.pb(p.n)
    exs = fe.Exact(p.matrix(), p.n)
    km = forms.kernel_measures(forms.kernel_residual(ex, p, pb), exs.A_I, float(np.trace(exs.port_matrix(pb.Q))))
    assert km["e_K_inf"] < 1e-12
    low = bm.datum_shift(bk, -300.0).layout("sine-hat", M).solve(ex)
    dev = max(float(np.abs(v - u + 300.0).max()) for u, v in zip(cp.fields, low.fields))
    assert dev <= 1e-9 * 300.0


def test_the_hats_option_refusals_and_save_load(tmp_path):
    """Unknown options and Stage 0's forms refuse other hats; a classical-hat expert called
    without operators is refused; the option survives `train.load_expert`, and a
    configuration written before it (no ``hats`` key) loads as learned hats."""
    with pytest.raises(ValueError):
        forms.KernelExpert("split", "unet", "0.3M", hats="exact")
    with pytest.raises(ValueError):
        forms.make("unet", "0.3M", "bubble", hats="classical")
    e = forms.KernelExpert("split", "unet", "0.3M", hats="classical")
    b = _operator_batch(32)
    with pytest.raises(ValueError, match="operator"):
        e(b["inputs"], b["scalars"], b["s0"], w=b["w"], g=b["g"])
    path = tmp_path / "e.pt"
    torch.save({"model": e.state_dict(), "config": e.config()}, path)
    assert train.load_expert(str(path)).hats == "classical"
    old = forms.KernelExpert("split", "unet", "0.3M")
    cfg = {k: v for k, v in old.config().items() if k != "hats"}
    torch.save({"model": old.state_dict(), "config": cfg}, path)
    assert train.load_expert(str(path)).hats == "learned"


# ---------------------------------------------------------------------------
# B2: the objective's mode covariance
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def labelled():
    """Two training-generator examples at n = 32 as a float64 batch, with exact labels."""
    exs = [data.draw(data.SEEDS["train"] + 930 + i, n=32) for i in range(2)]
    arrs = [data.arrays(e) for e in exs]
    b = {k: torch.tensor(np.stack([a[k] for a in arrs]), dtype=torch.float64)
         for k in ("a11", "a12", "a22", "sigma", "w", "g")}
    b["inputs"] = torch.tensor(np.stack([a["inputs"] for a in arrs]), dtype=torch.float32)
    b["scalars"] = torch.tensor(np.stack([a["scalars"] for a in arrs]), dtype=torch.float32)
    b["s0"] = torch.tensor(np.array([a["s0"] for a in arrs]), dtype=torch.float64)
    Q = ports.basis("sine-hat", 32, M).Q
    lab = [data.labels_from_arrays(a, Q) for a in arrs]
    labels = {"H": torch.tensor(np.stack([x["H"].T.reshape(4 * M, 31, 31) for x in lab])),
              "up": torch.tensor(np.stack([x["up"].reshape(31, 31) for x in lab])),
              "trace_Lambda": torch.tensor([x["trace_Lambda"] for x in lab], dtype=torch.float64),
              "up_energy": torch.tensor([x["up_energy"] for x in lab], dtype=torch.float64)}
    return b, labels


@pytest.mark.parametrize("objective", ["label-free", "supervised"])
def test_the_identity_covariance_is_the_old_objective_bit_for_bit(labelled, objective):
    """``mix = I`` gives the loss, its normaliser, the Gram trace and the gradient of every
    parameter bit for bit as no ``mix`` (the default)."""
    b, lab = labelled
    e = _randomise(forms.KernelExpert("split", "unet", "0.3M", m=M, bubble="corner"), 0.02)
    e.train()
    H, up = forms.call(e, b)
    o1 = model.objectives(b, H, up, "sine-hat", M, lab, objective, H0=e.last_principal)
    o2 = model.objectives(b, H, up, "sine-hat", M, lab, objective, H0=e.last_principal,
                          mix=torch.eye(4 * M, dtype=torch.float64))
    for k in ("loss", "ZH", "trace_gram", "e_H", "e_p"):
        assert torch.equal(o1[k], o2[k]), k
    prm = [p for p in e.parameters() if p.requires_grad]
    g1 = torch.autograd.grad(o1["loss"].sum(), prm, retain_graph=True, allow_unused=True)
    g2 = torch.autograd.grad(o2["loss"].sum(), prm, allow_unused=True)
    assert all((a is None and c is None) or torch.equal(a, c) for a, c in zip(g1, g2))


def test_the_prior_keeps_t26(labelled):
    """For the prior's ``F``, label-free minus supervised is the same constant for two
    different networks' modes: ``tr(Sigma Lambda~) = tr(Sigma Lambda*) + tr(Sigma E^T A E)``,
    so the exact modes stay the minimiser; and the prior's loss differs from the white one."""
    b, lab = labelled
    F = torch.tensor(forms.mode_mixing("generator-field", "sine-hat", 32, M, tuple(range(4 * M))))
    gaps = []
    for seed in (0, 1):
        e = _randomise(forms.KernelExpert("split", "unet", "0.3M", m=M, bubble="corner"), 0.05, seed)
        with torch.no_grad():
            H, up = forms.call(e, b)
        lf = model.objectives(b, H, up, "sine-hat", M, lab, "label-free", H0=e.last_principal, mix=F)
        su = model.objectives(b, H, up, "sine-hat", M, lab, "supervised", H0=e.last_principal, mix=F)
        white = model.objectives(b, H, up, "sine-hat", M, lab, "label-free", H0=e.last_principal)
        gaps.append(((lf["loss"] - su["loss"]) * lf["ZH"]).numpy())
        assert float((lf["loss"] - white["loss"]).abs().min()) > 1e-6
    assert np.abs(gaps[0] - gaps[1]).max() <= 1e-9 * np.abs(gaps[0]).max()


@pytest.mark.parametrize("k", range(8))
def test_the_trace_prior_commutes_with_the_squares_symmetries(k):
    """``P Sigma P^T = Sigma`` for `model.d4_ports`' signed permutation ``P``, so D4
    augmentation does not change the weighting; and `forms.mode_mixing` is the prior on
    the trained columns, floored, with no terms between them and the others."""
    n = 32
    S = forms.trace_covariance("generator-field", "sine-hat", n, M)
    perm, sign = model.d4_ports("sine-hat", n, M, k)
    P = np.zeros((4 * M, 4 * M))
    P[perm, np.arange(4 * M)] = sign
    assert np.abs(P @ S @ P.T - S).max() <= 1e-12 * np.abs(S).max()
    tr_ = forms.learned_columns("sine-hat", M)
    oth = [c for c in range(4 * M) if c not in tr_]
    F = forms.mode_mixing("generator-field", "sine-hat", n, M, tr_)
    FF = F @ F.T
    assert np.all(FF[np.ix_(tr_, oth)] == 0.0)
    B = S[np.ix_(tr_, tr_)]
    lam = np.linalg.eigvalsh(B)
    want = B + forms.MODE_PRIOR_FLOOR * lam.max() * np.eye(len(tr_))
    assert lam.min() > -1e-12 * lam.max()
    assert np.abs(FF[np.ix_(tr_, tr_)] - want).max() <= 1e-12 * lam.max()


# ---------------------------------------------------------------------------
# B3: the spectral path
# ---------------------------------------------------------------------------


def test_the_spectral_path_is_off_by_default_and_unchanged():
    """``spectral=0`` is the U-Net as before: the same parameters from the same seed and the
    same output bit for bit.  With the path on, the other parameters are drawn the same,
    and a network without it loads into it (``strict=False``) as the same function, bit
    for bit, since ``C_out`` starts at zero.  A spectral layer without ``cosine`` is the
    layer as before."""
    w = model.WIDTHS["0.3M"]
    torch.manual_seed(0)
    a = model.UNet(w, 7, 4)
    torch.manual_seed(0)
    b = model.UNet(w, 7, 4, spectral=0)
    torch.manual_seed(0)
    c = model.UNet(w, 7, 4, spectral=model.SPECTRAL_K)
    assert _same_state(a.state_dict(), b.state_dict())
    sa, sc = a.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sc[k]) for k in sa) and set(sc) - set(sa)
    _randomise(a, 0.05)
    b.load_state_dict(a.state_dict())
    c.load_state_dict(a.state_dict(), strict=False)
    for n in (32, 64):
        x = torch.randn(2, 4, n + 1, n + 1)
        s = torch.randn(2, 2)
        with torch.no_grad():
            ya = a(x, s)
            assert torch.equal(ya, b.eval()(x, s)) and torch.equal(ya, c.eval()(x, s))
    torch.manual_seed(0)
    l1 = model.SpectralLayer(4, 5, 3)
    torch.manual_seed(0)
    l2 = model.SpectralLayer(4, 5, 3, cosine=False)
    assert _same_state(l1.state_dict(), l2.state_dict())
    x = torch.randn(2, 4, 17, 17)
    with torch.no_grad():
        assert torch.equal(l1(x), l2(x))


def test_the_spectral_path_runs_under_bfloat16_autocast():
    """Decision D4 runs the network in bfloat16 autocast on a GPU.  Autocast lowers the
    cosine path's real mixing (a matmul) to bfloat16, and `torch.complex` takes no
    bfloat16: the box's first pricing died there (2026-11-07).  The path computes its
    spectrum in float32 with autocast off, so under the CPU's bfloat16 autocast the
    spectral U-Net answers, finite and near its float32 answer; and with autocast off the
    path is what it was, bit for bit, since float32 in means float32 throughout."""
    torch.manual_seed(0)
    net = _randomise(model.UNet(model.WIDTHS["0.3M"], 3, 4, spectral=model.SPECTRAL_K), 0.05)
    x = torch.randn(2, 4, 33, 33)
    s = torch.randn(2, 2)
    with torch.no_grad():
        y32 = net(x, s)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            y16 = net(x, s)
    assert torch.isfinite(y16).all()
    assert float((y16.float() - y32).abs().max()) <= 0.05 * float(y32.abs().max())
    lay = model.SpectralLayer(4, 4, 8, cosine=True)
    xs = torch.randn(2, 4, 9, 9)
    with torch.no_grad():
        ref = lay(xs)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            low = lay(xs.to(torch.bfloat16))
    assert low.dtype == torch.bfloat16 and torch.isfinite(low).all()
    assert float((low.float() - ref).abs().max()) <= 0.05 * float(ref.abs().max())


def _act(net, fx, fy, t):
    """The weights of ``net`` under symmetry (fx, fy, t): every 3 x 3 kernel moved as a
    field is, and the cosine weights transposed when the symmetry transposes."""
    g = copy.deepcopy(net)
    with torch.no_grad():
        for name, p in g.named_parameters():
            if p.dim() == 4 and tuple(p.shape[-2:]) == (3, 3):
                p.copy_(model._spatial(p.clone(), fx, fy, t))
            elif name.endswith("spec.wc") and t:
                p.copy_(p.clone().transpose(-1, -2))
    return g


@pytest.mark.parametrize("k", range(8))
def test_the_spectral_unet_commutes_with_the_squares_symmetries(k):
    """With every weight random (head and ``C_out`` included), symmetry k of the input, with
    the symmetry's action on the weights, gives symmetry k of the output to float32
    rounding, with and without the path: the path keeps the U-Net's D4 structure, on
    which the D4 augmentation relies."""
    fx, fy, t = k & 1, (k >> 1) & 1, (k >> 2) & 1
    gen = torch.Generator().manual_seed(k)
    x = torch.randn(2, 4, 33, 33, generator=gen)
    s = torch.randn(2, 2, generator=gen)
    for spectral in (0, model.SPECTRAL_K):
        net = _randomise(model.UNet(model.WIDTHS["0.3M"], 3, 4, spectral=spectral), 0.1, seed=k)
        with torch.no_grad():
            y = net(x, s)
            yg = _act(net, fx, fy, t).eval()(model._spatial(x, fx, fy, t), s)
        assert float(y.abs().max()) > 1e-3
        assert float((yg - model._spatial(y, fx, fy, t)).abs().max()) <= 1e-5 * float(y.abs().max())


def test_the_spectral_backbone_keeps_the_forms_exact_and_reaches_the_whole_piece():
    """`forms.KernelExpert` on ``"unet-spectral"`` in S's configuration, random weights: the
    uniform state is a fixed point to rounding and the answers are finite (the bubble
    keeps the traces exact as for any backbone).  At n = 128 the centre node's output
    reads an input node at a corner, 64 nodes away, through the path; the U-Net's
    receptive field (58 nodes) cannot, and its gradient there is exactly zero."""
    n = 32
    b = _operator_batch(n)
    for form in forms.FORMS:
        e = _randomise(forms.KernelExpert(form, "unet-spectral", "0.3M", m=M, bubble="corner", strips=True), 0.05)
        assert e.config()["backbone"] == "unet-spectral" and e.net.spectral == model.SPECTRAL_K
        with torch.no_grad():
            H, up = forms.call(e, b)
        assert torch.isfinite(H).all() and torch.isfinite(up).all()
        ks = forms.kernel_square(n, "sine-hat", M, "cpu")
        pre = float(e.last_kernel_residual.abs().max())
        assert float(forms.residual(H, e.last_level, b["s0"], ks).abs().max()) <= 1e-14 * (1.0 + pre)
    reach = {}
    for spectral in (0, model.SPECTRAL_K):
        net = _randomise(model.UNet(model.WIDTHS["0.3M"], 1, 4, spectral=spectral), 0.1)
        x = torch.randn(1, 4, 129, 129, requires_grad=True)
        net(x, torch.randn(1, 2))[0, 0, 64, 64].backward()
        reach[spectral] = float(x.grad[0, :, 0, 0].abs().sum())
    assert reach[0] == 0.0 and reach[model.SPECTRAL_K] > 0.0


# ---------------------------------------------------------------------------
# the options in training
# ---------------------------------------------------------------------------


def _shards(folder, split, seeds, labels=False, n=32):
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
    if labels:
        Q = ports.basis("sine-hat", n, M).Q
        lab = [data.labels_from_arrays(a, Q) for a in arrs]
        np.savez(folder / f"labels_{split}_n{n}_0000.npz",
                 H=np.stack([x["H"].T.astype(np.float32) for x in lab]),
                 up=np.stack([x["up"].astype(np.float32) for x in lab]),
                 trace_Lambda=np.array([x["trace_Lambda"] for x in lab]),
                 up_energy=np.array([x["up_energy"] for x in lab]), lseed=np.array(seeds))


def test_the_options_train_save_and_load(tmp_path):
    """Two steps of S's form with each option: the defaults are none of them; the prior
    changes the weights the steps reach (the objective changed) and is recorded with the
    arm; the classical hats and the spectral backbone train, save and load with their
    options."""
    d = tmp_path / "data"
    d.mkdir()
    _shards(d, "train", [data.SEEDS["train"] + 940 + i for i in range(4)])
    _shards(d, "val", [data.SEEDS["val"] + 940 + i for i in range(2)], labels=True)

    def run(name, **kw):
        arm = train.Arm(name, form="split", size="0.3M", p_size="0.3M", bubble="corner", strips=True,
                        steps=2, batch={32: 2}, val_every=2, strips_in_training=False, **kw)
        out = tmp_path / name
        train.train(arm, str(d), str(out), device="cpu", log_every=1)
        rec = json.load(open(out / "arm.json", encoding="utf-8"))
        return torch.load(out / "last.pt", map_location="cpu", weights_only=False), rec, out

    base, rb, _ = run("base")
    prior, rp, _ = run("prior", mode_prior="generator-field")
    assert rb["arm"]["mode_prior"] is None and rp["arm"]["mode_prior"] == "generator-field"
    assert not _same_state(base["model"], prior["model"])
    a = train.Arm("a")
    assert a.mode_prior is None and a.hats == "learned" and a.backbone == "unet"
    _hs, rh, out_h = run("hats", hats="classical")
    assert rh["config"]["hats"] == "classical" and train.load_expert(str(out_h / "best.pt")).hats == "classical"
    _ss, rs, out_s = run("spectral", backbone="unet-spectral")
    e = train.load_expert(str(out_s / "best.pt"))
    assert rs["config"]["backbone"] == "unet-spectral" and e.net.spectral == model.SPECTRAL_K

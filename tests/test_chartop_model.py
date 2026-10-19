"""The chart operator's torch side against its numpy side (Stage 0, W355): the DST-I,
the plain square's principal part, the energies and loads, the label-free objective
(T26 in float64), the D4 augmentation's tensor rules, and the three output forms.
An untrained network is used only for shapes and for the exact-trace property.

After Stage 0 (W356, W357): the kernel-exact forms of `forms`, with random weights
only.  The uniform state is a fixed point whatever the weights, steady or not; the
datum drops out of the particular field; the split form's modes never see the load;
and the identity the forms enforce holds for exact classical solves."""
from __future__ import annotations

import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import scipy.fft as sfft

torch = pytest.importorskip("torch")

from atlas.chartop import chart as ch
from atlas.chartop import data, fe, forms, host, model, ports, principal
from atlas.chartop.expert import LearnedExpert
from atlas.chartop.piece import PieceProblem, fo_scalars

torch.set_num_threads(1)


def _batch_from(examples):
    arrs = [data.arrays(e) for e in examples]
    out = {k: torch.tensor(np.stack([a[k] for a in arrs]), dtype=torch.float64)
           for k in ("a11", "a12", "a22", "sigma", "w", "g")}
    out["inputs"] = torch.tensor(np.stack([a["inputs"] for a in arrs]), dtype=torch.float32)
    out["scalars"] = torch.tensor(np.stack([a["scalars"] for a in arrs]), dtype=torch.float32)
    out["s0"] = torch.tensor(np.array([a["s0"] for a in arrs]), dtype=torch.float64)
    return out


@pytest.fixture(scope="module")
def examples():
    # two of one resolution, drawn from the training generator's own seeds
    seeds, out = data.SEEDS["train"], []
    s = 0
    while len(out) < 2:
        e = data.draw(seeds + 900 + s, n=32)
        out.append(e)
        s += 1
    return out


def test_dst1_is_scipys():
    x = torch.randn(3, 7, 15, dtype=torch.float64)
    y = model.dst1(x, -1)
    ref = sfft.dst(x.numpy(), type=1, axis=-1, norm="ortho")
    assert np.max(np.abs(y.numpy() - ref)) < 1e-13
    assert torch.allclose(model.dst1(y, -1), x, atol=1e-13)


@pytest.mark.parametrize("s0", [0.0, 37.0])
def test_principal_part_matches_numpy(s0):
    n = 32
    ps = model.PlainSquare(n, "sine-hat", 16, torch.device("cpu"))
    H0 = ps.principal(torch.tensor([s0], dtype=torch.float64))[0]
    ref = principal.modes(n, ports.basis("sine-hat", n, 16).Q, s0)
    got = H0.reshape(H0.shape[0], -1).T.numpy()
    assert np.max(np.abs(got - ref)) < 1e-12 * np.max(np.abs(ref))


def test_energy_mass_and_load_match_numpy(examples):
    b = _batch_from(examples)
    n = examples[0].n
    U = torch.randn(2, 3, n + 1, n + 1, dtype=torch.float64)
    Et = model.energy(b["a11"], b["a12"], b["a22"], b["sigma"], U)
    for i, e in enumerate(examples):
        # the stored coefficients are float32: assemble the reference from them
        M = fe.assemble(n, b["a11"][i].numpy(), b["a12"][i].numpy(), b["a22"][i].numpy(),
                        b["sigma"][i].numpy())
        for k in range(3):
            u = U[i, k].numpy().ravel()
            assert abs(Et[i, k].item() - u @ (M @ u)) < 1e-10 * abs(u @ (M @ u))
        Mw = fe.mass(n, e.problem.element_fields()["Jt"])
        bt = model._load(b["w"], b["g"])[i].numpy().ravel()
        assert np.max(np.abs(bt - Mw @ e.problem.g.ravel())) < 1e-6 * np.max(np.abs(bt))
        u = U[i, 0].numpy().ravel()
        mf = model.mass_form(b["w"], b["g"], U)[i, 0].item()
        assert abs(mf - e.problem.g.ravel() @ (Mw @ u)) < 1e-5 * max(1.0, abs(mf))


def test_label_free_minus_supervised_is_constant(examples):
    """T26 in the training code's own arithmetic (float64)."""
    b = _batch_from(examples)
    n = examples[0].n
    Q = ports.basis("sine-hat", n, 16).Q
    lab = [data.labels(e, Q) for e in examples]
    labels = {"H": torch.tensor(np.stack([l["H"].T.reshape(-1, n - 1, n - 1) for l in lab])),
              "up": torch.tensor(np.stack([l["up"].reshape(n - 1, n - 1) for l in lab])),
              "trace_Lambda": torch.tensor([l["trace_Lambda"] for l in lab]),
              "up_energy": torch.tensor([l["up_energy"] for l in lab])}
    diffs = []
    for _ in range(3):
        H = labels["H"] + 0.1 * torch.randn_like(labels["H"])
        up = labels["up"] + 0.1 * torch.randn_like(labels["up"])
        o1 = model.objectives(b, H, up, "sine-hat", 16, labels, "label-free")
        o2 = model.objectives(b, H, up, "sine-hat", 16, labels, "supervised")
        diffs.append((o1["loss"] - o2["loss"]).numpy())
    diffs = np.array(diffs)
    assert np.max(np.ptp(diffs, axis=0)) < 1e-9 * np.max(np.abs(diffs) + 1.0)
    # and the exact fields score zero error
    o = model.objectives(b, labels["H"], labels["up"], "sine-hat", 16, labels)
    assert o["e_H"].max().item() < 1e-10 and o["e_p"].max().item() < 1e-10


@pytest.mark.parametrize("k", range(8))
def test_d4_preserves_the_energy(examples, k):
    """E'(T u) = E(u): the transformed tensors are the transformed problem's."""
    b = _batch_from(examples)
    n = examples[0].n
    U = torch.randn(2, 2, n + 1, n + 1, dtype=torch.float64)
    E0 = model.energy(b["a11"], b["a12"], b["a22"], b["sigma"], U)
    bt = model.d4(b, k)
    Ut = model._spatial(U, k & 1, (k >> 1) & 1, (k >> 2) & 1)
    E1 = model.energy(bt["a11"], bt["a12"], bt["a22"], bt["sigma"], Ut)
    assert torch.allclose(E0, E1, rtol=1e-12)


@pytest.mark.parametrize("k", range(8))
def test_d4_mu_rule_matches_the_tensor_rule(k):
    """For a constant mu the transformed mu, through J G^-1(mu), is the transformed
    tensor."""
    n = 8
    mu = 0.3 + 0.2j
    a11, a12, a22 = ch.tensor_from_mu(np.full((n, n), mu))
    b = {"inputs": torch.zeros(1, 5, n + 1, n + 1), "a11": torch.tensor(a11)[None],
         "a12": torch.tensor(a12)[None], "a22": torch.tensor(a22)[None],
         "sigma": torch.zeros(1, n, n), "w": torch.ones(1, n, n), "g": torch.zeros(1, n + 1, n + 1)}
    b["inputs"][:, 1] = mu.real
    b["inputs"][:, 2] = mu.imag
    bt = model.d4(b, k)
    mu_t = complex(bt["inputs"][0, 1, 0, 0].item(), bt["inputs"][0, 2, 0, 0].item())
    t11, t12, t22 = ch.tensor_from_mu(np.array([mu_t]))
    assert np.allclose([t11[0], t12[0], t22[0]],
                       [bt["a11"][0, 0, 0].item(), bt["a12"][0, 0, 0].item(), bt["a22"][0, 0, 0].item()],
                       atol=1e-6)


@pytest.mark.parametrize("kind", ["sine-hat", "cosine"])
@pytest.mark.parametrize("k", [1, 2, 4, 5, 7])
def test_d4_labels_are_the_transformed_problems_exact_modes(examples, kind, k):
    """Augmenting a supervised batch: the signed permutation of `d4_ports` applied to
    the original's exact fields gives the transformed problem's exact fields."""
    e = examples[0]
    n = e.n
    Q = ports.basis(kind, n, 16).Q
    lab0 = data.labels(e, Q)
    b = _batch_from([e])
    lab = {"H": torch.tensor(lab0["H"].T.reshape(1, -1, n - 1, n - 1)),
           "up": torch.tensor(lab0["up"].reshape(1, n - 1, n - 1))}
    lt = model.d4_labels(lab, kind, 16, k)
    bt = model.d4(b, k)
    arr = {kk: bt[kk][0].numpy() for kk in ("a11", "a12", "a22", "sigma", "w", "g")}
    ref = data.labels_from_arrays(arr, Q)
    H_ref = ref["H"].T.reshape(-1, n - 1, n - 1)
    assert np.max(np.abs(lt["H"][0].numpy() - H_ref)) < 1e-9 * np.max(np.abs(H_ref))
    assert np.max(np.abs(lt["up"][0].numpy() - ref["up"].reshape(n - 1, n - 1))) < 1e-9 * (
        1 + np.max(np.abs(ref["up"])))


@pytest.mark.parametrize("form", model.FORMS)
@pytest.mark.parametrize("backbone,size", [("unet", "0.3M"), ("spectral", "1M")])
def test_shapes_and_untrained_output(form, backbone, size):
    for n in (32, 64):
        e = model.Expert(backbone, size, form)
        x = torch.randn(2, 5, n + 1, n + 1)
        s = torch.randn(2, 2)
        s0 = torch.tensor([0.0, 5.0], dtype=torch.float64)
        with torch.no_grad():
            H, up = e(x, s, s0)
        assert H.shape == (2, 64, n - 1, n - 1) and up.shape == (2, n - 1, n - 1)
        if form != "direct":
            # zero-initialised heads: the untrained bubble and fast-solve experts return
            # the plain square's exact modes (computed in float32 inside the network: its
            # rounding is part of the output) and a zero particular field
            ps = model.plain_square(n, "sine-hat", 16, "cpu")
            H0 = ps.principal(s0)
            assert torch.allclose(H, H0, atol=1e-5 * float(H0.abs().max()))
            assert torch.equal(H, e.last_principal)
            assert float(up.abs().max()) == 0.0


def test_parameter_counts():
    counts = {s: model.parameters(model.Expert("unet", s)) for s in model.WIDTHS}
    assert 0.2e6 < counts["0.3M"] < 0.4e6
    assert 0.8e6 < counts["1M"] < 1.4e6
    assert 2.5e6 < counts["3M"] < 3.5e6
    assert 0.6e6 < model.parameters(model.Expert("spectral", "1M")) < 1.6e6


# ---------------------------------------------------------------------------
# after Stage 0: the kernel-exact forms (W356, W357), random weights only
# ---------------------------------------------------------------------------


def _randomise(net, std, seed=0):
    torch.manual_seed(seed)
    for prm in net.parameters():
        torch.nn.init.normal_(prm, 0.0, std)
    return net.eval()


def _random_batch(n, seed=0):
    """Three examples, steady and two steps (s0 = 0, 3, 100), random channels, element
    weights and a source with a level."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(3, 5, n + 1, n + 1, generator=gen)
    s = torch.randn(3, 2, generator=gen)
    s0 = torch.tensor([0.0, 3.0, 100.0], dtype=torch.float64)
    w = torch.rand(3, n, n, generator=gen, dtype=torch.float64) + 0.5
    g = torch.randn(3, n + 1, n + 1, generator=gen, dtype=torch.float64) + 0.7
    return x, s, s0, w, g


@pytest.mark.parametrize("fo", [None, 0.01, 1.0, 100.0])
def test_exact_modes_make_the_uniform_state_a_fixed_point(examples, fo):
    """H c1 + s0 u_p(1) = 1 for exact classical solves in both port bases: the identity
    the kernel-exact forms enforce.  Steady it is H c1 = 1, the hats summing to one.
    For short steps the hats alone are far from one, so the steady constraint would be
    wrong there and the fixed point is the analogue."""
    p0 = examples[0].problem
    p = PieceProblem(p0.chart, log_kappa_hat=p0.log_kappa_hat, log_kappa_hat_e=p0.log_kappa_hat_e,
                     fo=fo, g=np.ones_like(p0.g))
    n = p.n
    ex = fe.Exact(p.matrix(), n)
    up1 = ex.particular(p.load()[fe.grid(n)[1]])
    for kind in ports.KINDS:
        c1 = forms.kernel_coefficients(kind, n, 16)
        H = ex.modes(ports.basis(kind, n, 16).Q)
        assert np.max(np.abs(H @ c1 + p.s0 * up1 - 1.0)) < 1e-12
        if fo == 0.01:
            assert np.max(np.abs(1.0 - H @ c1)) > 0.5


@pytest.mark.parametrize("kind", ports.KINDS)
def test_the_partition_of_unity(kind):
    """W sums to one with the kernel coefficients, is positive on the kernel columns and
    zero off them; for sine-hat it is the bilinear corner functions."""
    n, m = 16, 8
    ks = forms.kernel_square(n, kind, m, "cpu")
    one = (ks.c1[:, None, None] * ks.W).sum(0)
    assert float((one - 1.0).abs().max()) < 1e-14
    off = ks.c1 == 0
    assert float(ks.W[off].abs().max()) == 0.0 and float(ks.W[~off].min()) > 0.0
    if kind == "sine-hat":
        t = torch.arange(1, n, dtype=torch.float64) / n
        XI, ET = torch.meshgrid(t, t, indexing="xy")
        for c, f in enumerate([(1 - XI) * (1 - ET), XI * (1 - ET), XI * ET, (1 - XI) * ET]):
            assert float((ks.W[c] - f).abs().max()) < 1e-13


@pytest.mark.parametrize("std", [0.02, 0.3])
@pytest.mark.parametrize("kind", ports.KINDS)
@pytest.mark.parametrize("form", forms.FORMS)
def test_kernel_exact_whatever_the_weights(form, kind, std):
    """H c1 + s0 u_1 = 1 for random weights, steady and for two steps, in both port
    bases.  What remains after the projection is float64 rounding of what the network
    left before it, however large that was (std 0.3 makes outputs of order 1e7 to 1e8)."""
    n, m = 16, 8
    e = _randomise(forms.KernelExpert(form, "unet", "0.3M", kind=kind, m=m), std)
    x, s, s0, w, g = _random_batch(n)
    with torch.no_grad():
        H, up = e(x, s, s0, w=w, g=g)
    ks = forms.kernel_square(n, kind, m, "cpu")
    pre = float(e.last_kernel_residual.abs().max())
    assert pre > 1e-3                       # the raw output did break the identity
    assert float(forms.residual(H, e.last_level, s0, ks).abs().max()) <= 1e-14 * (1.0 + pre)
    # steady, the kernel columns' modes sum to one by themselves
    assert float((torch.einsum("k,kij->ij", ks.c1, H[0]) - 1.0).abs().max()) <= 1e-14 * (1.0 + pre)
    assert H.shape == (3, 4 * m, n - 1, n - 1) and up.shape == (3, n - 1, n - 1)
    assert torch.isfinite(H).all() and torch.isfinite(up).all()


def test_the_projection_moves_only_the_kernel_columns():
    """Stage 0's bubble with random weights, copied into the kernel form (its level row
    zero), on a sourceless input.  The projection leaves the side modes bit for bit as
    the network's own ``H0 + beta N`` and moves the hats by ``-W r``; and those side modes
    are the bubble's to the float32 rounding of a head one channel wider (the extra
    output changes the convolution's summation order), while the bubble's hats break
    the sum that the kernel form's keep."""
    n, m = 16, 8
    bub = _randomise(model.Expert("unet", "0.3M", "bubble", m=m), 0.02)
    ker = forms.KernelExpert("kernel", "unet", "0.3M", m=m).eval()
    sd = {k: v.clone() for k, v in bub.net.state_dict().items()}
    sd["head.weight"] = torch.cat([sd["head.weight"], torch.zeros_like(sd["head.weight"][:1])])
    sd["head.bias"] = torch.cat([sd["head.bias"], torch.zeros_like(sd["head.bias"][:1])])
    ker.net.load_state_dict(sd)
    x, s, s0, _w, _g = _random_batch(n)
    x[:, 4] = 0.0
    with torch.no_grad():
        Hb, _upb = bub(x, s, s0)
        Hk, upk = ker(x, s, s0)             # no source
        out = ker.net(x, s).to(torch.float64)
    ps = model.plain_square(n, "sine-hat", m, "cpu")
    H_raw = ps.principal(s0, torch.float32).to(torch.float64) + ps.beta[1:-1, 1:-1] * out[:, :-2, 1:-1, 1:-1]
    ks = forms.kernel_square(n, "sine-hat", m, "cpu")
    side = ks.c1 == 0
    assert torch.equal(Hk[:, side], H_raw[:, side])
    assert torch.equal(Hk[:, ~side], H_raw[:, ~side] - ks.W[None, ~side] * ker.last_kernel_residual[:, None])
    assert float((Hk[:, side] - Hb[:, side]).abs().max()) <= 1e-6 * float(Hb[:, side].abs().max())
    assert float(forms.residual(Hb, ker.last_level, s0, ks).abs().max()) > 1e-3
    assert float(forms.residual(Hk, ker.last_level, s0, ks).abs().max()) < 1e-13
    assert float(upk.abs().max()) == 0.0


@pytest.mark.parametrize("form", forms.FORMS)
def test_zero_heads_give_the_principal_part(form):
    """As initialised (zero heads) the forms return the plain square's modes, with the
    float32 rounding of their sum removed, and u_p = gbar u_1^0: the principal part's
    answer for the source's level."""
    n, m = 16, 8
    e = forms.KernelExpert(form, "unet", "0.3M", m=m).eval()
    x, s, s0, w, g = _random_batch(n)
    with torch.no_grad():
        H, up = e(x, s, s0, w=w, g=g)
    ps = model.plain_square(n, "sine-hat", m, "cpu")
    assert float((H - ps.principal(s0)).abs().max()) < 1e-5
    u10 = forms.level_principal(ps, s0)
    assert float((e.last_level - u10).abs().max()) < 1e-6 * float(u10.abs().max())
    assert torch.equal(up, e.last_gbar[:, None, None] * e.last_level)
    ks = forms.kernel_square(n, "sine-hat", m, "cpu")
    assert float(forms.residual(H, e.last_level, s0, ks).abs().max()) < 1e-14


@pytest.mark.parametrize("form", forms.FORMS)
def test_a_uniform_source_shift_moves_only_the_level_term(form):
    """g -> g + a leaves the network's input unchanged (to the rounding of g - gbar), so
    the modes and the level field stay and u_p moves by a u_1.  With the coefficients'
    move this is the datum dropping out: lowering every temperature by T0 (g by s0 T0)
    lowers the coupled answer by exactly T0."""
    n, m = 16, 8
    e = _randomise(forms.KernelExpert(form, "unet", "0.3M", m=m), 0.02)
    x, s, s0, w, g = _random_batch(n)
    a = torch.tensor([-300.0, 41.0, 7.5], dtype=torch.float64)
    with torch.no_grad():
        H1, up1 = e(x, s, s0, w=w, g=g)
        u1 = e.last_level.clone()
        H2, up2 = e(x, s, s0, w=w, g=g + a[:, None, None])
    assert float((H2 - H1).abs().max()) <= 1e-10 * float(H1.abs().max())
    assert float((e.last_level - u1).abs().max()) <= 1e-10 * float(u1.abs().max())
    assert float((up2 - up1 - a[:, None, None] * u1).abs().max()) <= 1e-10 * float(up2.abs().max())


def test_split_source_is_homogeneous_and_level_free():
    n = 16
    x, _s, _s0, w, g = _random_batch(n)
    gb, ch_, sc = forms.split_source(x, w, g)
    gb2, ch2, sc2 = forms.split_source(x, w, 3.0 * g)
    assert torch.allclose(gb2, 3.0 * gb, rtol=1e-14) and torch.allclose(sc2, 3.0 * sc, rtol=1e-14)
    assert torch.allclose(ch2, ch_, atol=1e-13)
    gb3, ch3, sc3 = forms.split_source(x, w, g - 5.0)
    assert torch.allclose(gb3, gb - 5.0, atol=1e-13) and torch.allclose(sc3, sc, rtol=1e-12)
    assert torch.allclose(ch3, ch_, atol=1e-12)
    # a unit-RMS channel; a uniform source is all level, and nothing is left of it
    assert torch.allclose(torch.sqrt((ch_ ** 2).mean((-2, -1))), torch.ones(3, dtype=torch.float64))
    gb4, ch4, sc4 = forms.split_source(x, w, torch.full_like(g, 2.5))
    assert torch.allclose(gb4, torch.full_like(gb4, 2.5), rtol=1e-15)
    assert float(ch4.abs().max()) == 0.0 and float(sc4.abs().max()) == 0.0
    assert all(float(t.abs().max()) == 0.0 for t in forms.split_source(x, None, None))


def test_the_split_forms_modes_never_see_the_load():
    """The split form's modes and level field are the same, bit for bit, for no source, a
    random source, another random source and a uniform one, whatever the fifth input
    channel holds; its particular field does change.  The single-network kernel form's
    modes follow the load's shape (W357's defect, which the split removes)."""
    n, m = 16, 8
    e = _randomise(forms.KernelExpert("split", "unet", "0.3M", m=m), 0.02)
    x, s, s0, w, g = _random_batch(n)
    gen = torch.Generator().manual_seed(5)
    loads = [None, g, torch.randn(g.shape, generator=gen, dtype=torch.float64),
             torch.full_like(g, 2.0)]
    outs = []
    with torch.no_grad():
        for gg in loads:
            xx = x.clone()
            xx[:, 4] = torch.randn(xx[:, 4].shape, generator=gen)
            H, up = e(xx, s, s0, w=w, g=gg)
            outs.append((H, e.last_level.clone(), up))
    for H, u1, _up in outs[1:]:
        assert torch.equal(H, outs[0][0]) and torch.equal(u1, outs[0][1])
    assert float((outs[2][2] - outs[1][2]).abs().max()) > 1e-6
    ker = _randomise(forms.KernelExpert("kernel", "unet", "0.3M", m=m), 0.02)
    with torch.no_grad():
        Ha, _ = ker(x, s, s0, w=w, g=loads[1])
        Hb, _ = ker(x, s, s0, w=w, g=loads[2])
    assert float((Ha - Hb).abs().max()) > 1e-6


@pytest.mark.parametrize("k", range(8))
@pytest.mark.parametrize("kind", ports.KINDS)
def test_the_projection_commutes_with_the_squares_symmetries(kind, k):
    """Pi(S H, S u_1) = S Pi(H, u_1) for the eight symmetries, S acting on the modes by
    `model.d4_ports`' signed permutation: kernel columns go to kernel columns with sign
    +1, and the partition of unity moves with them, so D4 augmentation stays exact."""
    n, m = 16, 8
    ks = forms.kernel_square(n, kind, m, "cpu")
    perm, sign = model.d4_ports(kind, n, m, k)
    c1 = forms.kernel_coefficients(kind, n, m)
    assert np.all(sign[c1 == 1] == 1.0) and np.all(c1[perm[c1 == 1]] == 1.0)
    fx, fy, t = k & 1, (k >> 1) & 1, (k >> 2) & 1
    gen = torch.Generator().manual_seed(k)
    H = torch.randn(2, 4 * m, n - 1, n - 1, generator=gen, dtype=torch.float64)
    u1 = torch.randn(2, n - 1, n - 1, generator=gen, dtype=torch.float64)
    s0 = torch.tensor([0.0, 4.0], dtype=torch.float64)

    def S(Hh):
        out = torch.empty_like(Hh)
        out[:, torch.as_tensor(perm)] = (model._spatial(Hh, fx, fy, t)
                                         * torch.as_tensor(sign)[None, :, None, None])
        return out
    P1, _ = forms.project(S(H), model._spatial(u1, fx, fy, t), s0, ks)
    P2, _ = forms.project(H, u1, s0, ks)
    assert float((P1 - S(P2)).abs().max()) < 1e-13


@pytest.mark.parametrize("form", forms.FORMS)
def test_the_kernel_forms_are_trainable(examples, form):
    """The label-free objective reaches every head through the projection: finite,
    non-zero gradients for the mode rows, the level row and the particular field, on a
    steady and a transient example whose sources have a level.  One backward pass, no
    optimiser step, nothing kept."""
    b = _batch_from(examples)
    b["s0"] = torch.tensor([0.0, 7.0], dtype=torch.float64)
    b["sigma"] = b["w"] * b["s0"][:, None, None]
    b["scalars"] = torch.tensor(np.stack([fo_scalars(None), fo_scalars(1.0 / 7.0)]))
    e = forms.KernelExpert(form, "unet", "0.3M")
    H, up = forms.call(e, b)
    o = model.objectives(b, H, up, "sine-hat", 16, None, "label-free", H0=e.last_principal)
    o["loss"].mean().backward()
    head = e.net.head.weight.grad[:, :, 0, 0]
    assert torch.isfinite(head).all()
    assert float(head[:64].abs().max()) > 0.0           # the modes
    assert float(head[-1].abs().max()) > 0.0            # the level field
    ph = e.pnet.head.weight.grad if form == "split" else head[-2]
    assert torch.isfinite(ph).all() and float(ph.abs().max()) > 0.0


def test_a_kernel_expert_saves_and_loads(tmp_path):
    from atlas.chartop import train
    e = _randomise(forms.KernelExpert("split", "unet", "0.3M", m=8, p_size="0.3M"), 0.02)
    path = tmp_path / "best.pt"
    torch.save({"model": e.state_dict(), "config": e.config()}, path)
    e2 = train.load_expert(str(path))
    assert isinstance(e2, forms.KernelExpert) and e2.config() == e.config()
    x, s, s0, w, g = _random_batch(16)
    with torch.no_grad():
        a, b_ = e(x, s, s0, w=w, g=g), e2(x, s, s0, w=w, g=g)
    assert torch.equal(a[0], b_[0]) and torch.equal(a[1], b_[1])


@pytest.mark.parametrize("fo", [None, 0.05, 20.0])
def test_the_uniform_state_measure_through_the_expert(examples, fo):
    """e_K through the host's interface (one call on the piece's uniform state): zero to
    rounding for exact solves and for both kernel-exact forms with random weights; not
    for Stage 0's bubble with weights of the same size, which it would have flagged."""
    p0 = examples[0].problem
    p = PieceProblem(p0.chart, log_kappa_hat=p0.log_kappa_hat, log_kappa_hat_e=p0.log_kappa_hat_e,
                     fo=fo)
    n = p.n
    pb = ports.basis("sine-hat", n, 16)
    ex = fe.Exact(p.matrix(), n)
    trL = float(np.trace(ex.port_matrix(pb.Q)))
    r = forms.kernel_residual(host.ClassicalExpert(), p, pb)
    assert forms.kernel_measures(r, ex.A_I, trL)["e_K_inf"] < 1e-12
    for form in forms.FORMS:
        net = _randomise(forms.KernelExpert(form, "unet", "0.3M"), 0.02)
        km = forms.kernel_measures(forms.kernel_residual(LearnedExpert(net), p, pb), ex.A_I, trL)
        assert km["e_K_inf"] < 1e-12 and km["e_K"] < 1e-12
    bub = _randomise(model.Expert("unet", "0.3M"), 0.02)
    km = forms.kernel_measures(forms.kernel_residual(LearnedExpert(bub), p, pb), ex.A_I, trL)
    assert km["e_K_inf"] > 1e-3 and km["e_K"] > 1e-6
    # e_H(theta): e_H at theta = 0, the level in quadrature at theta = 10
    assert forms.level_weighted_e_H(0.3, km["e_K"], 0.0) == pytest.approx(0.3)
    assert forms.level_weighted_e_H(0.3, km["e_K"], 10.0) == pytest.approx(
        math.sqrt(0.09 + 100.0 * km["r_energy"] / trL))


def test_a_march_keeps_the_level_field_with_the_modes(examples):
    """`LearnedExpert.respond(modes_from=...)` with the kernel form, whose level field
    follows the load's shape: the kept modes are paired with the kept level field, so
    the uniform state is still a fixed point in a later step.  Unpaired, it is not."""
    p0 = examples[0].problem
    first = PieceProblem(p0.chart, fo=0.5, g=p0.g)
    n = first.n
    pb = ports.basis("sine-hat", n, 16)
    c1 = forms.kernel_coefficients("sine-hat", n, 16)
    # weights large enough (std 0.1) that the single network's answer visibly follows
    # the source's shape; at 0.02 the unpaired defect is only 5e-7
    ex = LearnedExpert(_randomise(forms.KernelExpert("kernel", "unet", "0.3M"), 0.1))
    kept = ex.respond([first], lambda n_: pb)
    later = forms.uniform_state(first)
    r = ex.respond([later], lambda n_: pb, modes_from=kept)[0]
    assert np.max(np.abs(r.up + r.H @ c1 - 1.0)) < 1e-12
    fresh = ex.respond([later], lambda n_: pb)[0]
    assert np.max(np.abs(fresh.up + kept[0].H @ c1 - 1.0)) > 1e-4

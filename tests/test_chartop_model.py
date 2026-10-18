"""The chart operator's torch side against its numpy side (Stage 0, W355): the DST-I,
the plain square's principal part, the energies and loads, the label-free objective
(T26 in float64), the D4 augmentation's tensor rules, and the three output forms.
An untrained network is used only for shapes and for the exact-trace property."""
from __future__ import annotations

import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import scipy.fft as sfft

torch = pytest.importorskip("torch")

from atlas.chartop import chart as ch
from atlas.chartop import data, fe, model, ports, principal

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
            # the plain square's exact modes and a zero particular field
            ps = model.plain_square(n, "sine-hat", 16, "cpu")
            assert torch.allclose(H, ps.principal(s0), atol=1e-12)
            assert float(up.abs().max()) == 0.0


def test_parameter_counts():
    counts = {s: model.parameters(model.Expert("unet", s)) for s in model.WIDTHS}
    assert 0.2e6 < counts["0.3M"] < 0.4e6
    assert 0.8e6 < counts["1M"] < 1.4e6
    assert 2.5e6 < counts["3M"] < 3.5e6
    assert 0.6e6 < model.parameters(model.Expert("spectral", "1M")) < 1.6e6

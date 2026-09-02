"""Tier 26 -- the FROZEN CHECKPOINT as the fluid agent of PoC 1a.

`tests/test_tier21_wind_farm_design.py` pins the classical column of
[[poc1-results-differentiable-design]]: the torch rollout against
`wake_array.exposed_reference_solver`, bitwise.  This file pins the other
expert, and it has to answer a strictly harder question, because the checkpoint
arrives with its differentiability deliberately removed.  `FrozenFluidExpert`
takes numpy in, runs its forward inside ``torch.no_grad()``, and calls
``.detach().cpu().numpy()`` on the way out -- three cuts, and its own docstring
says they are there "so that a gradient cannot be taken by accident".

So the claim these tests exist to make checkable is:

    the taped column is the SAME arithmetic as the frozen wrapper's -- bitwise,
    on a real batched call and through a whole composed macro-step -- and the
    only thing that changed is that the tape survived it.

Everything else here follows from that: that the parameters stayed frozen, that
the objective is bit-reproducible so a finite difference has something to
measure, that the adjoint agrees with central differences at an ``h`` chosen for
a float32 forward pass rather than a float64 one, and that
`assembly.ProjectedAssembly`, `wake_array.exposed_reference_solver` and the
classical `Rollout` are all untouched by the swap.

    python -m pytest tests/test_tier26_poseidon_design.py -q

The checkpoint is ~85 MB and is fetched from HuggingFace on first construction;
every test that needs it skips rather than fails if the build repo or the
checkpoint is not there, which is `window_ns.load_reference`'s own convention.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from atlas.cases import wake_array as wa
from atlas.cases import wind_farm_design as wd

torch = pytest.importorskip("torch")


def _expert_or_skip():
    try:
        return wa.scaled_expert()
    except Exception as exc:                                # pragma: no cover
        pytest.skip(f"the frozen checkpoint is not available here: {exc}")


@pytest.fixture(scope="module")
def small():
    """A K12 case at a short horizon: the real rung, a cheap march.

    Not a reduced geometry -- the tiling, the window size, the overlap, the ramp
    and the macro-step are the case study's own, because a test on a geometry
    the PoC does not run is a test of a different composition.
    """
    _expert_or_skip()
    torch.set_num_threads(int(os.environ.get("ATLAS_TEST_THREADS", "8")))
    case = wd.case_k12(steps=3)
    return case, wd.rollout_for(case, "poseidon")


# ---------------------------------------------------------------------------
# 1. provenance -- is this the expert the case studies marched?
# ---------------------------------------------------------------------------


def test_taped_expert_is_bitwise_the_frozen_wrapper(small):
    """The whole substitution rests on this one assertion.

    `TapedPoseidon.step_batch` against `FrozenFluidExpert.step_many` with
    ``galilean=False, project=False`` -- the arrangement W98 settled -- on a
    real batch at the real window count.  Bitwise, not to a tolerance: the two
    are the same sequence of float32 operations on the same weights, and
    anything less than equality would mean one of them is not.
    """
    case, ro = small
    ex = wa.scaled_expert()
    rng = np.random.default_rng(20260902)
    uf = rng.standard_normal((ro.n_win, wa.N, wa.N)) * 0.05
    vf = rng.standard_normal((ro.n_win, wa.N, wa.N)) * 0.05
    a, b = ex.step_many(uf, vf, wa.MACRO_DT, galilean=False, project=False)
    with torch.no_grad():
        c, d = ro.expert.step_batch(torch.as_tensor(uf, dtype=wd.TORCH_DTYPE),
                                    torch.as_tensor(vf, dtype=wd.TORCH_DTYPE),
                                    wa.MACRO_DT)
    assert np.abs(c.numpy() - a).max() == 0.0
    assert np.abs(d.numpy() - b).max() == 0.0


def test_the_lead_time_is_exactly_native():
    """0.2 macro-step over a Scaling(4.0, 2.0) is a lead of 0.1.

    The checkpoint's smallest trained non-identity lead.  Nothing below it was
    ever trained, and `adapters` raises rather than extrapolating -- so this is
    a property of the geometry the case study chose, and it is asserted rather
    than trusted because `S_LEN` and `MACRO_DT` are separately declared.
    """
    _expert_or_skip()
    ad = wd._adapters()
    e = wd.taped_poseidon()
    assert e.lead(wa.MACRO_DT) == pytest.approx(ad.EXPERT_NATIVE_DT, abs=1e-12)


def test_the_checkpoint_is_frozen_and_stays_frozen(small):
    """20.8 M parameters, none of them requiring a gradient.

    "Frozen, independently-pretrained" is the load-bearing phrase of
    [[prior-art-and-novelty-atlas-0.1]] section 2.1, and this run is the one that
    claims it.  A backward pass that quietly accumulated parameter gradients
    would still produce the right dJ/dtheta and would make the sentence false.
    """
    case, ro = small
    assert ro.expert.n_params == 20774444
    assert ro.expert.n_grad_params == 0
    th = torch.as_tensor(wd.initial_design(case), dtype=wd.TORCH_DTYPE
                         ).requires_grad_(True)
    j, _ = wd.objective(th, ro, grad=True)
    j.backward()
    assert all(p.grad is None for p in ro.expert.model.parameters())


def test_reduced_precision_matmul_is_off(small):
    """TF32 would put the checkpoint's arithmetic at ~1e-3 relative.

    Asserted rather than assumed because it is a global torch flag: something
    else in the process can turn it back on, and the failure is silent and lands
    in the fourth significant figure of J -- exactly where the finite-difference
    check lives.
    """
    case, ro = small
    assert wd.TapedPoseidon.tf32_pinned
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


# ---------------------------------------------------------------------------
# 2. the composition layer -- the torch twins of two numpy operators
# ---------------------------------------------------------------------------


def test_transport_project_matches_wake_array(small):
    """`PoseidonRollout.transport_project` against `wa.transport_and_project`.

    The two global operators the checkpoint needs from the composition layer:
    one Leray projection (Nyquist-zeroed wavenumbers, because it is a
    derivative) and one spectral translation by ``U_INF dt`` (Nyquist kept,
    because it is an interpolation), on the domain extended by one window of
    tapered fluctuation.  W98 is the finding that says both must be global; this
    asserts the torch twin is the same operator, to float64 rounding.
    """
    case, ro = small
    rng = np.random.default_rng(7)
    ny, nx = case.shape
    pu = rng.standard_normal((ny, nx)) * 0.1
    pv = rng.standard_normal((ny, nx)) * 0.1
    na, nb = wa.transport_and_project(pu, pv, wa.MACRO_DT, wa.U_INF, project=True)
    with torch.no_grad():
        ta, tb = ro.transport_project(torch.as_tensor(pu, dtype=wd.TORCH_DTYPE),
                                      torch.as_tensor(pv, dtype=wd.TORCH_DTYPE))
    assert np.abs(ta.numpy() - na).max() < 1e-14
    assert np.abs(tb.numpy() - nb).max() < 1e-14


def test_macro_step_reproduces_the_w93_march(small):
    """One composed macro-step against `scripts/w93_wake_array.py`'s own loop.

    Written out here in numpy, line for line, rather than imported: the point of
    the assertion is that the torch column marches the arrangement the case
    study describes, and an assertion that calls the same function twice would
    not say that.
    """
    case, ro = small
    th_np = wd.initial_design(case)
    th = torch.as_tensor(th_np, dtype=wd.TORCH_DTYPE)
    u = torch.full(case.shape, wa.U_INF, dtype=wd.TORCH_DTYPE)
    v = torch.zeros_like(u)
    with torch.no_grad():
        fx, fy, _un, _T, _P = ro.disks.forcing(u, v, th)
        u2, v2, _p, _n = ro.macro_step(u, v, th)

    ex = wa.scaled_expert()
    ufs = ro.tiling.cut(u.numpy() - wa.U_INF)
    vfs = ro.tiling.cut(v.numpy())
    u1, v1 = ex.step_many(ufs, vfs, wa.MACRO_DT, galilean=False, project=False)
    u1 = u1 + (ufs.mean(axis=(1, 2)) - u1.mean(axis=(1, 2)))[:, None, None]
    v1 = v1 + (vfs.mean(axis=(1, 2)) - v1.mean(axis=(1, 2)))[:, None, None]
    au, av = ro.tiling.assemble(u1, v1)
    au = au + fx.numpy() * wa.MACRO_DT
    av = av + fy.numpy() * wa.MACRO_DT
    au, av = wa.transport_and_project(au, av, wa.MACRO_DT, wa.U_INF, project=True)
    au = wa.U_INF + au
    b = wd.BAND
    for f, val in ((au, wa.U_INF), (av, 0.0)):
        f[:, :b] = val
        f[:b, :] = val
        f[-b:, :] = val
    assert np.abs(u2.numpy() - au).max() < 1e-14
    assert np.abs(v2.numpy() - av).max() < 1e-14


def test_the_declared_assembly_is_still_the_projected_one(small):
    """The swap changed the expert and nothing the expert is composed BY.

    `PoseidonRollout` inherits `Rollout.__init__`, which constructs the declared
    `assembly.ProjectedAssembly` -- blend plus one global Leray projection --
    so the object on the record is the same one CS-7 and CS-8 marched.
    """
    case, ro = small
    from atlas import assembly as asm
    assert isinstance(ro.assembly, asm.ProjectedAssembly)
    assert ro.assembly is not None
    # ...and the classical agent is not quietly still in the loop.
    assert ro.solver is None


# ---------------------------------------------------------------------------
# 3. the objective, and whether a finite difference has anything to measure
# ---------------------------------------------------------------------------


def test_J_is_bit_reproducible(small):
    """Two evaluations at the same theta, bit for bit.

    OP-6's operational content.  The batch layout is pinned -- `Rollout.cut`
    stacks windows in `tiling.offsets` order always, and `_place` accumulates in
    that same fixed order rather than through `scatter_add` -- so the
    checkpoint's measured 6.6e-7 batch-position spread never enters, and a
    central difference is measuring a derivative rather than a reordering.
    """
    case, ro = small
    th = wd.initial_design(case)
    a, _ = wd.value_only(th, ro)
    b, _ = wd.value_only(th, ro)
    assert a == b


def test_value_and_grad_agrees_with_the_forward_value(small):
    """The taped path and the untaped path return the same J.

    Checkpointing recomputes each macro-step's forward during the backward, so
    this also says the recompute is the same computation -- which it is only
    because nothing in the column is stochastic.
    """
    case, ro = small
    th = wd.initial_design(case)
    j_fwd, _ = wd.value_only(th, ro)
    j_grad, g, _ = wd.value_and_grad(th, ro)
    assert j_fwd == j_grad
    assert np.all(np.isfinite(g))
    assert np.count_nonzero(g) == g.size


def test_adjoint_against_central_differences(small):
    """The adjoint against central differences, at an h a float32 expert allows.

    **The step size is the finding, not a detail, so this asserts a SWEEP.**
    The classical column's error curve bottoms out at h = 1e-5, where its
    float64 rollout still has four digits to spare.  This column's forward pass
    runs through 20.8 M float32 parameters, so J carries a relative noise floor
    near 1e-7 and the cancellation branch of the finite-difference curve arrives
    two to three decades earlier -- measured on the real rollout, the minimum is
    around h = 3e-3 rotor diameters and 1e-3 is already worse.  A test that
    pinned ONE h would be asserting a property of the arithmetic rather than of
    the adjoint, so it takes the best of two and says which won.
    """
    case, ro = small
    th = wd.initial_design(case)
    best = max((wd.fd_check(th, ro, n_probe=6, h=h) for h in (1e-2, 3e-3)),
               key=lambda c: c["cosine"])
    assert best["cosine"] > 0.999, best
    assert best["median_rel_err"] < 5e-2, best


# ---------------------------------------------------------------------------
# 4. the swap itself
# ---------------------------------------------------------------------------


def test_rollout_for_dispatches_and_refuses_the_unknown():
    case = wd.case_k12(steps=2)
    ro = wd.rollout_for(case, "reference_exposed")
    assert type(ro) is wd.Rollout
    with pytest.raises(ValueError, match="unknown fluid expert kind"):
        wd.rollout_for(case, "walrus")


def test_the_classical_column_is_untouched_by_the_swap():
    """`Rollout` still builds the exposed classical solver and no checkpoint.

    The regression this guards is the obvious one: a swap implemented by
    mutating shared state rather than by subclassing would leave the classical
    column marching the checkpoint, and every number on
    [[poc1-results-differentiable-design]] would silently become a number about
    a different run.
    """
    case = wd.case_k12(steps=2)
    ro = wd.Rollout(case)
    assert ro.solver is wd.tape_solver(wa.NU_REF, "cpu")
    assert not hasattr(ro, "expert")
    assert getattr(ro, "kind", None) is None


def test_the_two_columns_disagree_and_that_is_the_point(small):
    """OP-3 says the checkpoint over-dissipates wakes; a march should show it.

    Not a tolerance -- a **sign**.  Both columns march the same layout from the
    same freestream with the same disks; the checkpoint's wakes recover faster,
    so its downstream turbines see more inflow and the array's farm power comes
    out HIGHER than the classical column's.  If the two ever agreed to within
    the checkpoint's own turbine-free noise floor, either the swap did not
    happen or OP-3 stopped being true, and both are worth failing a test over.
    """
    case, ro = small
    th = wd.initial_design(case)
    j_pos, _ = wd.value_only(th, ro)
    j_ref, _ = wd.value_only(th, wd.rollout_for(case, "reference_exposed"))
    assert j_pos != j_ref
    assert abs(j_pos - j_ref) / abs(j_ref) > 1e-3


def test_the_march_stays_in_the_band(small):
    """W100's own stability rule, on the composed checkpoint column.

    The classical column needed the projected assembly to survive a long march;
    this asserts the same assembly holds a globally-receptive expert over the
    horizon the optimiser uses.  Short here (the fixture is 3 macro-steps); the
    70-step version is `--stage confirm` and is reported on the results page.
    """
    case, ro = small
    th = torch.as_tensor(wd.initial_design(case), dtype=wd.TORCH_DTYPE)
    u = torch.full(case.shape, wa.U_INF, dtype=wd.TORCH_DTYPE)
    v = torch.zeros_like(u)
    with torch.no_grad():
        for _ in range(case.steps):
            u, v, _p, _n = ro.macro_step(u, v, th)
            assert torch.isfinite(u).all() and torch.isfinite(v).all()
            assert float(u.abs().max()) < 3.0

# ---------------------------------------------------------------------------
# 5. the demo's toggle
# ---------------------------------------------------------------------------


def test_demo_poseidon_rollout_is_the_poc_column_at_the_default_inflow(small):
    """`DemoPoseidonRollout` is `PoseidonRollout`, bitwise, at u_inf = 1, alpha = 0.

    The same assertion `tests/test_tier22_demo.py` makes about the classical
    pair, and for the same reason: it is the entire licence for the subclass.
    An animation that is not the column the results page measured is a picture,
    not a demonstration.
    """
    from atlas.demo import engine as de
    case, ro = small
    dro = de.demo_rollout(case, "poseidon", u_inf=1.0, inflow_deg=0.0)
    assert isinstance(dro, wd.PoseidonRollout)
    th = torch.as_tensor(wd.initial_design(case), dtype=wd.TORCH_DTYPE)
    u = torch.full(case.shape, wa.U_INF, dtype=wd.TORCH_DTYPE)
    v = torch.zeros_like(u)
    with torch.no_grad():
        a0, b0, p0, _ = ro.macro_step(u, v, th)
        a1, b1, p1, _ = dro.macro_step(u, v, th)
    assert torch.equal(a0, a1)
    assert torch.equal(b0, b1)
    assert torch.equal(p0, p1)


def test_a_pointed_freestream_moves_the_transport_kernel(small):
    """The knob the checkpoint column has and the classical one does not.

    A yawed inflow changes the band on both columns; on this one it must ALSO
    change the spectral translation, because the agent does not advect and the
    composition layer owes it that.  A phase left at the construction-time
    freestream would translate the field one way while the boundary imposed
    another -- a mismatch that looks like a physical result.
    """
    from atlas.demo import engine as de
    case, _ro = small
    dro = de.demo_rollout(case, "poseidon", u_inf=1.0, inflow_deg=0.0)
    before = dro._phase.clone()
    dro.set_inflow(1.0, 10.0)
    assert not torch.equal(before, dro._phase)
    assert dro.free_v > 0.0


def test_the_demo_toggle_refuses_an_expert_it_does_not_have():
    from atlas.demo import engine as de
    case = wd.case_k12(steps=2)
    with pytest.raises(ValueError, match="unknown fluid expert"):
        de.demo_rollout(case, "walrus")
    assert de.DemoConfig(expert="walrus").clamped().expert == "reference_exposed"

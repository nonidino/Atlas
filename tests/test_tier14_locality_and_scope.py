"""Tier 14 -- W68, W69, W70, W71, W72, W74.

The 2026-08-29 session closed five holes and opened five.  Four of the five
turned out to be one question -- *how much of a probed block is the expert's
operator and how much is the boundary condition on top of it?* -- and the
measurement that answers it opened a sixth (W74) that none of them predicted.

Where a rule can be tested against the algebra it comes from, it is tested that
way here and not only through a case study, per `test_tier12_topology`'s pattern.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from atlas.capability import EllipticSubsolve, TimeDiscretization
from atlas.cases.poseidon import elliptic_signature
from atlas.probe import (OPERATOR_CONTENT_FLOOR, ProbeError, operator_content,
                         probe_base)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))


# ---------------------------------------------------------------------------
# W68 -- operator_content, against the algebra
# ---------------------------------------------------------------------------


def test_operator_content_is_zero_exactly_on_a_multiple_of_the_identity():
    """The definition, at the point the whole finding turns on."""
    for c in (1.0, -0.1933, 5.0001, 1e6):
        assert operator_content(c * np.eye(16)) == pytest.approx(0.0, abs=1e-15)


def test_operator_content_is_invariant_under_a_positive_rescale():
    """omega must not move with the flux convention.

    W66 measured that the wrong THERM half is a positive rescale of the right
    one -- `S_heat = 94.97 S_entropy` then, 1199x at the corrected base -- and
    that passivity is exactly invariant under it.  A statistic that MOVED under
    a rescale would be reporting the convention, which is the trap W66 is about.
    """
    rng = np.random.default_rng(0)
    S = rng.standard_normal((16, 16))
    base = operator_content(S)
    for c in (1e-3, 0.5, 94.97, 1199.0):
        assert operator_content(c * S) == pytest.approx(base, rel=1e-12)


def test_operator_content_is_one_when_the_diagonal_carries_nothing():
    """The other end: a traceless off-diagonal block is all operator."""
    S = np.zeros((4, 4))
    S[0, 1] = S[1, 0] = 1.0
    assert operator_content(S) == pytest.approx(1.0, rel=1e-12)


def test_a_film_coefficient_plus_a_small_operator_reports_their_ratio():
    """omega IS the ratio, which is why 300x underneath reads as 3e-3.

    Built here rather than measured.  Note what Frobenius does to the ratio: a
    tridiagonal tail whose ENTRIES are h/r contributes sqrt(2(n-1)) of them
    against the diagonal's sqrt(n), so omega is r-inverse times sqrt(2(n-1)/n),
    about 1.4x -- the per-entry ratio and omega are not the same number, and the
    shell's measured 1.97e-5 is the second.
    """
    n = 48
    h = 5.0
    tail = np.zeros((n, n))
    for i in range(n - 1):
        tail[i, i + 1] = tail[i + 1, i] = h / 3000.0
    S = h * np.eye(n) + tail
    om = operator_content(S)
    assert om < OPERATOR_CONTENT_FLOOR
    assert om == pytest.approx(np.linalg.norm(tail) / np.linalg.norm(S), rel=1e-12)


def test_elliptic_signature_declines_on_a_block_with_no_operator_content():
    """W68's fix: a third outcome, not a third threshold."""
    S = 5.0001 * np.eye(16)
    v = elliptic_signature(S)["verdict"]
    assert "NO OPERATOR RESOLVED" in v
    assert "EMBEDDED" not in v and "EXPOSED" not in v


def test_elliptic_signature_keeps_both_original_calibration_points():
    """The precondition must not disturb what section 8.3 measured."""
    exposed = np.diag([1.196, 1.0]) + 0.001 * np.array([[0.0, 1.0], [-1.0, 0.0]])
    embedded = np.diag([21.73, 1.0]) + 0.9 * np.array([[0.0, 1.0], [-1.0, 0.0]])
    assert "EXPOSED" in elliptic_signature(exposed)["verdict"]
    assert "EMBEDDED" in elliptic_signature(embedded)["verdict"]
    assert elliptic_signature(np.zeros((2, 2)))["verdict"] == "empty operator"


def test_elliptic_signature_floor_can_be_switched_off_explicitly():
    """`floor=0.0` restores the pre-W68 statistic, for the synthetic points only."""
    S = 5.0001 * np.eye(16)
    assert "NO OPERATOR RESOLVED" not in elliptic_signature(S, floor=0.0)["verdict"]


# ---------------------------------------------------------------------------
# W71 -- the closed form, checked against a synthetic DtN with a known symbol
# ---------------------------------------------------------------------------


def test_omega_tracks_the_symbol_spread_of_a_diagonal_operator():
    """W71's mechanism, isolated from the solver.

    In a Fourier basis a DtN block is diagonal and omega measures how far its
    symbol strays from flat.  Give it the parabolic symbol
    ``a(k) = a0 sqrt(1 + (k ell)^2)`` and omega must fall like ``(k_max ell)^2``
    as ell shrinks -- which is the (k ell)^2 factor of the fitted law, with the
    Biot number held at 1 by construction.
    """
    k = np.arange(1, 17, dtype=float)
    prev = None
    for ell in (1e-2, 1e-3, 1e-4):
        S = np.diag(np.sqrt(1.0 + (k * ell) ** 2))
        om = operator_content(S)
        if prev is not None:
            # ell down 10x -> (k ell)^2 down 100x
            assert prev / om == pytest.approx(100.0, rel=0.05)
        prev = om


# ---------------------------------------------------------------------------
# W69 -- required_halo, and the two new conformance tests
# ---------------------------------------------------------------------------


def _caps(**kw):
    from atlas.capability import ExpertCapabilities, PortType, port_decl
    from atlas.ports import ResponseHalf

    base = dict(
        expert_id="t",
        ports=[port_decl(name="p:MECH", port_type=PortType.MECH,
                         nondim={"velocity": 1.0, "traction": 1.0, "power_area": 1.0},
                         response_half=ResponseHalf.EFFORT)],
        substeps_per_macro_step=1,
        stencil_radius=1,
    )
    base.update(kw)
    return ExpertCapabilities(**base)


def test_required_halo_is_the_product_for_an_explicit_agent():
    c = _caps(time_discretization=TimeDiscretization.EXPLICIT,
              stencil_radius=2, substeps_per_macro_step=10)
    assert c.required_halo() == 20


def test_required_halo_is_undecidable_for_an_implicit_agent_with_a_stencil():
    """W69: one macro-step of an implicit scheme inverts a dense operator.

    `radius x substeps` is an EXPLICIT agent's domain of dependence and was being
    applied to both.  `_halo_rule` REFUSES an overlap below the number it
    returns, so the wrong value bought a passing halo check.
    """
    c = _caps(time_discretization=TimeDiscretization.IMPLICIT,
              stencil_radius=1, substeps_per_macro_step=1)
    assert c.required_halo() is None


def test_a_zero_stencil_algebraic_closure_keeps_a_halo_of_zero():
    """The guard is on the stencil, not on the label, and this is why.

    `wind_farm_real`'s actuator disk declares IMPLICIT because it has no time
    stepping at all.  It couples no cells, so there is no dense inverse to be
    global and its reach is 0 under either discretization -- the derivation says
    so, and the fix follows the derivation rather than preserving a verdict.
    """
    c = _caps(time_discretization=TimeDiscretization.IMPLICIT,
              stencil_radius=0, substeps_per_macro_step=1)
    assert c.required_halo() == 0


def test_wind_farm_real_still_compiles_after_the_halo_correction():
    """The one implicit agent on an overlapping graph in this vault."""
    from atlas import compile_scheme
    from atlas.cases import wind_farm_real

    st = np.load(os.path.join(_ROOT, "out", "tier0b", "s0_state.npz"))
    g = wind_farm_real.build(st["u"], st["v"])
    r = compile_scheme(g, probe_state="regression")
    assert r.verdict.value == "admit-uncertified"
    assert not r.decisions.refusals


def test_deterministic_is_falsified_by_a_jittering_response():
    from atlas.conformance import _test_deterministic
    from atlas.transfer import InterfaceSpace, Prolongation

    space = InterfaceSpace(seam_id="s", dim=4)
    P = Prolongation(agent_id="t", port_name="p:MECH", matrix=np.eye(4),
                     gram_V=np.eye(4))
    rng = np.random.default_rng(1)
    c = _caps(deterministic=True, reproducibility_floor=1e-12,
              boundary_response=lambda _p, x: np.asarray(x) + 1.0 + 1e-6 * rng.standard_normal(4))
    t = _test_deterministic(c, space, P)
    assert t.verdict.value == "refuse"
    assert "two identical calls differ" in t.message


def test_deterministic_passes_a_pure_function():
    from atlas.conformance import _test_deterministic
    from atlas.transfer import InterfaceSpace, Prolongation

    space = InterfaceSpace(seam_id="s", dim=4)
    P = Prolongation(agent_id="t", port_name="p:MECH", matrix=np.eye(4),
                     gram_V=np.eye(4))
    c = _caps(deterministic=True, boundary_response=lambda _p, x: 2.0 * np.asarray(x) + 1.0)
    assert _test_deterministic(c, space, P).verdict.value == "admit"


def test_support_radius_falsifies_an_under_declared_stencil():
    """A response that reaches further along the seam than the record allows."""
    from atlas.conformance import _test_support_radius
    from atlas.transfer import InterfaceSpace, Prolongation

    n = 32
    space = InterfaceSpace(seam_id="s", dim=n)
    P = Prolongation(agent_id="t", port_name="p:MECH", matrix=np.eye(n),
                     gram_V=np.eye(n))
    # a 5-cell moving average: the response reaches 5 cells from any poke
    def respond(_p, x):
        x = np.asarray(x, float)
        return sum(np.roll(x, k) for k in range(-5, 6)) / 11.0

    c = _caps(time_discretization=TimeDiscretization.EXPLICIT,
              stencil_radius=1, substeps_per_macro_step=1,
              boundary_response=respond)
    t = _test_support_radius(c, space, P)
    assert t.verdict.value == "refuse"
    assert t.measured == 5 and t.declared == 1


def test_support_radius_admits_a_response_inside_the_declared_reach():
    from atlas.conformance import _test_support_radius
    from atlas.transfer import InterfaceSpace, Prolongation

    n = 32
    space = InterfaceSpace(seam_id="s", dim=n)
    P = Prolongation(agent_id="t", port_name="p:MECH", matrix=np.eye(n),
                     gram_V=np.eye(n))
    c = _caps(time_discretization=TimeDiscretization.EXPLICIT,
              stencil_radius=3, substeps_per_macro_step=1,
              boundary_response=lambda _p, x: 2.0 * np.asarray(x))
    t = _test_support_radius(c, space, P)
    assert t.verdict.value == "admit"
    assert t.measured == 0


def test_governing_family_is_reported_as_unverifiable_rather_than_untested():
    """W69's third one.  The entry exists so the gap is visible in the report."""
    from atlas.conformance import _test_governing_family

    t = _test_governing_family(_caps(governing_family="navier-stokes-2d"))
    assert t.verdict.value == "admit-uncertified"
    assert t.measured == "no test exists"
    assert "THIRD unverifiable declaration" in t.message


def test_the_conformance_suite_reports_three_unverifiable_fields():
    """The count itself, so a fourth cannot be added without this failing."""
    from atlas.conformance import run_conformance

    cert = run_conformance(_caps(governing_family="x", validity=lambda *a: True))
    names = {t.field_name for t in cert.tests if t.cost == "none exists"}
    assert names == {"validity", "governing_family"}
    # response_half is the third; it is checked at L3/C9 across a PAIR of
    # records rather than on one, which is exactly why it has no FieldTest.
    from atlas.ports import ResponseHalf, check_response_half
    from atlas.ports import PortType

    assert check_response_half(PortType.MECH, ResponseHalf.EFFORT,
                               ResponseHalf.EFFORT).ok


# ---------------------------------------------------------------------------
# W74 -- the probe's linearization point
# ---------------------------------------------------------------------------


def test_probe_base_defaults_to_zeros_so_every_earlier_measurement_stands():
    c = _caps()
    assert np.array_equal(probe_base(c, "p:MECH", 5), np.zeros(5))


def test_probe_base_uses_the_declared_operating_point():
    c = _caps(probe_base=lambda _p: np.full(5, 900.0))
    assert np.array_equal(probe_base(c, "p:MECH", 5), np.full(5, 900.0))


def test_a_probe_base_of_the_wrong_length_raises_rather_than_broadcasting():
    c = _caps(probe_base=lambda _p: np.full(3, 900.0))
    with pytest.raises(ProbeError, match="expected 5"):
        probe_base(c, "p:MECH", 5)


def test_the_base_is_free_for_an_affine_expert_and_not_for_a_nonlinear_one():
    """W74's whole content, in four lines of algebra.

    The old probe subtracted ``respond(0)`` and called that a bias removal.  For
    an affine response the derivative is the same wherever it is taken, so the
    base is genuinely free.  For ``q/T`` -- which is what PORT_SPECS[THERM]
    declares, chosen so effort times flow is a power -- it is not.
    """
    from atlas.probe import _columns_finite_difference

    d = np.eye(3)

    affine = _caps(boundary_response=lambda _p, x: 3.0 * np.asarray(x) + 7.0)
    at_zero, _ = _columns_finite_difference(affine, "p:MECH", d, 1e-4)
    shifted = _caps(probe_base=lambda _p: np.full(3, 900.0),
                    boundary_response=lambda _p, x: 3.0 * np.asarray(x) + 7.0)
    at_900, _ = _columns_finite_difference(shifted, "p:MECH", d, 1e-4)
    assert np.allclose(at_zero, at_900, atol=1e-6)

    # the declared THERM bond, in miniature: flow = q/T with q linear in T
    def therm(_p, x):
        T = np.asarray(x, float) + 400.0        # the other side of the seam
        return (500.0 * (np.asarray(x, float) - 400.0)) / T

    a = _caps(boundary_response=therm)
    b = _caps(probe_base=lambda _p: np.full(3, 900.0), boundary_response=therm)
    S0, _ = _columns_finite_difference(a, "p:MECH", d, 1e-4)
    S9, _ = _columns_finite_difference(b, "p:MECH", d, 1e-4)
    ratio = float(np.linalg.norm(S0) / np.linalg.norm(S9))
    assert ratio > 3.0, f"the base should matter here and it moved only {ratio:.2f}x"


# ---------------------------------------------------------------------------
# W72 -- a positive control per escape, generated rather than listed
# ---------------------------------------------------------------------------


def test_every_escape_in_the_derived_table_has_a_detector(tmp_path):
    """The point of W72: a byte in the table with no detector is a FAILING test.

    The tab was found by accident after W38 had already prescribed scanning for
    it, because the list was hand-written and one entry was missed.  This
    generates a corrupted page per escape and asserts the scanner objects.
    """
    import vault_scan

    assert set(vault_scan.ESCAPES) >= set("abfnrtv"), vault_scan.ESCAPES
    for letter, byte in sorted(vault_scan.ESCAPES.items()):
        word = (vault_scan.TEX_WORDS.get(letter) or ("x",))[0]
        # what the page SHOULD say, and what a string literal turns it into
        good = f"The coefficient $\\{word}$ is measured below.\n"
        bad = good.replace(f"\\{word}", chr(byte) + word[1:])
        p = tmp_path / f"corrupt_{letter}.md"
        p.write_bytes(bad.encode("utf-8"))
        problems = vault_scan.scan(str(p))
        assert problems, (
            f"escape \\{letter} -> 0x{byte:02x} (as in \\{word}) produced a page "
            f"the scanner does not flag: {bad!r}"
        )


def test_the_scanner_accepts_the_uncorrupted_page(tmp_path):
    """The negative control the positive one is worthless without."""
    import vault_scan

    p = tmp_path / "clean.md"
    p.write_bytes(
        ("# Title\n\nThe coefficient $\\tau$ and $\\nu$ and $\\rho$ are fine.\n\n"
         "| a | b |\n|---|---|\n| $\\alpha$ | $\\beta$ |\n\n"
         "```\n$ unbalanced inside a fence\n```\n\n"
         "$$\n\\nabla \\cdot u = 0\n$$\n\n"
         "A price of \\$5 and a span `$\\langle U` quoted in code.\n"
         ).encode("utf-8")
    )
    assert vault_scan.scan(str(p)) == []


def test_a_split_table_row_is_caught_without_needing_three_contiguous_lines(tmp_path):
    """The wind-farm spec's W11 row, reproduced.

    It survived because the `\\r` became a plain LF (invisible to a byte scan),
    the split left a two-line fragment (below the ragged check's minimum), and
    the tail was `vert` (on no list).
    """
    import vault_scan

    p = tmp_path / "split.md"
    p.write_bytes(
        ("| **W11** | Residual | $\\lvert\\mathcal R(t)\nvert$ below 1% |\n"
         ).encode("utf-8")
    )
    problems = vault_scan.scan(str(p))
    assert any("never closes it" in x for x in problems), problems
    assert any("unpaired '$'" in x for x in problems), problems


def test_the_vault_itself_is_clean():
    """The standing verification, run as a test rather than by hand."""
    r = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "scripts", "vault_scan.py"),
         os.path.join(_ROOT, "wiki")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout[-3000:]
    assert "0 problems" in r.stdout


def test_every_vault_link_resolves_or_is_pinned():
    """The link half of the standing verification (2026-09-29).

    `vault_scan` reads bytes and never looks at links, so it passed a vault with
    13 dead ones.  Every dead link left on purpose is pinned in
    `link_scan.KNOWN` with its count; anything else fails, and so does a pin
    that no longer matches.
    """
    r = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "scripts", "link_scan.py"),
         os.path.join(_ROOT, "wiki")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout[-3000:]
    assert " 0 new, 0 stale" in r.stdout, r.stdout[-3000:]
    # the scan has to have read something for "0 new" to mean anything
    n_links = int(r.stdout.strip().splitlines()[-1].split(" files, ")[1].split(" links")[0])
    assert n_links > 1000, r.stdout[-500:]


def test_the_link_scanner_flags_a_dead_link_and_nothing_else(tmp_path):
    """Positive and negative controls in one vault: only `[[missing]]` is dead."""
    import link_scan

    sub = tmp_path / "concepts" / "deep"
    sub.mkdir(parents=True)
    (sub / "present.md").write_text("# Present\n", encoding="utf-8")
    (tmp_path / "figure.png").write_bytes(b"")
    (tmp_path / "page.md").write_text(
        "See [[present]], [[present|an alias]], [[present#A heading]] and ![[figure.png]].\n"
        "| a | [[present\\|in a table]] |\n"
        "Quoted, not linked: `[[quoted-dead]]`.\n"
        "```\n[[fenced-dead]]\n```\n"
        "And one that is really dead: [[missing]].\n",
        encoding="utf-8",
    )
    n_files, n_links, dead, new, stale = link_scan.check(str(tmp_path), known={})
    assert dead == [("page.md", 7, "missing")], dead   # the fence is lines 4-6
    assert new == dead and stale == []
    assert n_files == 2 and n_links == 6, (n_files, n_links)


def test_a_link_pin_goes_stale_when_its_link_is_fixed(tmp_path):
    """A pin must not outlive what it excuses, in either direction."""
    import link_scan

    (tmp_path / "page.md").write_text("[[gone]] and [[gone]]\n", encoding="utf-8")
    exact = {("page.md", "gone"): (2, "test")}
    assert link_scan.check(str(tmp_path), known=exact)[3:] == ([], [])
    fewer = {("page.md", "gone"): (1, "test")}
    assert link_scan.check(str(tmp_path), known=fewer)[4] == [(("page.md", "gone"), 1, 2)]
    (tmp_path / "gone.md").write_text("# Now it exists\n", encoding="utf-8")
    assert link_scan.check(str(tmp_path), known=exact)[4] == [(("page.md", "gone"), 2, 0)]


# ---------------------------------------------------------------------------
# W70 -- the scope statement is a statement, so it has to say the true thing
# ---------------------------------------------------------------------------


def test_the_shells_thermoelastic_coupling_is_one_way():
    """The claim VOLUMETRIC_COUPLING_SCOPE rests on, read off the solver."""
    import inspect

    from atlas.cases import thermal_seam as T

    _C2, TS, _TH, _GR = T.load_solvers()
    params = set(inspect.signature(TS.ThermoStruct2D.step_thermal).parameters)
    assert not (params & {"u", "disp", "displacement", "strain", "sigma",
                          "stress", "eps"})


def test_the_scope_statement_names_the_measurement_and_not_just_the_gap():
    from atlas.cases.thermal_seam import VOLUMETRIC_COUPLING_SCOPE

    assert "one-way" in VOLUMETRIC_COUPLING_SCOPE
    assert "4.8e-3" in VOLUMETRIC_COUPLING_SCOPE
    assert "W70" in VOLUMETRIC_COUPLING_SCOPE


def test_the_shell_is_still_declared_embedded_after_W68():
    """W68 changed what the STATISTIC says, not what the record declares."""
    from atlas.cases import thermal_seam as T

    caps = T.shell_capabilities(T.ShellAgent(dt=T.DT_SHELL))
    assert caps.elliptic_subsolve is EllipticSubsolve.EMBEDDED

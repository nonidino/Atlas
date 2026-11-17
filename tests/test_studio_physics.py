"""P3: the equation registry and the regime numbers (integrated-ui-plan §4, §8).

Done-when: every equation in the list either runs or says why not on hover, and every
regime number is checked against a hand computation on one case.  The hand
computations below use the numbers written out, not the registry's code.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest  # noqa: E402

from atlas.studio import protocol, registry_eq as R  # noqa: E402
from atlas.studio.session import Session  # noqa: E402
from atlas.workbench import registry  # noqa: E402
from atlas.workbench.spec import example_case  # noqa: E402


def test_every_equation_runs_or_says_why_not():
    rows = R.equations()
    assert len(rows) >= 40
    for e in rows:
        assert e["runs"] or len(e["why_not"]) > 20, e["id"]
        assert e["kind"] in ("elliptic", "parabolic", "hyperbolic", "mixed")
        assert e["group"] in R.GROUPS and e["html"]
    # every family that runs here is some equation's solver
    runs = {e["runs_with"] for e in rows if e["runs"]}
    assert runs == set(registry.available_ids())
    # the chart-operator tag is CO-D's alone, today
    assert {e["group"] for e in rows if e["co"]} == {"A"}
    # an equation that runs has its properties filled
    for e in rows:
        if e["runs"]:
            assert e["order"] and e["linear"] is not None and e["self_adjoint"] is not None, e["id"]
            assert e["conserves"] and e["uses"], e["id"]


def test_guarantee_classes_follow_the_catalog():
    by = R.BY_ID
    assert by["heat"].guarantee == "G0" and by["navier-stokes"].guarantee == "G2"
    assert by["heat-nonlinear"].guarantee == "G1" and by["acoustics"].guarantee == "G3"
    assert by["advection-diffusion"].guarantee.startswith("G0′")
    for cls in ("G0", "G0′", "G1", "G2", "G3"):
        assert R.CLASSES[cls]


def test_regime_numbers_against_a_hand_computation():
    # fast-heat: steel and copper, transient; copper diffuses fastest
    s = example_case("fast-heat")
    r = R.regime(s)
    alpha_cu = 400.0 / (8960.0 * 385.0)                      # 1.15955e-4 m^2/s
    dx, dt = 0.000625, s.run.macro_dt
    assert r["contrast"]["value"] == pytest.approx(400.0 / 45.0, rel=1e-12)
    assert r["alpha"]["material"] == "copper"
    assert r["fo_cell"]["value"] == pytest.approx(alpha_cu * dt / dx ** 2, rel=1e-12)
    area = 640 * 384 * dx * dx
    assert r["area"] == pytest.approx(area, rel=1e-12)
    assert r["fo_domain"]["value"] == pytest.approx(alpha_cu * dt / area, rel=1e-12)
    assert r["fo_domain"]["range"] == [1e-2, 1e2]
    assert r["fo_domain"]["within"] == (1e-2 <= alpha_cu * dt / area <= 1e2)
    # plume-2: q = 1 m^2/s; the shallow reach (depth 1.25, mixing 0.5) runs fastest
    s = example_case("plume-2")
    r = R.regime(s)
    L = 480 * 5.0
    assert r["pe"]["value"] == pytest.approx((1.0 / 1.25) * L / 0.5, rel=1e-12)
    assert r["pe"]["cell"] == pytest.approx((1.0 / 1.25) * 5.0 / 0.5, rel=1e-12)
    assert r["contrast"]["value"] == pytest.approx(2.0 / 0.5)
    # wake-array-3: Re_D = U D / nu with D = 1 (lengths in diameters): the referent's 255
    s = example_case("wake-array-3")
    r = R.regime(s)
    assert r["re"]["value"] == pytest.approx(1.0 / 0.00392, rel=1e-12)
    assert round(r["re"]["value"]) == 255
    # fast-sound: a Gaussian pulse of standard deviation 0.1 m on a 0.01 m grid
    s = example_case("fast-sound")
    r = R.regime(s)
    assert r["k"]["value"] == pytest.approx(30.0) and r["k"]["kh"] == pytest.approx(0.3)
    assert r["k"]["within"] is True                           # 0.3 < pi sqrt 2
    assert r["contrast"]["value"] == pytest.approx(1480.0 / 343.0)
    # cooled-block: steady (no Fourier number); the coolant's Peclet number
    s = example_case("cooled-block")
    r = R.regime(s)
    alpha_w = 0.6 / (1000.0 * 4180.0)
    assert r["fo_domain"]["value"] is None
    assert r["pe"]["value"] == pytest.approx(0.01 * 0.2 / alpha_w, rel=1e-12)
    assert r["contrast"]["value"] == pytest.approx(400.0 / 0.6)


def test_the_page_receives_the_equations_and_the_regime():
    cat = Session.catalog()
    assert len(cat["equations"]) == len(R.EQUATIONS) and set(cat["classes"]) >= {"G0", "G2"}
    s = Session(spec=example_case("fast-heat"), key="x")
    reply, pushes = protocol.handle(s, {"id": 1, "type": "state"})
    st = next(p for p in pushes if p["type"] == "state")
    assert st["regime"]["equation"] == "heat"
    assert st["regime"]["fo_cell"]["value"] == pytest.approx(R.regime(s.spec)["fo_cell"]["value"])
    assert math.isfinite(st["regime"]["contrast"]["value"])

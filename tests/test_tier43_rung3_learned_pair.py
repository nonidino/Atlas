"""Tier 43 -- rung 3: two learned experts of different families in one graph.

`f1-pathmap-and-end-goal` section 3 defines rung 3 as *"a second learned expert
joins an existing graph; first true multi-family coupling"*, and section 3.3
records that no graph in `atlas/cases` has ever held two.  CS-17 builds it, and
what this file pins is a **result**, not a defect: the graph compiles, with zero
refusals, and every cost it carries is attributable by a 2x2.

Three kinds of assertion.

**The measured result.**  Numbers are asserted against `out/rung3/rung3.json`
rather than retyped, on the Tier 39 discipline -- a figure in a page and a
figure in a rule must not be able to drift apart.  Run
`python scripts/rung3_learned_pair.py` first; the tests skip without it.

**The controls, and they are what make the result mean anything.**  CC is the
classical incumbent on the identical geometry; the 2x2 attributes every extra
decertification to one side or the other; and `L2/R10`'s positive control fires
in the same compile, so *the rule did not refuse* is evidence rather than
absence.

**The live checks.**  Coherence is a property of the declarations, so it is
checked live rather than read off the artifact: neither learned expert declares
a `THERM` port, which is why the brief's `thermal_seam` template is not
constructible and why the rung-3 seam is `MECH`.

Nothing here loads NeuberNet.  It is unlicensed and local-only, and the suite
must pass on a machine that has never had it.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.ports import PortType                                      # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT = os.path.join(_ROOT, "out", "rung3", "rung3.json")

CELLS = ("LL", "LC", "CL", "CC")


@pytest.fixture(scope="module")
def r3():
    if not os.path.exists(ARTIFACT):
        pytest.skip("run scripts/rung3_learned_pair.py")
    with open(ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# the headline
# --------------------------------------------------------------------------


def test_rung3_the_graph_COMPILES_and_that_is_the_result(r3):
    """Two learned experts of different families, and nothing refuses.

    The brief expected a refusal. Every cell of the 2x2 reaches
    `admit-uncertified` with an empty refusal list, including the one holding
    both checkpoints.
    """
    for key in CELLS:
        cell = r3["cells"][key]
        assert cell["verdict"] == "admit-uncertified", key
        assert cell["runnable"] is True, key
        assert cell["refusals"] == [], (key, cell["refusals"])


def test_rung3_R10_positive_control_fires_in_the_same_compile(r3):
    """*The rule did not refuse* is worth nothing without a cell where it does.

    Same two-agent shape, same rule, one declaration changed: with a second
    agent of the EMBEDDED agent's own `governing_family` present, `L2/R10`
    refuses. Rung 3's graph is the other case, so its zero refusals are W114's
    premise clearing rather than the rule being absent.
    """
    ctl = r3["r10_positive_control"]
    assert ctl["same_family"]["verdict"] == "refuse"
    assert ctl["same_family"]["runnable"] is False
    assert "L2/R10" in ctl["same_family"]["refusing_rules"]
    assert ctl["different_family"]["verdict"] == "admit-uncertified"
    assert ctl["different_family"]["refusing_rules"] == []


def test_rung3_the_measured_elliptic_declaration_changes_no_verdict(r3):
    """W60's axis, compiled all three ways rather than argued.

    `poseidon_capabilities` defaults `elliptic_subsolve` to NONE and W60 records
    that as undeclarable; W93 MEASURED it -- support_reach 64 of 128, with
    `elliptic_signature` agreeing -- so EMBEDDED is the measured value. R10
    refuses an EMBEDDED agent only when another agent of its family is present,
    and here there is none, so declaring the measured value refuses nothing.
    """
    axis = r3["elliptic_axis"]
    assert set(axis) == {"none", "embedded", "unknown"}
    for label, cell in axis.items():
        assert cell["verdict"] == "admit-uncertified", label
        assert cell["refusals"] == [], label
    # Declaring the MEASURED value is not merely harmless, it is cleaner: the
    # undeclared-pressure-solve branch stops firing and R10 says nothing.
    assert "L2/R10" in axis["none"]["decertifications"]
    assert "L2/R10" not in axis["embedded"]["decertifications"]
    assert "L2/R10/W60" in axis["unknown"]["decertifications"]


# --------------------------------------------------------------------------
# what the second learned expert costs, attributed
# --------------------------------------------------------------------------


def test_rung3_the_decertification_ladder_is_ten_eleven_twelve_thirteen(r3):
    counts = {k: len(r3["cells"][k]["decertifications"]) for k in CELLS}
    assert counts == {"CC": 10, "CL": 11, "LC": 12, "LL": 13}


def test_rung3_the_cost_of_two_learned_experts_is_EXACTLY_additive(r3):
    """The question rung 3 exists to ask, and the answer is: no interaction term.

    What the pair costs at the rule level is the union of what each costs alone
    -- set equality, not a count.
    """
    base = set(r3["cells"]["CC"]["decertifications"])
    only = {k: set(r3["cells"][k]["decertifications"]) - base for k in ("CL", "LC", "LL")}
    assert only["LL"] == only["CL"] | only["LC"]
    assert only["CL"] == {"L4/E7/passivity"}
    assert only["LC"] == {"L2/R10", "L4/R2b/W46"}
    assert only["CL"] & only["LC"] == set()


def test_rung3_passivity_is_attributable_to_the_LEARNED_solid(r3):
    """E7 fails wherever NeuberNet is and holds wherever it is not.

    The classical patch on the identical ring at the identical base has a
    passivity defect of exactly zero. This is not W165's non-associated pair --
    it is a third case, a learned operator that is not passive because nothing
    made it so -- and it is separated by a control rather than by argument.
    """
    defects = {k: r3["cells"][k]["passivity_defect"] for k in CELLS}
    assert defects["CC"] == 0.0 and defects["LC"] == 0.0
    assert defects["CL"] > 0.5 and defects["LL"] > 1.0
    e7 = {k: r3["cells"][k]["envelope"][6] for k in CELLS}
    assert e7 == {"LL": "fails", "CL": "fails", "LC": "holds", "CC": "holds"}


def test_rung3_the_reference_pair_survives_ONE_learned_expert_and_not_two(r3):
    """tau is defined in CC and undefined in all three cells holding a checkpoint.

    A reference pair is a PAIR. With one learned expert the classical side still
    declares `lambda_ref` and the missing half has a peer to appeal to; with two
    there is nothing on either side to appeal to. That is the difference the
    substitution campaign, which only ever swaps one side, cannot exhibit.
    """
    assert r3["cells"]["CC"]["tau_undefined_seams"] == []
    for key in ("LL", "LC", "CL"):
        assert r3["cells"][key]["tau_undefined_seams"] == ["fsi"], key
    msg = r3["cells"]["LL"]["messages"]["L1/E3/fsi"]
    assert "carry no lambda_ref" in msg
    assert "tau is emitted as UNDEFINED for both" in msg


# --------------------------------------------------------------------------
# the interface space, and the block the certificate cannot see
# --------------------------------------------------------------------------


def test_rung3_the_learned_expert_sets_the_interface_space(r3):
    """dim M = min_i m_i_eff, and a checkpoint measures far lower than its control.

    On one 29-sensor ring at one base through one instrument, NeuberNet's
    effective rank at 99% of the energy is 3 and the classical patch's is 25.
    """
    dm = r3["dim_M"]
    assert dm["neubernet_m_eff_measured"] == 3
    assert dm["classical_m_eff_measured"] == 25
    assert dm["declared_for_every_cell"] == 3
    assert dm["per_cell_if_each_declared_its_own"] == {"LL": 3, "CL": 3, "LC": 16, "CC": 16}
    # and the measurement that set it
    nn = r3["port"]["neubernet"]
    assert nn["n"] == 29 and nn["effective_rank_99"] == 3
    assert nn["repeat_call_relative"] == 0.0, "the checkpoint stopped being bitwise repeatable"


def test_rung3_the_fluid_block_is_invisible_to_the_certificate_in_every_cell(r3):
    """W97's class on a third graph, and W137's number again.

    The fluid's share of the assembled operator is 1.5e-5 with the learned
    fluid -- the order W137 measured at `wing_fsi`'s aero-structure seam -- and
    5.5e-4 with the classical one, so the seam is the solid's either way and
    123x more so when the fluid is a checkpoint.
    """
    share = {k: r3["cells"][k]["blocks"]["FLUID"] / r3["cells"][k]["norm_S"]
             for k in CELLS}
    assert share["LL"] == pytest.approx(1.49e-5, rel=0.05)
    assert share["CC"] == pytest.approx(5.46e-4, rel=0.05)
    for key in CELLS:
        assert share[key] < 1e-2, key
    # the learned fluid responds to its boundary data 123x less than the
    # classical one on the identical geometry
    ratio = r3["cells"]["CC"]["blocks"]["FLUID"] / r3["cells"]["LL"]["blocks"]["FLUID"]
    assert ratio == pytest.approx(123.0, rel=0.05)
    assert "the substitution certificate is BLIND at this seam" in \
        r3["cells"]["LL"]["messages"]["L4/block-share/fsi"]


# --------------------------------------------------------------------------
# coherence -- live, because it is a property of the declarations
# --------------------------------------------------------------------------


def test_the_thermal_seam_template_is_not_constructible():
    """Neither learned expert carries a temperature, so THERM has no counterpart.

    Checked live off `poseidon_capabilities` rather than read off the artifact,
    so a port added to the checkpoint's record fails this rather than sliding
    past it. NeuberNet is not loaded: its port is displacement-in,
    traction-out, which is MECH by construction and is asserted from the
    adapter's declared component list in the artifact test above.
    """
    import numpy as np

    from atlas.cases import poseidon as PO

    agent = PO.PoseidonAgent.__new__(PO.PoseidonAgent)   # no checkpoint load
    agent.agent_id = "P"
    agent.n = PO.EXPERT_RES
    agent.dt = PO.MACRO_DT
    agent.shared_faces = ("xhi",)
    agent.storage = lambda *a, **k: 0.0
    agent.respond = lambda *a, **k: np.zeros(PO.EXPERT_RES)
    caps = PO.poseidon_capabilities(agent)
    types = {p.port_type for p in caps.ports}
    assert types == {PortType.MECH}
    assert PortType.THERM not in types
    assert caps.governing_family == "incompressible-navier-stokes-2d"
    # and the record carries no referent, which is what breaks the pair
    assert caps.lambda_ref is None


def test_rung3_coherence_is_recorded_with_its_geometric_caveat(r3):
    """The seam is a fiction and the page says so; this pins that it said so.

    A fluid face is a wetted surface and NeuberNet's ring is an internal cut in
    a solid. `PortDecl.geometry` is free text, so no rule can tell them apart --
    `Agent.domain`'s problem one object down. The finding is the schema's
    silence, not the fixture's virtue, and it must stay written down.
    """
    coh = r3["coherence"]
    assert coh["either_declares_THERM"] is False
    assert coh["poseidon_port_types"] == ["MECH"]
    assert "NOT constructible" in coh["verdict"]
    assert "INTERNAL CUT" in coh["geometry_caveat"]
    assert "free text" in coh["geometry_caveat"]

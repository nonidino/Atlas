"""Tier 74 -- what a person needs to run this without being here.

[[poc3-racelab-showable]].

Two things, and both are section 12 criteria rather than polish:

  * **criterion 1** -- one clone, one command, a car simulating, on Windows AND
    on macOS or Linux.  The bundle's file lists stopped at Tier 57, so it shipped
    the porous column and none of the body-fitted one.
  * **criterion 2, for the six geometry knobs** -- a viewer could move one,
    commit it, and watch the column spend ninety seconds re-cutting the car
    before declining to march it (W293).  The refusal now happens before the
    rebuild, with its reason on the button.
"""

from __future__ import annotations

import os
import re

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

import atlas.demo_racelab.bodyfitted as BF

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(_ROOT, "atlas", "demo_racelab", "static", "bodyfitted.html")
BUILDER = os.path.join(_ROOT, "scripts", "build_racelab_bundle.py")
TEMPLATES = os.path.join(_ROOT, "atlas", "demo_racelab", "bundle")

#: The builder and the launcher templates exist only UPSTREAM: the bundle is
#: what they produce, so it does not carry them.  Every test about them skips
#: there and says why, which is the rule `test_tier57_racelab_bundle_records`
#: already follows -- so this file can be carried unchanged and the bundle has
#: every PoC 3 test.
upstream_only = pytest.mark.skipif(
    not os.path.isfile(BUILDER),
    reason="the bundle builder lives upstream; this bundle is its output")
templates_only = pytest.mark.skipif(
    not os.path.isdir(TEMPLATES),
    reason="the launcher templates live upstream; the bundle carries the "
           "files they produce, at its root")


# ---------------------------------------------------------------------------
# W293 -- refused before the rebuild, not after it
# ---------------------------------------------------------------------------


def test_nothing_pending_is_not_a_refusal():
    """`nothing_pending` and `no_spun_up_field` are different states and the
    page treats them differently: one greys the button quietly, the other greys
    it loudly with a reason."""
    c = BF.BodyFittedColumn().commit_check(_ROOT)
    assert c["ok"] is False
    assert c["kind"] == "nothing_pending"


def test_the_nominal_car_can_be_committed():
    """The control that keeps the refusal from being vacuous: the car this
    repository builds HAS a settled field, so a commit to it is allowed."""
    col = BF.BodyFittedColumn()
    col.pending_regrid = ["rake"]
    c = col.commit_check(_ROOT)
    if not c["ok"]:
        pytest.skip("this checkout has no settled field for the nominal car: %s"
                    % c.get("settled"))
    assert c["kind"] == "has_settled_field"
    assert c["settled"].endswith(".npz")


def test_a_moved_geometry_knob_is_refused_with_its_reason_and_a_remedy():
    """**The refusal costs nothing and names the car it would have built.**

    `geometry_fingerprint` is a function of the PARAMETERS, so the car a commit
    would produce can be named without building it -- which is what makes the
    check affordable on every frame rather than only on the click.
    """
    col = BF.BodyFittedColumn()
    st = col.knob_state()
    st.set("rake", 0.10)
    col.pending_regrid = ["rake"]
    c = col.commit_check(_ROOT)
    assert c["ok"] is False
    assert c["kind"] == "no_spun_up_field"
    assert "W293" in c["reason"]
    assert c["what_would_make_it_possible"]
    # it names the car it WOULD build, not the one marching
    assert re.search(r"_[0-9a-f]{12}\.npz$", c["settled"])

    # ... and moving the knob back restores the commit, so the refusal tracks
    # the geometry rather than latching
    st.set("rake", 0.0)
    back = col.commit_check(_ROOT)
    if back["ok"]:
        assert back["kind"] == "has_settled_field"
        assert back["settled"] != c["settled"]


def test_the_server_refuses_the_commit_before_it_rebuilds():
    """The order matters: `commit_check` must be consulted BEFORE `regrid`, or
    the viewer pays ninety seconds to be told no and loses the marching car."""
    pytest.importorskip("fastapi")
    import atlas.demo_racelab.bodyfitted_server as S

    src = open(S.__file__, encoding="utf-8").read()
    body = src.split("def _commit", 1)[1].split("def _publish", 1)[0]
    assert "commit_check" in body, "the commit does not consult the check"
    # the CALL, not the substring: `pending_regrid` contains "regrid" and sits
    # above the check, so a substring test here passes on the wrong match and
    # fails on the right code
    assert body.index("commit_check") < body.index("self.col.regrid("), \
        "the check must come before the rebuild, or it saves nothing"
    assert "commit_refused" in body


def test_the_page_greys_the_button_and_says_why():
    text = open(STATIC, encoding="utf-8").read()
    assert 'id="commitBlocked"' in text
    assert "commit_check" in text
    assert "Commit refused" in text
    assert "what_would_make_it_possible" in text
    # the refusal must not be drawn for the merely-nothing-pending case
    assert 'chk.kind !== "nothing_pending"' in text


# ---------------------------------------------------------------------------
# criterion 1 -- the bundle carries the column it claims to
# ---------------------------------------------------------------------------


def _builder_src() -> str:
    return open(BUILDER, encoding="utf-8").read()


@upstream_only
def test_the_bundle_ships_every_script_the_body_fitted_demo_imports():
    """`BodyFittedColumn.build` imports `tier62_car_solids` and
    `tier63_duct_openings` at build time, and `_devices` imports the second
    again.  Without them in the bundle the body-fitted page does not start --
    and nothing else in the bundle would notice."""
    import scripts.build_racelab_bundle as B  # noqa: F401  (import for parity)

    src = _builder_src()
    for s in ("tier62_car_solids.py", "tier63_duct_openings.py"):
        assert '"%s"' % s in src, s
    # every tier script the body-fitted column's tests import
    for n in range(58, 74):
        assert re.search(r'"tier%d_[a-z0-9_]+\.py"' % n, src), \
            "scripts/tier%d is not in the bundle" % n


@upstream_only
def test_the_bundle_ships_every_body_fitted_test():
    src = _builder_src()
    for n in range(58, 74):
        assert re.search(r'"test_tier%d_[a-z0-9_]+\.py"' % n, src), \
            "tests/test_tier%d is not in the bundle" % n


@upstream_only
def test_the_bundle_ships_the_records_those_tests_read():
    src = _builder_src()
    for n in range(6, 23):
        assert '"out/racelab%d/racelab%d.json"' % (n, n) in src, \
            "out/racelab%d is not in the bundle" % n


@upstream_only
def test_the_bundle_ships_the_field_the_body_fitted_demo_releases_from():
    """It is named for the car, so the name IS the check.  Without it the page
    settles for about four minutes before its first frame, or refuses."""
    src = _builder_src()
    assert "_bodyfitted_settled" in src
    assert "bodyfitted_t" in src
    # and the data scan must ALLOW it, or the build refuses its own bundle
    assert re.search(r"bodyfitted_t.*npz", src)
    import scripts.build_racelab_bundle as B

    rel, name = B._bodyfitted_settled()
    assert name.startswith("bodyfitted_t")
    assert any(p.match(rel) for p in B.BINARY_ALLOWED), \
        "the settled field is not in BINARY_ALLOWED, so the build would refuse it"


@upstream_only
def test_the_bundle_ships_the_pages_the_new_tests_open():
    """A page left out turns a passing test into a skip, silently."""
    src = _builder_src()
    for page in ("poc3-racelab-body-fitted-grids.md", "poc3-racelab-car-solids.md",
                 "poc3-racelab-car-union.md", "poc3-racelab-dashboard.md",
                 "poc3-racelab-certified-step.md",
                 "poc3-racelab-certified-screen.md"):
        assert '"%s"' % page in src, page


@upstream_only
def test_every_bundled_page_exists_in_the_vault():
    import scripts.build_racelab_bundle as B

    wiki = os.path.join(_ROOT, "wiki")
    have = {f for _dp, _dn, fn in os.walk(wiki) for f in fn}
    missing = [p for p in B.WIKI_PAGES if p not in have]
    assert not missing, "the bundle names pages the vault does not have: %s" % missing


@upstream_only
def test_every_bundled_script_and_test_exists():
    import scripts.build_racelab_bundle as B

    for s in B.SCRIPTS:
        assert os.path.isfile(os.path.join(_ROOT, "scripts", s)), s
    for t in B.TESTS:
        assert os.path.isfile(os.path.join(_ROOT, "tests", t)), t


# ---------------------------------------------------------------------------
# the launcher, and the check that is not the engine
# ---------------------------------------------------------------------------


def test_one_command_reaches_either_column():
    from atlas.demo_racelab.__main__ import COLUMNS, PORTS

    assert set(COLUMNS) == {"porous", "body-fitted"}
    assert PORTS["porous"] != PORTS["body-fitted"], \
        "two engines on one port is one engine"


def test_the_body_fitted_server_has_its_own_run_and_socket_check():
    """Passing the porous column's socket check says nothing about this one:
    Tier 71 found every upgrade to THIS module's /ws refused 403 while the
    porous column was fine throughout."""
    pytest.importorskip("fastapi")
    import atlas.demo_racelab.bodyfitted_server as S

    assert callable(S.run)
    assert callable(S.socket_check)
    assert S._PIXEL_PNG[:8] == b"\x89PNG\r\n\x1a\n"


@templates_only
def test_shapely_is_declared_because_nothing_import_time_can_see_it():
    """**The dependency the demo has always needed and never declared.**

    `car_solids` and `car_union` import `shapely` INSIDE the functions that use
    it, so it was in no requirements file, no import-time check missed it, and
    it was present here because Anaconda ships it. On a clean Linux machine the
    body-fitted page reached "cutting the solids" and stopped -- while
    `run.py --check` passed.
    """
    req = open(os.path.join(_ROOT, "atlas", "demo_racelab", "bundle",
                            "requirements.txt"), encoding="utf-8").read()
    assert re.search(r"^shapely==\d", req, re.M), \
        "shapely is not pinned, so a clean machine cannot cut the car's solids"

    run_py = os.path.join(_ROOT, "atlas", "demo_racelab", "bundle", "run.py")
    src = open(run_py, encoding="utf-8").read()
    required = src.split("REQUIRED = [", 1)[1].split("]", 1)[0]
    assert '"shapely"' in required, \
        "a missing shapely must be reported as a missing PACKAGE, not as a " \
        "progress overlay that stops"

    # and the check must import what the column imports while building, or it
    # goes on passing while the page cannot start
    body = src.split("def body_fitted_check", 1)[1]
    for mod in ("tier62_car_solids", "tier63_duct_openings", "car_solids",
                "car_union", "shapely"):
        assert mod in body, mod


def test_the_lazy_imports_are_still_where_this_test_thinks_they_are():
    """The control for the test above: if `shapely` ever moves to module level
    the reasoning changes, and this says so rather than quietly still passing."""
    for mod in ("car_solids", "car_union"):
        src = open(os.path.join(_ROOT, "atlas", "cases", "%s.py" % mod),
                   encoding="utf-8").read()
        head = src.split("def ", 1)[0]
        assert "import shapely" not in head and "from shapely" not in head, \
            "%s now imports shapely at module level; the packaging note in " \
            "requirements.txt explains a hazard that no longer exists" % mod
        assert "shapely" in src, "%s no longer uses shapely at all" % mod


@templates_only
def test_the_launcher_checks_the_body_fitted_column_on_both_paths():
    """The body-fitted column needs no learned expert -- W294 says it could not
    use one here anyway -- so a machine where scOT is unavailable must still
    have it checked."""
    run_py = os.path.join(_ROOT, "atlas", "demo_racelab", "bundle", "run.py")
    src = open(run_py, encoding="utf-8").read()
    assert "def body_fitted_check" in src
    assert src.count("body_fitted_check()") >= 2, \
        "the partial path returns before the body-fitted column is checked"
    assert "--column body-fitted" in src


@templates_only
def test_the_ignore_allowlist_is_generated_from_the_artifact_list():
    """**Two lists that had to agree, and did not.**

    The template's hand-written `out/` allowlist stopped at racelab5 while
    `ARTIFACTS` grew to racelab22, and the body-fitted settled field is in
    neither -- so the orphan branch would have carried a bundle whose page had
    nothing to release from, while every test here passed. The allowlist is now
    written by the build from the same list the copy uses.
    """
    src = _builder_src()
    assert "keep = [a for a, _w in ARTIFACTS]" in src, \
        "the allowlist is not generated from ARTIFACTS"
    template = open(os.path.join(_ROOT, "atlas", "demo_racelab", "bundle",
                                 "gitignore"), encoding="utf-8").read()
    # the template must NOT carry a hand-written allowlist any more, or the two
    # can disagree again
    assert "!out/racelab5/cache/settled.npz" not in template
    # ... and it must ignore what the demo WRITES while running
    assert "out/cache/raster_*.npz" in template


@upstream_only
def test_the_commit_asserts_every_artifact_actually_lands_in_the_branch():
    """`git add` on an ignored path says nothing. Without this assertion a rule
    that drops a file leaves a branch that builds, tests and demos here and is
    missing a file on a fresh clone."""
    src = _builder_src()
    body = src.split("def commit", 1)[1]
    assert "must_track" in body
    assert "ls-files" in body
    assert "would not get them" in body
    assert "_bodyfitted_settled()[0]" in body, \
        "the settled field is not force-added, so out/* would drop it"


@templates_only
@pytest.mark.parametrize("name", ["run.sh", "run.cmd", "run.py"])
def test_the_launcher_templates_are_present(name):
    assert os.path.isfile(os.path.join(_ROOT, "atlas", "demo_racelab", "bundle", name))


@templates_only
def test_run_sh_has_no_carriage_returns():
    """A CRLF `run.sh` dies on Linux before it prints anything, and the bundle
    is built on Windows."""
    raw = open(os.path.join(_ROOT, "atlas", "demo_racelab", "bundle", "run.sh"),
               "rb").read()
    assert b"\r\n" not in raw
    assert raw.startswith(b"#!/usr/bin/env bash\n")

"""PoC 3 phase 5 -- the bundle, and what a machine with nothing exposes.

What this file pins:

1. **W243: a switch that cannot switch is refused and greyed out, never left
   live.**  With the learned expert absent -- `scOT` could not be fetched --
   "all learned" used to tint every window learned and count them learned while
   the march went on classical.  The flip is now refused before the assignment
   moves, a rollout that cannot be built puts the old assignment back, and the
   page greys the control out with the reason.  The CONTROL is the same flip
   with an expert present, which must move.
2. **A settled field is matched to the car by the fingerprint it carries**, not
   by the directory it sits in.  The control is a matching field in an OLDER
   directory beating a non-matching one in a newer directory.
3. **The launchers say what they do**: every requirement pinned exactly, the
   `scOT` archive pinned to one commit in every launcher, the bundle's paths
   SET before anything imports a case study, and exit code 3 meaning the same
   "classical only" in all three.
4. **The builder's scans discriminate**: a planted use of the unlicensed
   structural checkpoint, a stray weight file and a vendored `scOT` are each
   caught, and prose that merely names the checkpoint is counted, not refused.
   These need `scripts/build_racelab_bundle.py` and skip in the bundle, which
   does not carry its own builder.

Every test here reads its launcher files from wherever this checkout keeps
them -- `atlas/demo_racelab/bundle/` in `atlas-0.1`, the root in the bundle --
so the same file means the same thing in both.
"""

from __future__ import annotations

import ast
import os
import re
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np                                                    # noqa: E402
import pytest                                                         # noqa: E402

_TPL = os.path.join(_ROOT, "atlas", "demo_racelab", "bundle")
IN_BUNDLE = not os.path.isdir(_TPL)
DEMO = os.path.join(_ROOT, "atlas", "demo_racelab")


def _launcher(name: str) -> str:
    """A launcher file, from atlas-0.1's template directory or the bundle root."""
    if not IN_BUNDLE:
        path = os.path.join(_TPL, name)
    else:
        path = os.path.join(_ROOT, {"gitignore": ".gitignore",
                                    "gitattributes": ".gitattributes"}.get(
                                        name, name))
    if not os.path.isfile(path):
        pytest.skip("%s is not in this checkout" % name)
    with open(path, encoding="utf-8") as fh:      # universal newlines
        return fh.read()


# ---------------------------------------------------------------------------
# 1. W243 -- the learned switch, with and without the learned expert
# ---------------------------------------------------------------------------


class _PresentStack:
    """Stands in for a built `WindowStack`: `ex` is set.  Never marched."""

    ex = object()


def _engine(learned: bool):
    from atlas.demo_racelab import engine as E
    e = E.Engine()
    if learned:
        e.stack, e.stack_error = _PresentStack(), None
    else:
        #: exactly the state `_stack()` leaves when the import fails, set
        #: before `_build` so no checkpoint is loaded
        e.stack, e.stack_error = None, "No module named 'scOT'"
    e._build()
    return e


def test_a_learned_preset_is_refused_when_the_expert_is_absent():
    e = _engine(learned=False)
    before = dict(e.assignment)
    e.post({"kind": "preset", "name": "learned"})
    e._drain()
    assert e.assignment == before, "the assignment moved and the march did not"
    assert set(e._roll.assignment.values()) == {"classical"}
    note = e.notes.get("learned_refused", "")
    assert "NOT AVAILABLE" in note and "scOT" in note, note


def test_a_single_window_flip_is_refused_the_same_way():
    e = _engine(learned=False)
    n = e.names[0]
    e.post({"kind": "mode", "window": n, "mode": "learned"})
    e._drain()
    assert e.assignment[n] == "classical"
    assert "learned_refused" in e.notes


def test_the_control_the_same_flip_moves_when_the_expert_is_present():
    """Without this the two tests above would pass on an engine that refused
    every flip."""
    e = _engine(learned=True)
    e.post({"kind": "preset", "name": "learned"})
    e._drain()
    assert set(e.assignment.values()) == {"learned"}
    assert set(e._roll.assignment.values()) == {"learned"}
    assert "learned_refused" not in e.notes


def test_a_rollout_that_cannot_be_built_puts_the_old_assignment_back():
    e = _engine(learned=True)
    before = dict(e.assignment)

    def broken(_assignment):
        raise RuntimeError("the rollout could not be built")

    e._rollout = broken
    e.post({"kind": "preset", "name": "wake-learned"})
    e._drain()
    assert e.assignment == before
    assert "NOT applied" in e.notes.get("error", "")


def test_the_frame_says_the_learned_expert_is_absent_and_why():
    e = _engine(learned=False)
    e.post({"kind": "preset", "name": "learned"})
    e._drain()
    e._publish()
    p = e.frame.payload
    assert p["learned_available"] is False
    assert "scOT" in (p["stack_error"] or "")
    assert "learned_refused" in p["notes"]


def test_the_page_greys_the_learned_controls_out_with_the_reason():
    """A control whose failure is invisible is this project's recurring defect
    (W233, W242); the page must READ the flag and DRAW the reason."""
    html = open(os.path.join(DEMO, "static", "index.html"),
                encoding="utf-8").read()
    assert "f.learned_available === false" in html
    assert 'id="learnednote"' in html and 'id="errnote"' in html
    i = html.index("const noLearned")
    block = html[i:i + 1800]
    assert "f.stack_error" in block, "the reason is never drawn"
    assert "noLearned && needsLearned(b)" in block, "nothing is disabled"
    assert "f.notes.learned_refused" in block and "f.notes.error" in block
    # the classical preset and the classical mode must stay live
    assert 'b.dataset.preset !== "classical"' in block
    assert 'b.dataset.mode !== "classical"' in block


# ---------------------------------------------------------------------------
# 1b. W251 -- the page's socket, through the real server
# ---------------------------------------------------------------------------


def test_W251_the_page_s_socket_carries_a_frame_through_the_real_server():
    """The first bundle pinned plain uvicorn, whose WebSocket support is
    nothing unless a library is installed beside it, so the page never got a
    frame -- and every engine-level test passed.  This one opens the socket."""
    from atlas.demo_racelab import server as S
    e = _engine(learned=False)
    e._publish()
    ok, detail = S.socket_check(e)
    assert ok, detail


def test_W251_the_control_a_server_with_no_websocket_library_fails_it(
        monkeypatch):
    """The exact state the first bundle shipped in: uvicorn's automatic
    WebSocket protocol resolves to nothing.  The check must say so -- or the
    test above would pass on a check that could not fail."""
    import uvicorn.protocols.websockets.auto as auto
    from atlas.demo_racelab import server as S
    monkeypatch.setattr(auto, "AutoWebSocketsProtocol", None)
    e = _engine(learned=False)
    e._publish()
    ok, detail = S.socket_check(e, timeout_s=10.0)
    assert not ok, detail


def test_W253_the_socket_check_does_not_go_through_the_environment_s_proxy(
        monkeypatch):
    """Found by the no-network check: with every proxy variable pointed at a
    closed port -- which is what a dead corporate proxy looks like -- the check
    reported the page's socket broken.  websockets 15 routes even a
    ``ws://127.0.0.1`` connection through the environment's proxy, so a user
    behind a proxy would have been told a working page cannot receive frames.
    The check talks to its own server on loopback and must connect directly."""
    from atlas.demo_racelab import server as S
    dead = "http://127.0.0.1:9"
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(var, dead)
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    e = _engine(learned=False)
    e._publish()
    ok, detail = S.socket_check(e, timeout_s=15.0)
    assert ok, detail


def test_W251_the_bundle_pins_a_websocket_library_and_the_self_test_uses_it():
    req = _launcher("requirements.txt")
    assert re.search(r"^websockets==\d", req, re.MULTILINE), \
        "no WebSocket library is pinned"
    src = _launcher("run.py")
    assert '("websockets",' in src, "the package check does not name it"
    body = src[src.index("def check()"):]
    assert body.count("socket_check(eng)") >= 2, \
        "the classical-only path or the full path does not open the socket"


# ---------------------------------------------------------------------------
# 2. a settled field belongs to a car
# ---------------------------------------------------------------------------


def _field(root, tier, fingerprint=None, value=1.0, outflow="default"):
    """A settled field on disk.  Since Tier 59 (W258) a field records its
    column's outlet condition; ``"default"`` writes the demo's own, and
    ``None`` writes none -- a field from before the condition was a choice."""
    from atlas.cases import racelab as RL
    d = os.path.join(root, "out", tier, "cache")
    os.makedirs(d, exist_ok=True)
    arrays = {"u": np.full((2, 3), value), "v": np.zeros((2, 3))}
    if fingerprint is not None:
        arrays["geometry"] = np.array(fingerprint)
    if outflow is not None:
        arrays["outflow"] = np.array(RL.OUTFLOW if outflow == "default" else outflow)
    np.savez(os.path.join(d, "settled"), **arrays)


def test_the_car_s_own_field_is_found_by_its_fingerprint(tmp_path):
    from atlas.demo_racelab.engine import find_release
    _field(tmp_path, "racelab5", "a" * 64, value=5.0)
    u, _v, note, mine = find_release(str(tmp_path), "a" * 64)
    assert mine is True and float(u[0, 0]) == 5.0
    assert "THIS car" in note


def test_a_matching_older_field_beats_a_newer_one_for_another_car(tmp_path):
    """The control that says the match is by fingerprint and not by order."""
    from atlas.demo_racelab.engine import find_release
    _field(tmp_path, "racelab5", "b" * 64, value=5.0)
    _field(tmp_path, "racelab4", "a" * 64, value=4.0)
    u, _v, _note, mine = find_release(str(tmp_path), "a" * 64)
    assert mine is True and float(u[0, 0]) == 4.0


def test_another_car_s_field_is_used_only_with_the_warning(tmp_path):
    from atlas.demo_racelab.engine import find_release
    _field(tmp_path, "racelab4", "b" * 64)
    _u, _v, note, mine = find_release(str(tmp_path), "a" * 64)
    assert mine is False
    assert "DIFFERENT CAR" in note and "--stages spinup" in note


def test_a_field_with_no_fingerprint_is_never_taken_for_this_car(tmp_path):
    from atlas.demo_racelab.engine import find_release
    _field(tmp_path, "racelab2", None)
    _u, _v, note, mine = find_release(str(tmp_path), "a" * 64)
    assert mine is False and "no fingerprint" in note


def test_no_field_at_all_is_not_a_no_but_nothing_to_ask(tmp_path):
    from atlas.demo_racelab.engine import find_release
    u, v, note, mine = find_release(str(tmp_path), "a" * 64)
    assert u is None and v is None and mine is None
    assert "FREESTREAM" in note


def test_W258_the_right_car_on_the_other_column_is_not_this_column_s(tmp_path):
    """Tier 59: a settled field belongs to a column as well as a car."""
    from atlas.cases import racelab as RL
    from atlas.demo_racelab.engine import find_release
    other = next(m for m in RL.OUTFLOW_MODES if m != RL.OUTFLOW)
    _field(tmp_path, "racelab8", "a" * 64, value=8.0, outflow=other)
    _u, _v, note, mine = find_release(str(tmp_path), "a" * 64)
    assert mine is False and "OTHER COLUMN" in note
    # the control: asked for that column, the same file is this car's
    u, _v, _n, mine = find_release(str(tmp_path), "a" * 64, outflow=other)
    assert mine is True and float(u[0, 0]) == 8.0
    # and a field recording no outlet condition is the pinned column's
    _field(tmp_path, "racelab5", "a" * 64, value=5.0, outflow=None)
    u, _v, _n, mine = find_release(str(tmp_path), "a" * 64, tiers=("racelab5",),
                                   outflow="pinned")
    assert mine is True and float(u[0, 0]) == 5.0


# ---------------------------------------------------------------------------
# 3. the launchers
# ---------------------------------------------------------------------------

_SCOT = re.compile(r"camlab-ethz/poseidon/archive/([0-9a-f]+)\.zip")


def test_every_requirement_is_pinned_exactly():
    lines = [ln.strip() for ln in _launcher("requirements.txt").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    assert lines
    for ln in lines:
        assert re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][A-Za-z0-9_.\-+]*", ln), ln
    names = {ln.split("==")[0].lower() for ln in lines}
    #: the classical column's import chain reaches PyYAML (the build repo's
    #: config), so it may not be left to arrive with the learned expert's stack
    assert "pyyaml" in names
    assert "torch" not in names, "torch is installed by the launchers, per OS"


def test_the_constraints_pin_everything_and_leave_torch_and_scot_alone():
    if IN_BUNDLE:
        path = os.path.join(_ROOT, "constraints.txt")
    else:
        path = os.path.join(_TPL, "constraints.txt")
    if not os.path.isfile(path):
        pytest.skip("no constraints.txt yet -- it is generated from a verified "
                    "install")
    lines = [ln.strip() for ln in open(path, encoding="utf-8").read().splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    assert len(lines) > 10
    for ln in lines:
        assert re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][A-Za-z0-9_.\-]*", ln), ln
        assert ln.split("==")[0].lower() not in ("torch", "scot"), ln


def test_scot_is_pinned_to_one_commit_everywhere_it_is_named():
    commits = set()
    for name in ("run.sh", "run.cmd", "requirements.txt"):
        found = _SCOT.findall(_launcher(name))
        assert found, "%s does not name the scOT archive" % name
        commits.update(found)
    assert len(commits) == 1, commits
    assert re.fullmatch(r"[0-9a-f]{40}", commits.pop())


def test_scot_is_installed_without_its_dependencies_and_not_fatally():
    sh, cmd = _launcher("run.sh"), _launcher("run.cmd")
    assert '--no-deps' in sh and '--no-deps' in cmd
    assert "scot-ok" in sh and "scot-ok" in cmd
    # a failed fetch continues to the self-test rather than exiting
    i = sh.index("installing scOT")
    assert "exit 1" not in sh[i:sh.index("--check", i)]


def test_exit_code_three_means_classical_only_in_all_three_launchers():
    tree = ast.parse(_launcher("run.py"))
    codes = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            names = [t.id for t in node.targets[0].elts]
            vals = [v.value for v in node.value.elts]
            codes.update(zip(names, vals))
    assert codes == {"OK": 0, "BROKEN": 1, "CLASSICAL_ONLY": 3}, codes
    assert '"$RC" -eq 3' in _launcher("run.sh")
    assert '"%RC%"=="3"' in _launcher("run.cmd")


def test_run_py_sets_the_bundle_s_paths_before_anything_imports_a_case():
    """SET, not defaulted: a machine that already points ATLAS_BUILD_REPO or
    HF_HOME elsewhere must still run this bundle's own copies."""
    src = _launcher("run.py")
    first_import = min(src.index(s) for s in ("from atlas", "import atlas")
                       if s in src)
    for var in ("ATLAS_BUILD_REPO", "HF_HOME", "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE"):
        k = src.index('os.environ["%s"] =' % var)
        assert k < first_import, var
        assert 'setdefault("%s"' % var not in src, var
    conf = _launcher("conftest.py")
    for var in ("ATLAS_BUILD_REPO", "HF_HOME"):
        assert 'os.environ["%s"] =' % var in conf, var


def test_run_py_asserts_tf32_off_and_filters_the_launcher_flags():
    src = _launcher("run.py")
    assert "assert torch.backends.cuda.matmul.allow_tf32 is False" in src
    assert "assert torch.backends.cudnn.allow_tf32 is False" in src
    assert 'LAUNCHER_ONLY = ("--check", "--reinstall")' in src


def test_run_sh_survives_macos_bash_3_2():
    """Expanding an EMPTY array under `set -u` is an unbound-variable error in
    bash 3.2, which macOS still ships -- it would kill the launcher before it
    installed anything."""
    sh = _launcher("run.sh")
    assert sh.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in sh
    assert not re.search(r'"\$\{[A-Za-z_]+\[@\]\}"', sh), \
        "an array expansion that dies under set -u on bash 3.2"


def test_the_launchers_install_the_cpu_wheel_and_say_the_gpu_is_unused():
    sh, cmd = _launcher("run.sh"), _launcher("run.cmd")
    for text in (sh, cmd):
        assert "download.pytorch.org/whl/cpu" in text
        assert "cu126" not in text and "cu12" not in text
        assert "does not use a GPU" in text


def test_the_line_endings_and_the_allowlist_are_declared():
    """**The allowlist moved, and the reason it moved is the point.**

    Tier 57 wrote the `out/` allowlist by hand in the template and this test
    pinned it there. Tier 74 found the consequence: the hand-written copy
    stopped at `racelab5` while the builder's `ARTIFACTS` list reached
    `racelab22`, so the two disagreed about what the bundle contains -- and the
    body-fitted column's settled field was in neither, which would have left a
    clone with a page that has nothing to release from.

    So the allowlist is now **generated by the build from `ARTIFACTS`**: it
    exists in the BUNDLE's `.gitignore`, and must NOT exist in the template,
    because a second hand-written copy is exactly what drifted. The assertion
    therefore differs by where it runs, and both halves matter.
    """
    ga = _launcher("gitattributes")
    for rule in ("*.sh text eol=lf", "*.cmd text eol=crlf",
                 "*.safetensors binary", "*.npz binary"):
        assert rule in ga, rule
    gi = _launcher("gitignore")
    assert ".venv/" in gi
    assert "out/*" in gi, "nothing under out/ is ignored by default"
    # what the demo WRITES while running is not part of the bundle
    assert "out/cache/raster_*.npz" in gi

    if IN_BUNDLE:
        # the generated allowlist, in the file the branch actually carries
        assert "!out/racelab5/cache/settled.npz" in gi
        assert re.search(r"!out/cache/bodyfitted_t\d+(\.\d+)?_[0-9a-f]{12}\.npz",
                         gi), "the body-fitted settled field is not force-kept"
    else:
        assert "!out/racelab5/cache/settled.npz" not in gi, (
            "the template carries a hand-written allowlist again; it is "
            "generated from ARTIFACTS by build_racelab_bundle, and the two "
            "copies drifted once already (Tier 74)")


# ---------------------------------------------------------------------------
# 4. the builder's scans, which exist only upstream
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def B():
    path = os.path.join(_ROOT, "scripts", "build_racelab_bundle.py")
    if not os.path.isfile(path):
        pytest.skip("this checkout does not carry the bundle builder (it is the "
                    "bundle); its scans are tested upstream")
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_racelab_bundle", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_NB = "neuber" + "net"


def _write(root, rel, text):
    p = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)


def test_the_scan_catches_a_use_of_the_unlicensed_checkpoint(B, tmp_path):
    _write(tmp_path, "a.py", "import %s\n" % _NB)
    _write(tmp_path, "b.py", "p = '~/.cache/%s/weights.pt'\n" % _NB)
    _write(tmp_path, "%s/readme.txt" % _NB, "x\n")
    r = B.scan_unlicensed_checkpoint(str(tmp_path))
    assert len(r["uses"]) >= 2
    assert r["files_or_dirs_named_for_it"]


def test_the_scan_counts_prose_and_does_not_refuse_it(B, tmp_path):
    """The control: a sentence saying why the checkpoint is absent is not a
    use of it -- `test_tier51_racelab_graph.py`'s rule, applied to the bundle."""
    _write(tmp_path, "a.md", "The %s checkpoint is unlicensed and absent.\n"
           % _NB.capitalize())
    r = B.scan_unlicensed_checkpoint(str(tmp_path))
    assert r["uses"] == [] and r["files_or_dirs_named_for_it"] == []
    assert r["prose_mentions_total"] == 1


def test_the_scan_catches_a_stray_weight_and_allows_the_field(B, tmp_path):
    _write(tmp_path, "notes.txt", "text\n")
    for rel in ("vendor/other/model.pt", "out/racelab5/cache/settled.npz",
                "out/racelab5/cache/arm_referent.npz"):
        p = os.path.join(tmp_path, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(b"\x00\x01\x02")
    r = B.scan_data(str(tmp_path))
    unexpected = {x["file"] for x in r["unexpected"]}
    allowed = {x["file"] for x in r["allowed"]}
    assert unexpected == {"vendor/other/model.pt",
                          "out/racelab5/cache/arm_referent.npz"}
    assert allowed == {"out/racelab5/cache/settled.npz"}


def test_the_scan_catches_a_vendored_scot(B, tmp_path):
    os.makedirs(os.path.join(tmp_path, "vendor", "scOT"))
    assert B.scan_scot(str(tmp_path))


def test_the_builder_does_not_push(B):
    src = open(B.__file__, encoding="utf-8").read()
    assert '"push"' not in src and "'push'" not in src
    assert "push --force <remote>" in src, "it should say how to publish"


def test_the_builder_carries_every_poc3_test_and_every_page_they_quote(B):
    for f in B.TESTS:
        assert os.path.isfile(os.path.join(_ROOT, "tests", f)), f
    assert os.path.basename(__file__) in B.TESTS
    wiki = os.path.join(_ROOT, *B.WIKI_DIR)
    for p in B.WIKI_PAGES:
        assert os.path.isfile(os.path.join(wiki, p)), p

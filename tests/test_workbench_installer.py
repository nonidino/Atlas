"""The workbench's one-command install (demo step 7, O2): the launcher templates
and the builder, checked without installing anything.

What is checked here is what PoC 3's bundle learned the expensive way:

* ``run.sh`` must survive the bash macOS ships (3.2): no arrays, every
  ``"$@"`` spelled ``${1+"$@"}`` under ``set -u``, LF endings;
* every third-party package the workbench imports, at any indentation, must
  be pinned in ``requirements.txt`` or be one of the declared optional ones
  ("Lazy imports are undeclared deps": a function-level import of shapely
  once stopped a Linux page while the self-test passed);
* the two launchers pin the same versions, and pass no flag of their own on
  to the workbench;
* the self-test marches one example of every type the page offers;
* the builder carries every workbench test and the scripts they import, and
  refuses NeuberNet, Poseidon's weights and every other binary file.

The bundle carries this file with the other workbench tests, but not the
templates or the builder, which live upstream in ``atlas-0.1``: there every
test here skips and says why.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

TEMPLATES = os.path.join(HERE, "atlas", "workbench", "bundle")
BUILDER = os.path.join(HERE, "scripts", "build_workbench_bundle.py")

if not (os.path.isdir(TEMPLATES) and os.path.isfile(BUILDER)):
    pytest.skip("the launcher templates and the builder live upstream, in atlas-0.1; this "
                "copy of the workbench carries the launchers they make, not them",
                allow_module_level=True)


def _text(name: str) -> str:
    with open(os.path.join(TEMPLATES, name), encoding="utf-8") as fh:
        return fh.read()


def _bytes(name: str) -> bytes:
    with open(os.path.join(TEMPLATES, name), "rb") as fh:
        return fh.read()


def _builder():
    """The builder, loaded by path: an ``import`` statement here would make the
    builder's own script-closure carry the builders into the bundle."""
    spec = importlib.util.spec_from_file_location("_wb_builder", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_py_literal(name: str):
    """A top-level literal of the template run.py, read without importing it
    (importing it SETS ATLAS_BUILD_REPO for this whole test process)."""
    tree = ast.parse(_text("run.py"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name
                                                for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# run.sh, for the bash macOS ships
# ---------------------------------------------------------------------------


def test_run_sh_is_lf_and_avoids_what_bash_32_cannot_do():
    raw = _bytes("run.sh")
    assert b"\r" not in raw, "a CR in run.sh breaks its shebang on a Mac"
    text = raw.decode("utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    code = "\n".join(ln.split("#", 1)[0] if not ln.lstrip().startswith("#") else ""
                     for ln in text.splitlines())
    banned = {r"\w+=\(": "an array", r"\$\{!": "indirect expansion",
              r"declare\s+-A": "an associative array", r"\bmapfile\b": "mapfile",
              r"\breadarray\b": "readarray", r",,\}": "lower-casing",
              r"\^\^\}": "upper-casing", r"\|&": "|&", r"&>>": "&>>",
              r"\bcoproc\b": "coproc"}
    for pat, what in banned.items():
        assert not re.search(pat, code), f"run.sh uses {what}, which bash 3.2 lacks"
    # every "$@" is ${1+"$@"}: with no argument, bash 3.2's set -u is not to be trusted
    bare = [m.start() for m in re.finditer(r'"\$@"', code)
            if code[max(0, m.start() - 4):m.start()] != "${1+"]
    assert not bare, "a bare \"$@\" in run.sh"


def test_run_sh_parses():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash on this machine to parse run.sh with")
    r = subprocess.run([bash, "-n", os.path.join(TEMPLATES, "run.sh")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------
# run.cmd, and the two launchers against each other
# ---------------------------------------------------------------------------


def test_run_cmd_jumps_only_to_labels_it_has():
    text = _text("run.cmd")
    labels = set(re.findall(r"^:([A-Za-z_]+)\s*$", text, re.M))
    targets = set(re.findall(r"\bgoto\s+:?([A-Za-z_]+)", text, re.I)) - {"eof"}
    calls = set(re.findall(r"\bcall\s+:([A-Za-z_]+)", text, re.I))
    assert targets <= labels and calls <= labels, (targets | calls) - labels


def test_both_launchers_pin_the_same_versions():
    sh, cmd = _text("run.sh"), _text("run.cmd")
    pins_sh = dict(re.findall(r'^(TORCH|GMSH|PIP)="([^"]+)"', sh, re.M))
    pins_cmd = dict(re.findall(r'^set "(TORCH|GMSH|PIPPIN)=([^"]+)"', cmd, re.M))
    assert pins_sh == {"TORCH": pins_cmd["TORCH"], "GMSH": pins_cmd["GMSH"],
                       "PIP": pins_cmd["PIPPIN"]}
    assert pins_sh["TORCH"] == "torch==2.7.1" and pins_sh["GMSH"] == "gmsh==4.15.0"
    req = _text("requirements.txt")
    for v in ("torch 2.7.1", "gmsh 4.15.0"):
        assert v in req, v
    assert "torch 2.7.1 publishes no Intel-macOS" in " ".join(_text("README.md").split())


def test_every_flag_the_launchers_read_is_stripped_before_the_workbench():
    only = set(_run_py_literal("LAUNCHER_ONLY"))
    sh_flags = set(re.findall(r"^\s+(--[a-z-]+)\)", _text("run.sh"), re.M))
    cmd_flags = set(re.findall(r'findstr /C:"(--[a-z-]+)"', _text("run.cmd")))
    assert sh_flags == cmd_flags == only == {"--check", "--reinstall", "--no-torch",
                                              "--no-gmsh"}


# ---------------------------------------------------------------------------
# what is installed
# ---------------------------------------------------------------------------


def _pins(name: str) -> dict[str, str]:
    out = {}
    for ln in _text(name).splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ln:
            m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.+\-]+)", ln)
            assert m, f"{name}: {ln!r} is not an exact pin"
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def test_requirements_are_exact_pins_and_the_constraints_agree():
    req = _pins("requirements.txt")
    assert {"numpy", "scipy", "pandas", "pydantic", "panel", "bokeh", "pytest"} <= set(req)
    if os.path.isfile(os.path.join(TEMPLATES, "constraints.txt")):
        cons = _pins("constraints.txt")
        for name, ver in req.items():
            assert cons.get(name) == ver, (name, ver, cons.get(name))
        assert "torch" not in cons, "torch is pinned by the launchers, with its +cpu label"
        # From param 2.3.0, with Panel 1.5.2, a widget's `name` is constant and the
        # header's relabelling raises: the first clean install served HTTP 500.
        major, minor = (int(x) for x in cons["param"].split(".")[:2])
        assert (major, minor) < (2, 3), cons["param"]


#: module name -> the distribution that provides it, where they differ or where
#: the module arrives with another pinned package
_DIST = {"tornado": "bokeh"}


def _third_party_imports(paths: list[str]) -> dict[str, set[str]]:
    """Every absolute import in code, at any depth of the syntax tree (so inside
    functions too); docstrings and comments are prose and are not read."""
    std = set(sys.stdlib_module_names) | {"__future__", "atlas"}
    found: dict[str, set[str]] = {}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for n in names:
                if n not in std:
                    found.setdefault(n, set()).add(os.path.relpath(p, HERE))
    return found


def test_every_package_the_workbench_imports_is_pinned_or_optional():
    """At ANY indentation: a function-level import is a dependency the module-level
    imports cannot see."""
    wb = os.path.join(HERE, "atlas", "workbench")
    paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(wb)
             if "bundle" not in dp.split(os.sep) and "__pycache__" not in dp
             for f in fn if f.endswith(".py")]
    paths += [os.path.join(HERE, "atlas", f) for f in os.listdir(os.path.join(HERE, "atlas"))
              if f.endswith(".py")]
    paths += [os.path.join(HERE, "atlas", "cases", f + ".py")
              for f in ("wake_array", "scaling_ladder", "window_ns", "poseidon")]
    found = _third_party_imports(paths)
    required = {m for m, _w in _run_py_literal("REQUIRED")}
    optional = {m for m, _w in _run_py_literal("OPTIONAL")}
    undeclared = {m: sorted(w) for m, w in found.items() if m not in required | optional}
    assert not undeclared, f"imported but neither required nor optional: {undeclared}"
    req = _pins("requirements.txt")
    for m in required:
        assert _DIST.get(m, m) in req, f"{m} is checked by the self-test but not pinned"


# ---------------------------------------------------------------------------
# the self-test
# ---------------------------------------------------------------------------


def test_the_self_test_fetches_its_own_page_directly_not_through_a_proxy():
    """urllib sends even a request for 127.0.0.1 to HTTP_PROXY unless NO_PROXY names
    it: with pip pointed at a closed proxy the self-test called a working page broken
    (the server listened and never saw a request).  Every fetch of the page goes
    through an opener with no proxy; `verify_workbench_bundle --optional-fails` runs
    the launcher behind a closed proxy and is the behavioural check."""
    text = _text("run.py")
    assert "urllib.request.urlopen(" not in text
    assert "ProxyHandler({})" in text and text.count("direct.open(") == 2


def test_the_self_test_marches_one_example_of_every_type():
    from atlas.workbench import registry
    from atlas.workbench.spec import EXAMPLES
    types = _run_py_literal("TYPES")
    assert [fid for fid, _k, _n in types] == registry.available_ids()
    for fid, key, steps in types:
        assert EXAMPLES[key].family == fid, (fid, key)
        assert 1 <= steps <= 2


# ---------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------


def test_the_builder_carries_every_workbench_test_and_the_scripts_they_import():
    W = _builder()
    tests = W.tests_to_carry()
    upstream = sorted(f for f in os.listdir(os.path.join(HERE, "tests"))
                      if re.match(r"^test_workbench_.*\.py$", f))
    assert tests == upstream + ["workbench_ui.py"]
    scripts = W.script_closure([os.path.join(HERE, "tests", t) for t in tests])
    assert "w346_rotor_count_speed.py" in scripts      # test_workbench_runner's control
    assert "w100_scaling_ladder.py" in scripts         # ... and what W346 imports
    assert "build_workbench_bundle.py" not in scripts  # this file loads it by path
    assert "build_racelab_bundle.py" not in scripts


def test_the_builder_leaves_out_only_packages_nothing_else_imports():
    W = _builder()
    pat = re.compile(r"(?:from|import)\s+(?:atlas\.|\.\.?)(%s)\b" % "|".join(W.EXCLUDE_ATLAS))
    hits = []
    for dp, _dn, fn in os.walk(os.path.join(HERE, "atlas")):
        rel = os.path.relpath(dp, os.path.join(HERE, "atlas")).split(os.sep)
        if rel[0] in W.EXCLUDE_ATLAS:
            continue
        for f in fn:
            if f.endswith(".py"):
                with open(os.path.join(dp, f), encoding="utf-8") as fh:
                    if pat.search(fh.read()):
                        hits.append(os.path.join(dp, f))
    assert not hits


def test_the_builder_refuses_weights_neubernet_and_model_caches(tmp_path):
    W = _builder()
    clean = tmp_path / "clean"
    (clean / "atlas").mkdir(parents=True)
    (clean / "atlas" / "x.py").write_text("# the learned windows were Poseidon-T\n")
    assert W.scan_data(str(clean))["unexpected"] == []
    assert W.scan_poseidon(str(clean))["paths"] == []
    assert W.B.scan_unlicensed_checkpoint(str(clean))["uses"] == []

    bad = tmp_path / "bad"
    hub = bad / "vendor" / "hf-cache" / "hub" / "models--camlab-ethz--Poseidon-T"
    hub.mkdir(parents=True)
    (hub / "model.safetensors").write_bytes(b"\x00\x01weights")
    (bad / "w.pt").write_bytes(b"\x00\x02")
    nb = "neuber" + "net"
    (bad / "loader.py").write_text("import %s\n" % nb)
    assert {r["file"] for r in W.scan_data(str(bad))["unexpected"]} == {
        "vendor/hf-cache/hub/models--camlab-ethz--Poseidon-T/model.safetensors", "w.pt"}
    assert W.scan_poseidon(str(bad))["paths"]
    assert W.B.scan_unlicensed_checkpoint(str(bad))["uses"]

    # the learned case's own weights and truth are allowed by exact path, nothing beside
    ok = tmp_path / "ok" / "out" / "learned-case"
    ok.mkdir(parents=True)
    (ok / "window-net.pt").write_bytes(b"\x00w")
    (ok / "truth-farm-12.npz").write_bytes(b"\x00t")
    (ok / "other.pt").write_bytes(b"\x00o")
    found = W.scan_data(str(tmp_path / "ok"))
    assert {r["file"] for r in found["allowed"]} == {"out/learned-case/window-net.pt",
                                                     "out/learned-case/truth-farm-12.npz"}
    assert {r["file"] for r in found["unexpected"]} == {"out/learned-case/other.pt"}


def test_the_branch_keeps_its_line_endings_and_ignores_what_it_makes():
    attrs = _text("gitattributes")
    assert "*.sh text eol=lf" in attrs and "*.cmd text eol=crlf" in attrs
    ignore = _text("gitignore").splitlines()
    assert ".venv/" in ignore and "out/*" in ignore and "!out/learned-case/" in ignore
    W = _builder()
    assert W.LAUNCHERS["gitattributes"] == ".gitattributes"
    assert W.LAUNCHERS["gitignore"] == ".gitignore"
    for t in W.LAUNCHERS:
        assert t in W.OPTIONAL_TEMPLATES or os.path.isfile(os.path.join(TEMPLATES, t)), t

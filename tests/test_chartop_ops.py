"""Stage 0's three operational defects (W360), each with its test.

1. **The pool followed ``os.cpu_count()``.**  The rented A100 box reported 192 CPUs on a
   quota of 23 and ran 191 workers, at 4.2 s per example against 2.0 s with 63 on a 64-core
   box.  `scripts/chartop_cpus.py` takes the smallest of the cgroup quotas, the affinity mask
   and the CPU count; the runner, the payload and the scripts that size a pool use it.
2. **A cost run showed a gate's measure.**  E5's record carried the learned answer's energy
   error, E4's measure, before the one evaluation.
3. **An interrupted evaluation could not resume.**  It registered its weights before it
   started, so a rerun was refused, and the only way past the refusal was to edit the
   registry.  `scripts/chartop_registry.py` keeps checkpoints and ``--resume`` continues the
   same evaluation: the same weights, gates module and test split, checked by hash.  An
   evaluation that finished is still refused.

The CPU helper, the registry and the payload need only the standard library.  The cost and
evaluation scripts are driven with untrained networks and stubbed stages: what is checked
holds for any weights.
"""
from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import types
import zipfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import chartop_cpus as cpus  # noqa: E402
import chartop_registry as registry  # noqa: E402

RUN_SH = os.path.join(ROOT, "scripts", "chartop_box_run.sh")


# ---------------------------------------------------------------------------
# 1. the worker count follows the container's CPU quota
# ---------------------------------------------------------------------------


def _cgroup(path, v2=None, v1=None, mount="cpu"):
    path.mkdir(parents=True, exist_ok=True)
    if v2 is not None:
        (path / "cpu.max").write_text(v2)
    if v1 is not None:
        d = path / mount
        d.mkdir()
        (d / "cpu.cfs_quota_us").write_text(v1[0])
        (d / "cpu.cfs_period_us").write_text(v1[1])
    return str(path)


def _host(monkeypatch, path, cpu_count, affinity, v2=None, v1=None):
    """A machine as the helper sees it: its cgroup mount, os.cpu_count() and affinity mask
    (``None``: the platform has no sched_getaffinity, as Windows has none)."""
    monkeypatch.setattr(cpus, "CGROUP_ROOT", _cgroup(path, v2=v2, v1=v1))
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
    if affinity is None:
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    else:
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(affinity)), raising=False)


@pytest.mark.parametrize("text, expect", [
    ("2304000 100000\n", 23),                 # the A100 box: 23.04 CPUs
    ("max 100000\n", None),                   # no quota
    ("150000 100000\n", 1), ("50000 100000\n", 1), ("250000 100000\n", 2),   # down, never to zero
    ("200000\n", 2),                          # the period is optional in the kernel's syntax
    ("", None), ("garbage\n", None), ("abc 100000\n", None), ("100000 0\n", None),
    ("0 100000\n", None), ("-1 100000\n", None)])
def test_cgroup_v2_quota(tmp_path, text, expect):
    assert cpus.cgroup_v2_cpus(_cgroup(tmp_path, v2=text)) == expect


@pytest.mark.parametrize("mount", ["cpu", "cpu,cpuacct", "cpuacct,cpu"])
def test_cgroup_v1_quota(tmp_path, mount):
    assert cpus.cgroup_v1_cpus(_cgroup(tmp_path, v1=("2300000\n", "100000\n"), mount=mount)) == 23


@pytest.mark.parametrize("quota, period", [("-1\n", "100000\n"), ("100000\n", "0\n"), ("junk\n", "100000\n")])
def test_cgroup_v1_without_a_quota(tmp_path, quota, period):
    assert cpus.cgroup_v1_cpus(_cgroup(tmp_path, v1=(quota, period))) is None


def test_cgroup_v1_needs_both_files(tmp_path):
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("200000")
    assert cpus.cgroup_v1_cpus(str(tmp_path)) is None


def test_the_a100_box_gets_its_quota_not_the_host(monkeypatch, tmp_path):
    _host(monkeypatch, tmp_path, 192, 192, v2="2304000 100000\n")
    assert cpus.sources() == {"cgroup_v2": 23, "cgroup_v1": None, "affinity": 192, "cpu_count": 192}
    assert cpus.available_cpus() == 23
    assert cpus.default_workers(1) == 22 and cpus.default_workers(2) == 21
    assert max(1, os.cpu_count() - 1) == 191            # what the runner and the generator took before


@pytest.mark.parametrize("cpu_count, affinity, v2, v1, expect", [
    (64, 64, None, None, 64),                           # no quota: the 3090 box, 63 workers
    (192, 192, "2304000 100000", None, 23),             # a cgroup v2 quota under the host
    (192, 192, None, ("2300000", "100000"), 23),        # a cgroup v1 quota
    (192, 8, "2304000 100000", None, 8),                # an affinity mask under the quota
    (4, 4, "2304000 100000", None, 4),                  # a quota above the machine
    (192, 192, "2304000 100000", ("500000", "100000"), 5),   # the smaller of the two versions
    (22, None, None, None, 22),                         # Windows: no cgroup files, no sched_getaffinity
    (None, None, None, None, 1),                        # os.cpu_count() may say nothing
])
def test_the_smallest_source_wins(monkeypatch, tmp_path, cpu_count, affinity, v2, v1, expect):
    _host(monkeypatch, tmp_path, cpu_count, affinity, v2=v2, v1=v1)
    assert cpus.available_cpus() == expect
    assert cpus.default_workers(0) == expect
    assert cpus.default_workers(1) == max(1, expect - 1)


def test_a_pool_has_at_least_one_worker(monkeypatch, tmp_path):
    _host(monkeypatch, tmp_path, 1, 1)
    assert cpus.default_workers(5) == 1


def test_windows_reads_the_cpu_count_alone(monkeypatch, tmp_path):
    """No cgroup mount (the folder does not exist) and no sched_getaffinity."""
    monkeypatch.setattr(cpus, "CGROUP_ROOT", str(tmp_path / "no" / "such" / "mount"))
    monkeypatch.setattr(os, "cpu_count", lambda: 22)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    assert cpus.sources() == {"cgroup_v2": None, "cgroup_v1": None, "affinity": None, "cpu_count": 22}
    assert cpus.default_workers(2) == 20


def test_the_command_prints_the_worker_count():
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "chartop_cpus.py")]
    out = subprocess.run(cmd + ["--reserve", "2"], capture_output=True, text=True, check=True).stdout
    assert out.strip() == str(cpus.default_workers(2))
    explained = json.loads(subprocess.run(cmd + ["--explain"], capture_output=True, text=True,
                                          check=True).stdout)
    assert explained["workers"] == cpus.default_workers(1)
    assert set(explained["sources"]) == {"cgroup_v2", "cgroup_v1", "affinity", "cpu_count"}


def test_the_generator_and_the_evaluation_size_their_pools_from_the_quota(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    import chartop_data
    import chartop_evaluate
    _host(monkeypatch, tmp_path, 192, 192, v2="2304000 100000\n")
    got = chartop_data.build_parser().parse_args(["--split", "val", "--count", "1", "--out", "x"])
    assert got.workers == 21                            # 23 less the two the default keeps back
    assert chartop_evaluate.build_parser().parse_args([]).workers == 21
    explicit = chartop_data.build_parser().parse_args(["--split", "val", "--count", "1", "--out", "x",
                                                       "--workers", "5"])
    assert explicit.workers == 5


def test_no_chartop_script_reads_the_cpu_count_for_a_pool():
    """Every ``os.cpu_count()`` in the chartop scripts is on a list; a new one is a pool sized
    the way W360 found wrong until someone has read it."""
    allowed = {("chartop_price.py", '"cpus": os.cpu_count(),')}      # a field of the price record
    for path in sorted(glob.glob(os.path.join(ROOT, "scripts", "chartop_*.py"))):
        name = os.path.basename(path)
        if name == "chartop_cpus.py":
            continue
        text = open(path, encoding="utf-8").read()
        lines = text.splitlines()
        for node in ast.walk(ast.parse(text)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "cpu_count"):
                line = lines[node.lineno - 1].strip()
                assert any(name == n and frag in line for n, frag in allowed), f"{name}:{node.lineno}: {line}"


def test_the_runner_sizes_its_pool_with_the_quota_helper():
    lines = open(RUN_SH, encoding="utf-8").read().splitlines()
    default = [ln for ln in lines if ln.startswith("WORKERS=")]
    assert len(default) == 1 and "chartop_cpus.py" in default[0] and "nproc" not in default[0]
    # nproc survives only as the loud fallback, inside the test for an empty answer
    guard = next(i for i, ln in enumerate(lines) if ln.startswith('if [ -z "$WORKERS" ]'))
    uses = [i for i, ln in enumerate(lines) if "nproc" in ln and not ln.lstrip().startswith("#")]
    assert uses and all(i > guard for i in uses)
    assert "chartop_cpus.py --reserve 1 --explain" in "\n".join(lines)       # the box's log says why


def test_the_payload_ships_the_helper_and_what_the_shipped_scripts_import(monkeypatch, tmp_path):
    import chartop_box_payload as payload
    assert "chartop_cpus.py" in payload.SCRIPTS
    for script in payload.SCRIPTS:
        src = open(os.path.join(ROOT, "scripts", script), encoding="utf-8").read()
        for mod in re.findall(r"^\s*(?:import|from) (chartop_\w+)", src, re.M):
            assert mod + ".py" in payload.SCRIPTS, f"{script} imports {mod}, which the payload does not ship"
    out = tmp_path / "payload.tar.gz"
    monkeypatch.setattr(payload, "OUT", str(out))
    monkeypatch.setattr(sys, "argv", ["chartop_box_payload.py"])
    assert payload.main() == 0
    with tarfile.open(out) as tar:
        assert "chartop/scripts/chartop_cpus.py" in tar.getnames()
        assert "chartop/run.sh" in tar.getnames()


# ---------------------------------------------------------------------------
# 2. a cost run records no gate's measure
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def e5():
    pytest.importorskip("torch")
    import chartop_e5
    return chartop_e5


def _untrained_expert():
    import torch
    from atlas.chartop import data, model
    from atlas.chartop.expert import LearnedExpert
    torch.manual_seed(0)
    net = model.Expert("unet", "0.3M")
    for p in net.parameters():
        torch.nn.init.normal_(p, 0.0, 0.02)
    return LearnedExpert(net, envelope=data.ENVELOPE)


def test_a_cost_run_records_no_gate_measure_unless_asked(e5):
    ex = _untrained_expert()
    rec = e5.bench_times("bend-3", "step", ex, 1, n=32)
    assert set(e5.GATE_MEASURE_KEYS).isdisjoint(rec)
    assert "energy_rel" not in json.dumps(rec)                      # nor under any other name
    assert all(k.endswith("_s") for k in rec if k.startswith("learned"))      # the learned arm: times only
    assert rec["coarse_max_over_range_approx"] > 0                  # the competitor's error stays
    asked = e5.bench_times("bend-3", "step", ex, 1, gate_measures=True, n=32)
    assert asked["learned_energy_rel"] > 0
    assert set(asked) - set(rec) == {"learned_energy_rel"}


def test_the_design_loop_records_no_gate_measure_unless_asked(e5):
    ex = _untrained_expert()
    plain = e5.design_loop(ex, 1, n=32, variants=2)
    assert len(plain["variants"]) == 2
    assert all("learned_energy_rel" not in row for row in plain["variants"])
    asked = e5.design_loop(ex, 1, gate_measures=True, n=32, variants=2)
    assert all(row["learned_energy_rel"] > 0 for row in asked["variants"])


def test_the_flag_is_off_by_default_and_reaches_both_measurements(e5, monkeypatch, tmp_path):
    parser = e5.build_parser()
    assert parser.parse_args([]).with_gate_measures is False
    assert parser.parse_args(["--with-gate-measures"]).with_gate_measures is True
    for flag, expect in (([], False), (["--with-gate-measures"], True)):
        seen = []
        out = tmp_path / ("on" if expect else "off")
        monkeypatch.setattr(e5, "load_expert", lambda path, device: object())
        monkeypatch.setattr(e5, "LearnedExpert", lambda net, **kw: object())
        monkeypatch.setattr(e5, "power_state", lambda: {})
        monkeypatch.setattr(e5, "bench_times", lambda name, problem, ex, reps, gate_measures=None, n=64:
                            seen.append(gate_measures) or {"name": name})
        monkeypatch.setattr(e5, "design_loop", lambda ex, reps, gate_measures=None, **kw:
                            seen.append(gate_measures) or {"totals_s": {}})
        monkeypatch.setattr(sys, "argv", ["chartop_e5.py", "--out", str(out), "--threads", "1",
                                          "--reps", "1"] + flag)
        assert e5.main() == 0
        assert seen and all(s is expect for s in seen)             # six benchmark problems and the loop
        rec = json.load(open(glob.glob(str(out / "e5-*.json"))[0], encoding="utf-8"))
        assert rec["gate_measures"] is expect


# ---------------------------------------------------------------------------
# 3. the registry and the checkpoints
# ---------------------------------------------------------------------------

HASHES = {"B": "b" * 64, "Bp": "c" * 64}
IDENT = {"gates_sha256": "a" * 64, "test_count": 2400, "dual_test": 100}


def _begin(out, hashes=None, **kw):
    kw.setdefault("identity", IDENT)
    kw.setdefault("session", {"torch": "2.7.1", "torch_threads": 8})
    return registry.begin(str(out), hashes or HASHES, **kw)


def _registry(out):
    path = out / "evaluated.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def test_a_first_evaluation_is_registered_and_opens_its_checkpoints(tmp_path):
    ev = _begin(tmp_path, stamp="20261018-100000")
    assert not ev.resumed and ev.stamp == "20261018-100000"
    assert sorted(_registry(tmp_path).values()) == sorted(HASHES.values())
    assert all(re.fullmatch(r"\d{4}-\d\d-\d\d:(B|Bp)", k) for k in _registry(tmp_path))
    state = json.loads((tmp_path / "evaluation-20261018-100000-checkpoints" / "state.json").read_text())
    assert state["weights"] == HASHES and state["identity"] == IDENT and state["completed"] is None
    assert state["sessions"][0]["resume"] is False and state["test"] == {"sha256": None}


def test_an_interrupted_evaluation_resumes_it_does_not_restart(tmp_path):
    ev = _begin(tmp_path, stamp="s1")
    assert ev.json_stage("unit-a.json", lambda: {"x": 1}) == {"x": 1}
    ev.save("raw.bin", b"bytes")                                   # the process dies here
    with pytest.raises(registry.Refused, match=r"did not finish.*--resume.*2 checkpoints"):
        _begin(tmp_path)
    again = _begin(tmp_path, resume=True)
    assert again.resumed and again.stamp == "s1" and again.session == 1
    assert again.load("raw.bin") == b"bytes"
    assert again.json_stage("unit-a.json", lambda: pytest.fail("a unit that was done is done again")) == {"x": 1}
    assert again.reused == ["raw.bin", "unit-a.json"]
    prov = again.provenance()
    assert [s["index"] for s in prov["sessions"]] == [0, 1] and prov["units"] == {"raw.bin": 0, "unit-a.json": 0}
    assert len(_registry(tmp_path)) == 2                           # the resume added nothing


def test_a_finished_evaluation_is_refused_with_or_without_resume(tmp_path):
    ev = _begin(tmp_path, stamp="s1")
    record = tmp_path / "evaluation-s1.json"
    record.write_text("{}")
    ev.complete(str(record))
    for resume in (False, True):
        with pytest.raises(registry.Refused, match="runs once"):
            _begin(tmp_path, resume=resume)
    done = json.loads((tmp_path / "evaluation-s1-checkpoints" / "state.json").read_text())["completed"]
    assert done["record"] == "evaluation-s1.json" and len(done["sha256"]) == 64


def test_a_record_on_disk_means_finished_even_if_the_state_was_never_closed(tmp_path):
    _begin(tmp_path, stamp="s1")
    (tmp_path / "evaluation-s1.json").write_text("{}")             # died between the record and the state
    for resume in (False, True):
        with pytest.raises(registry.Refused, match="runs once"):
            _begin(tmp_path, resume=resume)


def test_weights_from_before_the_checkpoints_cannot_be_resumed_and_the_file_is_untouched(tmp_path):
    legacy = json.dumps({"2026-10-18:B": HASHES["B"], "2026-10-18:Bp": HASHES["Bp"]}, indent=1)
    (tmp_path / "evaluated.json").write_text(legacy)
    with pytest.raises(registry.Refused, match="runs once"):
        _begin(tmp_path)
    with pytest.raises(registry.Refused, match="nothing to resume"):
        _begin(tmp_path, resume=True)
    assert (tmp_path / "evaluated.json").read_text() == legacy
    assert not glob.glob(str(tmp_path / "evaluation-*"))


def test_some_of_the_arms_begun_is_refused_and_resume_names_the_arm_that_differs(tmp_path):
    _begin(tmp_path, stamp="s1")
    other = {"B": HASHES["B"], "Bp": "d" * 64}
    with pytest.raises(registry.Refused, match="runs once"):
        _begin(tmp_path, other)
    with pytest.raises(registry.Refused, match=r"other weights in arm\(s\) \['Bp'\]"):
        _begin(tmp_path, other, resume=True)
    fewer = {"B": HASHES["B"]}
    with pytest.raises(registry.Refused, match=r"\['Bp'\]"):
        _begin(tmp_path, fewer, resume=True)


def test_resume_with_nothing_begun_is_refused_and_creates_nothing(tmp_path):
    with pytest.raises(registry.Refused, match="nothing to resume"):
        _begin(tmp_path, resume=True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("changed", [{"gates_sha256": "f" * 64}, {"test_count": 6}, {"dual_test": 3}])
def test_resume_is_refused_when_the_identity_changed(tmp_path, changed):
    _begin(tmp_path, stamp="s1")
    with pytest.raises(registry.Refused, match=next(iter(changed))):
        _begin(tmp_path, resume=True, identity=dict(IDENT, **changed))
    _begin(tmp_path, resume=True)                                   # and the refusal changed nothing


def test_the_test_split_is_checked_by_hash(tmp_path):
    ev = _begin(tmp_path, stamp="s1")
    ev.confirm_test_split("1" * 64)
    ev.confirm_test_split("1" * 64)
    again = _begin(tmp_path, resume=True)
    again.confirm_test_split("1" * 64)
    with pytest.raises(registry.Refused, match="not the one this evaluation began with"):
        again.confirm_test_split("2" * 64)


def test_a_split_not_yet_hashed_when_the_run_died_is_hashed_by_the_resume(tmp_path):
    _begin(tmp_path, stamp="s1")                                    # died while the split was drawn
    again = _begin(tmp_path, resume=True)
    assert again.state["test"]["sha256"] is None
    again.confirm_test_split("3" * 64)
    assert _begin(tmp_path, resume=True).state["test"]["sha256"] == "3" * 64


def test_a_checkpoint_that_is_not_the_recorded_file_is_computed_again(tmp_path):
    ev = _begin(tmp_path, stamp="s1")
    ev.json_stage("unit.json", lambda: {"v": 1})
    path = tmp_path / "evaluation-s1-checkpoints" / "unit.json"
    path.write_bytes(path.read_bytes().replace(b"1", b"2"))         # a file that is not the one recorded
    calls = []
    second = _begin(tmp_path, resume=True)
    assert second.json_stage("unit.json", lambda: calls.append(1) or {"v": 1}) == {"v": 1}
    assert calls == [1] and second.reused == []
    third = _begin(tmp_path, resume=True)
    assert third.json_stage("unit.json", lambda: pytest.fail("the rewritten checkpoint is good")) == {"v": 1}
    assert third.provenance()["units"] == {"unit.json": 1}          # the second session made it


def test_a_checkpoint_the_state_never_recorded_is_ignored(tmp_path):
    ev = _begin(tmp_path, stamp="s1")
    (tmp_path / "evaluation-s1-checkpoints" / "orphan.json").write_text('{"v": 9}')    # died before the state
    assert _begin(tmp_path, resume=True).load("orphan.json") is None
    assert ev.json_stage("orphan.json", lambda: {"v": 1}) == {"v": 1}


def test_a_unit_hands_on_what_json_makes_of_it_whether_computed_or_read_back(tmp_path):
    value = {"pair": (1, 2), 3: 4.5, "nan": float("nan")}
    fresh = _begin(tmp_path, stamp="s1").json_stage("u.json", lambda: value)
    read = _begin(tmp_path, resume=True).json_stage("u.json", lambda: pytest.fail("recomputed"))
    assert fresh["pair"] == [1, 2] and fresh["3"] == 4.5 and np.isnan(fresh["nan"])
    assert json.dumps(fresh) == json.dumps(read)


def test_checkpoint_names_are_file_names():
    ev = registry.Evaluation("unused", {"sessions": [{}], "checkpoints": {}}, False)
    for bad in ("../x", "a/b", "a b", ""):
        with pytest.raises(ValueError):
            ev.load(bad)


def test_a_second_evaluation_on_the_same_day_does_not_overwrite_the_first_entry(tmp_path):
    first = _begin(tmp_path, stamp="s1")
    (tmp_path / "evaluation-s1.json").write_text("{}")
    first.complete(str(tmp_path / "evaluation-s1.json"))
    other = {"B": "d" * 64, "Bp": "e" * 64}
    _begin(tmp_path, other, stamp="s2")
    reg = _registry(tmp_path)
    assert sorted(reg.values()) == sorted(list(HASHES.values()) + list(other.values()))
    assert len(reg) == 4


def test_the_entries_already_in_the_registry_are_kept_as_they_are(tmp_path):
    """Stage 0's file is final: a later evaluation only adds to the end of it."""
    old = {f"2026-10-18:{a}": f"{i:064x}" for i, a in enumerate(["B", "Bp", "C1", "C2-sup", "C5-0.3M"])}
    text = json.dumps(old, indent=1)                                # the format the script always wrote
    (tmp_path / "evaluated.json").write_text(text, encoding="utf-8")
    _begin(tmp_path, {"B": "d" * 64, "Bp": "e" * 64}, stamp="s1")
    new_text = (tmp_path / "evaluated.json").read_text(encoding="utf-8")
    assert new_text.startswith(text[:-2])                           # every old line, byte for byte
    assert list(json.loads(new_text).items())[:len(old)] == list(old.items())


def test_a_crash_between_the_state_and_the_registry_still_counts_as_begun(tmp_path):
    _begin(tmp_path, stamp="s1")
    (tmp_path / "evaluated.json").unlink()                          # the registry write never happened
    with pytest.raises(registry.Refused, match="did not finish"):
        _begin(tmp_path)
    _begin(tmp_path, resume=True)
    assert sorted(_registry(tmp_path).values()) == sorted(HASHES.values())     # the resume registers them


def test_a_state_that_cannot_be_read_blocks_a_new_evaluation(tmp_path):
    folder = tmp_path / "evaluation-s1-checkpoints"
    folder.mkdir()
    (folder / "state.json").write_text("{not json")
    with pytest.raises(registry.Refused, match="cannot be read"):
        _begin(tmp_path)


def test_a_write_is_retried_while_onedrive_holds_the_file(monkeypatch, tmp_path):
    calls = []
    real = os.replace

    def held(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(13, "the file is held")
        return real(src, dst)
    monkeypatch.setattr(os, "replace", held)
    slept = []
    monkeypatch.setattr(registry.time, "sleep", slept.append)
    registry.write_atomic(str(tmp_path / "x.json"), b"whole")
    assert len(calls) == 3 and len(slept) == 2 and (tmp_path / "x.json").read_bytes() == b"whole"
    assert not (tmp_path / "x.json.tmp").exists()


def test_a_write_that_never_succeeds_gives_up_and_a_missing_folder_is_not_waited_for(monkeypatch, tmp_path):
    slept = []
    monkeypatch.setattr(registry.time, "sleep", slept.append)
    with monkeypatch.context() as held_for_good:
        held_for_good.setattr(registry, "RETRIES", 3)

        def held(src, dst):
            raise PermissionError(13, "the file is held")
        held_for_good.setattr(os, "replace", held)
        with pytest.raises(PermissionError):
            registry.write_atomic(str(tmp_path / "x"), b"")
    assert len(slept) == 2
    slept.clear()
    with pytest.raises(FileNotFoundError):
        registry.write_atomic(str(tmp_path / "no" / "such" / "folder" / "x"), b"")
    assert slept == []


def test_a_read_is_retried_while_onedrive_holds_the_file(monkeypatch, tmp_path):
    path = tmp_path / "x"
    path.write_bytes(b"data")
    real, calls = open, []

    def held(file, *a, **kw):
        if os.fspath(file) == str(path):
            calls.append(1)
            if len(calls) < 3:
                raise PermissionError(13, "the file is held")
        return real(file, *a, **kw)
    monkeypatch.setattr("builtins.open", held)
    monkeypatch.setattr(registry.time, "sleep", lambda s: None)
    assert registry.read_bytes(str(path)) == b"data" and len(calls) == 3


def _zip(path, members, date_time):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(zipfile.ZipInfo(name, date_time=date_time), data)


def test_the_split_hash_is_of_the_contents_not_of_the_zip_container(tmp_path):
    members = {"seed.npy": b"\x01\x02\x03", "x.npy": b"abc" * 100000}
    (tmp_path / "later").mkdir()
    (tmp_path / "other").mkdir()
    _zip(tmp_path / "a.npz", members, (2026, 10, 18, 1, 2, 4))
    _zip(tmp_path / "later" / "a.npz", members, (2020, 1, 1, 0, 0, 0))              # drawn again, at another time
    _zip(tmp_path / "c.npz", dict(members, **{"x.npy": b"abd" * 100000}), (2026, 10, 18, 1, 2, 4))
    _zip(tmp_path / "other" / "z.npz", members, (2026, 10, 18, 1, 2, 4))            # the same bytes, another name
    first = registry.npz_content_sha256([str(tmp_path / "a.npz")])
    assert registry.npz_content_sha256([str(tmp_path / "later" / "a.npz")]) == first
    assert registry.npz_content_sha256([str(tmp_path / "c.npz")]) != first          # one byte of one array
    assert registry.npz_content_sha256([str(tmp_path / "other" / "z.npz")]) != first
    one, two = str(tmp_path / "a.npz"), str(tmp_path / "other" / "z.npz")
    assert registry.npz_content_sha256([one, two]) == registry.npz_content_sha256([two, one])


# ---------------------------------------------------------------------------
# 3, continued: the evaluation script over the registry
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evaluate():
    pytest.importorskip("torch")
    import chartop_evaluate
    return chartop_evaluate


def _fake_split(folder, count, per_shard=2, labels=True, offset=0):
    """Shards holding exactly the registered test seeds (all that the completeness check reads)."""
    from atlas.chartop import gates as G
    folder.mkdir(parents=True, exist_ok=True)
    seeds = list(range(G.SEED_BASE["test"] + offset, G.SEED_BASE["test"] + offset + count))
    for k in range(0, count, per_shard):
        s = np.array(seeds[k:k + per_shard])
        name = f"test_n32_{k // per_shard:04d}.npz"
        np.savez(folder / name, seed=s, x=s * 2.0)
        if labels:
            for pre in ("labels_", "labels-cosine_"):
                np.savez(folder / (pre + name), lseed=s, H=s * 3.0)
    return folder


def test_a_split_is_whole_when_it_holds_exactly_the_registered_seeds(evaluate, tmp_path):
    assert evaluate.split_complete(str(_fake_split(tmp_path / "ok", 6)), 6)
    assert not evaluate.split_complete(str(tmp_path / "missing"), 6)
    assert not evaluate.split_complete(str(_fake_split(tmp_path / "short", 4)), 6)        # cut off
    assert not evaluate.split_complete(str(_fake_split(tmp_path / "long", 6)), 4)         # seeds beyond the count
    assert not evaluate.split_complete(str(_fake_split(tmp_path / "nolabels", 6, labels=False)), 6)
    cut = _fake_split(tmp_path / "cut", 6)
    os.remove(cut / "labels-cosine_test_n32_0001.npz")
    assert not evaluate.split_complete(str(cut), 6)                                      # one basis unlabelled
    foreign = _fake_split(tmp_path / "foreign", 6)
    np.savez(foreign / "test_n64_0000.npz", seed=np.array([3_000_000]))
    for pre in ("labels_", "labels-cosine_"):
        np.savez(foreign / (pre + "test_n64_0000.npz"), lseed=np.array([3_000_000]))
    assert not evaluate.split_complete(str(foreign), 6)                                  # a seed twice
    junk = _fake_split(tmp_path / "junk", 6)
    (junk / "test_n32_0001.npz").write_bytes(b"not a zip")
    assert not evaluate.split_complete(str(junk), 6)                                     # a shard cut mid-write


def test_a_write_cut_off_is_not_a_shard(evaluate, tmp_path):
    whole = _fake_split(tmp_path / "t", 4)
    (whole / "test_n32_0001.npz.tmp.npz").write_bytes(b"cut off")
    assert [os.path.basename(f) for f in evaluate.split_shards(str(whole))] == ["test_n32_0000.npz",
                                                                                "test_n32_0001.npz"]
    assert evaluate.split_complete(str(whole), 4)
    assert len(evaluate.split_files(str(whole))) == 6                  # two shards and four labels files


def test_the_test_split_is_drawn_only_when_it_is_not_whole(evaluate, tmp_path, monkeypatch):
    folder = tmp_path / "t"
    calls = []

    def draw(cmd, check):
        calls.append(cmd)
        _fake_split(folder, 4)                                          # the draw completes it
    monkeypatch.setattr(evaluate.subprocess, "run", draw)
    _fake_split(folder, 2)                                              # a split cut off by an interruption
    evaluate.ensure_test(str(folder), 4, 1)
    assert len(calls) == 1 and "--count" in calls[0] and "4" in calls[0]
    evaluate.ensure_test(str(folder), 4, 1)
    assert len(calls) == 1                                              # whole: left alone


def test_a_split_that_is_still_wrong_after_the_draw_is_an_error(evaluate, tmp_path, monkeypatch):
    folder = tmp_path / "t"
    monkeypatch.setattr(evaluate.subprocess, "run", lambda cmd, check: _fake_split(folder, 4))
    folder.mkdir()
    foreign = np.array([3_000_100, 3_000_101])                          # another run's shard, under its own name
    np.savez(folder / "test_n64_0007.npz", seed=foreign)
    for pre in ("labels_", "labels-cosine_"):
        np.savez(folder / (pre + "test_n64_0007.npz"), lseed=foreign)
    with pytest.raises(SystemExit, match="move any other shards out of it"):
        evaluate.ensure_test(str(folder), 4, 1)


@pytest.fixture(scope="module")
def mini_split(tmp_path_factory):
    """Six real examples at n = 32 on the registered test seeds, with both port bases' labels, in
    the shard format `scripts/chartop_data.py` writes."""
    pytest.importorskip("torch")
    import chartop_data
    from atlas.chartop import data, gates as G
    folder = tmp_path_factory.mktemp("split")
    real = data.draw
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(data, "draw", lambda seed, **kw: real(seed, n=32, **kw))
        chunk = [chartop_data._draw_one((G.SEED_BASE["test"] + i, "test", ["sine-hat", "cosine"]))
                 for i in range(6)]
    arrs = {key: np.stack([c[1][key] for c in chunk]) for key in chunk[0][1]}
    for key in chunk[0][2]:
        arrs[key] = np.array([c[2][key] for c in chunk])
    np.savez(folder / "test_n32_0000.npz", **arrs)
    for kind, pre in (("sine-hat", "labels_"), ("cosine", "labels-cosine_")):
        lab = {key: np.stack([c[3][kind][key] for c in chunk]) for key in ("H", "up")}
        lab["trace_Lambda"] = np.array([c[3][kind]["trace_Lambda"] for c in chunk])
        lab["up_energy"] = np.array([c[3][kind]["up_energy"] for c in chunk])
        lab["lseed"] = np.array([c[2]["seed"] for c in chunk])
        np.savez(folder / (pre + "test_n32_0000.npz"), **lab)
    return folder


def test_e1_rows_are_checkpointed_per_shard_and_read_back(evaluate, mini_split, tmp_path, monkeypatch):
    import torch
    from atlas.chartop import model
    assert evaluate.split_complete(str(mini_split), 6)
    nets = {"B": model.Expert("unet", "0.3M"), "C6": model.Expert("unet", "0.3M", kind="cosine")}
    for net in nets.values():
        for p in net.parameters():
            torch.nn.init.normal_(p, 0.0, 0.02)
    hashes = {"B": "1" * 64, "C6": "2" * 64}
    first = registry.begin(str(tmp_path), hashes, identity=IDENT, session={}, stamp="s1")
    rows = evaluate.e1_streamed(nets, str(mini_split), first)
    assert sorted(first.state["checkpoints"]) == ["e1-cosine-test_n32_0000.npz", "e1-sine-hat-test_n32_0000.npz"]
    assert set(rows) == {"B", "C6"} and len(rows["B"]["e_H"]) == 6
    whole = evaluate.e1_streamed(nets, str(mini_split))                # no checkpoints at all
    for arm in rows:
        for key in rows[arm]:
            np.testing.assert_array_equal(rows[arm][key], whole[arm][key])

    real = evaluate.e1_rows
    monkeypatch.setattr(evaluate, "e1_rows", lambda *a, **k: pytest.fail("a shard that was done is done again"))
    resumed = registry.begin(str(tmp_path), hashes, resume=True, identity=IDENT, session={})
    again = evaluate.e1_streamed(nets, str(mini_split), resumed)
    assert len(resumed.reused) == 2
    for arm in rows:
        assert set(again[arm]) == set(rows[arm])
        for key in rows[arm]:
            np.testing.assert_array_equal(rows[arm][key], again[arm][key])

    # one checkpoint is not the file the state recorded: only its shard is computed again
    path = tmp_path / "evaluation-s1-checkpoints" / "e1-cosine-test_n32_0000.npz"
    path.write_bytes(path.read_bytes()[:-8])
    calls = []
    monkeypatch.setattr(evaluate, "e1_rows", lambda expert, test: calls.append(expert.kind) or real(expert, test))
    third = registry.begin(str(tmp_path), hashes, resume=True, identity=IDENT, session={})
    healed = evaluate.e1_streamed(nets, str(mini_split), third)
    assert calls == ["cosine"] and third.reused == ["e1-sine-hat-test_n32_0000.npz"]
    for arm in rows:
        for key in rows[arm]:
            np.testing.assert_array_equal(rows[arm][key], healed[arm][key])


class Stages:
    """chartop_evaluate with its numerical stages replaced by cheap deterministic ones that log
    their calls.  What stays real: the registry, the split's completeness and hash, the E1
    shards' reading and checkpoints, every unit's checkpoint, the gates' judgements and the
    record's assembly."""

    def __init__(self, evaluate, monkeypatch):
        self.log, self.crash_at = [], None
        mp = monkeypatch
        mp.setattr(evaluate, "DUAL_CHUNK", 2)
        mp.setattr(evaluate.tr, "load_expert", lambda p, dev: types.SimpleNamespace(
            kind="sine-hat", config=lambda: {"weights": os.path.basename(os.path.dirname(p))}))
        mp.setattr(evaluate, "LearnedExpert", lambda net, envelope=None: types.SimpleNamespace(net=net, calls=0))
        mp.setattr(evaluate, "e1_rows", self.e1_rows)
        mp.setattr(evaluate, "benchmark_problem", self.benchmark_problem)
        mp.setattr(evaluate, "dual_records", self.dual_records)
        mp.setattr(evaluate.data, "draw", lambda s: types.SimpleNamespace(problem=("test-piece", s)))
        mp.setattr(evaluate, "kt_record", self.kt_record)
        mp.setattr(evaluate, "rollout_record", self.rollout_record)

    def hit(self, *key):
        self.log.append(key)
        if self.crash_at == key:
            raise RuntimeError("the power went at " + repr(key))

    def e1_rows(self, expert, test):
        self.hit("e1_rows")
        (n, t), = test.items()
        seed = np.asarray(t["meta_seed"])
        return {"e_H": 0.3 + 1e-3 * (seed % 97), "e_p": 0.05 + 1e-3 * (seed % 89),
                "sup_mu": np.asarray(t["meta_sup_mu"], float), "freq": np.asarray(t["meta_freq"]),
                "n": np.full(len(seed), n), "seed": seed, "fo": np.where(seed % 2 == 0, -1.0, 5.0),
                "family": np.asarray(t["meta_family"])}

    def benchmark_problem(self, name, problem, experts):
        self.hit("bench", name, problem)
        base = (len(name) + 3 * len(problem)) / 100.0
        arms = {a: {"energy_rel": base + 0.01 * i, "calls_per_piece": 1.0, "min_eig": 0.5,
                    "e3": {"met": True}, "e4": {"met": base < 0.1}, "se1": {"met": True},
                    "se3": {"met": True}} for i, a in enumerate(experts)}
        return {"name": name, "problem": problem, "pieces": 2, "charts": ["a"],
                "truncation_energy_rel": 1e-3, "elapsed_s": 0.0, "arms": arms,
                "B_solution": {"c": [0.25, 0.5]},
                "_coupled_B": types.SimpleNamespace(c=np.array([0.25, 0.5])),
                "_layout": types.SimpleNamespace(R=[np.eye(2), 2 * np.eye(2)], P=2),
                "_bench": types.SimpleNamespace(problems=[("piece", name, 0), ("piece", name, 1)]),
                "_mono": object()}

    def dual_records(self, ex, problems, traces):
        self.hit("dual", problems[0])             # the first piece names the unit
        return [{"problem": p, "min_eig_rel": {"lo": np.float64(0.5 + 0.01 * i), "hi": 0.25},
                 "trace": None if traces is None else traces[i]} for i, p in enumerate(problems)]

    def kt_record(self, name, ex):
        self.hit("kt", name)
        return {"classical_iterations": 3, "kt": {"met": True}, "learned_history": [0.5, 0.25]}

    def rollout_record(self, ex):
        self.hit("rollout")
        return {"benchmark": "s-channel", "steps": 20, "records": [{"step": 1, "e4": {"met": True}}]}


@pytest.fixture()
def setup(evaluate, mini_split, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    for arm, blob in (("B", b"weights of B"), ("Bp", b"weights of Bp")):
        (runs / arm).mkdir(parents=True)
        (runs / arm / "best.pt").write_bytes(blob)
    stages = Stages(evaluate, monkeypatch)

    def argv(out, extra=(), split=mini_split):
        return ["--runs", str(runs), "--out", str(out), "--test-dir", str(split), "--test-count", "6",
                "--dual-test", "3", "--threads", "1", "--workers", "1", *extra]
    return types.SimpleNamespace(evaluate=evaluate, stages=stages, argv=argv, tmp=tmp_path, runs=runs,
                                 split=mini_split)


def _record(out):
    (path,) = glob.glob(str(out / "evaluation-*[0-9].json"))
    return json.loads(open(path, encoding="utf-8").read())


def test_an_interrupted_evaluation_resumes_and_matches_an_uninterrupted_one(setup, capsys):
    ev, stages = setup.evaluate, setup.stages
    whole, cut = setup.tmp / "whole", setup.tmp / "cut"

    assert ev.main(setup.argv(whole)) == 0
    complete = stages.log[:]
    for key in (("bench", "insert-round", "step"), ("dual", ("piece", "insert-round", 0)),
                ("dual", ("test-piece", 3_000_000)), ("dual", ("test-piece", 3_000_004)),
                ("kt", "bend-3"), ("rollout",)):
        assert key in complete
    stages.log.clear()

    stages.crash_at = ("bench", "s-channel", "step")                 # the power goes mid-benchmarks
    with pytest.raises(RuntimeError, match="power went"):
        ev.main(setup.argv(cut))
    before = stages.log[:]
    assert "run the same command with --resume" in capsys.readouterr().out
    assert not glob.glob(str(cut / "evaluation-*[0-9].json"))        # nothing was written as the record
    with pytest.raises(SystemExit, match="--resume"):
        ev.main(setup.argv(cut))                                     # and it cannot be started again
    assert stages.log == before

    stages.log.clear()
    stages.crash_at = None
    assert ev.main(setup.argv(cut, ["--resume"])) == 0
    after = stages.log[:]
    done = set(before) - {("bench", "s-channel", "step")}
    assert done and not done & set(after)                            # a finished unit is not done again
    assert ("e1_rows",) not in after                                 # the E1 shard came from its checkpoint
    assert ("bench", "s-channel", "step") in after and ("rollout",) in after
    assert set(before) | set(after) == set(complete)                 # between them, every unit

    resumed, whole_rec = _record(cut), _record(whole)
    prov = resumed.pop("resumed")
    assert [s["resume"] for s in prov["sessions"]] == [False, True]
    assert prov["units"]["bench-bend-3-steady.json"] == 0 and prov["units"]["rollout.json"] == 1
    assert prov["units"]["bench-s-channel-step.json"] == 1
    for rec in (resumed, whole_rec):
        rec.pop("written")
    assert resumed == whole_rec                                      # the numbers of an uninterrupted run
    assert whole_rec["DUAL"]["benchmark_pieces"] == 6 and whole_rec["DUAL"]["test_pieces"] == 3
    assert whole_rec["test_sha256"] and "resumed" not in whole_rec
    assert glob.glob(str(cut / "evaluation-*-test-rows.npz")) and glob.glob(str(cut / "evaluation-*[0-9].txt"))

    for flags in ([], ["--resume"]):                                 # and a finished evaluation is final
        stages.log.clear()
        with pytest.raises(SystemExit, match="runs once"):
            ev.main(setup.argv(cut, flags))
        assert stages.log == []
    assert len(json.loads((cut / "evaluated.json").read_text())) == 2


def test_resume_refuses_what_makes_it_a_different_evaluation(setup, monkeypatch, capsys):
    ev, stages = setup.evaluate, setup.stages
    out = setup.tmp / "out"
    own = setup.tmp / "own-split"
    shutil.copytree(setup.split, own)
    stages.crash_at = ("rollout",)
    with pytest.raises(RuntimeError):
        ev.main(setup.argv(out, split=own))
    stages.crash_at = None
    stages.log.clear()
    with pytest.raises(SystemExit, match="dual_test"):
        ev.main(setup.argv(out, ["--resume", "--dual-test", "4"], split=own))
    with pytest.raises(SystemExit, match="test_count"):
        ev.main(setup.argv(out, ["--resume", "--test-count", "5"], split=own))
    with monkeypatch.context() as other_gates:
        other_gates.setattr(ev, "gates_sha256", lambda: "0" * 64)
        with pytest.raises(SystemExit, match="gates_sha256"):
            ev.main(setup.argv(out, ["--resume"], split=own))
    shard = own / "test_n32_0000.npz"
    with np.load(shard) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["sup_mu"] = arrays["sup_mu"] + 1e-3                       # the same seeds, another split
    np.savez(shard, **arrays)
    capsys.readouterr()
    with pytest.raises(SystemExit, match="not the one this evaluation began with"):
        ev.main(setup.argv(out, ["--resume"], split=own))
    assert "stopped:" not in capsys.readouterr().out                 # a refusal does not offer --resume
    assert stages.log == []                                          # none of the refusals ran a stage
    shutil.copy(setup.split / "test_n32_0000.npz", shard)            # the original shard restored
    assert ev.main(setup.argv(out, ["--resume"], split=own)) == 0


def test_weights_evaluated_before_the_checkpoints_existed_are_refused(setup):
    ev = setup.evaluate
    out = setup.tmp / "legacy"
    out.mkdir()
    blob = (setup.runs / "B" / "best.pt").read_bytes()
    legacy = json.dumps({"2026-10-18:B": hashlib.sha256(blob).hexdigest()}, indent=1)
    (out / "evaluated.json").write_text(legacy)
    with pytest.raises(SystemExit, match="runs once"):
        ev.main(setup.argv(out))
    with pytest.raises(SystemExit, match="nothing to resume"):
        ev.main(setup.argv(out, ["--resume"]))
    assert (out / "evaluated.json").read_text() == legacy and setup.stages.log == []


def test_resume_with_nothing_to_resume_starts_nothing(setup):
    out = setup.tmp / "fresh"
    with pytest.raises(SystemExit, match="nothing to resume"):
        setup.evaluate.main(setup.argv(out, ["--resume"]))
    assert not out.exists() and setup.stages.log == []

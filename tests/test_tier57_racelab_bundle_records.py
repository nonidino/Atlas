"""Tier 57 -- the bundle's verification, pinned to its own records.

Everything [[poc3-racelab-bundle]] says about the four rounds is read out of
``out/racelab_bundle/verify_*.json``, which `scripts/verify_racelab_bundle.py`
and `scripts/verify_racelab_offline.py` wrote on fresh clones.  These tests
hold the page to those records, and the records to what a passing bundle looks
like -- including the failed rounds, which are kept because each one IS a
finding (W250, W251, W253).

The records are records OF the bundle and are not carried inside it, so in the
bundle every test here skips and says so.
"""

from __future__ import annotations

import json
import os
import re

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(_ROOT, "out", "racelab_bundle")
PAGE = os.path.join(_ROOT, "wiki", "concepts", "Atlas 0.1", "common",
                    "poc3-racelab-bundle.md")
PLATFORMS = ("windows-amd64", "linux-x86_64")


def _rec(name: str) -> dict:
    path = os.path.join(REC, name)
    if not os.path.isfile(path):
        pytest.skip("%s is absent -- the verification records live upstream, "
                    "not in the bundle they verify" % name)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _page() -> str:
    if not os.path.isfile(PAGE):
        pytest.skip("the case study page is absent")
    with open(PAGE, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# round 4: what passing looks like, on both platforms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plat", PLATFORMS)
def test_round_4_passes_every_check_on_a_fresh_clone(plat):
    r = _rec("verify_%s.json" % plat)
    assert r["run_sh_mode_in_the_index"] == "100755"
    assert r["run_sh_has_a_carriage_return"] is False
    assert r["run_cmd_lines_end_crlf"] is True
    idn = r["identity"]
    assert idn["different"] == []
    assert idn["identical"] >= 164
    assert idn["every_poc3_test_is_carried"] is True
    assert idn["settled.npz_identical_to_atlas_0.1_s_cache"] is True
    assert idn["weights_sha256_matches_SOURCE_COMMITS"] is True
    nb = r["scan_unlicensed_checkpoint"]
    assert nb["uses"] == [] and nb["files_or_dirs_named_for_it"] == []
    assert r["scan_data"]["unexpected"] == [] and r["scan_scot"] == []
    assert r["static_checks_pass"] is True
    assert r["source_commits"]["dirty"] is False


@pytest.mark.parametrize("plat", PLATFORMS)
def test_round_4_launcher_exits_0_and_the_page_s_socket_carries_a_frame(plat):
    lc = _rec("verify_%s.json" % plat)["launch_check"]
    assert lc["exit_code"] == 0, lc["tail"][-600:]
    assert "a JSON frame and a PNG frame arrived over /ws" in lc["tail"]
    assert "the page can receive the march" in lc["tail"]


def test_linux_ran_the_bottom_of_the_supported_python_range():
    lc = _rec("verify_linux-x86_64.json")["launch_check"]
    assert (lc["python_given"] or "").endswith("python3.10")
    assert "python   3.10." in lc["tail"]


def test_the_two_platforms_agree_on_the_learned_column_to_the_printed_precision():
    got = {}
    for plat in PLATFORMS:
        tail = _rec("verify_%s.json" % plat)["launch_check"]["tail"]
        rms = re.search(r"rms against it ([0-9.]+)", tail)
        err = re.search(r"median ([0-9.]+), max ([0-9.]+)", tail)
        assert rms and err, tail[-800:]
        got[plat] = (rms.group(1), err.group(1), err.group(2))
    assert got["windows-amd64"] == got["linux-x86_64"], got


def test_with_no_network_the_classical_column_still_runs():
    r = _rec("verify_windows-amd64_offline.json")
    a, b = r["launcher_offline"], r["socket_refused"]
    assert a["exit_code"] == 3
    assert a["said_the_fetch_failed"] and a["classical_marched"]
    assert a["scot_stamp_written"] is False
    assert "a JSON frame and a PNG frame arrived over /ws" in a["tail"]
    assert b["exit_code"] == 3
    assert b["connection_attempts"] == [], "a connection left the machine"
    assert b["loopback_connections_allowed"], \
        "the self-test's own socket check never ran under the refusal"
    assert r["passes"] is True


def test_the_identity_check_was_broken_on_purpose_and_caught_it():
    r = _rec("verify_identity_control.json")
    assert r["caught_exactly_the_altered_file"] is True
    assert r["different"] == ["tests/test_tier56_racelab_drawn_car.py"]


# ---------------------------------------------------------------------------
# the failed rounds, each of which is a finding
# ---------------------------------------------------------------------------


def test_round_1_recorded_W250_linux_refusing_its_own_settled_field():
    r = _rec("verify_linux-x86_64_round1_W250.json")
    assert r["static_checks_pass"] is True
    lc = r["launch_check"]
    assert lc["exit_code"] == 1
    assert "DIFFERENT car" in lc["tail"]


def test_round_1_recorded_stock_ubuntu_naming_the_venv_package():
    lc = _rec("verify_linux-x86_64_round1_stock-python.json")["launch_check"]
    assert lc["exit_code"] == 1
    assert "apt install python3.12-venv" in lc["tail"]


@pytest.mark.parametrize("plat", PLATFORMS)
def test_round_2_passed_without_ever_opening_the_page_s_socket(plat):
    """W251's whole point, preserved: the self-test printed OK on a bundle whose
    page could not receive a frame, because it did not look."""
    lc = _rec("verify_%s_round2_W251.json" % plat)["launch_check"]
    assert lc["exit_code"] == 0
    assert "live connection" not in lc["tail"]


def test_round_3_offline_recorded_W253_the_check_going_through_the_proxy():
    r = _rec("verify_windows-amd64_offline_round3_W253.json")
    assert r["launcher_offline"]["exit_code"] == 1
    assert "ConnectionRefusedError" in r["launcher_offline"]["tail"]
    assert r["socket_refused"]["connection_attempts"] == []


# ---------------------------------------------------------------------------
# the final build, verified statically after the write-up landed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plat", PLATFORMS)
def test_the_final_build_passes_every_static_check(plat):
    r = _rec("verify_%s_final.json" % plat)
    assert r["static_checks_pass"] is True
    assert r["identity"]["different"] == []
    assert r["run_sh_mode_in_the_index"] == "100755"


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------


def test_the_page_quotes_the_records():
    page = _page()
    win = _rec("verify_windows-amd64.json")
    n = win["identity"]["identical"]
    #: the page writes numbers in LaTeX, so "164 of 164" is "$164$ of $164$"
    assert re.search(r"\$?%d\$? of \$?%d\$?" % (n, n), page), n
    assert win["source_commits"]["weights_sha256"][:8] in page
    for fp in ("993c358a", "f591a83c", "a1f67a8e"):
        assert fp in page, fp
    for w in ("W243", "W250", "W251", "W252", "W253"):
        assert w in page, w


def test_the_page_names_what_the_tier_did_not_do():
    page = _page()
    assert "What this tier did NOT do, named" in page
    assert "not pushed" in page

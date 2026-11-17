"""The studio's shell (integrated UI plan, phase P0): the server, its protocol, the
page's files, the vendored libraries, and every example opened and saved.

The page itself is checked in a real browser at every phase; these tests cover what
can be checked without one: every message the page can send, applied to a session
and checked through `spec.check`; the page served with every file it names (and every
module those import) from this machine, with the MIME types a browser requires; the
vendored files against their pinned hashes; and a case file opened and saved bit
for bit.
"""

from __future__ import annotations

import json
import os
import re

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("shapely")

from fastapi.testclient import TestClient  # noqa: E402

from atlas.studio import fonts, protocol, vendor  # noqa: E402
from atlas.studio.server import STATIC, create_app  # noqa: E402
from atlas.studio.session import Refused, Session, apply_ops  # noqa: E402
from atlas.workbench.spec import EXAMPLES, CaseSpec, check, example_case  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    cases = tmp_path_factory.mktemp("cases")
    with TestClient(create_app(str(cases))) as c:
        yield c


@pytest.fixture()
def session(tmp_path):
    return Session(cases_dir=str(tmp_path), spec=example_case("wall-2"), key="wall-2")


# -- the vendored libraries and the fonts --------------------------------------------

def test_every_vendored_file_hashes_to_its_pin():
    assert vendor.check() == []


def test_every_vendored_package_carries_its_licence():
    for pkg in vendor.PACKAGES:
        assert any(re.search(r"(^|/)LICENSE(\.txt)?$", p) for p, _h in pkg.files), pkg.name
    notices = open(os.path.join(ROOT, "THIRD-PARTY-NOTICES.md"), encoding="utf-8").read()
    for pkg in vendor.PACKAGES:
        assert pkg.name in notices and pkg.version in notices, pkg.name


def test_ranade_is_never_committed():
    """ITF Free Font License, section 02: no distribution through a repository."""
    import subprocess
    out = subprocess.run(["git", "check-ignore", "atlas/studio/static/fonts/ranade/x.woff2"],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, "the Ranade folder is not git-ignored"
    tracked = subprocess.run(["git", "ls-files", "atlas/studio/static/fonts/ranade"],
                             cwd=ROOT, capture_output=True, text=True).stdout
    assert tracked.strip() == ""


def test_the_font_css_names_only_files_that_exist():
    css = fonts.font_css()
    for url in re.findall(r"url\('([^']+)'\)", css):
        rel = url.split("/static/", 1)[1]
        assert os.path.isfile(os.path.join(STATIC, *rel.split("/"))), url
    assert "Instrument Sans" in css and "JetBrains Mono" in css
    assert ("Ranade" in css) == fonts.ranade_installed()


# -- the page and its files -----------------------------------------------------------

def _imports(js: str) -> list[str]:
    return re.findall(r"""(?:^|\n)\s*(?:import|export)\s[^'"]*?from\s+['"]([^'"]+)['"]""", js) + \
        re.findall(r"""(?:^|\n)\s*import\s+['"]([^'"]+)['"]""", js)


def test_the_page_and_every_file_it_needs_come_from_this_machine(client):
    r = client.get("/?v=1")
    assert r.status_code == 200 and "<title>Atlas</title>" in r.text
    refs = (re.findall(r'<script[^>]*\ssrc="([^"]+)"', r.text)
            + re.findall(r'<link[^>]*\shref="([^"]+)"', r.text))
    assert refs
    for u in refs:
        assert not re.match(r"[a-z]+://", u) and not u.startswith("//"), f"outside: {u}"
    seen, todo = set(), [u for u in refs if not u.startswith("data:")]
    mimes = {}
    while todo:
        u = todo.pop()
        path = u.split("?", 1)[0]
        if path in seen:
            continue
        seen.add(path)
        resp = client.get(path)
        assert resp.status_code == 200, path
        mimes[path] = resp.headers["content-type"].split(";")[0]
        if path.endswith(".js") and path.startswith("/static/js/"):
            base = path.rsplit("/", 1)[0]
            for imp in _imports(resp.text):
                assert imp.startswith("."), f"{path} imports {imp} from outside the page"
                todo.append(os.path.normpath(base + "/" + imp).replace("\\", "/"))
    assert all(mimes[p] == "text/javascript" for p in mimes if p.endswith(".js")), mimes
    assert mimes["/fonts.css"] == "text/css"
    # every module the page imports was found and served
    assert "/static/js/views/geometry.js" in seen and "/static/js/canvas.js" in seen


def test_the_wasm_and_the_fonts_have_the_types_a_browser_requires(client):
    r = client.get("/static/vendor/planegcs/dist/planegcs_dist/planegcs.wasm")
    assert r.status_code == 200 and r.headers["content-type"] == "application/wasm"
    assert r.headers["cache-control"] == "no-cache"
    r = client.get("/static/fonts/jetbrains-mono/jetbrains-mono-latin-400-normal.woff2")
    assert r.status_code == 200 and r.headers["content-type"] == "font/woff2"
    assert client.get("/favicon.ico").status_code == 200


def test_no_page_file_names_the_network():
    """The page opens with the network cut: nothing of ours reaches outside."""
    bad = []
    for dp, _dn, fn in os.walk(STATIC):
        if os.sep + "vendor" in dp + os.sep:
            continue
        for f in fn:
            if f.endswith((".html", ".js", ".css")):
                text = open(os.path.join(dp, f), encoding="utf-8").read()
                for m in re.finditer(r"(https?:)?//[a-z0-9.-]+\.[a-z]{2,}", text):
                    if not m.group(0).endswith("www.w3.org"):    # SVG's namespace, never fetched
                        bad.append((f, m.group(0)))
    assert bad == []


def test_every_message_the_page_sends_has_a_handler():
    sent = set()
    for dp, _dn, fn in os.walk(os.path.join(STATIC, "js")):
        for f in fn:
            text = open(os.path.join(dp, f), encoding="utf-8").read()
            sent |= set(re.findall(r"(?:send|request)\('([a-z]+\.[a-z_]+)'", text))
    assert sent, "no messages found"
    assert sent <= set(protocol.HANDLERS), sorted(sent - set(protocol.HANDLERS))


# -- the socket -----------------------------------------------------------------------------

def _until(ws, kind, n=200):
    for _ in range(n):
        m = ws.receive_json()
        if m.get("type") == kind:
            return m
    raise AssertionError(f"no {kind} message")


def test_hello_gives_a_session_the_catalog_and_the_state(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "session": None})
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["session"] and hello["resumed"] is False
        cat = _until(ws, "catalog")
        assert sum(f["runs"] for f in cat["families"]) == 8
        assert len(cat["examples"]) == len(EXAMPLES)
        st = _until(ws, "state")
        assert {"case", "meta", "issues", "status", "drawing"} <= set(st)
        CaseSpec.model_validate(st["case"])
        sid = hello["session"]
        ws.send_json({"id": 1, "type": "case.example", "key": "bend-3"})
        st = _until(ws, "state")
        assert st["meta"]["key"] == "bend-3"
        assert _until(ws, "reply")["ok"] is True
    # the same page reconnecting gets its own case back
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "session": sid})
        hello = ws.receive_json()
        assert hello["resumed"] is True and hello["session"] == sid
        assert _until(ws, "state")["meta"]["key"] == "bend-3"


def test_a_run_over_the_socket_ends_in_cards_with_no_raw_seconds(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "session": None})
        ws.receive_json()
        ws.send_json({"id": 1, "type": "case.example", "key": "wall-2"})
        _until(ws, "reply")
        ws.send_json({"id": 2, "type": "edit.apply", "label": "one step",
                      "ops": [{"op": "set", "path": ["run", "steps"], "value": 1}]})
        _until(ws, "reply")
        ws.send_json({"id": 3, "type": "run.start", "arms": ["serial", "full"]})
        done = None
        fields = None
        for _ in range(2000):
            m = ws.receive_json()
            if m["type"] == "fields":
                fields = m
            if m["type"] == "run" and m["status"] in ("done", "failed", "stopped"):
                done = m
                break
        assert done is not None and done["status"] == "done", done
        cards = done["cards"]
        assert cards["checks"] and cards["failed"] == 0
        text = json.dumps(cards)
        for raw in ("seconds_per_step", '"mean"', '"median"', "wall_seconds", "build_seconds"):
            assert raw not in text, raw
        assert fields is not None and set(fields["arms"]) >= {"serial", "full"}


# -- the session: every message, through spec.check -----------------------------------------

def test_every_handler_refuses_what_it_does_not_take(session):
    for name, h in protocol.HANDLERS.items():
        reply, _ = protocol.handle(session, {"id": 1, "type": name, "bogus": 1})
        assert reply["ok"] is False and "does not take" in reply["error"], name
        for arg, (_typ, required) in h.args.items():
            if required:
                reply, _ = protocol.handle(session, {"id": 2, "type": name})
                assert reply["ok"] is False and "needs" in reply["error"], name
                reply, _ = protocol.handle(session, {"id": 3, "type": name, arg: 12.5})
                assert reply["ok"] is False and "wrong type" in reply["error"], name
    assert protocol.handle(session, {"id": 4, "type": "nope"})[0]["ok"] is False
    assert protocol.handle(session, ["not", "an", "object"])[0]["ok"] is False


def test_edits_as_data_and_undo(session):
    before = session.spec.to_json()
    r, pushes = protocol.handle(session, {"id": 1, "type": "edit.apply", "label": "steps",
                                          "ops": [{"op": "set", "path": ["run", "steps"],
                                                   "value": 3}]})
    assert r["ok"] and session.spec.run.steps == 3 and pushes[0]["type"] == "state"
    assert [i.severity for i in check(session.spec)] == [i["severity"]
                                                          for i in pushes[0]["issues"]]
    r, _ = protocol.handle(session, {"id": 2, "type": "edit.undo"})
    assert session.spec.to_json() == before
    r, _ = protocol.handle(session, {"id": 3, "type": "edit.redo"})
    assert session.spec.run.steps == 3
    # a refused edit changes nothing, and says so
    r, pushes = protocol.handle(session, {"id": 4, "type": "edit.apply",
                                          "ops": [{"op": "set", "path": ["run", "steps"],
                                                   "value": -1}]})
    assert r["ok"] is False and session.spec.run.steps == 3
    assert any(p["type"] == "notice" and p["kind"] == "error" for p in pushes)


def test_apply_ops_set_insert_remove_move():
    doc = {"a": [1, 2, 3], "b": {"c": 1}}
    apply_ops(doc, [{"op": "set", "path": ["b", "c"], "value": 2},
                    {"op": "insert", "path": ["a"], "index": 0, "value": 0},
                    {"op": "remove", "path": ["a", 3]},
                    {"op": "move", "path": ["a"], "from": 0, "to": 2},
                    {"op": "set", "path": ["b", "d"], "value": 5}])
    assert doc == {"a": [1, 2, 0], "b": {"c": 2, "d": 5}}
    for bad in ([{"op": "set", "path": ["zz", "y"], "value": 1}],
                [{"op": "remove", "path": ["a", 9]}],
                [{"op": "explode", "path": []}], [], "x"):
        with pytest.raises(Refused):
            apply_ops({"a": [1]}, bad)


def test_file_names_stay_in_the_cases_folder(session):
    for name in ("../evil", "a/b", "C:\\x", "", ".hidden"):
        r, _ = protocol.handle(session, {"id": 1, "type": "case.save", "name": name})
        assert r["ok"] is False, name
    r, _ = protocol.handle(session, {"id": 2, "type": "case.open", "name": "missing"})
    assert r["ok"] is False


@pytest.mark.parametrize("key", sorted(EXAMPLES))
def test_every_example_opens_and_saves_bit_for_bit(tmp_path, key):
    """P0's done-when: every existing example opens and saves unchanged.  Saved, then
    opened and saved again, the file is the same bytes; and what was saved is the
    example as the engine builds it."""
    s = Session(cases_dir=str(tmp_path), spec=example_case("wall-2"), key="x")
    s.example(key)
    path = s.save(key)
    first = open(path, "rb").read()
    want = tmp_path / "engine.json"
    example_case(key).save(str(want))
    assert first == want.read_bytes()
    s2 = Session(cases_dir=str(tmp_path), spec=example_case("wall-2"), key="y")
    s2.open(key)
    s2.save(key + "-again")
    assert open(os.path.join(str(tmp_path), key + "-again.json"), "rb").read() == first
    st = s2.state()
    assert st["meta"]["file"] == key + "-again.json" and st["meta"]["dirty"] is False


def test_a_new_case_of_every_kind_is_ready_to_run(tmp_path):
    from atlas.workbench import registry
    s = Session(cases_dir=str(tmp_path), spec=example_case("wall-2"), key="x")
    for fid in registry.available_ids():
        s.start_new(fid)
        errors = [i.message for i in check(s.spec) if i.severity == "error"]
        assert errors == [], (fid, errors)
        assert s.state()["status"]["kind"] in ("ok", "warn")
    s.undo()
    assert s.history and s.future


def test_the_learned_case_is_read_only(tmp_path):
    s = Session(cases_dir=str(tmp_path), spec=example_case("wall-2"), key="x")
    s.example("learned-farm")
    before = s.spec.to_json()
    r, pushes = protocol.handle(s, {"id": 1, "type": "edit.apply",
                                    "ops": [{"op": "set", "path": ["run", "steps"], "value": 2}]})
    assert r["ok"] is False and s.spec.to_json() == before


def test_quantised_fields_keep_the_outside_and_the_range():
    import base64

    import numpy as np
    f = np.array([[0.0, 1.0], [np.nan, 0.5]])
    q = protocol.quantise(f)
    a = np.frombuffer(base64.b64decode(q["data"]), dtype="<u2").reshape(2, 2)
    assert q["min"] == 0.0 and q["max"] == 1.0
    assert a[1, 0] == 65535 and a[0, 0] == 0 and a[0, 1] == 65534 and a[1, 1] == 32767

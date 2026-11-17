"""Start Atlas from this folder: one command, then a browser tab.

    python run.py                 # serves the studio on http://127.0.0.1:8020/ and opens it
    python run.py --no-open       # ... without opening a browser
    python run.py --port 8031     # on another port (default: the first free from 8020)
    python run.py --check         # the self-test alone, no server

`run.cmd` (Windows) and `run.sh` (macOS, Linux) build `.venv`, install the
pinned packages into it, run this file's self-test once, and then serve.  Use
them, unless `.venv` is already built; then this file is what they run.

**The build repository's solver is vendored.**  The wind farm's fluid expert,
`reference.WindowNS`, lives in a second repository and is found through
`ATLAS_BUILD_REPO`.  This folder carries the one package of it the workbench
loads, at `vendor/src/atlas/cases/windfarm/`, in the layout its loader expects,
and this file SETS `ATLAS_BUILD_REPO` to `vendor/` rather than defaulting it: a
machine that already points the variable at another checkout would otherwise
run this copy against different solver code (PoC 3's lesson).  The self-test
asserts the solver was loaded from here.

**What the self-test does**, in the order that makes a failure diagnosable:

  1. every package the workbench imports imports, and the optional ones (torch,
     Gmsh) are reported separately;
  2. the wind farm's solver loads FROM THIS FOLDER (the path is printed and
     asserted to be under `vendor/`);
  3. **each of the eight simulation types** compiles through the page's own
     compile and marches a few steps of every arm it offers through the page's
     own runner; the fields are finite inside the domain, and every exact
     control the type registers (its "bit for bit" identities, such as threaded
     equals serial) holds;
  4. **the learned case** (the wind farm with a trained network in every window):
     its weights load and one learned macro-step runs, finite and
     divergence-free, and the registered evaluation's verdict is printed; without
     torch it is reported absent, which is optional;
  5. **the page**, the studio (the integrated UI, `atlas/studio`, which this
     folder serves since 2026-11-16; the Panel page is retired): its vendored files
     against their pinned sha256, Ranade fetched from Fontshare when the network
     allows (optional: the page falls back to Instrument Sans), its own server
     started on a free port, its page and every file the page and its modules
     name fetched as a browser would, and a case opened, edited and run for one
     step over the page's own WebSocket.  A self-test that drove only the engine
     once passed on a bundle whose page could never receive a frame (PoC 3,
     W251), so the page is opened here, not assumed.

Nothing is compared with a stored number: the point is that the answers are
produced on this machine.  Exit 0: all of it.  Exit 3: all of it, and an
optional package is absent (the launchers still serve).  Anything else:
broken, and the launchers do not serve.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

os.environ["ATLAS_BUILD_REPO"] = os.path.join(HERE, "vendor")
# torch and numpy can each bring their own OpenMP runtime on Windows and macOS;
# with both loaded, the first linear-algebra call aborts the process without this.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, HERE)

#: Every package the workbench imports, by module name, and what for.  Checked
#: by importing rather than assumed: a `pip install` that failed half-way leaves
#: an environment that looks built and is not.
REQUIRED = [
    ("numpy", "the arrays"),
    ("scipy", "sparse solves, transforms and image operations, in every type"),
    ("pandas", "the results tables"),
    ("pydantic", "the case file's schema"),
    ("bokeh", "the retired Panel page, which its tests still drive"),
    ("panel", "the retired Panel page, which its tests still drive"),
    ("tornado", "the retired Panel page's server"),
    ("fastapi", "the studio's server"),
    ("uvicorn", "the studio's server process"),
    ("websockets", "the studio page's WebSocket"),
    ("shapely", "the studio's geometry: each region's cells"),
]

#: Installed by the launchers when they can be.  Without them every other part
#: of the workbench runs, and the self-test exits 3 instead of 0.
OPTIONAL = [
    ("torch", "the learned case, CPU-only; nothing else needs it"),
    ("gmsh", "File > Import geometry from Gmsh"),
]

#: One example per simulation type.  The type's Fast example where it is quick
#: to march, otherwise the type's smallest example; a step or two of every arm.
TYPES = [
    ("incompressible-2d", "wake-array-3", 2),
    ("conduction-2d", "fast-heat", 2),
    ("electric-2d", "plate-circuit", 2),
    ("transport-2d", "plume-2", 2),
    ("acoustics-2d", "fast-sound", 2),
    ("elasticity-2d", "plate-hole", 1),
    ("thermoelastic-2d", "bimetal-arc", 2),
    ("conjugate-heat-2d", "cooled-block", 2),
]

#: Exit codes, named once.  The launchers act on them.
OK, BROKEN, PARTIAL = 0, 1, 3

#: Flags that belong to `run.sh` / `run.cmd`.  The launchers pass their whole
#: argument list through, and the workbench would refuse these (PoC 2's bundle
#: rebuilt its environment and then died on `unrecognized arguments`).
LAUNCHER_ONLY = ("--check", "--reinstall", "--no-torch", "--no-gmsh")


def _version(mod) -> str:
    return str(getattr(mod, "__version__", "") or "")


def _short(exc: BaseException, n: int = 110) -> str:
    text = "%s: %s" % (type(exc).__name__, exc)
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n - 3] + "..."


def _under(path: str, root: str) -> bool:
    try:
        return (os.path.commonpath([os.path.abspath(path), os.path.abspath(root)])
                == os.path.abspath(root))
    except ValueError:                       # different drives on Windows
        return False


def _packages(optional: bool = True, quiet: bool = False):
    """Import every package by name; return (missing required, absent optional)."""
    import importlib
    missing, absent = [], []
    if not quiet:
        print("  packages")
    for name, why in REQUIRED:
        try:
            m = importlib.import_module(name)
            if not quiet:
                print("    ok       %-9s %-12s %s" % (name, _version(m), why))
        except Exception as exc:
            missing.append((name, why, exc))
            print("    MISSING  %-9s %-12s %s" % (name, "", why))
    if optional:
        for name, why in OPTIONAL:
            try:
                m = importlib.import_module(name)
                if not quiet:
                    print("    ok       %-9s %-12s %s (optional)" % (name, _version(m), why))
            except Exception as exc:
                absent.append((name, why, exc))
                if not quiet:
                    print("    absent   %-9s %-12s %s (optional): %s"
                          % (name, "", why, _short(exc, 70)))
    if missing:
        print("\n  This folder is not installed. Run .\\run.cmd (Windows) or ./run.sh "
              "(macOS, Linux),\n  which builds .venv and installs requirements.txt into "
              "it. If you are running\n  python directly, use the one inside .venv.")
        for name, _why, exc in missing:
            print("    %s: %s" % (name, _short(exc)))
    return missing, absent


def _solver() -> bool:
    """The wind farm's fluid expert, loaded the way the workbench loads it."""
    print("\n  the build repository's solver (the wind farm's fluid expert)")
    vendor = os.environ["ATLAS_BUILD_REPO"]
    try:
        from atlas.cases import window_ns as WN
        mod = WN.load_reference()
    except Exception as exc:
        print("    FAILED   reference.WindowNS   %s" % _short(exc))
        return False
    where = getattr(mod, "__file__", "?")
    inside = _under(where, vendor)
    print("    %s   reference.WindowNS   %s" % ("ok  " if inside else "WRONG",
                                               os.path.relpath(where, HERE)
                                               if inside else where))
    #: every module of the private package, not only the one asked for: a
    #: sibling resolved from another checkout would be a different solver
    stray = sorted(n for n, m in list(sys.modules.items())
                   if n.split(".")[0] == "atlas_windfarm_reference"
                   and getattr(m, "__file__", None)
                   and not _under(m.__file__, vendor))
    if not inside or stray:
        print("    [!] not loaded from this folder's vendor/: %s"
              % (", ".join(stray) or where))
        return False
    return True


def _finite_inside(fields: dict, arms) -> list[str]:
    """A drawn shape's field is NaN outside the shape by design, so finiteness is
    asked inside it: no infinity, some finite value, and the same NaN cells in
    every arm as in the full domain."""
    import numpy as np
    bad = []
    full = fields.get("full")
    for a in arms:
        f = fields.get(a)
        if f is None:
            bad.append("%s: no field" % a)
            continue
        f = np.asarray(f, dtype=float)
        if np.isinf(f).any():
            bad.append("%s: an infinite value" % a)
        if not np.isfinite(f).any():
            bad.append("%s: no finite value" % a)
        if full is not None and f.shape == np.shape(full) and not np.array_equal(
                np.isnan(f), np.isnan(np.asarray(full, dtype=float))):
            bad.append("%s: NaN where the full domain has a value" % a)
    return bad


def _types(threads: int) -> bool:
    """Each type: the page's compile, then a short run of every arm."""
    import math
    import time

    from atlas.workbench import registry, runner
    from atlas.workbench.compile import compile_case
    from atlas.workbench.spec import check as case_check
    from atlas.workbench.spec import example_case

    print("\n  the eight simulation types  (the page's own compile, then a short run of "
          "every arm\n  through the page's own runner, on %d thread%s)"
          % (threads, "s" * (threads != 1)))
    ok = True
    for fid, key, steps in TYPES:
        label = registry.short_label(fid)
        try:
            s = example_case(key)
            errors = [i.message for i in case_check(s) if i.severity == "error"]
            if errors:
                print("    [!]  %-18s %-14s the example does not pass its own checks: %s"
                      % (label, key, errors[0][:80]))
                ok = False
                continue
            s.run.steps = steps
            t0 = time.perf_counter()
            cs = compile_case(s)
            tc = time.perf_counter() - t0
            verdict = cs.verdict + (" before the compiler" if cs.refused_before else "")
            t1 = time.perf_counter()
            r = runner.CaseRun(s, arms=("serial", "parallel", "full"), steps=steps,
                               threads=threads)
            r.run_blocking()
            tr = time.perf_counter() - t1
        except Exception as exc:
            print("    [!]  %-18s %-14s %s" % (label, key, _short(exc)))
            ok = False
            continue
        res = r.results or {}
        bad = []
        exact = []
        if r.status != "done":
            bad.append("the run ended %s: %s" % (r.status, r.progress().error))
        else:
            bad += _finite_inside(r.progress().fields, r.arms)
            # The type's own EXACT controls -- the checks it registers with no
            # tolerance, "bit for bit" -- hold after any number of steps, so they
            # are asked here.  Which arms must agree is the type's to say: a split
            # by physics runs a synchronous and a lagged split, which differ by
            # design, and registers the identities they do satisfy.
            exact = [c for c in res.get("checks") or [] if c.get("tolerance") is None]
            for c in exact:
                if c.get("passed") is False:
                    bad.append("%s: FAILED (%s)" % (c.get("title"), c.get("detail") or ""))
            for a, d in (res.get("field_difference") or {}).items():
                if not all(isinstance(d.get(k), float) and math.isfinite(d[k])
                           for k in ("rms", "max")):
                    bad.append("%s against the full domain is not finite" % a)
        held = sum(1 for c in exact if c.get("passed") is True)
        agree = "  %d exact control%s held" % (held, "s" * (held != 1)) if held else ""
        print("    %s %-18s %-14s compile %-27s %5.1f s   %d step%s of %-20s %5.1f s%s"
              % ("ok  " if not bad else "[!] ", label, key, verdict, tc, steps,
                 "s" * (steps != 1), ", ".join(r.arms), tr, agree))
        for b in bad:
            print("         %s" % b)
        ok = ok and not bad
    return ok


def _learned(threads: int) -> str:
    """The learned case (demo item 1.5) through the page's own adapter: the trained
    weights load and one learned macro-step of farm-12 runs.  Returns "ok",
    "absent" (torch or the weights are not here: optional) or "broken"."""
    import hashlib
    import time
    print("\n  the learned case  (the wind farm with a trained network in every window; "
          "a fixed case)")
    try:
        import numpy as np
        from atlas.workbench import learned_case as LC
        from atlas.workbench.spec import example_case
        spec = example_case(LC.KEY)
        arms, why = LC.available_arms(spec)
        if "learned" not in arms:
            print("    absent   %s" % why.get("learned"))
            return "absent"
        t0 = time.perf_counter()
        case = LC.LearnedCase(spec, arms=("learned",), threads=threads)
        s = case.step("learned", case.initial("learned"))
        mass = case.observe("learned", s)["mass"]
        case.close()
        took = time.perf_counter() - t0
        if not (np.all(np.isfinite(s.u)) and np.all(np.isfinite(s.v)) and mass <= 1e-9):
            print("    [!]  the learned step is not finite, or not divergence-free (%.1e)" % mass)
            return "broken"
        sha = hashlib.sha256(open(LC.WEIGHTS, "rb").read()).hexdigest()[:12]
        print("    ok   the trained weights (sha256 %s...) load and step farm-12: %.1f s, "
              "divergence %.1e" % (sha, took, mass))
        gate = LC.gate_record()
        if gate is None:
            print("    note the registered evaluation is not in this copy")
        else:
            v = gate.get("verdict") or {}
            failed = [k for k in ("G1", "G2", "G3", "G4", "G5", "G6")
                      if (v.get(k) or {}).get("passed") is not True]
            print("    the registered evaluation (%s): %s"
                  % (str(gate.get("at", ""))[:10], "every bar met" if v.get("all") else
                     "not every bar met (%s); the page's card says so" % ", ".join(failed)))
        return "ok"
    except Exception as exc:
        print("    [!]  %s" % _short(exc))
        return "broken"


def _free_port(start: int = 8020, count: int = 20) -> int:
    """The first port from `start` that nothing answers on and this process can bind."""
    import socket
    for p in range(start, start + count):
        try:
            socket.create_connection(("127.0.0.1", p), timeout=0.3).close()
            continue                         # something answers there
        except OSError:
            pass
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", p))
            return p
        except OSError:
            continue
        finally:
            s.close()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _studio() -> bool:
    """The studio (atlas/studio, the integrated UI): its vendored files against their
    pinned hashes, then its own server on a free port, its page fetched with every
    file the page and its modules name, and a case opened, edited and run for a step
    over the page's own WebSocket, with the run's cards received.  A page that loads
    and never gets a frame is the failure this exists to catch (PoC 3, W251)."""
    import json
    import re
    import socket
    import subprocess
    import time
    import urllib.request

    print("\n  the studio  (its own server, its page, and a case run over its WebSocket)")
    try:
        from atlas.studio import fonts, vendor
    except Exception as exc:
        print("    [!]  the studio does not import: %s" % _short(exc))
        return False
    bad = vendor.check()
    if bad:
        print("    [!]  vendored files changed: %s" % "; ".join(bad[:3]))
        return False
    print("    ok   %d vendored files match their pinned sha256" % len(vendor.files()))
    fonts.fetch_ranade(log=print)                    # optional: Instrument Sans otherwise
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    port = _free_port(8060)
    base = "http://127.0.0.1:%d" % port
    logdir = os.path.join(HERE, "out", "workbench")
    os.makedirs(logdir, exist_ok=True)
    logpath = os.path.join(logdir, "selftest-studio.log")
    log = open(logpath, "w", encoding="utf-8", errors="replace")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "atlas.studio", "--port", str(port)],
                            cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
    ok = True
    try:
        t0 = time.perf_counter()
        while True:
            if proc.poll() is not None:
                print("    [!]  the studio's server exited (code %s) before it listened"
                      % proc.returncode)
                return False
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                if time.perf_counter() - t0 > 120:
                    print("    [!]  the studio's server did not listen within two minutes")
                    return False
                time.sleep(0.3)
        html = direct.open(base + "/", timeout=60).read().decode("utf-8", "replace")
        if "<title>Atlas</title>" not in html:
            print("    [!]  %s/ answered, but not with the studio's page" % base)
            return False
        refs = (re.findall(r'<script[^>]*\ssrc="([^"]+)"', html)
                + re.findall(r'<link[^>]*\shref="([^"]+)"', html))
        outside = [u for u in refs if re.match(r"([a-z]+:)?//", u)]
        todo = [u.split("?")[0] for u in refs if u not in outside and not u.startswith("data:")]
        seen, failed = set(), []
        while todo:
            u = todo.pop()
            if u in seen:
                continue
            seen.add(u)
            try:
                with direct.open(base + u, timeout=60) as r:
                    body = r.read()
                    kind = r.headers.get("Content-Type", "")
                if not body or (u.endswith(".js") and "javascript" not in kind):
                    failed.append(u)
                    continue
            except Exception:
                failed.append(u)
                continue
            if u.startswith("/static/js/"):
                folder = u.rsplit("/", 1)[0]
                for imp in re.findall(r"""(?:import|export)\s[^'"]*?from\s+['"](\.[^'"]+)['"]""",
                                      body.decode("utf-8", "replace")):
                    todo.append(os.path.normpath(folder + "/" + imp).replace("\\", "/"))
        if failed or outside:
            print("    [!]  the page's files: %d did not load (%s); %d named on the network"
                  % (len(failed), ", ".join(failed[:3]), len(outside)))
            ok = False
        else:
            print("    ok   the page and all %d files it and its modules name, from this "
                  "machine" % len(seen))
        from websockets.sync.client import connect
        t1 = time.perf_counter()
        with connect("ws://127.0.0.1:%d/ws" % port, open_timeout=60,
                     max_size=None, proxy=None) as ws:
            ws.send(json.dumps({"type": "hello", "session": None}))
            got = {}

            def until(kind, limit=120.0):
                end = time.perf_counter() + limit
                while time.perf_counter() < end:
                    m = json.loads(ws.recv(timeout=max(0.1, end - time.perf_counter())))
                    got[m.get("type")] = m
                    if m.get("type") == kind and (kind != "run" or m.get("status") in
                                                  ("done", "failed", "stopped")):
                        return m
                raise TimeoutError("no %s message within %.0f s" % (kind, limit))
            until("state")
            for mid, msg in enumerate((
                    {"type": "case.example", "key": "wall-2"},
                    {"type": "edit.apply", "label": "one step",
                     "ops": [{"op": "set", "path": ["run", "steps"], "value": 1}]},
                    {"type": "run.start", "arms": ["serial", "full"]}), 1):
                ws.send(json.dumps(dict(msg, id=mid)))
                r = until("reply")
                if not r.get("ok"):
                    raise RuntimeError("%s refused: %s" % (msg["type"], r.get("error")))
            run = until("run", 300.0)
        if run.get("status") != "done" or not (run.get("cards") or {}).get("checks"):
            print("    [!]  the run over the socket ended %s: %s"
                  % (run.get("status"), run.get("error")))
            ok = False
        else:
            c = run["cards"]
            print("    ok   over the page's WebSocket: a case opened, edited and run, its "
                  "cards back (%d of %d checks passed, %.1f s)"
                  % (c["passed"], len(c["checks"]), time.perf_counter() - t1))
    except Exception as exc:
        print("    [!]  %s" % _short(exc))
        ok = False
    finally:
        proc.terminate()
        try:
            proc.wait(15)
        except Exception:
            proc.kill()
        log.close()
        if not ok:
            print("    the studio server's own output is in %s" % os.path.relpath(logpath, HERE))
    return ok


def check() -> int:
    import platform
    import time

    t0 = time.perf_counter()
    print("Atlas self-test  (Python %s on %s %s, %s logical CPUs)\n"
          % (sys.version.split()[0], platform.system(), platform.machine(),
             os.cpu_count()))
    missing, absent = _packages()
    if missing:
        return BROKEN
    if not _solver():
        return BROKEN
    threads = max(1, min(2, os.cpu_count() or 1))
    types_ok = _types(threads)
    learned = _learned(threads)
    page_ok = _studio()
    took = time.perf_counter() - t0
    if not (types_ok and page_ok) or learned == "broken":
        print("\n  BROKEN -- see the [!] lines above. (%.0f s)" % took)
        return BROKEN
    if absent:
        print("\n  PARTIAL -- every type ran, the solver came from vendor/, and the page "
              "answers;\n  absent and optional: %s. (%.0f s)"
              % ("; ".join("%s (%s)" % (n, w) for n, w, _e in absent), took))
        return PARTIAL
    print("\n  OK -- every type ran, the solver came from vendor/, and the page answers. "
          "(%.0f s)" % took)
    return OK


def main(argv=None) -> int:
    # A Windows console is cp1252: one non-ASCII character in an exception
    # message would otherwise turn a diagnosis into a UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except Exception:
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--check" in argv:
        return check()
    missing, _absent = _packages(optional=False, quiet=True)
    if missing:
        return BROKEN
    args = [a for a in argv if a not in LAUNCHER_ONLY]
    no_open = "--no-open" in args
    args = [a for a in args if a != "--no-open"]
    if not any(a == "--port" or a.startswith("--port=") for a in args):
        args += ["--port", str(_free_port())]
    if not no_open and "--open" not in args:
        args.append("--open")
    from atlas.studio.__main__ import main as studio_main
    studio_main(args)
    return OK


if __name__ == "__main__":
    raise SystemExit(main())

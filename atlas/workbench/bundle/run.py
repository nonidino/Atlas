"""Start the Atlas Workbench from this folder: one command, then a browser tab.

    python run.py                 # serves http://127.0.0.1:8020/ and opens it
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
  4. **the page**: this folder's own server is started on a free port, the
     page is fetched as a browser would fetch it, every script and stylesheet
     it names is fetched from the same server, and its document is pulled over
     the page's WebSocket with Bokeh's own client.  A self-test that drove only
     the engine once passed on a bundle whose page could never receive a frame
     (PoC 3, W251), so the page is opened here, not assumed.

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
    ("bokeh", "the plots, and the server the page runs on"),
    ("panel", "the page"),
    ("tornado", "the server, and the page's WebSocket"),
]

#: Installed by the launchers when they can be.  Without them every other part
#: of the workbench runs, and the self-test exits 3 instead of 0.
OPTIONAL = [
    ("torch", "the learned case (CPU-only); nothing else uses it"),
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


def _page() -> bool:
    """Serve the page from this folder and fetch it the way a browser does."""
    import re
    import socket
    import subprocess
    import time
    import urllib.request

    print("\n  the page  (this folder's own server, on a free local port, fetched as a "
          "browser would)")
    port = _free_port(8040)
    base = "http://127.0.0.1:%d" % port
    logdir = os.path.join(HERE, "out", "workbench")
    os.makedirs(logdir, exist_ok=True)
    logpath = os.path.join(logdir, "selftest-server.log")
    log = open(logpath, "w", encoding="utf-8", errors="replace")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "run.py"), "--no-open",
                             "--port", str(port)], cwd=HERE, env=env, stdout=log,
                            stderr=subprocess.STDOUT)
    ok = True
    try:
        t0 = time.perf_counter()
        while True:
            if proc.poll() is not None:
                print("    [!]  the server exited (code %s) before it listened" % proc.returncode)
                ok = False
                return ok
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                if time.perf_counter() - t0 > 120:
                    print("    [!]  the server did not listen within two minutes")
                    ok = False
                    return ok
                time.sleep(0.3)
        print("    ok   the server listened on port %d after %.1f s"
              % (port, time.perf_counter() - t0))

        t1 = time.perf_counter()
        html = urllib.request.urlopen(base + "/", timeout=120).read().decode(
            "utf-8", "replace")
        if "<title>Atlas Workbench</title>" not in html:
            print("    [!]  %s/ answered, but not with the workbench's page" % base)
            return False
        print("    ok   %s/ answered with the page in %.1f s" % (base, time.perf_counter() - t1))

        refs = (re.findall(r'<script[^>]*\ssrc="([^"]+)"', html)
                + re.findall(r'<link[^>]*\shref="([^"]+)"', html))
        local = [u for u in refs if not re.match(r"[a-z]+://", u) or u.startswith(base)]
        outside = [u for u in refs if u not in local]
        failed = []
        for u in local:
            full = u if u.startswith(base) else base + (u if u.startswith("/") else "/" + u)
            try:
                with urllib.request.urlopen(full, timeout=60) as r:
                    if r.status != 200 or not r.read():
                        failed.append(u)
            except Exception:
                failed.append(u)
        if failed:
            print("    [!]  %d of the page's %d scripts and stylesheets did not load: %s"
                  % (len(failed), len(local), ", ".join(failed[:3])))
            ok = False
        else:
            print("    ok   all %d of its scripts and stylesheets, from this machine"
                  % len(local))
        if outside:
            print("    note the page also names %d resource%s on the network: %s"
                  % (len(outside), "s" * (len(outside) != 1), ", ".join(outside[:3])))

        # Panel's own models have to be registered here for the document to decode
        import panel  # noqa: F401
        import panel.models  # noqa: F401
        from bokeh.client import pull_session
        t2 = time.perf_counter()
        with pull_session(url=base + "/") as session:
            doc = session.document
            labels = {str(getattr(m, "label", "")) for m in doc.models}
            n = len(list(doc.models))
        want = {"Fast example": "the header's Fast example button",
                "Run": "the Run button"}
        seen = [w for k, w in want.items() if any(k in lab for lab in labels)]
        if len(seen) != len(want) or not n:
            print("    [!]  the document came over the WebSocket without %s"
                  % ", ".join(w for k, w in want.items() if w not in seen))
            ok = False
        else:
            print("    ok   its document, over the page's WebSocket: %d models, with %s"
                  " (%.1f s)" % (n, " and ".join(seen), time.perf_counter() - t2))
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
            print("    the server's own output is in %s" % os.path.relpath(logpath, HERE))
    return ok


def check() -> int:
    import platform
    import time

    t0 = time.perf_counter()
    print("Atlas Workbench self-test  (Python %s on %s %s, %s logical CPUs)\n"
          % (sys.version.split()[0], platform.system(), platform.machine(),
             os.cpu_count()))
    missing, absent = _packages()
    if missing:
        return BROKEN
    if not _solver():
        return BROKEN
    threads = max(1, min(2, os.cpu_count() or 1))
    types_ok = _types(threads)
    page_ok = _page()
    took = time.perf_counter() - t0
    if not (types_ok and page_ok):
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
    from atlas.workbench.__main__ import main as workbench_main
    workbench_main(args)
    return OK


if __name__ == "__main__":
    raise SystemExit(main())

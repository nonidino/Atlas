"""Start the PoC 3 RaceLab demo from this bundle. No installation, no paths.

    python run.py                          # then open http://127.0.0.1:8013/
    python run.py --open                   # ... and open a browser for you
    python run.py --check                  # self-test only, no server

    python run.py --column body-fitted     # the other column, on :8014

**There are two columns and neither replaces the other.**  The default,
`porous`, is fourteen rectangular windows on one lattice: it is where a window
can be clicked and flipped to a learned expert.  `body-fitted` is twelve overset
grids with the car as real walls -- the solids, the cooling duct, the radiator
and turbine, and the **certified mode**, whose limit is the classical answer by
a proof rather than a measurement.  They listen on different ports and can run
side by side.

**The body-fitted column takes about a minute to appear the first time.**  It
cuts twelve curvilinear grids, finds their donors and builds one composite
pressure system, and then builds a probe operator for the picture, which is
cached afterwards (about 64 s, once).  The settled field it releases from IS in
this bundle, so it does not have to march four minutes to reach one.

Everything this needs is in this directory, including three things that would
normally have to be found on the machine:

  * **the build repository's solvers** -- the fluid expert `reference.WindowNS`
    and the conduction and structural solvers the block and the wing use --
    vendored under `vendor/src/atlas/` in the exact layout their loaders expect,
    so all this file does is point `ATLAS_BUILD_REPO` at `vendor/`;
  * **the frozen Poseidon-T checkpoint**, 83 MB of weights that would otherwise
    be downloaded from the Hugging Face hub -- vendored under `vendor/hf-cache/`
    in the layout the hub's own cache uses, so the adapter's
    `ScOT.from_pretrained("camlab-ethz/Poseidon-T")` resolves it offline with no
    change to any source file.  Those weights are **CC-BY-NC-4.0**: research use
    only.  See `vendor/POSEIDON-T-LICENCE.md`;
  * **the settled field the demo releases from**, `out/racelab5/cache/
    settled.npz`, which carries the fingerprint of the car it was settled
    around.  Without it the demo would release from a uniform freestream and
    show a transient no recorded number was measured at.

All of it is set here, before anything imports a case study, and **set rather
than defaulted**: a machine that already has `ATLAS_BUILD_REPO` or `HF_HOME`
pointing somewhere else would otherwise run this bundle against a different
copy of the solvers or look for the weights in a cache that does not have them.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

os.environ["ATLAS_BUILD_REPO"] = os.path.join(HERE, "vendor")
#: The bundled checkpoint.  `HF_HUB_OFFLINE` is what makes it THE checkpoint:
#: without it the hub would try the network first, and a demo that silently
#: downloads 83 MB on a train is a demo that does not run on a train.
os.environ["HF_HOME"] = os.path.join(HERE, "vendor", "hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# torch and numpy can each bring their own OpenMP on Windows and macOS; without
# this the process aborts at the first linear-algebra call.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, HERE)

#: Everything `requirements.txt` installs that the CLASSICAL column needs.
#: Checked by name rather than assumed, because the failure this is here to
#: prevent is a launcher that reports success and a server that dies on its
#: first request: a `pip install` that failed halfway leaves a virtual
#: environment that looks built and is not.
REQUIRED = [
    ("numpy", "the arrays"),
    ("scipy", "the solvers' sparse linear algebra and transforms"),
    ("torch", "the composed march"),
    ("yaml", "the build repository's solver package reads its config (PyYAML)"),
    ("PIL", "PNG encoding of the field (pillow)"),
    ("fastapi", "the server"),
    ("uvicorn", "the server"),
    ("websockets", "the page's live connection: without it uvicorn refuses "
                   "every WebSocket upgrade and the page never receives a "
                   "frame (W251)"),
    ("shapely", "cutting the car's panels into closed solids, and the device "
                "rings -- the BODY-FITTED column's first build step. It is "
                "imported INSIDE the functions that use it (`car_solids`, "
                "`car_union`), so nothing at import time missed it and the "
                "Linux bundle got as far as 'cutting the solids' before "
                "stopping. Found by starting that column's page on a clean "
                "machine; the self-test had passed"),
]

#: What the LEARNED column needs on top.  Missing, the classical column still
#: runs and the page greys the learned switch out with the reason.
LEARNED = [
    ("transformers", "the checkpoint's Swin backbone"),
    ("safetensors", "reading the checkpoint"),
    ("huggingface_hub", "resolving the bundled checkpoint offline"),
    ("scOT", "the Poseidon-T model class (camlab-ethz/poseidon), installed by "
             "the launcher from a pinned commit because it has no licence to "
             "redistribute"),
]

HUB_DIR = os.path.join(HERE, "vendor", "hf-cache", "hub",
                       "models--camlab-ethz--Poseidon-T")
SETTLED = os.path.join(HERE, "out", "racelab5", "cache", "settled.npz")

#: Exit codes, named once.  The launchers act on them.
OK, BROKEN, CLASSICAL_ONLY = 0, 1, 3


def _import_all(rows, label):
    import importlib
    missing = []
    print("  %s" % label)
    for name, why in rows:
        try:
            m = importlib.import_module(name)
            print("    ok       %-16s %s" % (name, getattr(m, "__version__", "")))
        except Exception as exc:
            missing.append((name, why, exc))
            print("    MISSING  %-16s -- %s" % (name, why))
    return missing


def _packages() -> tuple[list, list]:
    """Import every dependency by name, and say which one is missing."""
    req = _import_all(REQUIRED, "packages the classical column needs")
    lrn = _import_all(LEARNED, "packages the learned column needs")
    if req:
        print("\n  This bundle is not installed. Run ./run.sh (Windows: .\\run.cmd),"
              " which builds\n  .venv and installs requirements.txt into it. If "
              "you are running python directly,\n  use the interpreter inside "
              ".venv.")
        for name, _why, exc in req:
            print("    %s: %s: %s" % (name, type(exc).__name__, exc))
    return req, lrn


def _tf32_off():
    """No float32 matmul is ever silently widened to TF32 in this process."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def socket_check(eng):
    """`demo_racelab.server.socket_check` -- the page's live connection, checked
    through the demo's own server (W251).  Imported here, not defined here, so
    the tests exercise the same function the self-test does."""
    from atlas.demo_racelab.server import socket_check as check
    return check(eng)


def _under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path),
                                   os.path.abspath(root)]) == os.path.abspath(root)
    except ValueError:
        return False


def check() -> int:
    """Prove the whole thing on this machine, in about a minute.

    In the order that makes a failure diagnosable:

      1. every package imports, and the ones only the learned column needs are
         reported separately;
      2. **the build repository's solvers load FROM THIS BUNDLE** -- the path
         each was loaded from is printed and asserted to be under `vendor/`,
         because a solver loaded from somewhere else is a different test;
      3. the settled field is present and **was settled around this car**;
      4. the graph compiles and the verdict comes out;
      5. **both columns march**, alternately, through the demo's own engine:
         the all-classical column, then every window flipped to learned beside
         the all-classical referent, and the per-window one-step error;
      6. **the page can receive them**: the demo's own server, on a real port,
         hands a real WebSocket client a frame (W251).

    Nothing is compared against a stored number; the point is that the answers
    are produced here.  Exit 0: both columns marched.  Exit 3: the classical
    column marched and the learned expert is not available here.  Anything
    else: broken.
    """
    req, lrn = _packages()
    if req:
        return BROKEN

    import time

    import numpy as np
    import torch

    _tf32_off()
    torch.set_num_threads(2)
    print("\n  python   %s  (%s)" % (sys.version.split()[0], sys.platform))
    print("  torch    %s   numpy %s   threads %d   TF32 off (asserted)"
          % (torch.__version__, np.__version__, torch.get_num_threads()))
    print("  gpu      not used: every column in this demo runs on the CPU")

    print("\n  the build repository's solvers")
    from atlas.cases import cooling_loop as CL
    from atlas.cases import window_ns as WN
    vendor = os.environ["ATLAS_BUILD_REPO"]
    for label, fn in (("reference.WindowNS", WN.load_reference),
                      ("thermostruct2d", CL.load_solvers)):
        try:
            mod = fn()
        except Exception as exc:
            print("    FAILED   %-20s %s: %s" % (label, type(exc).__name__, exc))
            return BROKEN
        where = getattr(mod, "__file__", "?")
        inside = _under(where, vendor)
        print("    %s   %-20s %s" % ("ok  " if inside else "WRONG", label,
                                     os.path.relpath(where, HERE)
                                     if inside else where))
        if not inside:
            print("    [!] that solver was NOT loaded from this bundle's vendor/")
            return BROKEN

    from atlas.cases import racelab as RL
    print("\n  the car and the field it releases from")
    fp = RL.geometry_fingerprint()
    print("    car      %s  (%d plates, %d wheels)"
          % (fp[:16], len(RL.load_geometry()["plates"]),
             len(RL.load_geometry()["wheels"])))
    if not os.path.isfile(SETTLED):
        print("    [!] %s is missing, so the demo would release from the "
              "freestream" % os.path.relpath(SETTLED, HERE))
        return BROKEN
    d = np.load(SETTLED)
    have = str(d["geometry"]) if "geometry" in d.files else None
    print("    field    %s  (%s)" % ((have or "no fingerprint")[:16],
                                     os.path.relpath(SETTLED, HERE)))
    if have != fp:
        print("    [!] the settled field was settled around a DIFFERENT car")
        return BROKEN

    have_weights = os.path.isfile(os.path.join(HUB_DIR, "refs", "main"))
    snap = (open(os.path.join(HUB_DIR, "refs", "main")).read().strip()
            if have_weights else None)
    print("    weights  %s" % ("Poseidon-T snapshot %s, offline" % snap[:12]
                               if snap else "ABSENT from vendor/hf-cache"))

    from atlas.demo_racelab.engine import Engine, RaceConfig
    t0 = time.perf_counter()
    eng = Engine(RaceConfig())
    eng._build()
    print("    release  %s" % eng.notes.get("release"))
    learned = bool(eng.stack is not None and eng.stack.ex is not None)
    print("    learned  %s" % ("available" if learned else
                                "NOT available: %s" % (eng.stack_error or
                                                       "the stack was not built")))
    print("    built in %.1f s" % (time.perf_counter() - t0))

    print("\n  the graph")
    t0 = time.perf_counter()
    eng._compile_once()
    v = eng.seam_verdicts
    if "error" in v:
        print("    FAILED   the compile raised: %s" % v["error"])
        return BROKEN
    print("    %s, refusing at %s, %d agents over %d seams  (%.1f s)"
          % (v["graph_verdict"], ", ".join(v["refusals"]) or "nothing",
             v["n_agents"], v["n_seams"], time.perf_counter() - t0))

    def march(n):
        live, ref = [], []
        with torch.no_grad():
            for _ in range(n):
                eng._march_once()
                live.append(eng.step_s)
                ref.append(eng.base_step_s)
        return live, ref

    print("\n  the classical column  (all 14 windows WindowNS, beside the "
          "all-classical referent)")
    live, ref = march(4)
    if eng.rms_vs_referent is None or eng.rms_vs_referent != 0.0:
        print("    [!] an all-classical column differs from its own referent: "
              "rms %r" % eng.rms_vs_referent)
        return BROKEN
    print("    %.0f ms / macro-step, rms against the referent %.1e (bitwise, "
          "as it must be)" % (1000 * float(np.median(live[1:])),
                              eng.rms_vs_referent))
    print("    outside the envelope on %d macro-steps" % eng.outside_steps)
    if eng.outside_steps:
        print("    [!] the classical column left its declared envelope: %s"
              % (eng.outside or {}).get("why"))
        return BROKEN

    if not learned:
        eng._publish()
        print("\n  the learned column  NOT MARCHED -- the learned expert is not "
              "available here.")
        missing = [n for n, _w, _e in lrn]
        if missing:
            print("    missing: %s" % ", ".join(missing))
        print("    The classical column runs; the page greys the learned switch "
              "out with this reason.")
        print("\n  the page's live connection  (the demo's own server, a real "
              "port, a real WebSocket)")
        ok, detail = socket_check(eng)
        print("    %s   %s" % ("ok  " if ok else "[!] ", detail))
        if not ok:
            print("    The page would receive nothing, so the classical column "
                  "could not be seen either.")
            return BROKEN
        # the body-fitted column needs no learned expert -- W294 says it could
        # not use one here anyway -- so it is checked on this path too
        if not body_fitted_check():
            return BROKEN
        print("\n  PARTIAL -- the classical column marched and the page can "
              "receive it; the learned one could not be built.")
        print("  The body-fitted column is unaffected: python run.py "
              "--column body-fitted")
        return CLASSICAL_ONLY

    print("\n  the learned column  (all 14 windows Poseidon-T, beside the "
          "all-classical referent)")
    eng.post({"kind": "preset", "name": "learned"})
    eng._drain()
    if set(eng.assignment.values()) != {"learned"}:
        print("    [!] the learned preset was not applied: %s"
              % (eng.notes.get("learned_refused") or eng.notes.get("error")))
        return BROKEN
    live, ref = march(4)
    print("    %.0f ms / macro-step against the referent's %.0f, rms against it "
          "%.3g" % (1000 * float(np.median(live[1:])),
                    1000 * float(np.median(ref[1:])), eng.rms_vs_referent))
    print("    outside the envelope on %d of 4 macro-steps%s"
          % (eng.outside_steps, " -- a learned column leaves it within a "
             "handful, which the page stamps" if eng.outside_steps else ""))
    if not eng.measure_windows():
        print("    [!] the per-window one-step error could not be measured: %s"
              % eng.stack_error)
        return BROKEN
    errs = [x for x in eng.window_error.values() if x is not None]
    if not errs or not all(np.isfinite(errs)):
        print("    [!] the per-window one-step error is not finite: %r" % errs)
        return BROKEN
    print("    one-step error against WindowNS on the same state: median %.3f, "
          "max %.3f over %d windows"
          % (float(np.median(errs)), float(np.max(errs)), len(errs)))
    eng._publish()
    if not eng.frame.png:
        print("    [!] the field did not encode to a PNG")
        return BROKEN

    print("\n  the page's live connection  (the demo's own server, a real port, "
          "a real WebSocket)")
    ok, detail = socket_check(eng)
    print("    %s   %s" % ("ok  " if ok else "[!] ", detail))
    if not ok:
        print("    The page would receive nothing: no field, no numbers, and "
              "every control would send into nothing.")
        return BROKEN

    if not body_fitted_check():
        return BROKEN

    print("\n  OK -- both columns marched, the checkpoint loaded offline, the "
          "solvers came from vendor/, and the page can receive the march.")
    print("  Now run:  python run.py           (porous, :8013)")
    print("       or:  python run.py --column body-fitted   (:8014)")
    return OK


def body_fitted_check() -> bool:
    """The OTHER column: its settled field, its scripts, and its own socket.

    **Passing the porous column's socket check says nothing about this one.**
    Tier 71 found every upgrade to `bodyfitted_server`'s ``/ws`` refused 403
    while its page and its JSON routes answered 200, and the porous column was
    fine throughout. So this asks THIS app, on a real port, for a frame.

    The composite is NOT built here: ninety seconds would make the self-test
    unusable, and what this check exists to catch lives in the transport and in
    what is on disk. What it does check on disk is the pair the body-fitted page
    cannot start without -- the settled field named for this car, and the two
    tier scripts `BodyFittedColumn.build` imports.
    """
    print("\n  the body-fitted column  (the solids, the duct, the certified "
          "mode -- python run.py --column body-fitted)")
    ok = True
    try:
        from atlas.demo_racelab.bodyfitted import BodyFittedColumn
        from atlas.cases import racelab as RL

        here = os.path.dirname(os.path.abspath(__file__))
        settled = BodyFittedColumn().settled_path(here)
        have = os.path.isfile(settled)
        print("    %s   the settled field it releases from: %s"
              % ("ok  " if have else "[!] ", os.path.basename(settled)))
        if not have:
            print("      Without it the column settles from t = 8 for about "
                  "four minutes before the first frame, or declines to march.")
            ok = False
        else:
            print("      (the car is %s, and the field's name carries it)"
                  % RL.geometry_fingerprint()[:12])
        for s in ("tier62_car_solids.py", "tier63_duct_openings.py"):
            p = os.path.join(here, "scripts", s)
            if not os.path.isfile(p):
                print("    [!]    scripts/%s is missing -- the column imports "
                      "it while building" % s)
                ok = False

        # **What the column imports, imported.**  Checking that files exist is
        # not checking that the column starts: this test passed on a clean
        # Linux machine while the page stopped at "cutting the solids" with no
        # `shapely` -- an import that lives INSIDE the function that uses it,
        # so nothing at module level went looking for it.
        #
        # `shapely` itself is in REQUIRED above, which is where a missing
        # package belongs and is what now fails loudly.  This imports the
        # modules and the two tier scripts the column reaches for while
        # building, because a bundle can also be missing one of THOSE -- and
        # doing it here costs milliseconds.  Cutting the solids for real would
        # cost 63 s on every self-test, which is not what this check is for.
        import sys as _sys
        if os.path.join(here, "scripts") not in _sys.path:
            _sys.path.insert(0, os.path.join(here, "scripts"))
        import tier62_car_solids                      # noqa: F401
        import tier63_duct_openings                   # noqa: F401
        from atlas.cases import car_solids            # noqa: F401
        from atlas.cases import car_union             # noqa: F401
        import shapely                                # noqa: F401
        print("    ok     what it imports while building resolves "
              "(shapely %s, both tier scripts)" % shapely.__version__)
    except ImportError as exc:
        print("    [!]    something the body-fitted column imports is missing: "
              "%s" % exc)
        print("      Its page would start, show a progress overlay, and stop "
              "-- which is how this was found, with the rest of the self-test "
              "passing.")
        return False
    except Exception as exc:                                  # pragma: no cover
        print("    [!]    the column could not be inspected: %s: %s"
              % (type(exc).__name__, str(exc)[:200]))
        return False

    try:
        from atlas.demo_racelab.bodyfitted_server import socket_check as bf_check
        got, detail = bf_check()
    except Exception as exc:                                  # pragma: no cover
        got, detail = False, "%s: %s" % (type(exc).__name__, str(exc)[:200])
    print("    %s   its own page's live connection: %s"
          % ("ok  " if got else "[!] ", detail))
    if not got:
        print("      Its page would load, render, and never receive a frame.")
        ok = False
    return ok


#: Flags that belong to `run.sh` / `run.cmd` and mean nothing to the app.  The
#: launchers pass their whole argument list through, so without this
#: `./run.sh --reinstall` rebuilds the environment and then dies on
#: `unrecognized arguments: --reinstall` -- PoC 2's bundle found that one.
LAUNCHER_ONLY = ("--check", "--reinstall")


def main() -> int:
    # A Windows console is cp1252, and one non-ASCII character in an exception
    # message -- a path, a quote a library prints -- would otherwise turn a
    # diagnosis into a UnicodeEncodeError.  `reconfigure` changes the error
    # handler in place; it does not wrap or close the stream.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except Exception:
            pass
    if "--check" in sys.argv:
        return check()
    req, _lrn = _packages()
    if req:
        return BROKEN
    _tf32_off()
    print()
    from atlas.demo_racelab.__main__ import main as demo_main
    demo_main([a for a in sys.argv[1:] if a not in LAUNCHER_ONLY])
    return OK


if __name__ == "__main__":
    raise SystemExit(main())

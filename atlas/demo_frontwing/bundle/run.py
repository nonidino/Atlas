"""Start the PoC 2 front-wing demo from this bundle. No installation, no paths.

    python run.py                     # then open http://127.0.0.1:8012/
    python run.py --open              # ... and open a browser for you
    python run.py --check             # self-test only, no server
    python run.py --tiling single     # the single-window referent

Everything this needs is in this directory, including the two solvers that
normally live in a different repository:

  * the fluid expert `reference.WindowNS`, at `vendor/src/atlas/cases/windfarm/`;
  * the structural expert `thermostruct2d.ThermoStruct2D`, at
    `vendor/src/atlas/solvers/`.

Both are found through `ATLAS_BUILD_REPO`, which this file sets before anything
imports a case study, because both are resolved at call time and both default to
a path on the machine this was developed on.

**Nothing is downloaded, at install time or at run time.** That is the one large
difference from the PoC 1a bundle, and it is a property of what this demo shows
rather than a convenience: every expert here is a classical solver, and the beat
that offers a frozen neural operator does so by handing the compiler its
*declaration* -- which is where the verdict comes from -- rather than its
weights. So there is no checkpoint, no `scOT`, no Hugging Face cache and no
network dependency of any kind past `pip install`.

Also bundled, because a live number should never be the only number on screen:

  * `out/w141/settled.npz` -- the settled flow field the demo releases from, so
    the first frame is the state every recorded number was measured at rather
    than a transient;
  * `out/w141/w141.json` -- the full-scale run's own artifact. Every "recorded"
    figure on screen is read out of it at load time. The demo never writes it
    and never falls back to a hard-coded copy of a number in it.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

os.environ.setdefault("ATLAS_BUILD_REPO", os.path.join(HERE, "vendor"))
# torch and numpy can each bring their own OpenMP on Windows and macOS; without
# this the process aborts at the first linear-algebra call.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, HERE)

#: Everything `requirements.txt` installs, and what fails if it is missing.
#: Checked by name rather than assumed, because the failure this is here to
#: prevent is a launcher that reports success and a server that dies on its
#: first request: a `pip install` that failed halfway leaves a virtual
#: environment that looks built and is not.
REQUIRED = [
    ("numpy", "the arrays"),
    ("scipy", "the structural expert's sparse linear algebra"),
    ("torch", "the composed march, and the adjoint through both seams"),
    ("PIL", "PNG encoding of the flow field (pillow)"),
    ("fastapi", "the server"),
    ("uvicorn", "the server"),
    ("cma", "beat 4's population baseline. Without it the race falls back to a "
            "plain evolution strategy, which is a WEAKER baseline and would "
            "flatter the gradient column, so the screen says which one ran"),
]


def _packages() -> int:
    import importlib

    missing = []
    print("  packages")
    for name, why in REQUIRED:
        try:
            m = importlib.import_module(name)
            print(f"    ok   {name:<10} {getattr(m, '__version__', '')}")
        except Exception as exc:
            missing.append((name, why, exc))
            print(f"    MISSING  {name:<10} -- {why}")
    if missing:
        print("\n  This bundle is not installed. Run ./run.sh (Windows: "
              ".\\run.cmd), which builds\n  .venv and installs requirements.txt "
              "into it. If you are running python directly,\n  use the "
              "interpreter inside .venv.")
        for name, _why, exc in missing:
            print(f"    {name}: {type(exc).__name__}: {exc}")
        return 1
    return 0


def check() -> int:
    """Prove the whole thing on this machine, in about a minute.

    Four things, in the order that makes a failure diagnosable:

      1. every package imports;
      2. **both external solvers load** -- the fluid expert and the structural
         one, from `vendor/`, which is the step that fails on a machine where
         `ATLAS_BUILD_REPO` is wrong;
      3. the graph compiles and the per-seam verdicts come out, including the
         substitution beat's three columns, which needs no marching at all;
      4. a few macro-steps march, and the power balance closes.

    Nothing is compared against a stored number. The point is that the answers
    are produced here.
    """
    rc = _packages()
    if rc:
        return rc

    import time

    import numpy as np
    import torch

    torch.set_num_threads(1)
    print(f"\n  python   {sys.version.split()[0]}  ({sys.platform})")
    print(f"  torch    {torch.__version__}   numpy {np.__version__}")
    print(f"  threads  {torch.get_num_threads()}")
    print(f"  solvers  {os.environ['ATLAS_BUILD_REPO']}")

    from atlas.cases import front_wing as F
    from atlas.demo_frontwing import substitution as SUB
    from atlas.demo_frontwing.engine import Engine, DemoConfig, load_recorded

    print("\n  the two external solvers")
    from atlas.cases.window_ns import load_reference
    from atlas.cases.thermal_strain import load_solvers
    ref = load_reference()
    ts = load_solvers()
    print(f"    ok   reference.WindowNS      {ref.__file__}")
    print(f"    ok   thermostruct2d          {ts.__file__}")

    rec = load_recorded()
    print(f"\n  the recorded artifact  "
          f"{'ok, ' + str(rec.get('generated')) if rec.get('available') else 'ABSENT -- the recorded columns will be blank'}")

    print("\n  beat 1: the substitution, from declarations alone")
    u = np.full((144, 208), F.U_INF)
    v = np.zeros((144, 208))
    t0 = time.perf_counter()
    sub = SUB.evaluate(u, v, dict(F.DESIGN_REF), F.H0_REF,
                       tiling=F.DEFAULT_TILING)
    for k in sub["order"]:
        r = sub["rows"][k]
        print(f"    {r['name'][:46]:<46} {r['n_red']} refused, "
              f"{r['n_amber']} uncertified")
    print(f"    ({time.perf_counter() - t0:.1f} s, three compiles, no weights)")
    reds = [sub["rows"][k]["n_red"] for k in sub["order"]]
    if not (reds[-1] > reds[0]):
        print("    [!] the substitution did not change the verdict map here, "
              "which it should on a six-window graph. Something is wrong.")
        return 1

    print("\n  the live march")
    eng = Engine(DemoConfig(tiling="six").clamped())
    ms = []
    for _ in range(6):
        t0 = time.perf_counter()
        eng._march_once()
        ms.append((time.perf_counter() - t0) * 1000.0)
    b = eng.balance.payload()
    r = eng.frame.payload["readout"]
    print(f"    {float(np.median(ms[2:])):.0f} ms / macro-step on this machine")
    print(f"    downforce {r['downforce']:.6f}   ride height "
          f"{r['ride_height']:.5f}   peak von Mises {r['vm_max']:.1f}")
    if b.get("last"):
        print(f"    power balance, corrected: {b['last']['corrected']:.3e} "
              f"(recorded, settled: {b['recorded']['settled']:.2e})")
    eng.stop()

    if not np.isfinite(r["downforce"]) or r["downforce"] <= 0:
        print("    [!] the march did not produce a sensible downforce.")
        return 1

    print("\n  OK -- both solvers loaded, the compiler answered, the march ran.")
    print("  Now run:  python run.py")
    return 0


#: Flags that belong to `run.sh` / `run.cmd` and mean nothing to the app.
#: The launchers pass their whole argument list through, so without this
#: `./run.sh --reinstall` rebuilds the virtual environment and then dies on
#: `unrecognized arguments: --reinstall` -- the install worked and the user is
#: told it did not. (The PoC 1a bundle's `run.py` has the same defect.)
LAUNCHER_ONLY = ("--check", "--reinstall")


def main() -> int:
    if "--check" in sys.argv:
        return check()
    if _packages():
        return 1
    print()
    from atlas.demo_frontwing.cli import main as demo_main
    return demo_main([a for a in sys.argv[1:] if a not in LAUNCHER_ONLY])


if __name__ == "__main__":
    raise SystemExit(main())

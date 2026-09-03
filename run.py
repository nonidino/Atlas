"""Start the PoC 1a wind-farm demo from this bundle. No installation, no paths.

    python run.py                       # then open http://127.0.0.1:8011/
    python run.py --domain large --turbines 25 --open
    python run.py --device cuda         # if you have one
    python run.py --check               # self-test only, no server

Everything this needs is in this directory, including the two things that would
normally have to be found on the machine:

  * the classical fluid expert `reference.WindowNS`, which lives in a different
    repository -- vendored under `vendor/` in the exact layout its loader
    expects, so all this file does is point `ATLAS_BUILD_REPO` at it;
  * the **frozen Poseidon-T checkpoint**, 83 MB of weights that would otherwise
    be downloaded from the Hugging Face hub on first use -- vendored under
    `vendor/hf-cache/` in the layout the hub's own cache uses, so
    `ScOT.from_pretrained("camlab-ethz/Poseidon-T")` resolves it offline with no
    change to any source file.  Those weights are **CC-BY-NC-4.0**: research use
    only.  See `vendor/POSEIDON-T-LICENCE.md`.

Both are set before anything imports the case studies, because both are read at
call time and both default to paths on the machine this was developed on.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

os.environ.setdefault("ATLAS_BUILD_REPO", os.path.join(HERE, "vendor"))
#: The bundled checkpoint. `HF_HUB_OFFLINE` is what makes it *the* checkpoint:
#: without it the hub would try the network first, and a demo that silently
#: downloads 83 MB on a train is a demo that does not run on a train.
os.environ.setdefault("HF_HOME", os.path.join(HERE, "vendor", "hf-cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# torch and numpy can each bring their own OpenMP on Windows and macOS; without
# this the process aborts at the first linear-algebra call.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, HERE)

#: Everything `requirements.txt` installs, and what fails if it is missing.
#: Checked by name rather than assumed, because the failure mode this is here to
#: prevent is a launcher that reports success and a server that dies on its
#: first request.
REQUIRED = [
    ("numpy", "the arrays"),
    ("scipy", "the classical solver's linear algebra"),
    ("torch", "every composed column, and the adjoint"),
    ("PIL", "PNG encoding of the fields (pillow)"),
    ("fastapi", "the server"),
    ("uvicorn", "the server"),
    ("transformers", "the checkpoint's Swin backbone"),
    ("safetensors", "reading the checkpoint"),
    ("huggingface_hub", "resolving the bundled checkpoint offline"),
    ("scOT", "the Poseidon-T model class (camlab-ethz/poseidon)"),
]


def _packages() -> int:
    """Import every dependency by name and say which one is missing.

    Loudly, and before anything else runs: `pip install` failing halfway through
    leaves a virtual environment that looks built and is not, and the symptom
    without this is a traceback five minutes later from inside a solver.
    """
    import importlib

    missing = []
    print("  packages")
    for name, why in REQUIRED:
        try:
            m = importlib.import_module(name)
            v = getattr(m, "__version__", "")
            print(f"    ok   {name:<16} {v}")
        except Exception as exc:
            missing.append((name, why, exc))
            print(f"    MISSING  {name:<16} -- {why}")
    if missing:
        print("\n  This bundle is not installed. Run ./run.sh (Windows: .\\run.cmd), which "
              "builds .venv\n  and installs requirements.txt into it. If you are "
              "running python directly,\n  use the interpreter inside .venv.")
        for name, _why, exc in missing:
            print(f"    {name}: {type(exc).__name__}: {exc}")
        return 1
    return 0


def check() -> int:
    """Prove the whole claim on this machine, in about a minute.

    Loads both fluid experts and the undivided classical solver, marches a few
    macro-steps of each **alternately** -- so each has the machine to itself,
    which is the same discipline the demo's own measurement panel uses -- and
    prints what they cost here. Nothing is compared against a stored number:
    the point is that the ratio is produced by this machine.
    """
    rc = _packages()
    if rc:
        return rc

    import time

    import numpy as np
    import torch

    # The same thread count the server runs with. Left at torch's default the
    # self-test reported 784 ms against the server's 437 ms for the same step on
    # the same box, because the composed step saturates near eight cores and
    # sixteen of them are contention -- a self-test that does not use the
    # demo's own settings is measuring something the demo never does.
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))

    print(f"\n  python   {sys.version.split()[0]}  ({sys.platform})")
    print(f"  torch    {torch.__version__}   numpy {np.__version__}")
    print(f"  threads  {torch.get_num_threads()}   cuda {torch.cuda.is_available()}")
    print(f"  expert   {os.environ['ATLAS_BUILD_REPO']}")
    print(f"  weights  {os.environ['HF_HOME']}  (offline)")

    from atlas.cases import scaling_ladder as sl
    from atlas.cases import wake_array as wa
    from atlas.cases import wind_farm_design as wd
    from atlas.demo import engine as de

    cfg = de.DemoConfig(domain="medium", k=12).clamped()
    if cfg.expert != "poseidon":
        print("\n  [!] the frozen checkpoint could not be built here, so the "
              "composed column\n      below is the CLASSICAL one and the speed "
              "ratio is not the demo's claim.")
    case = cfg.case()
    print(f"\n  {case.n_windows} windows, {case.shape[1]}x{case.shape[0]} cells, "
          f"{cfg.k} turbines, expert = {cfg.expert}")

    ro = de.demo_rollout(case, cfg.expert)
    th = torch.as_tensor(wd.project_design(de.layout_grid(case), case),
                         dtype=wd.TORCH_DTYPE)
    u, v = ro.freestream()
    mono = sl.reference_monolith(case.tiling.nx, case.tiling.ny, wa.NU_REF)
    uu = np.full(case.shape, 1.0)
    vv = np.zeros(case.shape)
    band = ro._band.cpu().numpy()

    cms, kms = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        with torch.no_grad():
            u, v, power, _ = ro.macro_step(u, v, th)
        cms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        tu = torch.as_tensor(uu, dtype=wd.TORCH_DTYPE)
        tv = torch.as_tensor(vv, dtype=wd.TORCH_DTYPE)
        fx, fy, _un, _T, p = ro.disks.forcing(tu, tv, th)
        u1, v1 = mono.step_batch(uu[None], vv[None], wa.MACRO_DT, bc0=None,
                                 force=(fx.numpy()[None], fy.numpy()[None]))
        uu, vv = np.where(band, 1.0, u1[0]), np.where(band, 0.0, v1[0])
        kms.append((time.perf_counter() - t0) * 1000.0)

    # If there is a GPU, the self-test uses it too: a bundle that installed a
    # CUDA wheel and then only ever exercised the CPU has not checked the thing
    # it just spent four gigabytes on.
    if torch.cuda.is_available():
        try:
            gro = de.demo_rollout(case, cfg.expert, device="cuda")
            gth = th.to("cuda")
            gu, gv = gro.freestream()
            gms = []
            with torch.no_grad():
                for _ in range(3):
                    t0 = time.perf_counter()
                    gu, gv, gp, _ = gro.macro_step(gu, gv, gth)
                    torch.cuda.synchronize()   # or the timer measures the queue
                    gms.append((time.perf_counter() - t0) * 1000.0)
            print("")
            print(f"  cuda     {torch.cuda.get_device_name(0)}")
            print(f"    composed graph on the GPU {float(np.median(gms[1:])):8.0f} "
                  f"ms / macro-step, farm power {float(gp.sum()):.3f}")
        except Exception as exc:                            # pragma: no cover
            print("")
            print(f"  [!] a GPU is present but the composed column failed on "
                  f"it: {type(exc).__name__}: {exc}")
            return 1

    c, m = float(np.median(cms[1:])), float(np.median(kms[1:]))
    print(f"\n  composed graph            {c:8.0f} ms / macro-step")
    print(f"  undivided classical       {m:8.0f} ms / macro-step")
    print(f"  ratio                     {m / c:8.2f}x  "
          f"({'composed' if m > c else 'undivided'} is the faster one here)")
    print(f"  farm power  composed {float(power.sum()):.3f}   "
          f"undivided {float(p.sum()):.3f}   (3 steps from still air, not settled)")
    print("\n  OK -- both columns marched and the checkpoint loaded offline.")
    print("  Now run:  python run.py")
    return 0


def main() -> int:
    if "--check" in sys.argv:
        return check()
    if _packages():
        return 1
    print()
    from atlas.demo.cli import main as demo_main
    return demo_main([a for a in sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())

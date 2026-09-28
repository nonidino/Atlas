"""Which physics families the workbench knows, and how far each is wired.

Step 1 of [[outcome-c4-path-to-declarative-cases]]: Python written once per
solver, never per case.  In the shell this is a catalogue, not yet a factory: it
names each family, the solvers and devices it would use (all of which exist in
`atlas/cases/`), and **what is missing before the workbench can run it**, so that
the GUI can show an unavailable option with its reason instead of hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Family:
    id: str
    label: str
    status: str                 # "ready-to-wire" | "planned"
    solvers: tuple[str, ...]
    devices: tuple[str, ...] = ()
    has_full_domain: bool = False
    note: str = ""
    sources: tuple[str, ...] = field(default_factory=tuple)


FAMILIES: tuple[Family, ...] = (
    Family(
        id="incompressible-2d",
        label="2-D incompressible flow with actuator disks (wind farm)",
        status="ready-to-wire",
        solvers=("WindowNS, elliptic part exposed (one window)",
                 "RectangularNS on the undivided domain (the full-domain reference)"),
        devices=("actuator-disk",),
        has_full_domain=True,
        note=("Every piece exists and is tested: CS-7's tilings, the projected assembly, "
              "W346's serial and threaded columns and its timing harness. The workbench "
              "runner that drives them from a case file is plan step 3 and is not "
              "built yet."),
        sources=("atlas/cases/scaling_ladder.py", "atlas/cases/wake_array.py",
                 "scripts/w346_rotor_count_speed.py"),
    ),
    Family(
        id="conduction-loop",
        label="Conduction block on a coolant loop (CS-13)",
        status="planned",
        solvers=("thermostruct2d.step_thermal", "lumped coolant legs"),
        note=("Needs a registry entry with a real boundary response and a runner for "
              "a field coupled to lumped legs (plan scope B). No full-domain reference: "
              "the loop is lumped."),
        sources=("atlas/cases/cooling_loop.py",),
    ),
    Family(
        id="wing-fsi",
        label="Flexible wing in two-way flow (CS-12)",
        status="planned",
        solvers=("FSIRollout fluid windows", "ThermoStruct2D.solve_mechanical"),
        note=("The case's resolution is module constants; a workbench version needs "
              "them as parameters (Tier 89 built a half-resolution copy by patching "
              "them). Plan scope B."),
        sources=("atlas/cases/wing_fsi.py",),
    ),
    Family(
        id="learned-poseidon",
        label="Learned windows: Poseidon-T (frozen)",
        status="planned",
        solvers=("adapters.FrozenFluidExpert",),
        note=("Uniform 128 x 128 windows at one native step only, and CC-BY-NC-4.0 "
              "weights (research use). Plan scope D."),
        sources=("atlas/cases/poseidon.py",),
    ),
)


def family(fid: str) -> Family:
    for f in FAMILIES:
        if f.id == fid:
            return f
    raise KeyError(fid)


def available_ids() -> list[str]:
    return [f.id for f in FAMILIES if f.status == "ready-to-wire"]


__all__ = ["Family", "FAMILIES", "family", "available_ids"]

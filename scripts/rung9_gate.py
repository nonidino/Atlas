"""Rung 9's gate, re-read against the graph that exists (Tier 47, the brief's Task 3).

[[f1-pathmap-and-end-goal]] section 3 gives rung 9 -- the full-vehicle graph --
the gate *"R closes; composition error sub-linear in N"*.  Tier 46 built the
graph: `front_wing`'s tiling and both circuits, joined, eighteen agents across
five families (`atlas/cases/integration_union.py`).  This asks whether each
clause has a subject on that graph, measures what decides it, and records the
numbers the proposed replacement is held to.

Six measurements, each with the control that makes it evidence:

  1. **N on the vehicle graph.**  Is "composition error as a function of N" a
     function on this graph at all?  From Tier 46's own record of every union.
  2. **What R(t) needs, against what the records declare**: storage,
     dissipation, the fluid domain's physical boundaries, and the units of the
     clocks and lengths the three subsystems are written in.
  3. **J2's receiving subsystem's own balance** -- the block's first law with
     the mount term in it and without it: CS-9 section 6's rule, the one CS-12
     closed its gate on.  Control: the unjoined block, with and without its own
     declared source.
  4. **J3's power path**: the power the flow gives up through the rotor's
     upstream face against the power the shaft delivers -- in the rotor's native
     geometry (the control), in the host, and with the disk re-sized for the host.
  5. **What a host-sized rotor costs the powertrain**: `CircuitSolve` at the
     rotor's native width (the control, which must reproduce Tier 46's operating
     point) and at its host width.
  6. **J1 carries no power**, so its check is the coolant loop's response to
     the air through the core, against the declared radiator (the control, which
     must be identical at the build state).

Writes ``out/rung9/rung9.json`` after every section.  numpy and the build
repo's closed forms; the fluid is read off ``out/w141/settled.npz`` and never
marched here.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections                                                     # noqa: E402
import dataclasses                                                     # noqa: E402
import io                                                              # noqa: E402
import json                                                            # noqa: E402
import sys                                                             # noqa: E402
import time                                                            # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np                                                     # noqa: E402

from atlas.capability import ExpertCapabilities                        # noqa: E402
from atlas.graph import CaseGraph                                      # noqa: E402
from atlas.cases import cooling_loop as CL                             # noqa: E402
from atlas.cases import front_wing as FW                               # noqa: E402
from atlas.cases import ground_effect as GE                            # noqa: E402
from atlas.cases import integration_union as IU                        # noqa: E402
from atlas.cases import powertrain as PT                               # noqa: E402
from atlas.cases import wake_array as WA                               # noqa: E402
from atlas.cases import wing_fsi as W                                  # noqa: E402

OUT = os.path.join(HERE, "out", "rung9")
W172 = os.path.join(HERE, "out", "w172", "w172.json")
FACES = ("xlo", "xhi", "ylo", "yhi")


def _retry(fn, attempts=40, pause=0.25):
    for k in range(attempts):
        try:
            return fn()
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(pause)


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 64 else {"shape": list(x.shape),
                                                "norm": float(np.linalg.norm(x))}
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def persist(res):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "rung9.json")
    tmp = path + ".tmp"

    def write():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(clean(res), fh, indent=1)

    _retry(write)
    _retry(lambda: os.replace(tmp, path))
    return path


def settled(name="w141"):
    d = np.load(os.path.join(HERE, "out", name, "settled.npz"))
    return d["u"], d["v"]


# ===========================================================================
# 1. N on the vehicle graph
# ===========================================================================


def n_on_the_vehicle() -> dict:
    """Every union Tier 46 built, keyed by N: which physics sits at each N."""
    with open(W172, encoding="utf-8") as fh:
        w172 = json.load(fh)
    rows = []
    for label, rec in w172["unions"].items():
        c = rec["count"]
        rows.append({"label": label, "N": c["K"], "families": c["families"],
                     "subsystems": rec["subsystems"], "joins": rec["joins"],
                     "ports": c["ports"], "seams": c["seams"]})
    disjoint = [r for r in rows if not r["joins"]]
    by_n = collections.defaultdict(list)
    for r in disjoint:
        by_n[r["N"]].append(r)
    return {
        "unions": rows,
        "disjoint_by_N": {str(n): [{"subsystems": r["subsystems"],
                                    "families": r["families"]} for r in rs]
                          for n, rs in sorted(by_n.items())},
        "N_values": sorted(by_n),
        "N_with_more_than_one_graph": sorted(n for n, rs in by_n.items() if len(rs) > 1),
        "families_at_N13": sorted(r["families"] for r in by_n.get(13, [])),
        "physics_added_with_every_increase_in_N": all(
            len(set(map(tuple, (r["subsystems"] for r in by_n[n])))) == len(by_n[n])
            for n in by_n),
    }


# ===========================================================================
# 2. what R(t) needs, against what the records declare
# ===========================================================================


def dissipation_evidence(ex) -> tuple[bool, str]:
    """Whether an expert dissipates energy, read off its OWN constants."""
    if isinstance(ex, IU.CoreRadiator):
        return True, (f"pressure drop 1/2 K u^2 with K = {ex.k_core:g}, and heat "
                      f"rejected to ambient at UA = {ex.ua:g} W/K")
    if isinstance(ex, CL.RadiatorLeg):
        return True, f"heat rejected to ambient, decay {ex.decay:.6f} per pass"
    if isinstance(ex, (CL.LineLeg, CL.PumpLeg)):
        return (ex.ua > 0.0), f"heat lost to ambient at UA = {ex.ua:g} W/K"
    if isinstance(ex, CL.CoolantLeg):
        return False, "picks up the wall heat and loses none"
    if isinstance(ex, PT._Element):
        return (ex.resistance > 0.0), f"I^2 R with R = {ex.resistance:g}"
    if isinstance(ex, WA.RotorDisk):
        return False, "local-induction form: P = T U_d, all of it to the shaft"
    if isinstance(ex, CL.BlockAgent):
        return False, "conduction moves heat and destroys no energy"
    if isinstance(ex, W.FSIFlowWindow):
        return (ex.nu > 0.0), f"viscous, nu = {ex.nu:g}"
    if isinstance(ex, W.WingStructure):
        return False, "quasi-static elasticity stores and returns"
    if isinstance(ex, GE.Suspension):
        return False, "a spring stores and returns"
    return False, type(ex).__name__


def r_prerequisites(g: CaseGraph, experts: dict) -> dict:
    cap_fields = [f.name for f in dataclasses.fields(ExpertCapabilities)]
    graph_fields = [f.name for f in dataclasses.fields(CaseGraph)]
    agents = []
    for a in g.agents:
        caps = a.capabilities
        dis, why = dissipation_evidence(experts.get(a.agent_id))
        agents.append({"agent": a.agent_id, "family": caps.governing_family,
                       "storage_declared": caps.storage is not None,
                       "dissipative_by_its_own_constants": dis, "why": why,
                       "dt_native": caps.dt_native, "L_native": caps.L_native})
    windows = {n: experts[n] for n in W.DEFAULT_TILING.names}
    boundary = {n: sorted(set(FACES) - set(ex.shared_faces)) for n, ex in windows.items()}
    clocks = collections.defaultdict(set)
    parts = {"front_wing": set(W.DEFAULT_TILING.names) | {"STRUCT", "SUSP"},
             "cooling_loop": {"BLOCK", "PASS", "HOT", "RAD", "COLD"},
             "powertrain": {"ROTOR", "MGU", "BUS", "BATT", "INV"}}
    for a in g.agents:
        sub = next(s for s, ids in parts.items() if a.agent_id in ids)
        clocks[sub].add(a.capabilities.dt_native)
    return {
        "capability_fields_that_could_carry_dissipation": [f for f in cap_fields
                                                           if "dissip" in f.lower()],
        "graph_fields_that_could_carry_dissipation": [f for f in graph_fields
                                                      if "dissip" in f.lower()],
        "capability_fields_that_could_carry_a_unit": [
            f for f in cap_fields if "unit" in f.lower() or f.lower().endswith("_si")],
        "agents": agents,
        "n_storage_declared": sum(1 for r in agents if r["storage_declared"]),
        "n_dissipative_by_their_own_constants": sum(
            1 for r in agents if r["dissipative_by_its_own_constants"]),
        "open_ports": [list(p) for p in g.open_ports()],
        "window_faces_that_are_domain_boundaries": boundary,
        "n_window_faces_that_are_domain_boundaries": sum(len(v) for v in boundary.values()),
        "clocks_by_subsystem": {k: sorted(v) for k, v in clocks.items()},
        "clock_units_as_the_modules_state_them": {
            "front_wing": "convective: the tiling's length unit (64 cells, DX = 1/64) "
                          "over U_inf = 1",
            "powertrain": "convective: wake_array's rotor diameter D = 1 over "
                          "U_inf = 1 (disk.py: D = U_inf = rho = 1)",
            "cooling_loop": "seconds (MACRO_DT = 0.05 s)"},
    }


# ===========================================================================
# 3. J2 -- the receiving subsystem's own balance, with and without the term
# ===========================================================================


def _with_and_without(bal: dict, source_key: str) -> dict:
    scale = max(abs(bal["q_outer"]), abs(bal["q_wall"]), 1.0e-30)
    without = abs(bal["stored_rate"] - (0.0 - bal["q_wall"])) / scale
    return {**bal, "source_term": source_key,
            "relative_without_the_source_term": without,
            "factor_without_over_with": without / max(bal["relative"], 1.0e-300)}


def j2_balance(info_alone: dict, info_joined: dict) -> dict:
    tc = np.full(CL.N_SEAM, CL.T_COOLANT_0)
    return {
        "control: BlockAgent, its declared source": _with_and_without(
            CL.BlockAgent().energy_balance(tc), "t_source behind H_OUTER"),
        "J2 without J3: mount at the reference heat": _with_and_without(
            IU.MountedBlock(q_machine=info_alone["q_machine"]).energy_balance(tc),
            "the machine's casing behind h_mount"),
        "J2 with J3: mount at the joined operating point": _with_and_without(
            IU.MountedBlock(q_machine=info_joined["q_machine"]).energy_balance(tc),
            "the machine's casing behind h_mount"),
        "q_machine_W": {"without_J3": info_alone["q_machine"],
                        "with_J3": info_joined["q_machine"]},
    }


# ===========================================================================
# 4. J3 -- the power the flow gives up against the power the shaft delivers
# ===========================================================================


def _disk_module():
    rotor = WA.RotorDisk(agent_id="PROBE", u_ref=np.ones(WA.ROTOR_CELLS))
    return rotor._mod, rotor.a


def _face_accounting(u_up: np.ndarray, ds: float, width: float,
                     u_down: np.ndarray | None = None) -> dict:
    mod, a = _disk_module()
    disk = mod.ActuatorDisk(a=a, area=width)
    st = disk(float(np.mean(u_up)))
    #: the traction RotorDisk returns, per unit area: 1/2 rho C_T' u^2 -- no width
    tau_up = 0.5 * disk.rho * mod.c_t_prime(st.a) * u_up ** 2
    thrust_face = float(np.sum(tau_up) * ds)
    power_face = float(np.sum(tau_up * u_up) * ds)
    nonuni = float(np.mean(u_up ** 3) / np.mean(u_up) ** 3)
    row = {
        "cells": int(u_up.size), "cell_measure": ds, "face_length": ds * u_up.size,
        "disk_width": width, "u_disk": st.u_disk,
        "thrust_through_the_face": thrust_face, "thrust_of_the_disk": st.thrust,
        "power_through_the_upstream_face": power_face,
        "shaft_power": st.power, "omega": st.omega, "torque": st.torque,
        "power_ratio_face_over_shaft": power_face / st.power,
        "thrust_ratio_face_over_disk": thrust_face / st.thrust,
        "inflow_nonuniformity_mean_u3_over_mean_u_cubed": nonuni,
        "power_ratio_over_nonuniformity": power_face / st.power / nonuni,
        #: what the two declared faces carry at the rotor's own base, where both
        #: faces are linearized: +tau on the upstream face, -tau on the other
        "declared_two_face_net_power_at_the_rotors_base":
            float(np.sum(tau_up * u_up - tau_up * u_up) * ds),
    }
    if u_down is not None:
        tau_down = 0.5 * disk.rho * mod.c_t_prime(st.a) * u_down ** 2
        row["declared_two_face_net_power_at_the_host_rings"] = float(
            np.sum(tau_up * u_up - tau_down * u_down) * ds)
    return row


def j3_power(u: np.ndarray) -> dict:
    t = W.DEFAULT_TILING
    k = t.names.index(IU.ROTOR_SITE.left)
    ox, oy = t.offsets[k]
    u_up = IU.ring_velocity(u, IU.ROTOR_SITE)
    u_down = np.asarray(u, dtype=float)[oy + IU.ROTOR_SITE.cells, ox + t.wx - 1].copy()
    ones = np.ones(WA.ROTOR_CELLS)
    host_width = WA.ROTOR_CELLS * W.DX
    return {
        "control: native geometry (wake_array), uniform inflow": _face_accounting(
            ones, WA.DX, 1.0),
        "host geometry, uniform inflow, the disk as powertrain declares it": _face_accounting(
            ones, W.DX, 1.0),
        "host geometry, the settled inflow, the disk as powertrain declares it": _face_accounting(
            u_up, W.DX, 1.0, u_down),
        "host geometry, the settled inflow, the disk re-sized to its host face": _face_accounting(
            u_up, W.DX, host_width, u_down),
        "host_face_length": host_width,
        "declared_disk_width": 1.0,
        "circuit_solve_builds_its_disk_at_the_default_width": _circuit_solve_width_is_hard_coded(),
    }


def _circuit_solve_width_is_hard_coded() -> bool:
    import inspect

    src = inspect.getsource(PT.CircuitSolve.torque_at)
    return "ActuatorDisk(a=float(a))" in src and "area" not in src


# ===========================================================================
# 5. what a host-sized rotor costs the powertrain
# ===========================================================================


class SizedCircuitSolve(PT.CircuitSolve):
    """`powertrain.CircuitSolve` with the disk's swept width declared.

    The parent builds `ActuatorDisk(a=a)` at the donor's default width, so its
    operating point is the operating point of a rotor one diameter wide wherever
    the rotor is placed.  This overrides exactly the two reads of the disk and
    nothing else, so at ``width = 1`` it must reproduce the parent bitwise.
    """

    def __init__(self, width: float, **kw):
        super().__init__(**kw)
        self.width = float(width)

    def torque_at(self, a: float) -> float:
        disk = self.rotor._mod.ActuatorDisk(a=float(a), area=self.width)
        return float(disk(self.u_ref).torque)

    def omega(self) -> float:
        disk = self.rotor._mod.ActuatorDisk(a=self.rotor.a, area=self.width)
        return float(disk(self.u_ref).omega)


def sized_operating_points(u: np.ndarray) -> dict:
    u_up = IU.ring_velocity(u, IU.ROTOR_SITE)
    u_mean = float(np.mean(u_up))
    out = {}
    parent = PT.CircuitSolve(rotor=WA.RotorDisk(agent_id="ROTOR", u_ref=u_up),
                             elements=PT.make_elements(), u_ref=u_mean).solve()
    out["parent CircuitSolve (Tier 46's J3)"] = parent.as_dict()
    for label, width in (("width 1, the declared rotor", 1.0),
                         ("width 0.5, the host face", WA.ROTOR_CELLS * W.DX)):
        r = SizedCircuitSolve(width, rotor=WA.RotorDisk(agent_id="ROTOR", u_ref=u_up),
                              elements=PT.make_elements(), u_ref=u_mean).solve()
        d = r.as_dict()
        mod, _a = _disk_module()
        best = max(float(mod.ActuatorDisk(a=aa, area=width)(u_mean).torque)
                   for aa in np.linspace(PT.A_MIN, PT.A_MAX, 400))
        d["largest_torque_the_disk_can_deliver_in_its_clamp"] = best
        d["torque_the_machine_demands"] = r.torque_machine
        d["demand_over_largest_supply"] = r.torque_machine / max(best, 1.0e-300)
        out[label] = d
    out["width_1_reproduces_the_parent"] = all(
        out["width 1, the declared rotor"][k] == out["parent CircuitSolve (Tier 46's J3)"][k]
        for k in ("omega", "induction", "current", "p_mech"))
    return out


# ===========================================================================
# 6. J1 carries no power: the coolant loop's response to the air
# ===========================================================================


def j1_loop_response(u: np.ndarray) -> dict:
    u_air = IU.ring_velocity(u, IU.CORE_SITE)

    def run(core_scale: float | None):
        ls = CL.LoopSolve()
        if core_scale is not None:
            core = IU.CoreRadiator(u_air=u_air)
            core.u_air = core.u_air * core_scale
            ls.legs["RAD"] = core
        r = ls.solve()
        return {"t_return": r.t_return, "q_block": r.q_block,
                "q_rejected": dict(r.q_rejected), "loop_balance_residual": r.residual,
                "block_first_law_relative": r.energy_balance["relative"],
                "ua_rad": (ls.legs["RAD"].ua if isinstance(ls.legs["RAD"], IU.CoreRadiator)
                           else CL.UA_RAD)}

    declared = run(None)
    at_build = run(1.0)
    faster = run(1.1)
    ds = W.DX
    core_flow_power = float(np.sum(0.5 * IU.K_CORE * u_air ** 3) * ds)
    return {
        "control: the declared radiator": declared,
        "the core in the flow, at the build state": at_build,
        "the core in the flow, air +10%": faster,
        "identical_at_the_build_state": (declared["t_return"] == at_build["t_return"]
                                         and declared["q_rejected"] == at_build["q_rejected"]),
        "t_return_change_K_for_air_plus_10pct": faster["t_return"] - at_build["t_return"],
        "rad_rejection_change_W_for_air_plus_10pct":
            faster["q_rejected"]["RAD"] - at_build["q_rejected"]["RAD"],
        "flow_work_on_the_core_in_tiling_units_per_span": core_flow_power,
        "power_crossing_into_the_coolant_through_J1": 0.0,
    }


# ===========================================================================
# referents
# ===========================================================================


def referents(g: CaseGraph) -> dict:
    fam = {a.agent_id: a.capabilities.governing_family for a in g.agents}
    joins = {}
    for c in g.connections:
        if c.seam_id.split("_", 1)[0] in IU.JOINS:
            joins[c.seam_id] = {"families": sorted({fam[c.a[0]], fam[c.b[0]]}),
                                "same_family": fam[c.a[0]] == fam[c.b[0]]}
    return {
        "tiling_monolith_declared": bool(W.SINGLE_TILING.is_single),
        "front_wing_has_a_referent_rollout": hasattr(FW, "referent_rollout"),
        "cooling_loop_has_a_closed_form": hasattr(CL.LoopSolve, "_closed_form"),
        "powertrain_has_a_closed_form": hasattr(PT.CircuitSolve, "_closed_form"),
        "joining_seams": joins,
    }


def main() -> None:
    t0 = time.perf_counter()
    u, v = settled()
    res = {"what": "rung 9's gate, re-read against the joined union",
           "date": time.strftime("%Y-%m-%d"),
           "state": "out/w141/settled.npz; the fluid is read, never marched"}

    res["n_on_the_vehicle"] = n_on_the_vehicle()
    nv = res["n_on_the_vehicle"]
    print("1. N values", nv["N_values"], "with more than one graph",
          nv["N_with_more_than_one_graph"], "families at N=13", nv["families_at_N13"],
          flush=True)
    persist(res)

    g, meta = IU.build(u, v)
    _g2, meta_alone = IU.build(u, v, subsystems=("cooling_loop", "powertrain"),
                               joins=("J2",))
    res["r_prerequisites"] = r_prerequisites(g, meta["experts"])
    rp = res["r_prerequisites"]
    print("2. dissipation fields", rp["capability_fields_that_could_carry_dissipation"],
          rp["graph_fields_that_could_carry_dissipation"],
          "storage declared", rp["n_storage_declared"], "dissipative",
          rp["n_dissipative_by_their_own_constants"], "boundary faces",
          rp["n_window_faces_that_are_domain_boundaries"], "clocks",
          rp["clocks_by_subsystem"], flush=True)
    res["referents"] = referents(g)
    persist(res)

    res["j2_balance"] = j2_balance(meta_alone["info"], meta["info"])
    for k, row in res["j2_balance"].items():
        if isinstance(row, dict) and "relative" in row:
            print("3. %-50s with %.3e without %.3e factor %.3e" % (
                k, row["relative"], row["relative_without_the_source_term"],
                row["factor_without_over_with"]), flush=True)
    persist(res)

    res["j3_power"] = j3_power(u)
    for k, row in res["j3_power"].items():
        if isinstance(row, dict):
            print("4. %-70s P_face/P_shaft %.6f  /nonuni %.6f  net2face@base %.3e" % (
                k, row["power_ratio_face_over_shaft"], row["power_ratio_over_nonuniformity"],
                row["declared_two_face_net_power_at_the_rotors_base"]), flush=True)
    print("   CircuitSolve hard-codes the disk width:",
          res["j3_power"]["circuit_solve_builds_its_disk_at_the_default_width"], flush=True)
    persist(res)

    res["sized_operating_points"] = sized_operating_points(u)
    for k, row in res["sized_operating_points"].items():
        if isinstance(row, dict):
            print("5. %-40s omega %.4f a %.4f I %.4f mgu %s rotor %s demand/supply %s" % (
                k, row["omega"], row["induction"], row["current"], row["mgu_valid"],
                row["rotor_valid"], row.get("demand_over_largest_supply")), flush=True)
    print("   width 1 reproduces the parent:",
          res["sized_operating_points"]["width_1_reproduces_the_parent"], flush=True)
    persist(res)

    res["j1_loop_response"] = j1_loop_response(u)
    j1 = res["j1_loop_response"]
    print("6. identical at build", j1["identical_at_the_build_state"], "dT_return",
          j1["t_return_change_K_for_air_plus_10pct"], "dQ_rad",
          j1["rad_rejection_change_W_for_air_plus_10pct"], flush=True)

    res["elapsed_seconds"] = time.perf_counter() - t0
    path = persist(res)
    print("wrote", path, "in %.1f s" % res["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    main()

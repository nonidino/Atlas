"""HDF5 episode schema (spec §3.6, extended by decisions D4/D5).

    episode_XXXX.h5
      /meta               sweep point, the resolved timing, the D1-D5 decisions,
                          git sha, config fingerprint, solver versions
      /refs/{agent}       rho_ref, u_ref, T_ref, L_ref, p_ref     # frozen per episode
      /cond/{agent}       [T, n_cond]
      /fields/{agent}     [T, C, Nz, Ny]   nondimensionalized
      /iface/{edge}       [T, C_edge, N_iface]  fluxes + traces, SOURCE side
      /iface_dst/{edge}   [T, C_edge, N_iface]  destination side, conservation edges
      /iface_exch/{edge}  [T]   integrated mass flux of the payload the destination
                                is running on -- the lag term of gate G5, isolated
      /iface_t_exch/{edge}[T]  when the payload the destination is using was computed
      /times              [T]   simulation time of each snapshot
      /rigid              [T, 7]
      /loads              [T, 4]

**`/iface` is not optional.** The interface flux fields are the supervision
signal for the edge layer and for the Phase-3 conservation constraint; a corpus
that saves only interiors cannot train or validate them, and regenerating is
expensive. `write_episode` therefore refuses to write an episode with an empty
`/iface`, rather than producing a file that looks fine until Phase 3.

**Every geometrically real interface is recorded; only a subset is declared**
(decision D4). Each `/iface/{edge}` dataset carries `declared`, `dt_exch` and
`coupled` attributes, so the model can select the declared seven while ablation
A4 still has the dense graph available.

**`/iface_t_exch` is what makes a held constant distinguishable from a live
signal** (decision D5). Coupling exchanges at dt_exch = max(dt_p, dt_q), so with
dt_snap = 1 ms the shell-touching edges are stale for up to 50 snapshots. An
interface loss trained on those without masking is fitting a staircase.

Interface channels, per edge and per interface station:
    (mass_flux, mom_z_flux, mom_y_flux, energy_flux, heat_flux, p_trace, T_trace, length)
On a solid-side record (the shell aft face) mass and energy flux are structurally
zero and the interface is carried by the momentum-flux (traction) and heat-flux
channels.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

SCHEMA_VERSION = 2
IFACE_CHANNELS = ("mass_flux", "mom_z_flux", "mom_y_flux", "energy_flux",
                  "heat_flux", "p_trace", "T_trace", "seg_length")
RIGID_FIELDS = ("x", "y", "theta", "vx", "vy", "omega", "m")
LOAD_FIELDS = ("Fz_thrust", "Fy_thrust", "Fz_aero", "Fy_aero")


def write_episode(path: str | Path, meta: dict, refs: dict, cond: dict,
                  fields: dict, iface: dict, rigid: np.ndarray,
                  loads: np.ndarray, iface_t_exch: dict | None = None,
                  iface_dst: dict | None = None, iface_exch: dict | None = None,
                  iface_meta: dict | None = None,
                  times: np.ndarray | None = None) -> Path:
    path = Path(path)
    if not iface:
        raise ValueError(
            "refusing to write an episode with no /iface group: interface fluxes are "
            "the supervision signal for the edge and conservation layers (spec §3.4)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("meta")
        g.attrs["schema_version"] = SCHEMA_VERSION
        for k, v in meta.items():
            g.attrs[k] = v
        gr = f.create_group("refs")
        for a, r in refs.items():
            sub = gr.create_group(a)
            for k, v in r.items():
                sub.attrs[k] = float(v)
        for name, src in (("cond", cond), ("fields", fields), ("iface", iface),
                          ("iface_dst", iface_dst or {}),
                          ("iface_exch", iface_exch or {}),
                          ("iface_t_exch", iface_t_exch or {})):
            grp = f.create_group(name)
            for k, v in src.items():
                d = grp.create_dataset(k, data=np.asarray(v, dtype=np.float32),
                                       compression="gzip", compression_opts=4)
                if name == "iface" and iface_meta and k in iface_meta:
                    for ak, av in iface_meta[k].items():
                        d.attrs[ak] = av
        if times is not None:
            f.create_dataset("times", data=np.asarray(times, dtype=np.float64))
        f.create_dataset("rigid", data=np.asarray(rigid, dtype=np.float32))
        f.create_dataset("loads", data=np.asarray(loads, dtype=np.float32))
        f.attrs["iface_channels"] = list(IFACE_CHANNELS)
        f.attrs["rigid_fields"] = list(RIGID_FIELDS)
    return path


def read_episode(path: str | Path) -> dict:
    with h5py.File(Path(path), "r") as f:
        out = {
            "meta": dict(f["meta"].attrs),
            "refs": {a: dict(f["refs"][a].attrs) for a in f["refs"]},
            "cond": {k: f["cond"][k][...] for k in f["cond"]},
            "fields": {k: f["fields"][k][...] for k in f["fields"]},
            "iface": {k: f["iface"][k][...] for k in f["iface"]},
            "iface_attrs": {k: dict(f["iface"][k].attrs) for k in f["iface"]},
            "rigid": f["rigid"][...],
            "loads": f["loads"][...],
            "iface_channels": [str(s) for s in f.attrs["iface_channels"]],
        }
        for name in ("iface_dst", "iface_exch", "iface_t_exch"):
            out[name] = {k: f[name][k][...] for k in f[name]} if name in f else {}
        out["times"] = f["times"][...] if "times" in f else None
    return out


def declared_edges(d: dict) -> list[str]:
    """The subset of recorded interfaces the model is allowed to see."""
    return sorted(k for k, a in d["iface_attrs"].items()
                  if bool(a.get("declared", True)))


def validate_episode(path: str | Path, cfg, n_snapshots: int | None = None) -> list[str]:
    """M2 per-episode checks. Returns a list of problems; empty means valid."""
    d = read_episode(path)
    problems = []
    if str(d["meta"].get("kind", "coupled")) == "solo":
        return _validate_solo(d, n_snapshots)
    agents = set(cfg.agent_ids)
    edges = {f"{e.src}-{e.dst}" for e in cfg.edge_list}
    recorded = {f"{r.src}-{r.dst}" for r in getattr(cfg, "recorded_interfaces", ())}

    missing = agents - set(d["fields"])
    if missing:
        problems.append(f"missing agents in /fields: {sorted(missing)}")
    missing = edges - set(d["iface"])
    if missing:
        problems.append(f"missing edges in /iface: {sorted(missing)}")
    missing = recorded - set(d["iface"])
    if missing:
        problems.append(f"missing recorded-but-undeclared interfaces (D4): {sorted(missing)}")

    T = d["rigid"].shape[0]
    if n_snapshots is not None and T != n_snapshots:
        problems.append(f"{T} snapshots, expected {n_snapshots}")
    for a, v in d["fields"].items():
        if v.shape[0] != T:
            problems.append(f"fields/{a} has {v.shape[0]} snapshots, rigid has {T}")
        if not np.isfinite(v).all():
            problems.append(f"fields/{a} contains non-finite values")
    for k, v in d["iface"].items():
        if not np.isfinite(v).all():
            problems.append(f"iface/{k} contains non-finite values")

    if int(d["meta"].get("schema_version", 1)) >= 2:
        problems += _validate_timing(d, T)
    return problems


def _validate_solo(d: dict, n_snapshots: int | None) -> list[str]:
    """T2 runs carry only the agents in their group, so the coupled completeness
    checks do not apply. What still must hold is that the file is finite, its
    clock is uniform, and it is unmistakably labelled: a solo run must never be
    mistaken for a coupled episode, because it has no trajectory and no
    cross-group interfaces."""
    problems = []
    T = d["times"].shape[0] if d["times"] is not None else 0
    if not T:
        return ["solo episode without /times"]
    if n_snapshots is not None and T != n_snapshots:
        problems.append(f"{T} snapshots, expected {n_snapshots}")
    if not d["fields"]:
        problems.append("solo episode with no /fields")
    for a, v in d["fields"].items():
        if v.shape[0] != T:
            problems.append(f"fields/{a} has {v.shape[0]} snapshots, times has {T}")
        if not np.isfinite(v).all():
            problems.append(f"fields/{a} contains non-finite values")
    for k, v in d["iface"].items():
        if not np.isfinite(v).all():
            problems.append(f"iface/{k} contains non-finite values")
    if T > 1:
        step = np.diff(d["times"])
        if not np.allclose(step, float(d["meta"]["dt_snap"]), rtol=1e-6, atol=1e-12):
            problems.append("snapshot spacing is not dt_snap")
    if bool(d["meta"].get("full_fidelity", False)):
        problems.append("a solo run may not be labelled full_fidelity")
    return problems


def _validate_timing(d: dict, T: int) -> list[str]:
    """D1/D5 checks: the snapshot clock and the exchange stamps must agree with the
    timing recorded in /meta, or the hold-last policy is unauditable after the fact."""
    problems = []
    t = d["times"]
    meta = d["meta"]
    if t is None:
        return ["schema v2 episode without /times"]
    dt_snap = float(meta["dt_snap"])
    if T > 1:
        step = np.diff(t)
        if not np.allclose(step, dt_snap, rtol=1e-6, atol=1e-12):
            problems.append(f"snapshot spacing is not dt_snap={dt_snap}")
    for k, ts in d["iface_t_exch"].items():
        if ts.shape[0] != T:
            problems.append(f"iface_t_exch/{k} has {ts.shape[0]} entries, expected {T}")
            continue
        if np.any(np.diff(ts) < -1e-9):
            problems.append(f"iface_t_exch/{k} goes backwards in time")
        if np.any(ts > t + 1e-6):
            problems.append(f"iface_t_exch/{k} is stamped in the future")
        lag = float(np.max(t - ts))
        dt_exch = float(d["iface_attrs"].get(k, {}).get("dt_exch", dt_snap))
        if lag > dt_exch + 1e-6:
            problems.append(
                f"iface_t_exch/{k} lags {lag:.4g} s, more than its cadence {dt_exch:.4g} s")
    return problems


def mass_budget_residual(path: str | Path) -> float:
    """|dm_system + integral(mdot_exit dt)| / m0 -- the M2 global conservation check.

    Uses the rigid-body mass history against the mass flux recorded on the e-f
    interface, so it grades the two independent records against each other. The
    integration step is the SNAPSHOT spacing, not dt_macro: under route B' those
    differ by a factor of 50, and using the wrong one silently scales the residual."""
    d = read_episode(path)
    rigid = d["rigid"]
    m0, mf = float(rigid[0, 6]), float(rigid[-1, 6])
    ch = d["iface_channels"].index("mass_flux")
    ln = d["iface_channels"].index("seg_length")
    ef = d["iface"]["e-f"]                       # [T, C, N]
    mdot = (ef[:, ch, :] * ef[:, ln, :]).sum(-1)
    dt = float(d["meta"].get("dt_snap", d["meta"]["dt_macro"]))
    expelled = float(np.trapezoid(mdot, dx=dt)) if hasattr(np, "trapezoid") else \
        float(np.trapz(mdot, dx=dt))
    return abs((mf - m0) + expelled) / max(m0, 1e-9)


def interface_flux_consistency(path: str | Path, skip_frac: float = 0.25) -> dict[str, float]:
    """Gate G5: the relative mass-flux mismatch across each conservation-typed
    interface, source side against destination side.

    The generator couples loosely, so each side sees the other lagged by one
    exchange. Phase 3 then imposes HARD flux matching at those same interfaces.
    If this lag error exceeds the constraint's tolerance, the constraint is
    fighting its own supervision -- and the symptom (a conservation loss that
    plateaus at a stubborn nonzero floor) gives no hint that the data, not the
    model, is the cause. The global mass budget cannot see this: it is a
    whole-system integral, and per-interface errors cancel inside it.

    `skip_frac` drops the leading fraction of the episode. Every episode starts
    from a uniform state, so the first snapshots contain an 8x jump in exit mass
    flux that no coupling scheme can track and that would otherwise dominate the
    maximum -- measured 0.880 over the whole episode against 0.465 on the tail."""
    d = read_episode(path)
    if not d["iface_dst"]:
        raise ValueError("no /iface_dst records: regenerate with schema v2, or G5 "
                         "cannot be evaluated (corpus completion plan §4)")
    ch = d["iface_channels"].index("mass_flux")
    ln = d["iface_channels"].index("seg_length")
    out = {}
    for k, dst in d["iface_dst"].items():
        src = d["iface"][k]
        sl = _tail(src.shape[0], skip_frac)
        a = (src[sl, ch, :] * src[sl, ln, :]).sum(-1)
        b = (dst[sl, ch, :] * dst[sl, ln, :]).sum(-1)
        scale = np.maximum(np.abs(a), 1e-12)
        out[k] = float(np.max(np.abs(a - b) / scale))
    return out


def _tail(n: int, skip_frac: float) -> slice:
    return slice(min(int(n * skip_frac), max(n - 1, 0)), None)


def interface_lag_error(path: str | Path, skip_frac: float = 0.25) -> dict[str, float]:
    """The coupling-LAG component of G5, alone.

    `interface_flux_consistency` compares two INTERIOR cells either side of the
    interface, so its value mixes three sources: the exchange lag, the half-cell
    offset across a strong gradient, and the transverse grid mismatch. Only the
    lag is a property of the coupling scheme -- the other two are discretization
    and would survive a perfectly synchronous solver. This function compares the
    live source-side integral against the payload the destination is actually
    running on, which leaves the lag and nothing else. It is the number to hold
    against the Phase-3 conservation tolerance.

    Measured on a coarse probe episode, tail of the run: **0.000 with the
    conservative interface remap, 0.302 with pointwise sampling** -- the number
    this function exists to keep at zero. Over the whole episode both read 0.880,
    which is the start-up transient, not the coupling; hence `skip_frac`."""
    d = read_episode(path)
    if not d.get("iface_exch"):
        raise ValueError("no /iface_exch records: regenerate with schema v2")
    ch = d["iface_channels"].index("mass_flux")
    ln = d["iface_channels"].index("seg_length")
    out = {}
    for k, held in d["iface_exch"].items():
        src = d["iface"][k]
        sl = _tail(src.shape[0], skip_frac)
        held = held[sl]
        live = (src[sl, ch, :] * src[sl, ln, :]).sum(-1)
        scale = np.maximum(np.abs(live), 1e-12)
        out[k] = float(np.max(np.abs(live - held) / scale))
    return out


__all__ = [
    "write_episode", "read_episode", "validate_episode", "mass_budget_residual",
    "interface_flux_consistency", "interface_lag_error", "declared_edges",
    "IFACE_CHANNELS", "RIGID_FIELDS", "LOAD_FIELDS", "SCHEMA_VERSION",
]

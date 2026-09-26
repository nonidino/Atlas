"""The rocket's seams, pinned: geometry that meets, and a coupler that wires by
position in each block's own gas model.

Written 2026-09-23 with the fixes, after an audit of the first marched episode
(research vault: `scripts/w321_seam_audit.py`, page `rocket-episode-seam-audit`).
Each test names the defect it closes and, where a control is cheap, shows the old
behaviour is really gone rather than merely renamed.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from atlas.config import load_config
from atlas.data import generate as gen
from atlas.data import sweep
from atlas.geometry.contours import nozzle_inner, shell_outer
from atlas.solvers import atmosphere, thermo, trajectory
from atlas.solvers.compressible2d import BC, NG, Compressible2D, GasConfig
from atlas.solvers.grid import (Block, axial_breaks, build_blocks, hole_band,
                                segment_counts, shell_faces)
from atlas.solvers.thermostruct2d import ShellMesh, SolidMaterial, ThermoStruct2D


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def episode(cfg):
    """Built, not marched: the coupler's wiring can be read without a solve."""
    pt = sweep.corner_cases()[0]
    return gen.CoupledEpisode(gen.EpisodeSpec(point=pt, n_macro=2, dt_macro=1e-5,
                                              coarsen=4), cfg)


# ------------------------------------------------------------------ geometry


def test_segment_counts_are_proportional_and_total(cfg):
    c = segment_counts([4.0, 0.12, 0.18, 0.10, 0.30], 232)
    assert c.sum() == 232 and (c >= 1).all()
    assert list(c) == [197, 6, 9, 5, 15]
    assert list(segment_counts([4.0, 0.12, 0.18, 0.10, 0.30], 58)) == [49, 2, 2, 1, 4]


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_every_block_has_a_node_at_every_break_it_spans(cfg, k):
    """A block with no node at a kink of h_in draws that wall as a chord across
    it. The airframe and `d` had a node at neither kink at ANY resolution."""
    breaks = axial_breaks(cfg)
    for a in cfg.agents:
        if a.grid[0] % k or a.grid[1] % k:
            continue
        z = build_blocks(a, cfg, k)[0].nodes[:, 0, 0]
        for b in breaks:
            if a.z[0] < b < a.z[1]:
                assert np.abs(z - b).min() < 1e-12, (a.id, k, b)


def test_blocks_that_already_met_the_rule_are_bit_identical(cfg):
    """`b` puts node 72 on the converging-section start and has since Phase 0;
    rebuilding the rule changed nothing that already obeyed it."""
    for aid in "abefg":
        z = build_blocks(cfg.agent(aid), cfg)[0].nodes[:, 0, 0]
        a = cfg.agent(aid)
        assert np.array_equal(z, np.linspace(a.z[0], a.z[1], a.grid[0] + 1)), aid


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_the_three_polylines_at_the_nozzle_are_one_curve(cfg, k):
    """Gas wall, shell inner face, shell outer face, and `d`'s inner boundary
    must trace the same contour. Measured before the fix at coarsen 4: a 14.7 mm
    gap between gas and shell against an 8 mm skin, 17.4 mm between shell and
    `d`, 12.4 mm overlaps; 2.7 mm / 3.5 mm at full resolution."""
    geo = cfg.geometry
    blk = {a.id: build_blocks(a, cfg, k) for a in cfg.agents}
    zz = np.linspace(0.0, geo.z_exit, 7001)
    gz = np.concatenate([blk["a"][0].nodes[:, -1, 0], blk["b"][0].nodes[1:, -1, 0],
                         blk["e"][0].nodes[1:, -1, 0]])
    gy = np.concatenate([blk["a"][0].nodes[:, -1, 1], blk["b"][0].nodes[1:, -1, 1],
                         blk["e"][0].nodes[1:, -1, 1]])
    c1, d1 = blk["c"][1], blk["d"][1]
    gas = np.interp(zz, gz, gy)
    s_in = np.interp(zz, c1.nodes[:, 0, 0], c1.nodes[:, 0, 1])
    s_out = np.interp(zz, c1.nodes[:, -1, 0], c1.nodes[:, -1, 1])
    d_in = np.interp(zz, d1.nodes[:, 0, 0], d1.nodes[:, 0, 1])
    assert np.abs(gas - nozzle_inner(zz, geo)).max() < 1e-12
    assert np.abs(s_in - gas).max() < 1e-12
    assert np.abs(s_out - shell_outer(zz, geo)).max() < 1e-12
    assert np.abs(d_in - s_out).max() < 1e-12


@pytest.mark.parametrize("n", [96, 48, 24, 12])
def test_the_plume_hole_is_exactly_f_at_every_resolution(cfg, n):
    """The hole's edges are node lines at |y| = plume_halfwidth and the band is
    centred. It was [-0.59375, 0.59375] at 96 cells (blanked by centre) and
    [-0.500, 0.625] at coarsen 4 (the fine mask subsampled)."""
    geo = cfg.geometry
    y, j0, j1 = hole_band(n, cfg.agent("g").y_halfwidth, geo.plume_halfwidth)
    assert y.size == n + 1
    assert y[j0] == pytest.approx(-geo.plume_halfwidth, abs=1e-12)
    assert y[j1] == pytest.approx(geo.plume_halfwidth, abs=1e-12)
    assert j0 == n - j1, "symmetric"
    assert np.all(np.diff(y) > 0)


def test_g_blanks_exactly_the_band_and_f_fills_it(cfg):
    for k in (1, 4):
        g = build_blocks(cfg.agent("g"), cfg, k)[0]
        f = build_blocks(cfg.agent("f"), cfg, k)[0]
        js = np.flatnonzero(g.blanked[0])
        assert g.nodes[0, js[0], 1] == pytest.approx(f.nodes[0, 0, 1], abs=1e-12)
        assert g.nodes[0, js[-1] + 1, 1] == pytest.approx(f.nodes[0, -1, 1], abs=1e-12)
        assert np.array_equal(g.nodes[:, 0, 0], f.nodes[:, 0, 0]), "same z stations"


def test_full_resolution_hole_keeps_its_cells_so_the_token_budget_holds(cfg):
    """Cells 29..66 were the hole before and are the hole now, so the patch
    layout -- and the spec's 152 `g` tokens -- did not move."""
    g = build_blocks(cfg.agent("g"), cfg)[0]
    assert list(np.flatnonzero(g.blanked[0])) == list(range(29, 67))


def test_coarse_blocks_are_rebuilt_not_subsampled(cfg):
    """The wall-normal stretch is compounded (ratio**k), which reproduces every
    k-th fine node exactly; the axial breaks are re-allocated at the coarse count."""
    d = cfg.agent("d")
    fine, coarse = build_blocks(d, cfg, 1)[1], build_blocks(d, cfg, 4)[1]
    iz = [int(np.argmin(np.abs(fine.nodes[:, 0, 0] - z))) for z in coarse.nodes[:, 0, 0]]
    common = [i for i, z in zip(iz, coarse.nodes[:, 0, 0])
              if abs(fine.nodes[i, 0, 0] - z) < 1e-12]
    assert common, "some stations coincide"
    for i in common:
        j = int(np.argmin(np.abs(coarse.nodes[:, 0, 0] - fine.nodes[i, 0, 0])))
        np.testing.assert_allclose(coarse.nodes[j, :, 1], fine.nodes[i, ::4, 1], atol=1e-12)


def test_coverage_is_exact_in_the_engine_window(cfg):
    """Rasterised at 1 mm: every point in the engine window is claimed by exactly
    one block, apart from the declared unmodelled regions. The audit measured
    33.5 cm^2 unclaimed and 16.6 cm^2 claimed twice before the fix (coarsen 4)."""
    k = 4
    patches = [b for a in cfg.agents for b in build_blocks(a, cfg, k)]
    dx = 1e-3
    zs = np.arange(-0.1 + dx / 2, 1.0, dx)
    ys = np.arange(-0.2 + dx / 2, 0.2, dx)
    ZZ, YY = np.meshgrid(zs, ys, indexing="ij")
    cnt = np.zeros(ZZ.shape, np.int16)
    for blk in patches:
        n = blk.nodes
        q = np.stack([n[:-1, :-1], n[1:, :-1], n[1:, 1:], n[:-1, 1:]], axis=2)[~blk.blanked]
        for quad in q:
            i0, i1 = np.searchsorted(zs, quad[:, 0].min()), np.searchsorted(zs, quad[:, 0].max())
            j0, j1 = np.searchsorted(ys, quad[:, 1].min()), np.searchsorted(ys, quad[:, 1].max())
            if i1 <= i0 or j1 <= j0:
                continue
            PZ, PY = ZZ[i0:i1, j0:j1], YY[i0:i1, j0:j1]
            inside = np.ones(PZ.shape, bool)
            for m in range(4):
                (az, ay), (bz, by), (cz, cy) = quad[m], quad[(m + 1) % 4], quad[(m + 2) % 4]
                side = np.sign((bz - az) * (cy - ay) - (by - ay) * (cz - az))
                inside &= ((bz - az) * (PY - ay) - (by - ay) * (PZ - az)) * side >= 0
            cnt[i0:i1, j0:j1] += inside
    unm = np.zeros(ZZ.shape, bool)
    for r in cfg.unmodelled:
        unm |= (ZZ >= r.z[0]) & (ZZ <= r.z[1]) & (np.abs(YY) <= r.y_halfwidth)
    # a raster point that lands exactly on a shared edge counts twice; none of
    # these offsets put one there, so both counts are exact
    assert int(((cnt == 0) & ~unm).sum()) == 0
    assert int((cnt >= 2).sum()) == 0


# -------------------------------------------------------------------- shell


def test_shell_faces_read_the_orientation_off_the_geometry(cfg):
    lo, hi = build_blocks(cfg.agent("c"), cfg)
    assert shell_faces(lo.nodes) == (-1, 0), "the lower panel is mirrored"
    assert shell_faces(hi.nodes) == (0, -1)


def test_shellmesh_rejects_an_interior_face():
    nodes = np.stack(np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 0.1, 3),
                                 indexing="ij"), axis=-1)
    with pytest.raises(ValueError):
        ShellMesh(nodes, inner_j=1, outer_j=-1)


def _mirror_pair(cfg, oriented: bool):
    lo, hi = build_blocks(cfg.agent("c"), cfg, 4)
    out = []
    for b in (lo, hi):
        mesh = ShellMesh(b.nodes, *shell_faces(b.nodes)) if oriented else ShellMesh(b.nodes)
        ts = ThermoStruct2D(mesh, SolidMaterial())
        ni = mesh.shape[0]
        p_in = np.linspace(8.0e6, 2.0e6, ni)          # anything non-uniform
        u, _ = ts.solve_mechanical(np.full(mesh.n_nodes, 288.15), p_in, 2.5e3)
        out.append(u.reshape(b.nodes.shape[0], b.nodes.shape[1], 2))
    return out


def test_the_two_panels_respond_as_mirror_images(cfg):
    """Symmetric loads, symmetric response: the lower panel's uy is minus the
    upper's, station by station and layer by layer."""
    lo, hi = _mirror_pair(cfg, oriented=True)
    scale = np.abs(hi[..., 1]).max()
    np.testing.assert_allclose(lo[:, ::-1, 1], -hi[..., 1], atol=1e-9 * scale)
    np.testing.assert_allclose(lo[:, ::-1, 0], hi[..., 0], atol=1e-9 * scale)


def test_diagnosis_the_default_faces_load_the_lower_panel_inside_out(cfg):
    """The defect, kept: with the default inner_j = 0 on both panels, the lower
    panel takes the inner pressure on its OUTER skin and both panels bend the
    same way -- the W321 episode's uy agreed to 1.41e-4 between panels, by up
    to 4.17 m."""
    lo, hi = _mirror_pair(cfg, oriented=False)
    scale = np.abs(hi[..., 1]).max()
    assert np.abs(lo[:, ::-1, 1] - hi[..., 1]).max() < 1e-3 * scale      # same sign
    assert np.abs(lo[:, ::-1, 1] + hi[..., 1]).max() > 1.0 * scale        # not mirrored


# ---------------------------------------------------------------- W301 wall


class _Spy(Compressible2D):
    fix = True

    def _isothermal_wall_conduction(self, Fi, Fj, T, Qz, Qy, k):
        if self.fix:
            super()._isothermal_wall_conduction(Fi, Fj, T, Qz, Qy, k)
        self.Fj = Fj.copy()


def _wall_flux(T_wall: float, T_gas: float, fix: bool = True):
    nz, ny = 6, 6
    z, y = np.linspace(0, 0.06, nz + 1), np.linspace(0, 0.03, ny + 1)
    Z, Y = np.meshgrid(z, y, indexing="ij")
    blk = Block("w", np.stack([Z, Y], -1), np.zeros((nz, ny), bool))
    cfg = GasConfig(gamma=1.22, R=361.0)
    s = _Spy(blk, cfg, bcs={"jmin": BC("wall_noslip", {"T_wall": np.full(nz, T_wall)}),
                            "jmax": BC("wall_noslip"), "imin": BC("extrapolate"),
                            "imax": BC("extrapolate")})
    s.fix = fix
    p = 8.0e6
    W = np.zeros((nz, ny, 4))
    W[..., 0] = p / (cfg.R * T_gas)
    W[..., 3] = p                     # quiescent: no viscous work in the wall face's flux
    U = thermo.prim_to_cons(W, cfg.gamma)
    s._viscous_residual(s.ghost(U))
    mu = thermo.sutherland(T_gas, cfg.mu_ref, cfg.T_mu_ref, cfg.sutherland_S)
    k = mu * cfg.cp / cfg.Pr
    dn = 0.5 * blk.vol[:, 0] / blk.a_j[:, 0]
    return s.Fj[:, 0, 3], k * (T_gas - T_wall) / dn * blk.a_j[:, 0]


def test_W301_the_wall_face_conducts_against_the_wall_it_was_given():
    """A 3000 K gas against a 300 K wall: the ghost saturates at 20 K, and the
    face flux is now k (T_i - T_w)/dn exactly -- what `generate._wall_flux` hands
    the shell -- instead of whatever the clamped ghost implied."""
    got, want = _wall_flux(300.0, 3000.0)
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_W301_the_channel_now_transmits_below_saturation_and_did_not_before():
    a, _ = _wall_flux(300.0, 3000.0)
    b, _ = _wall_flux(600.0, 3000.0)
    assert np.all(a > b), "a colder wall draws more heat"
    np.testing.assert_allclose((a - b) / a, 300.0 / 2700.0, rtol=1e-9)
    # the diagnosis: without the fix both walls are below (T_i + 20)/2, the ghost
    # clamps to the same 20 K, and the face flux cannot tell them apart
    a0, _ = _wall_flux(300.0, 3000.0, fix=False)
    b0, _ = _wall_flux(600.0, 3000.0, fix=False)
    np.testing.assert_array_equal(a0, b0)


# -------------------------------------------------------------- the coupler


def test_hole_state_takes_the_two_sides_separately(cfg):
    g = build_blocks(cfg.agent("g"), cfg, 4)[0]
    s = Compressible2D(g, GasConfig(gamma=1.4, R=287.0), bcs={})
    nz, ny = g.shape
    U = thermo.prim_to_cons(np.tile([0.04, 1000.0, 0.0, 2500.0], (nz, ny, 1)), 1.4)
    lo = thermo.prim_to_cons(np.tile([0.05, 900.0, 0.0, 3000.0], (nz, NG, 1)), 1.4)
    hi = thermo.prim_to_cons(np.tile([0.06, 800.0, 0.0, 3500.0], (nz, NG, 1)), 1.4)
    s.hole_state = (lo, hi)
    Ue = s.ghost(U)
    j0, j1 = s._hole
    np.testing.assert_allclose(Ue[NG:nz + NG, NG + j0], lo[:, 0])
    np.testing.assert_allclose(Ue[NG:nz + NG, NG + j1], hi[:, 0])
    s.hole_state = lo[:, 0]                           # the original contract still works
    Ue = s.ghost(U)
    np.testing.assert_allclose(Ue[NG:nz + NG, NG + j1], lo[:, 0])


def test_as_gas_keeps_pressure_temperature_and_velocity():
    air = GasConfig(gamma=1.4, R=287.053)
    gas = GasConfig(gamma=1.22, R=361.0)
    W = np.array([[0.04, 1044.0, 3.0, 2549.0]])
    U = thermo.prim_to_cons(W, air.gamma)
    V = gen._as_gas(U, air, gas)
    Wv = thermo.cons_to_prim(V, gas.gamma)
    assert Wv[0, 3] == pytest.approx(W[0, 3], rel=1e-12)
    assert Wv[0, 3] / (Wv[0, 0] * gas.R) == pytest.approx(W[0, 3] / (W[0, 0] * air.R), rel=1e-12)
    np.testing.assert_allclose(Wv[0, 1:3], W[0, 1:3], rtol=1e-12)
    # the diagnosis: the same conserved vector read directly by the other model
    Wr = thermo.cons_to_prim(U, gas.gamma)
    assert Wr[0, 3] / W[0, 3] == pytest.approx(0.22 / 0.4, rel=1e-9)


def test_the_shell_reads_its_inner_wall_at_its_own_z(episode):
    """Every inner-face segment over the engine takes the gas wall at the SAME z;
    the tank barrel takes the declared adiabatic, ambient-pressure wall. With a
    pressure ramp in z along the engine walls, each segment must read the ramp
    at its own midpoint. The old coupler stretched b's 0.28 m of wall over the
    whole 4.7 m airframe by array index, so the station at z = 0.06 read b near
    its throat end."""
    ep = episode
    saved = {a: [U.copy() for U in ep.U[a]] for a in gen.ENGINE_AGENTS}
    slope, p0 = 1.0e7, 1.0e6
    try:
        for a in gen.ENGINE_AGENTS:
            blk = ep.blocks[a][0]
            W = thermo.cons_to_prim(ep.U[a][0], ep.gas_cfg.gamma)
            W[..., 3] = p0 + slope * blk.centroid[..., 0]
            ep.U[a][0] = thermo.prim_to_cons(W, ep.gas_cfg.gamma)
        _, p_inf, _, _ = ep._atm()
        for k in (0, 1):
            h, T, p = ep._shell_inner_loads(k, p_inf)
            zs, _ = ep._shell_face(k, "inner")
            zm = 0.5 * (zs[:-1] + zs[1:])
            barrel = zm < 0.0
            assert barrel.any() and (~barrel).any()
            assert np.all(h[barrel] == 0.0)
            np.testing.assert_allclose(p[barrel], p_inf, rtol=1e-12)
            cell = 0.01                                    # the gas walls' axial size here
            assert np.abs(p[~barrel] - (p0 + slope * zm[~barrel])).max() <= slope * cell
    finally:
        for a, Us in saved.items():
            ep.U[a] = Us


def test_gas_walls_see_the_shell_temperature_at_their_z(episode):
    ep = episode
    saved = [T.copy() for T in ep.T_shell]
    try:
        for k, m in enumerate(ep.shell_mesh):
            z = m.nodes[..., 0].reshape(-1)
            ep.T_shell[k] = 300.0 + 100.0 * (z - z.min())          # a ramp in z
        for side in ("jmin", "jmax"):
            zc = 0.5 * (ep._wall_edges("b", 0, side)[:-1] + ep._wall_edges("b", 0, side)[1:])
            Tw = ep._wall_T("b", 0, side)
            np.testing.assert_allclose(Tw, 300.0 + 100.0 * (zc + 4.0), rtol=1e-12)
    finally:
        ep.T_shell = saved


def test_loads_on_a_symmetric_body_have_drag_and_no_side_force(episode):
    """The aero integral with the body's outward normal on both panels. The
    index-oriented normal gave exactly zero drag and a side force twice the
    upper panel's (1781.8 N/m in the W321 episode)."""
    ep = episode
    saved = [U.copy() for U in ep.U["d"]]
    try:
        for k, blk in enumerate(ep.blocks["d"]):
            p = 3000.0 + 800.0 * np.abs(blk.centroid[..., 1]) + 50.0 * blk.centroid[..., 0]
            W = np.stack([np.full(p.shape, 0.04), np.full(p.shape, 1000.0),
                          np.zeros(p.shape), p], axis=-1)
            ep.U["d"][k] = thermo.prim_to_cons(W, ep.air_cfg.gamma)
        ep._compute_loads()
        Fz, Fy = ep.loads_body[2], ep.loads_body[3]
        assert Fz > 0.0, "drag points to the tail"
        assert abs(Fy) < 1e-9 * abs(Fz), "no side force on a symmetric body"
    finally:
        ep.U["d"] = saved


def test_the_body_frame_puts_the_nose_at_minus_z(episode):
    """W322: at theta = pi/2 thrust points UP and drag DOWN. Fixing only the
    thrust's sign would have left the drag pushing the vehicle forward."""
    ep = episode
    ez, ey = ep._body_axes()
    th = float(ep.rigid[2])
    np.testing.assert_allclose(ez, [-np.cos(th), -np.sin(th)], atol=1e-15)
    assert ez[0] * ey[1] - ez[1] * ey[0] == pytest.approx(1.0), "a proper rotation"
    ep._compute_loads()
    assert ep.loads[1] > 0.0, "thrust up"


def test_the_freestream_is_the_relative_wind_and_carries_alpha(cfg):
    pt = next(p for p in sweep.corner_cases() if p.alpha != 0.0)
    ep = gen.CoupledEpisode(gen.EpisodeSpec(point=pt, n_macro=2, dt_macro=1e-5,
                                            coarsen=4), cfg)
    rho, u, v, p = ep._freestream()
    V = np.hypot(*ep.rigid[3:5])
    a = np.radians(pt.alpha)
    assert u == pytest.approx(V * np.cos(a), rel=1e-12)
    assert v == pytest.approx(-V * np.sin(a), rel=1e-12)
    # and it follows the vehicle: double the speed, double the wind
    ep.rigid = ep.rigid.copy()
    ep.rigid[3:5] *= 2.0
    assert ep._freestream()[1] == pytest.approx(2.0 * u, rel=1e-12)


def test_interface_records_cover_both_sides_with_outward_normals(episode, cfg):
    ep = episode
    rec = ep._iface_records()
    nb = ep.blocks["b"][0].shape[0]
    nd = ep.blocks["d"][0].shape[0]
    nf = ep.blocks["f"][0].shape[0]
    assert rec["b-c"].shape[1] == 2 * nb
    n_air = int((ep._airframe_frac(ep._wall_edges("d", 0, ep._d_wall_side(0))) > 0).sum())
    assert rec["c-d"].shape[1] == 2 * n_air < 2 * nd, "the slot is not the airframe (W333)"
    assert rec["g-f"].shape[1] == 2 * nf
    # impermeable walls carry no mass
    assert np.all(rec["b-c"][0] == 0.0) and np.all(rec["c-d"][0] == 0.0)
    # a symmetric state: the two walls' pressure forces cancel in y
    for key in ("b-c", "c-d"):
        fy = (rec[key][2] * rec[key][7]).sum()
        fy_scale = (np.abs(rec[key][2]) * rec[key][7]).sum()
        assert abs(fy) < 1e-9 * fy_scale, key
    # d-g covers only its declared span: the seg lengths add up to 2 * (1.5 - 0.6)
    geo = cfg.geometry
    assert rec["d-g"][7].sum() == pytest.approx(2 * (geo.farfield_halfwidth - geo.plume_halfwidth),
                                                rel=1e-12)


def test_undeclared_interfaces_are_recorded_and_tile_the_planes(episode, cfg):
    """a-c, e-c and d-f are exchanged by the coupler and recorded in every
    episode without being declared to the model. d's outlet plane is split
    between d-g (declared) and d-f (recorded) with nothing lost or counted twice."""
    rec = episode._iface_records()
    for key in ("a-c", "e-c", "d-f"):
        assert key in rec and np.isfinite(rec[key]).all(), key
    assert {f"{r.src}-{r.dst}" for r in cfg.recorded_interfaces} == {"a-c", "e-c", "d-f"}
    assert not ({f"{e.src}-{e.dst}" for e in cfg.edge_list} & {"a-c", "e-c", "d-f"})
    na, ne = episode.blocks["a"][0].shape[0], episode.blocks["e"][0].shape[0]
    assert rec["a-c"].shape[1] == 2 * na and rec["e-c"].shape[1] == 2 * ne
    geo = cfg.geometry
    outlet = 2.0 * (geo.farfield_halfwidth - float(shell_outer(np.array([geo.z_exit]), geo)[0]))
    total = rec["d-g"][7].sum() + rec["d-f"][7].sum()
    assert total == pytest.approx(outlet, rel=1e-12)


def test_config_declares_the_tank_wall():
    cfg = load_config()
    tank = next(r for r in cfg.unmodelled if r.name == "tank_interior")
    assert tank.wall_h == 0.0 and tank.wall_p_gauge == 0.0
    bad = dataclasses.replace(cfg, unmodelled=tuple(
        dataclasses.replace(r, wall_h=None) for r in cfg.unmodelled))
    with pytest.raises(ValueError):
        gen.CoupledEpisode(gen.EpisodeSpec(point=sweep.corner_cases()[0], n_macro=2,
                                           dt_macro=1e-5, coarsen=4), bad)


@pytest.mark.slow
def test_a_short_march_climbs_under_thrust_and_stays_finite(cfg):
    pt = sweep.corner_cases()[0]
    ep = gen.CoupledEpisode(gen.EpisodeSpec(point=pt, n_macro=3, dt_macro=2e-5,
                                            coarsen=4), cfg)
    v0 = float(ep.rigid[4])
    r = ep.run()
    assert np.isfinite(r["rigid"]).all()
    assert r["loads"][-1, 1] > 0.0
    assert r["rigid"][-1, 4] > v0, "thrust exceeds weight and drag"
    for k, v in r["iface"].items():
        assert np.isfinite(v).all(), k


# ------------------------------------------------ W332: the wall's inviscid flux


def _wall_block(T_wall, v_wall):
    """A 6x4 channel of 2500 K gas at 5 MPa with an isothermal wall at jmin,
    the gas moving along it at 800 m/s and INTO it at -v_wall."""
    nz, ny = 6, 4
    z, y = np.linspace(0, 0.06, nz + 1), np.linspace(0, 0.03, ny + 1)
    Z, Y = np.meshgrid(z, y, indexing="ij")
    blk = Block("w", np.stack([Z, Y], -1), np.zeros((nz, ny), bool))
    gas = GasConfig(gamma=1.22, R=361.0)
    s = Compressible2D(blk, gas, bcs={"jmin": BC("wall_noslip", {"T_wall": np.full(nz, T_wall)}),
                                      "jmax": BC("wall_noslip"), "imin": BC("extrapolate"),
                                      "imax": BC("extrapolate")})
    p, T = 5.0e6, 2500.0
    W = np.zeros((nz, ny, 4))
    W[..., 0] = p / (gas.R * T)
    W[..., 1] = 800.0
    W[..., 2] = v_wall
    W[..., 3] = p
    return s, thermo.prim_to_cons(W, gas.gamma)


def _jmin_mass_flux(s, Ue):
    """The inviscid mass flux through the jmin face, from a ghosted state."""
    from atlas.solvers.riemann import FLUXES
    nz, _ = s.block.shape
    L, R = s._faces(Ue[NG:nz + NG, :], axis=1)
    F = FLUXES[s.cfg.riemann](L, R, s.block.n_j, s.cfg.gamma) * s.block.a_j[..., None]
    return F[:, 0, 0]


def test_W332_an_isothermal_wall_passes_no_mass():
    """The inviscid flux sees the wall as a mirror at the cell's own density:
    no mass through the face, whatever the wall's temperature."""
    s, U = _wall_block(288.0, -50.0)
    scale = float(U[..., 0].max()) * 50.0 * float(s.block.a_j[:, 0].max())
    mirror = _jmin_mass_flux(s, s.ghost(U, thermal=False))
    assert np.abs(mirror).max() < 1e-12 * scale
    # the diagnosis: the ghost the viscous operator uses is ~125x denser than
    # the cell (T_g clamps at 20 K), so handed to the Riemann solver the pair
    # is not a reflection -- the contact moves off the wall at ~49 m/s and the
    # face injected mass at ~100x the cell's own normal mass flux
    thermal = _jmin_mass_flux(s, s.ghost(U))
    assert np.abs(thermal).max() > 10.0 * scale


def test_W332_the_residual_hands_the_inviscid_flux_the_mirror():
    """`residual` = inviscid(mirror ghost) + viscous(isothermal ghost)."""
    s, U = _wall_block(288.0, -50.0)
    want = s._inviscid_residual(s.ghost(U, thermal=False)) + s._viscous_residual(s.ghost(U))
    np.testing.assert_array_equal(s.residual(U), want)
    # a block with no isothermal wall builds one ghost array, as before
    s.bcs["jmin"] = BC("wall_noslip")
    Ue = s.ghost(U)
    np.testing.assert_array_equal(s.residual(U),
                                  s._inviscid_residual(Ue) + s._viscous_residual(Ue))


@pytest.mark.slow
def test_W332_the_engine_stays_mirror_symmetric():
    """The inviscid engine, walls at one temperature, marched 20 ms: with the
    isothermal ghost in the inviscid flux, the diverging nozzle's asymmetry grew
    from round-off by ~1e5 per 5 ms (1e-13, 3e-8, 5e-3) and broke the whole
    vehicle's mirror symmetry; adiabatic and slip walls, and the HLLE and
    Rusanov fluxes, stayed at round-off on the same march."""
    pt = sweep.corner_cases()[0]
    ep = gen.CoupledEpisode(gen.EpisodeSpec(point=pt, n_macro=5, dt_macro=5e-3, coarsen=4,
                                            inviscid=True), load_config())
    ep._wall_T = lambda agent, k, side: np.full(ep.blocks[agent][k].shape[0], 288.15)
    worst = 0.0
    for _ in range(4):
        ep._wire_engine()
        ep._advance_gas(gen.ENGINE_AGENTS, ep.dt_macro)
        W = thermo.cons_to_prim(ep.U["e"][0], ep.gas_cfg.gamma)
        p = W[..., 3]
        worst = max(worst, float(np.abs(p - p[:, ::-1]).max() / np.abs(p).max()))
    assert worst < 1e-10


# ------------------------------------------------------- W333: the forebody slot


def test_W333_isothermal_mask_leaves_those_stations_a_plain_mirror():
    s, U = _wall_block(288.0, 0.0)
    nz = s.block.shape[0]
    m = np.arange(nz) >= 3
    s.bcs["jmin"] = BC("wall_noslip", {"T_wall": np.full(nz, 288.0), "isothermal_mask": m})
    Ue, Um = s.ghost(U), s.ghost(U, thermal=False)
    np.testing.assert_array_equal(Ue[NG:nz + NG, :NG][~m], Um[NG:nz + NG, :NG][~m])
    assert not np.allclose(Ue[NG:nz + NG, :NG][m], Um[NG:nz + NG, :NG][m])


def test_W333_d_wall_ahead_of_the_nose_is_slip_and_adiabatic(episode, cfg):
    """`d`'s inner boundary ahead of the nose borders the declared forebody
    slot. It used to be a no-slip wall at the nose's temperature all the way to
    the inlet -- a metre of flat plate before the vehicle began."""
    ep = episode
    ep._wire_external()
    z0 = float(cfg.geometry.shell_z[0])
    for k in range(2):
        side = ep._d_wall_side(k)
        bc = ep.sol["d"][k].bcs[side]
        z = ep._wall_edges("d", k, side)
        ahead = 0.5 * (z[:-1] + z[1:]) < z0
        assert ahead.any() and (~ahead).any()
        np.testing.assert_array_equal(bc.params["no_slip_mask"], ~ahead)
        np.testing.assert_array_equal(bc.params["isothermal_mask"], ~ahead)
        assert np.abs(z - z0).min() < 1e-12, "the nose is a node line of d"


def test_W333_the_c_d_record_is_the_airframe_outer_face(episode):
    """c-d used to include d's wall over the forebody slot: 2 m of an 11.45 m
    record that borders no airframe."""
    ep = episode
    rec = ep._iface_records()["c-d"]
    outer = 0.0
    for m in ep.shell_mesh:
        outer += np.linalg.norm(np.diff(m.nodes[:, m.outer_j], axis=0), axis=-1).sum()
    assert rec[7].sum() == pytest.approx(outer, rel=1e-12)


def test_W333_the_slot_carries_no_load(episode):
    """Pressure and shear on the slot's wall used to count on the vehicle."""
    ep = episode
    saved = [U.copy() for U in ep.U["d"]]
    try:
        ep._compute_loads()
        base = ep.loads_body.copy()
        z0 = float(ep.geo.shell_z[0])
        for k, blk in enumerate(ep.blocks["d"]):
            ahead = blk.centroid[:, 0, 0] < z0
            U = ep.U["d"][k].copy()                  # only the slot's cells change
            Wa = thermo.cons_to_prim(U[ahead], ep.air_cfg.gamma)
            Wa[..., 1] *= 3.0
            Wa[..., 3] *= 2.0
            U[ahead] = thermo.prim_to_cons(Wa, ep.air_cfg.gamma)
            ep.U["d"][k] = U
        ep._compute_loads()
        np.testing.assert_array_equal(ep.loads_body[2:], base[2:])
    finally:
        ep.U["d"] = saved


def test_W333_config_declares_the_slot_wall():
    cfg = load_config()
    slot = next(r for r in cfg.unmodelled if r.name == "forebody_slot")
    assert slot.gas_wall == "slip_adiabatic"
    bad = dataclasses.replace(cfg, unmodelled=tuple(
        dataclasses.replace(r, gas_wall=None) for r in cfg.unmodelled))
    with pytest.raises(ValueError):
        gen.CoupledEpisode(gen.EpisodeSpec(point=sweep.corner_cases()[0], n_macro=2,
                                           dt_macro=1e-5, coarsen=4), bad)

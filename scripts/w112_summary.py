"""Read `out/w112/w112.json` and print exactly the numbers the wiki pages quote.

    python scripts/w112_summary.py [--out out/w112]

Kept as a script rather than done by hand for the reason `tier0-measurements`
gives every time: a number transcribed into a wiki page is a number that can
drift from the artifact it came from, and the repair for that is a reader that
runs.  Every table in `tier0-measurements` 21 and in
`case-study-seam-placement-atlas-0.1` is printed here, in the order it appears
there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def f(x, n=4):
    if x is None:
        return "--"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int,)) or (isinstance(x, float) and x == int(x) and abs(x) < 1e6):
        return str(int(x))
    return f"{x:.{n}g}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w112"))
    args = ap.parse_args()
    with open(os.path.join(args.out, "w112.json"), encoding="utf-8") as fh:
        d = json.load(fh)

    print("=" * 78)
    print("W112 / CS-9*  --  wall", f(d["wall_seconds"]), "s")
    print("=" * 78)

    c = d["controls"]
    print("\n## controls")
    print(f"  ZERO       defect = {c['ZERO']['composed_defect']!r}  "
          f"exactly zero: {c['ZERO']['exactly_zero']}  "
          f"({c['ZERO']['n_seams']} seams, {c['ZERO']['n_overlap_pairs']} pairs)")
    r = c["REPRODUCE"]
    print(f"  REPRODUCE  boxes identical: {r['boxes_identical']}  "
          f"weights differ by {r['weights_max_abs_difference']!r}  "
          f"PoU residual {f(r['partition_of_unity_residual'])}  "
          f"{r['n_overlap_pairs']} overlapping pairs, {r['n_seams']} seams")
    n = c["NOISE"]
    print(f"  NOISE      repeat bitwise: {n['defect_repeat_identical']}; "
          f"Q under a halved FD step moves {f(n['Q_fd_step_relative_spread_max'])} "
          f"(max), {f(n['Q_fd_step_relative_spread_mean'])} (mean)")

    for tag, g in d["geometries"].items():
        print(f"\n## {tag}  domain {g['nx']}x{g['ny']}  dt {f(g['dt'])}  "
              f"{g['n_candidates_scored']} candidates")
        hc = g["hand_chosen"]
        print(f"   hand-chosen {hc['label']}  {hc['n_col']}x{hc['n_row']}, "
              f"{hc['n_seams']} seams, {hc['n_overlap_pairs']} pairs, "
              f"halo_cells {hc['cost']['halo_cells']}, uniform={hc['is_uniform']}")
        for pname, p in g["pools"].items():
            print(f"   pool {pname:6s} {p['n_candidates']:4d}  [{p['budget']}]")

        by_pool: dict[str, list] = {}
        for key, rep in g["searches"].items():
            by_pool.setdefault(rep["pool"], []).append((key.split("::", 1)[1], rep))
        for pname, rows in by_pool.items():
            print(f"\n   --- pool {pname} " + "-" * 40)
            base = rows[0][1]
            print(f"       hand-chosen defect {f(base.get('hand_chosen_defect'))}, "
                  f"best available {f(base['best_defect'])}, "
                  f"worst {f(base['worst_defect'])}, "
                  f"spread {f(base['spread_factor'])}x, "
                  f"hand ranks {base.get('hand_chosen_rank')} of "
                  f"{base['n_candidates']}")
            print(f"       {'criterion':26s} {'chose':30s} {'shape':7s} "
                  f"{'seams':5s} {'defect':>10s} {'/hand':>8s} {'/best':>8s} "
                  f"{'lost':>6s} {'rank':>7s}")
            for crit, rep in rows:
                sh = rep.get("chosen_shape")
                print(f"       {crit:26s} {rep['chosen']:30s} "
                      f"{str(sh[0]) + 'x' + str(sh[1]) if sh else '--':7s} "
                      f"{rep.get('chosen_n_seams', '--'):5} "
                      f"{f(rep['chosen_defect']):>10s} "
                      f"{f(rep.get('chosen_over_hand_chosen')):>8s} "
                      f"{f(rep['penalty_vs_best']):>8s} "
                      f"{f(rep['fraction_of_available_range_lost'], 3):>6s} "
                      f"{f(rep['rank_correlation_vs_defect'], 4):>7s}")

        if "KNOWN" in g:
            k = g["KNOWN"]
            print(f"\n   --- KNOWN (wake at row {k['wake_row']}) " + "-" * 26)
            print(f"       defect argmin row {k['argmin_defect_row']} "
                  f"({k['distance_of_defect_argmin_from_wake']} rows away); "
                  f"Q argmin {k['argmin_Q_row']}; Qhat argmin "
                  f"{k['argmin_Q_hat_row']}")
            print(f"       cutting nearest the wake (row {k['nearest_cut_to_wake']}): "
                  f"{f(k['defect_cutting_on_the_wake'])}; farthest "
                  f"(row {k['farthest_cut']}): {f(k['defect_cutting_away'])}; "
                  f"ratio {f(k['on_over_away'])}x")

    print("\n## basis orbit (interface-transfer-theory 9's flagged weakness)")
    b = d["basis"]
    print(f"   {b['n_seams_measured']} real seams: orbit ratio min "
          f"{f(b['orbit_ratio_min'])}x, median {f(b['orbit_ratio_median'])}x, "
          f"max {f(b['orbit_ratio_max'])}x")
    print(f"   beta invariant to {f(b['beta_invariance_worst'])} (worst)")
    worst = max(b["rows"], key=lambda r: r["orbit_ratio"])
    print(f"   worst seam: {worst['geometry']} {worst['decomposition']} "
          f"{worst['seam_id']}: Q declared {f(worst['Q_declared_fourier'])}, "
          f"eigenbasis {f(worst['Q_in_eigenbasis_of_sym_part'])}, "
          f"random frames [{f(worst['Q_min_over_random_frames'])}, "
          f"{f(worst['Q_max_over_random_frames'])}]")

    print("\n## horizon")
    for tag, h in d.get("horizon", {}).items():
        print(f"   {tag}: {h['n_scored']} scored, one-interval vs "
              f"{d['horizon_intervals']}-interval defect ranks at "
              f"{f(h['one_interval_vs_marched'])}, marched defect in "
              f"[{f(h['marched_defect_min'])}, {f(h['marched_defect_max'])}], "
              f"{h['n_nonfinite']} non-finite")
        for crit, row in h["criteria"].items():
            keys = [k for k in row if k.startswith("vs_")]
            print(f"      {crit:26s} " + "  ".join(
                f"{k} {f(row[k], 4)}" for k in keys))

    print("\n## disagreements (criterion vs measured defect)")
    for tag, blk in d.get("disagreements", {}).items():
        print(f"   {tag}: {blk['n_sign_disagreements']} sign, "
              f"{blk['n_argmin_disagreements']} argmin, of {len(blk['rows'])}")
        for r in blk["rows"]:
            if r["sign_disagrees"] or r["argmin_disagrees"]:
                print(f"      {r['budget']:6s} {r['criterion']:26s} "
                      f"rank {f(r['rank_correlation'], 4):>8s}  "
                      f"penalty {f(r['penalty_vs_best'])}x  "
                      f"sign={r['sign_disagrees']} argmin={r['argmin_disagrees']}")

    if "shell" in d:
        s = d["shell"]
        print("\n## CS-9 -- the shared thermal-strain domain")
        print(f"   {s['n_z']} elements, streak at z = {s['streak_z_m']} m of "
              f"width {s['streak_width_m']} m, halo {s['halo_elements']}, "
              f"T in [{f(s['T_range'][0], 6)}, {f(s['T_range'][1], 6)}] K, "
              f"{s['n_candidates']} candidates (exhaustive)")
        z = s.get("ZERO_shell")
        if z:
            print(f"   ZERO: one segment -> conduction {f(z['conduction_defect'])}, "
                  f"elasticity {f(z['elasticity_defect'])}, both machine zero: "
                  f"{z['both_machine_zero']}")
        k = s.get("KNOWN_shell")
        if k:
            print(f"   KNOWN: distance-from-streak vs conduction defect ranks at "
                  f"{f(k['distance_vs_conduction_defect_rank'], 4)}; vs elasticity "
                  f"{f(k['distance_vs_elasticity_defect_rank'], 4)}")
            print(f"          conduction at the streak "
                  f"{f(k['conduction_at_the_streak'])} vs farthest "
                  f"{f(k['conduction_farthest_from_it'])} = "
                  f"{f(k['conduction_penalty_for_cutting_through_it'])}x")
            print(f"          elasticity at the streak "
                  f"{f(k['elasticity_at_the_streak'])} vs farthest "
                  f"{f(k['elasticity_farthest_from_it'])}")
        for name in ("by_conduction", "by_elasticity"):
            r = s[name]
            print(f"   {name}: argmin {r['argmin']} (z {r['argmin_z_cuts']}), "
                  f"other family's argmin {r['other_family_argmin']} "
                  f"(z {r['other_family_argmin_z_cuts']}), agree="
                  f"{r['families_agree']}")
            print(f"      following this family costs the other "
                  f"{f(r['cost_of_following_this_family_for_the_other'])}x")
        print(f"   rank correlation BETWEEN the two families: "
              f"{f(s['by_conduction']['rank_correlation_between_families'], 4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

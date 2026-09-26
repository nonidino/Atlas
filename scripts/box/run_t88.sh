#!/bin/bash
# Tier 88: the fixed episode (W337-W340) audited, and the records the fixes
# touch re-derived, on a rented box -- each re-derivation twice, as it is
# (out/t88) and with all four fixes reverted on the same tree (out/t88_revert),
# the attribution control. The march itself runs on the laptop.
# Every job logs to logs/<job>.log; logs/<job>.done holds "exit_code seconds".
#   BOX_ROOT=/root/job nohup bash run_t88.sh <workers_converged> <workers_gf> > runner.log 2>&1 < /dev/null &
set -u
ROOT=${BOX_ROOT:-/root/job}
cd "$ROOT/vault"
export ATLAS_BUILD_REPO="$ROOT/build"
export ATLAS_BUILD_REPO_IDENTITY="$(cat "$ROOT/IDENTITY")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
WC=${1:-12}
WG=${2:-4}
mkdir -p out/w336 out/t88 out/t88_revert "$ROOT/logs"
date +%s > "$ROOT/logs/START"

job() {
  local name=$1; shift
  ( t0=$(date +%s); "$@" > "$ROOT/logs/$name.log" 2>&1; rc=$?
    echo "$rc $(( $(date +%s) - t0 ))" > "$ROOT/logs/$name.done" ) &
}
S=scripts/w336_episode_residuals.py
R="python scripts/box/with_revert.py all"

# ---- the audit (W336's stages on the fixed code), the longest first ---------
job grid_engine_c1   python $S --stage grid_engine   --coarsen 1 --t-end 0.03
job audit            python $S --stage audit
job grid_external_c1 python $S --stage grid_external --coarsen 1 --t-end 0.03
job dt_2p5ms         python $S --stage dt --dt-macro 2.5e-3 --t-end 0.06
job dt_10ms          python $S --stage dt --dt-macro 1e-2   --t-end 0.06
job grid_engine_c2   python $S --stage grid_engine   --coarsen 2 --t-end 0.03
job grid_external_c2 python $S --stage grid_external --coarsen 2 --t-end 0.03
job grid_engine_c4   python $S --stage grid_engine   --coarsen 4 --t-end 0.03
job grid_external_c4 python $S --stage grid_external --coarsen 4 --t-end 0.03
job control_wall     python $S --stage control_wall
job control_old      python $S --stage control_old
job diag             python scripts/w336_thermal_seam_diag.py
job diag_revert      python scripts/w336_thermal_seam_diag.py --revert W339 \
                     --json out/w336/thermal_seam_diag_revert_W339.json

# ---- the re-derivation, fixed and reverted (W334's job list) ---------------
for V in fixed revert; do
  if [ $V = fixed ]; then P="python"; O=out/t88; else P=$R; O=out/t88_revert; fi
  job ${V}_w314_anchor env W314_FORK=1 W314_RECORD=$O/w314_anchor_cache.json \
      $P scripts/w314_rocket_multirate_defect.py --stages anchor --json $O/w314_anchor_only.json
  job ${V}_w314_core env W314_FORK=1 \
      $P scripts/w314_rocket_multirate_defect.py \
      --stages setup,r9,rate,slope,sigma_law,ratio,trace_rank,two_way,transfer,knee,constants \
      --json $O/w314.json
  job ${V}_w312_converged $P scripts/w312_advec_converged.py --workers "$WC" --json $O/w312_converged.json
  job ${V}_w312_gf $P scripts/w312_advec_converged.py --seams g-f --workers "$WG" --json $O/w312_gf.json
  job ${V}_w312_plane $P scripts/w312_plane_seam_converged.py --modes 8 --json $O/w312.json
  job ${V}_w312_under $P scripts/w312_plane_seam_converged.py --modes 8 --dt-scale 1e-3 --json $O/w312_underresolved.json
  job ${V}_w300 $P scripts/w300_rocket_bc_seam.py --stage all --json $O/w300.json
  job ${V}_w301 $P scripts/w301_saturation_sweep.py --json $O/w301.json
  # --with-real: W334's record carries the `real` level (its runner list did
  # not say so; it was run by hand, box_logs/w302_real.log). The first Tier 88
  # launch left it out and was re-run with it the same hour.
  job ${V}_w302 $P scripts/w302_declaration_ledger.py --with-real --json $O/w302.json
  job ${V}_w303 $P scripts/w303_cht_convention.py --json $O/w303.json
  job ${V}_w305 $P scripts/w305_trajectory_and_mech.py --json $O/w305.json
  job ${V}_w306 $P scripts/w306_sign_structure.py --json $O/w306.json
  job ${V}_w310 $P scripts/w310_rocket_r0_complete.py --json $O/w310.json
  # w304 first, then Tier 77's four-configuration table at w304's own root
  ( t0=$(date +%s)
    $P scripts/w304_seam_base.py --json $O/w304.json > "$ROOT/logs/${V}_w304.log" 2>&1
    echo "$? $(( $(date +%s) - t0 ))" > "$ROOT/logs/${V}_w304.done"
    t0=$(date +%s)
    root=$(python -c "import json; print(json.load(open('$O/w304.json'))['lamstar']['difference_root'])")
    $P scripts/w334_tier77_table.py --new-root "$root" --json $O/w334_tier77_table.json \
        > "$ROOT/logs/${V}_t77.log" 2>&1
    echo "$? $(( $(date +%s) - t0 ))" > "$ROOT/logs/${V}_t77.done" ) &
  # fold the anchor into the core's record, through the real stage code
  ( while [ ! -f "$ROOT/logs/${V}_w314_anchor.done" ] || [ ! -f "$ROOT/logs/${V}_w314_core.done" ]; do sleep 15; done
    t0=$(date +%s)
    env W314_REPLAY=$O/w314_anchor_cache.json \
      $P scripts/w314_rocket_multirate_defect.py --stages anchor --json $O/w314.json \
      > "$ROOT/logs/${V}_w314_replay.log" 2>&1
    echo "$? $(( $(date +%s) - t0 ))" > "$ROOT/logs/${V}_w314_replay.done" ) &
done

wait
date +%s > "$ROOT/logs/ALL.done"

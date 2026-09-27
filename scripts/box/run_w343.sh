#!/bin/bash
# W343: the d-g seam made two-way (build repo c1ccac6 + the two-way seam). Every
# stage the seam can reach is re-run: the march, the audit, the coupling-step
# study, the old-wall control, the thermal diagnostic and its W339 control, and
# the external flow alone at three grids (d's outlet now reads g). Plus the
# W343 control (the one-way seam put back, two steps, against Tier 88's audit)
# and the coarsen-4 engine alone, which the seam cannot reach: it must come out
# bitwise equal to Tier 88's, so the engine's coarsen-2 and coarsen-1 records
# carry over.
#   BOX_ROOT=/root/job nohup bash run_w343.sh > runner.log 2>&1 < /dev/null &
set -u
ROOT=${BOX_ROOT:-/root/job}
cd "$ROOT/vault"
export ATLAS_BUILD_REPO="$ROOT/build"
export ATLAS_BUILD_REPO_IDENTITY="$(cat "$ROOT/IDENTITY")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
mkdir -p out/w336 out/w321_box "$ROOT/logs"
date +%s > "$ROOT/logs/START"

job() {
  local name=$1; shift
  ( t0=$(date +%s); "$@" > "$ROOT/logs/$name.log" 2>&1; rc=$?
    echo "$rc $(( $(date +%s) - t0 ))" > "$ROOT/logs/$name.done" ) &
}
S=scripts/w336_episode_residuals.py
job grid_external_c1 python $S --stage grid_external --coarsen 1 --t-end 0.03
job march            python scripts/w321_rocket_episode.py --coarsen 4 --n-macro 30 --out out/w321_box
job audit            python $S --stage audit
job dt_2p5ms         python $S --stage dt --dt-macro 2.5e-3 --t-end 0.06
job dt_10ms          python $S --stage dt --dt-macro 1e-2   --t-end 0.06
job grid_external_c2 python $S --stage grid_external --coarsen 2 --t-end 0.03
job grid_external_c4 python $S --stage grid_external --coarsen 4 --t-end 0.03
job grid_engine_c4   python $S --stage grid_engine   --coarsen 4 --t-end 0.03
job control_wall     python $S --stage control_wall
job control_w343     python $S --stage control_old --revert W343 --json control_w343.json
job diag             python scripts/w336_thermal_seam_diag.py
job diag_revert      python scripts/w336_thermal_seam_diag.py --revert W339 \
                     --json out/w336/thermal_seam_diag_revert_W339.json
wait
date +%s > "$ROOT/logs/ALL.done"

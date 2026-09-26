#!/bin/bash
# Tier 88, second build (76edd8b+dirty:409bd71838fd6dc0): the one-way seams
# (e->f, d->f, d->g) hand over one layer again, the two-way seams two. Only the
# coupled stages can see that change -- f and g feed nothing back to the vehicle
# -- so only they are re-run here: the march, the audit, the coupling-step
# study, the old-wall control and the thermal diagnostic. The coarsen-4 grid
# stages are re-run as the check that the rest of the first build's records
# (run_t88.sh: the grid study at coarsen 2 and 1, the re-derivation) carry over:
# they must come out bitwise equal to the first build's.
#   BOX_ROOT=/root/job2 nohup bash run_t88b.sh > runner.log 2>&1 < /dev/null &
set -u
ROOT=${BOX_ROOT:-/root/job2}
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
job march            python scripts/w321_rocket_episode.py --coarsen 4 --n-macro 30 --out out/w321_box
job audit            python $S --stage audit
job dt_2p5ms         python $S --stage dt --dt-macro 2.5e-3 --t-end 0.06
job dt_10ms          python $S --stage dt --dt-macro 1e-2   --t-end 0.06
job control_wall     python $S --stage control_wall
job diag             python scripts/w336_thermal_seam_diag.py
job grid_engine_c4   python $S --stage grid_engine   --coarsen 4 --t-end 0.03
job grid_external_c4 python $S --stage grid_external --coarsen 4 --t-end 0.03
wait
date +%s > "$ROOT/logs/ALL.done"

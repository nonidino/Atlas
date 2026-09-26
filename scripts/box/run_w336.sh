#!/bin/bash
# W336: the rocket episode's residuals and error budget, on a rented box.
# Every job writes to vault/out/w336/ and logs to logs/<job>.log; logs/<job>.done
# holds "exit_code seconds" when it ends. The report runs after all of them.
set -u
ROOT=${BOX_ROOT:-/root/job}
cd "$ROOT/vault"
export ATLAS_BUILD_REPO="$ROOT/build"
export ATLAS_BUILD_REPO_IDENTITY="$(cat "$ROOT/IDENTITY")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
mkdir -p out/w336 "$ROOT/logs"
date +%s > "$ROOT/logs/START"

job() {
  local name=$1; shift
  ( t0=$(date +%s); "$@" > "$ROOT/logs/$name.log" 2>&1; rc=$?
    echo "$rc $(( $(date +%s) - t0 ))" > "$ROOT/logs/$name.done" ) &
}

S=scripts/w336_episode_residuals.py
# the longest first
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
wait
python $S --stage report > "$ROOT/logs/report.log" 2>&1
echo "$? $(( $(date +%s) - $(cat "$ROOT/logs/START") ))" > "$ROOT/logs/ALL.done"

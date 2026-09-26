#!/bin/bash
# W334: re-derive every Tier 76-85 gas-seam record with the fixed solver
# (W301 + W332), on a rented box. Every job writes to vault/out/w334/ and logs
# to logs/<job>.log; logs/<job>.done holds "exit_code seconds" when it ends.
#   bash run_box.sh <workers_converged> <workers_gf>
set -u
ROOT=${BOX_ROOT:-/root/job}
cd "$ROOT/vault"
export ATLAS_BUILD_REPO="$ROOT/build"
export ATLAS_BUILD_REPO_IDENTITY="$(cat "$ROOT/IDENTITY")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
WC=${1:-16}
WG=${2:-6}
mkdir -p out/w334 "$ROOT/logs"
date +%s > "$ROOT/logs/START"

job() {
  local name=$1; shift
  ( t0=$(date +%s); "$@" > "$ROOT/logs/$name.log" 2>&1; rc=$?
    echo "$rc $(( $(date +%s) - t0 ))" > "$ROOT/logs/$name.done" ) &
}

# ---- the long ones -------------------------------------------------------
# the declared 50 ms interval marches in its own process; its result is folded
# into the core's record afterwards through the real stage code (replay)
job w314_anchor env W314_FORK=1 W314_RECORD=out/w334/w314_anchor_cache.json \
    python scripts/w314_rocket_multirate_defect.py --stages anchor --json out/w334/w314_anchor_only.json
job w314_core env W314_FORK=1 \
    python scripts/w314_rocket_multirate_defect.py \
    --stages setup,r9,rate,slope,sigma_law,ratio,trace_rank,two_way,transfer,knee,constants \
    --json out/w334/w314.json
job w312_converged python scripts/w312_advec_converged.py --workers "$WC" --json out/w334/w312_converged.json
job w312_gf python scripts/w312_advec_converged.py --seams g-f --workers "$WG" --json out/w334/w312_gf.json
job w312_plane python scripts/w312_plane_seam_converged.py --modes 8 --json out/w334/w312.json

# ---- the short ones ------------------------------------------------------
job w312_under python scripts/w312_plane_seam_converged.py --modes 8 --dt-scale 1e-3 --json out/w334/w312_underresolved.json
job w300 python scripts/w300_rocket_bc_seam.py --stage all --json out/w334/w300.json
job w301 python scripts/w301_saturation_sweep.py --json out/w334/w301.json
job w302 python scripts/w302_declaration_ledger.py --json out/w334/w302.json
job w303 python scripts/w303_cht_convention.py --json out/w334/w303.json
job w304 python scripts/w304_seam_base.py --json out/w334/w304.json
job w305 python scripts/w305_trajectory_and_mech.py --json out/w334/w305.json
job w306 python scripts/w306_sign_structure.py --json out/w334/w306.json
job w310 python scripts/w310_rocket_r0_complete.py --json out/w334/w310.json

# ---- fold the anchor into the core's record --------------------------------
( while [ ! -f "$ROOT/logs/w314_anchor.done" ] || [ ! -f "$ROOT/logs/w314_core.done" ]; do sleep 15; done
  t0=$(date +%s)
  env W314_REPLAY=out/w334/w314_anchor_cache.json \
    python scripts/w314_rocket_multirate_defect.py --stages anchor --json out/w334/w314.json \
    > "$ROOT/logs/w314_replay.log" 2>&1
  echo "$? $(( $(date +%s) - t0 ))" > "$ROOT/logs/w314_replay.done" ) &

wait
date +%s > "$ROOT/logs/ALL.done"

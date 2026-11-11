#!/usr/bin/env bash
# TL's one evaluation, stage B, the price, on the rented machine (scripts/chartop_evaluate_twolevel.py).
#
#   STAMP=<stage A's stamp> DEADLINE_S=7200 setsid nohup bash scripts/chartop_evaluate_twolevel_box.sh \
#     < /dev/null > eval-box.log 2>&1 &
#
# Run from the payload's root (scripts/chartop_price_twolevel_payload.py, built from stage A's commit
# and checked with --verify-payload), with S's weights at weights/S/best.pt and TL-s0's at
# weights/TL-s0/best.pt.  First the pinned runner with STAGES=smoke: it installs the pinned versions,
# records the machine in env.log, checks S's weights against their recorded sha256, arms the box's
# own deadline (DEADLINE_S: the box STOPS itself, so a session that ends early cannot leave it
# billing) and runs the smoke price run.  Only if that passed, the price stage, with TL-s0's
# weights; stage C checks their sha256 against stage A's.  price.done holds its exit code, and
# EVAL.done comes last, so the laptop polls files rather than an ssh session.
set -u
cd "$(dirname "$0")/.."
: "${STAMP:?set STAMP to the stamp of stage A}"
TL_WEIGHTS="${TL_WEIGHTS:-weights/TL-s0/best.pt}"
STAGES=smoke DEADLINE_S="${DEADLINE_S:-7200}" bash run.sh > run.log 2>&1
status=$(cat ALL.done 2>/dev/null || echo 1)
if [ "$status" = "0" ]; then
  export PYTHONPATH="$PWD" KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8
  python scripts/chartop_evaluate_twolevel.py --price --stamp "$STAMP" --weights "$TL_WEIGHTS" > price.log 2>&1
  status=$?
  echo "$status" > price.done
fi
echo "$status" > EVAL.done

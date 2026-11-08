#!/usr/bin/env bash
# R2's one evaluation, stage B, the price, on the rented machine (scripts/chartop_evaluate_squares.py).
#
#   STAMP=<stage A's stamp> DEADLINE_S=7200 setsid nohup bash scripts/chartop_evaluate_squares_box.sh \
#     < /dev/null > eval-box.log 2>&1 &
#
# Run from the payload's root.  First the pinned runner with STAGES=smoke: it installs the pinned
# versions, records the machine in env.log, arms the box's own deadline (DEADLINE_S) and runs the
# smoke price run.  Only if that passed, the price stage.  price.done holds its exit code, and
# EVAL.done comes last, so the laptop polls files rather than an ssh session.
set -u
cd "$(dirname "$0")/.."
: "${STAMP:?set STAMP to the stamp of stage A}"
STAGES=smoke DEADLINE_S="${DEADLINE_S:-7200}" bash run.sh > run.log 2>&1
status=$(cat ALL.done 2>/dev/null || echo 1)
if [ "$status" = "0" ]; then
  export PYTHONPATH="$PWD" KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8
  python scripts/chartop_evaluate_squares.py --price --stamp "$STAMP" > price.log 2>&1
  status=$?
  echo "$status" > price.done
fi
echo "$status" > EVAL.done

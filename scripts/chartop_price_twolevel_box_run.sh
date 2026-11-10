#!/usr/bin/env bash
# TL's price protocol on a rented whole machine (gates_twolevel, gate PR; R2's protocol).
#
#   DEADLINE_S=7200 setsid nohup bash run.sh < /dev/null > run.log 2>&1 &
#   STAGES="smoke" DEADLINE_S=1800 setsid nohup bash run.sh < /dev/null > run.log 2>&1 &
#
# As R2's runner (chartop_price_box_run.sh): installs the laptop's versions (torch 2.7.1+cpu,
# numpy 1.26.4, scipy 1.13.1, pydantic 2.12.5, psutil, threadpoolctl 3.5.0), records the machine
# in env.log, checks S's weights (uploaded to weights/S/best.pt beside the payload) against their
# recorded sha256, then runs the stages: "smoke" (one small price run, which imports everything the
# dry run imports) and "dry-run" (scripts/chartop_price_twolevel.py --dry-run).  A stage runs only
# if the one before it exited 0.  Every stage writes <stage>.done with its exit code, and ALL.done
# comes last, so the laptop polls files rather than an ssh session.  The box gives itself a
# deadline: after DEADLINE_S seconds it STOPS itself (storage-only billing; the disk and the records
# survive), so a session that ends early cannot leave it billing (memory "vastai-gpu-offload").
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$PWD" KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8
STAGES="${STAGES:-smoke dry-run}"
DEADLINE_S="${DEADLINE_S:-7200}"
export WEIGHTS="${WEIGHTS:-weights/S/best.pt}"

eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^CONTAINER_(ID|API_KEY)=' | sed 's/^/export /')"
(
  python -m pip install -q vastai > deadline-pip.log 2>&1
  sleep "$DEADLINE_S"
  echo "deadline reached at $(date -u +%FT%TZ): stopping instance ${CONTAINER_ID:-?}"
  vastai stop instance "${CONTAINER_ID:-}" --api-key "${CONTAINER_API_KEY:-}"
) > deadline.log 2>&1 &
echo "deadline armed: ${DEADLINE_S}s, instance ${CONTAINER_ID:-unknown}, stages: ${STAGES}" > deadline-armed.log

python -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 > pip.log 2>&1 \
  && python -m pip install -q numpy==1.26.4 scipy==1.13.1 psutil pydantic==2.12.5 pydantic_core==2.41.5 \
       annotated-types==0.6.0 typing_extensions==4.15.0 threadpoolctl==3.5.0 >> pip.log 2>&1
status=$?
echo $status > pip.done

python - > env.log 2>&1 <<'PY'
import os, platform, numpy, scipy, torch, psutil
print("identity", open("IDENTITY").read().strip())
print("python", platform.python_version(), "torch", torch.__version__, "numpy", numpy.__version__,
      "scipy", scipy.__version__, "psutil", psutil.__version__)
print("cpus", os.cpu_count(), "affinity", len(os.sched_getaffinity(0)), "battery", psutil.sensors_battery())
for line in open("/proc/cpuinfo"):
    if line.startswith("model name"):
        print(line.strip())
        break
PY
python scripts/chartop_cpus.py --reserve 1 --explain >> env.log 2>&1

if [ "$status" -eq 0 ]; then
  python - > weights.log 2>&1 <<'PY'
import hashlib, os, sys
sys.path.insert(0, os.getcwd())
from atlas.chartop import gates_twolevel as GT
w = os.environ.get("WEIGHTS", "weights/S/best.pt")
h = hashlib.sha256(open(w, "rb").read()).hexdigest()
print(w, h)
sys.exit(0 if h == GT.INIT_SHA256 else 3)
PY
  status=$?
  echo $status > weights.done
fi

for st in $STAGES; do
  [ "$status" -eq 0 ] || break
  case "$st" in
    smoke)
      python scripts/chartop_price_twolevel.py --dry-run --runs 1 --reps 1 --n 32 \
        --problems workbench:plate-hole:steady --out smoke > smoke.log 2>&1
      status=$?
      echo $status > smoke.done ;;
    dry-run)
      python scripts/chartop_price_twolevel.py --dry-run --weights "$WEIGHTS" > dryrun.log 2>&1
      status=$?
      echo $status > dryrun.done ;;
    *)
      echo "unknown stage $st" > "$st.done"
      status=2 ;;
  esac
done
echo "$status" > ALL.done

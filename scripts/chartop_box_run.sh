#!/usr/bin/env bash
# Stage 0 (W355) on a rented box: install the pins, then run the requested stages.
#
#   STAGES="price" DEADLINE_S=2400 setsid nohup bash run.sh < /dev/null > run.log 2>&1 &
#   STAGES="data train" DATA_SPLITS="val train train-conformal" ONLY="B,Bp,C1" \
#     DEADLINE_S=23400 setsid nohup bash run.sh < /dev/null > run.log 2>&1 &
#
# Every stage writes <stage>.done with its exit code, and ALL.done at the end, so the
# laptop polls files rather than an ssh session.  The box gives itself a deadline:
# after DEADLINE_S seconds it STOPS itself (storage-only billing; the disk and the
# results survive), so a session that ends early cannot leave it billing at the GPU
# rate (the vault's memory "vastai-gpu-offload").  The data counts come from plan.json.
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$PWD" KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8
STAGES="${STAGES:-price}"
DATA_SPLITS="${DATA_SPLITS:-val train train-conformal train-c2}"
ONLY="${ONLY:-}"
DEADLINE_S="${DEADLINE_S:-3600}"
# The pool follows the container's CPU quota, not nproc: the A100 box reported 192 CPUs on a
# quota of 23 and ran 191 workers at 4.2 s per example, against 2.0 s with 63 on a 64-core box.
# chartop_cpus.py reads the cgroup quota, the affinity mask and the CPU count (W360).
WORKERS="${WORKERS:-$(python scripts/chartop_cpus.py --reserve 1 2>/dev/null)}"
if [ -z "$WORKERS" ]; then
  echo "scripts/chartop_cpus.py failed: WORKERS falls back to nproc - 1, which ignores a CPU quota" >&2
  WORKERS=$(( $(nproc) - 1 ))
fi

eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^CONTAINER_(ID|API_KEY)=' | sed 's/^/export /')"
(
  python -m pip install -q vastai > deadline-pip.log 2>&1
  sleep "$DEADLINE_S"
  echo "deadline reached at $(date -u +%FT%TZ): stopping instance ${CONTAINER_ID:-?}"
  vastai stop instance "${CONTAINER_ID:-}" --api-key "${CONTAINER_API_KEY:-}"
) > deadline.log 2>&1 &
echo "deadline armed: ${DEADLINE_S}s, instance ${CONTAINER_ID:-unknown}, stages: ${STAGES}" > deadline-armed.log

python -m pip install -q numpy==1.26.4 scipy==1.13.1 > pip.log 2>&1
echo $? > pip.done
python - > env.log 2>&1 <<'PY'
import os, platform, numpy, scipy, torch
print("python", platform.python_version(), "numpy", numpy.__version__, "scipy", scipy.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("cpus", os.cpu_count(), "affinity", len(os.sched_getaffinity(0)))
PY
python scripts/chartop_cpus.py --reserve 1 --explain >> env.log 2>&1
echo "WORKERS=$WORKERS" >> env.log

count() { python -c "import json; print(json.load(open('plan.json'))['data']['$1'])"; }

for st in $STAGES; do
  case "$st" in
    price)
      python scripts/chartop_price.py --out out > price.log 2>&1
      echo $? > price.done ;;
    data)
      mkdir -p data
      code=0
      for sp in $DATA_SPLITS; do
        case "$sp" in
          val) python scripts/chartop_data.py --split val --count "$(count val)" --workers "$WORKERS" \
                 --labels sine-hat,cosine --out data > data-val.log 2>&1 || code=1 ;;
          train) python scripts/chartop_data.py --split train --count "$(count train)" \
                 --workers "$WORKERS" --out data > data-train.log 2>&1 || code=1 ;;
          train-conformal) python scripts/chartop_data.py --split train-conformal \
                 --count "$(count train-conformal)" --workers "$WORKERS" --out data \
                 > data-conformal.log 2>&1 || code=1 ;;
          train-c2) python scripts/chartop_data.py --split train --count "$(count train-c2)" \
                 --name train-c2 --labels sine-hat --workers "$WORKERS" --out data \
                 > data-c2.log 2>&1 || code=1 ;;
        esac
      done
      echo $code > data.done ;;
    train)
      python scripts/chartop_train.py --plan plan.json --data data --runs runs --only "$ONLY" \
        > train.log 2>&1
      echo $? > train.done ;;
  esac
done
touch ALL.done

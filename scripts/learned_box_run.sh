#!/usr/bin/env bash
# The learned case on a rented box (demo step 8): install the pins, generate the
# registered layouts' data on the CPUs, train on the GPU.
#
#   WORKERS=24 MINUTES=75 setsid nohup bash learned_box_run.sh < /dev/null > run.log 2>&1 &
#
# Every stage writes <stage>.done with its exit code, and ALL.done at the end, so
# the laptop polls files rather than an ssh session.  The box gives itself a
# deadline: after DEADLINE_S seconds it STOPS itself (storage-only billing, the
# disk and the results survive), so a session that ends early cannot leave it
# billing at the GPU rate (the vault's memory "vastai-gpu-offload", W336).
set -u
cd "$(dirname "$0")"
export ATLAS_BUILD_REPO="$PWD/vendor"
export PYTHONPATH="$PWD"
export KMP_DUPLICATE_LIB_OK=TRUE
WORKERS="${WORKERS:-24}"
MINUTES="${MINUTES:-75}"
DEADLINE_S="${DEADLINE_S:-14400}"

# the container's id and key live in PID 1's environment, not in an ssh shell's
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^CONTAINER_(ID|API_KEY)=' | sed 's/^/export /')"
(
  python -m pip install -q vastai > deadline-pip.log 2>&1
  sleep "$DEADLINE_S"
  echo "deadline reached at $(date -u +%FT%TZ): stopping instance ${CONTAINER_ID:-?}"
  vastai stop instance "${CONTAINER_ID:-}" --api-key "${CONTAINER_API_KEY:-}"
) > deadline.log 2>&1 &
echo "deadline armed: ${DEADLINE_S}s, instance ${CONTAINER_ID:-unknown}" > deadline-armed.log

python -m pip install -q numpy==1.26.4 scipy==1.13.1 pydantic==2.12.5 -c constraints.txt \
  > pip.log 2>&1
echo $? > pip.done
python - > env.log 2>&1 <<'PY'
import os, platform, numpy, scipy, torch
print("python", platform.python_version(), "numpy", numpy.__version__, "scipy", scipy.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("cpus", os.cpu_count(), "affinity", len(os.sched_getaffinity(0)))
PY

python scripts/learned_data.py --set all --workers "$WORKERS" > data.log 2>&1
echo $? > data.done
cp out/learned-case/data/manifest.json out/learned-case/data-manifest.json 2>/dev/null

python scripts/learned_train.py --minutes "$MINUTES" > train.log 2>&1
echo $? > train.done
touch ALL.done

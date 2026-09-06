#!/usr/bin/env bash
# One command, macOS or Linux: build a virtual environment beside this file,
# install everything the demo needs into it at pinned versions, self-test, and
# start the server.
#
#   ./run.sh                        open http://127.0.0.1:8012/
#   ./run.sh --open                 ... and open a browser
#   ./run.sh --check                self-test only, no server
#   ./run.sh --reinstall            throw the venv away and build it again
#
# Nothing is installed outside ./.venv and nothing on your system python is
# touched. Delete the .venv directory to undo everything this does.
#
# Nothing is downloaded at run time. There is no checkpoint: every expert here
# is a classical solver, and the beat that offers a frozen neural operator does
# it by handing the compiler that operator's DECLARATION, which is where the
# verdict comes from.
set -euo pipefail
cd "$(dirname "$0")"

TORCH="torch==2.7.1"

# ---- a python we can actually use -----------------------------------------
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
      if "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null; then
        PY="$c"; break
      fi
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "No usable python found. This bundle pins its dependencies to versions"
  echo "published for Python 3.10, 3.11 and 3.12."
  echo "  macOS:  brew install python@3.12     (or python.org/downloads)"
  echo "  Linux:  sudo apt install python3.12 python3.12-venv"
  echo "Then:  PYTHON=/path/to/python3.12 ./run.sh"
  exit 1
fi

if ! "$PY" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)'; then
  echo "Found $("$PY" --version), and this bundle needs Python 3.10-3.12:"
  echo "numpy 1.26.4 and the other pins have no wheels outside that range."
  echo "Set PYTHON=/path/to/python3.12 and run again."
  exit 1
fi

for a in "$@"; do
  if [ "$a" = "--reinstall" ]; then rm -rf .venv; fi
done

VENV=".venv"
if [ ! -d "$VENV" ]; then
  echo "==> creating $VENV with $("$PY" --version)"
  "$PY" -m venv "$VENV" || {
    echo "venv failed. On Debian/Ubuntu: sudo apt install python3-venv"; exit 1; }
fi
VPY="$VENV/bin/python"

# ---- the install, pinned, and torch chosen for THIS machine ----------------
# A stamp file, so a second run starts in a second instead of re-resolving pip.
if [ ! -f "$VENV/.deps-ok" ] || [ requirements.txt -nt "$VENV/.deps-ok" ]; then
  echo "==> installing dependencies (a few minutes the first time; torch is large)"
  "$VPY" -m pip install --upgrade pip >/dev/null

  if [ "$(uname -s)" = "Darwin" ]; then
    # macOS wheels on PyPI are CPU/MPS; there is nothing to choose. The demo
    # does not use MPS: the composed column is float64 and MPS has no float64.
    echo "    torch: macOS, CPU wheel from PyPI"
    "$VPY" -m pip install "$TORCH"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "    torch: NVIDIA driver detected, CUDA 12.6 wheel"
    "$VPY" -m pip install "$TORCH" --index-url https://download.pytorch.org/whl/cu126
  else
    echo "    torch: no NVIDIA driver, CPU-only wheel (a tenth of the size)"
    "$VPY" -m pip install "$TORCH" --index-url https://download.pytorch.org/whl/cpu
  fi

  "$VPY" -m pip install -r requirements.txt
  touch "$VENV/.deps-ok"
fi

for a in "$@"; do
  if [ "$a" = "--check" ]; then exec "$VPY" run.py --check; fi
done

echo "==> self-test"
"$VPY" run.py --check

echo
echo "==> starting the demo"
exec "$VPY" run.py "$@"

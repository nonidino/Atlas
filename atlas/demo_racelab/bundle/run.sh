#!/usr/bin/env bash
# One command, macOS or Linux: build a virtual environment beside this file,
# install everything the demo needs into it at pinned versions, self-test, and
# start the server.
#
#   ./run.sh                        then open http://127.0.0.1:8013/
#   ./run.sh --open                 ... and open a browser
#   ./run.sh --check                self-test only, no server
#   ./run.sh --reinstall            throw the venv away and build it again
#
# Nothing is installed outside ./.venv and nothing on your system python is
# touched. Delete the .venv directory to undo everything this does.
#
# The Poseidon-T checkpoint is IN this checkout (vendor/hf-cache, 83 MB), so
# nothing is downloaded at run time. Its weights are CC-BY-NC-4.0: research use
# only -- see vendor/POSEIDON-T-LICENCE.md.
#
# Exit codes of the self-test, which this script acts on:
#   0  both columns marched
#   3  the CLASSICAL column marched and the learned expert is not available here
#      (usually: scOT could not be fetched). The server still starts, and the
#      page greys the learned switch out with the reason.
#   anything else: something is broken, and the server is not started.
set -euo pipefail
cd "$(dirname "$0")"

TORCH="torch==2.7.1"
PIP="pip==24.3.1"
SCOT="https://github.com/camlab-ethz/poseidon/archive/b8fa28f59bd7f7673323f28d11a12c6f3a215c61.zip"

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
  if ! "$PY" -m venv "$VENV"; then
    rm -rf "$VENV"
    PV="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    echo
    echo "Could not create a virtual environment with $PY."
    echo "Debian and Ubuntu ship python without its venv module:"
    echo "    sudo apt install python${PV}-venv"
    echo "or point this script at another interpreter:"
    echo "    PYTHON=/path/to/python3.12 ./run.sh"
    exit 1
  fi
fi
VPY="$VENV/bin/python"

# ---- the install, pinned ----------------------------------------------------
# A stamp file, so a second run starts in a second instead of re-resolving pip.
#
# `-c constraints.txt` pins what the top-level packages pull in. It is spelled
# out twice below rather than kept in an array, because macOS still ships bash
# 3.2, where expanding an EMPTY array under `set -u` is an "unbound variable"
# error -- which would kill this script on a Mac before it installed anything.
C1=""; C2=""
if [ -f constraints.txt ]; then C1="-c"; C2="constraints.txt"; fi
if [ ! -f "$VENV/.deps-ok" ] || [ requirements.txt -nt "$VENV/.deps-ok" ] \
   || { [ -f constraints.txt ] && [ constraints.txt -nt "$VENV/.deps-ok" ]; }; then
  echo "==> installing dependencies (a few minutes the first time; torch is large)"
  "$VPY" -m pip install --upgrade "$PIP" >/dev/null 2>&1 \
    || echo "    (could not install $PIP; continuing with $("$VPY" -m pip --version))"

  if [ "$(uname -s)" = "Darwin" ]; then
    # macOS wheels on PyPI are CPU/MPS; there is nothing to choose. The demo
    # does not use MPS: the composed march is float64 and MPS has no float64.
    echo "    torch: macOS, wheel from PyPI"
    "$VPY" -m pip install "$TORCH" ${C1:+"$C1"} ${C2:+"$C2"}
  else
    # The CPU-only wheel, whether or not there is a GPU: nothing in this demo
    # runs on one, so the CUDA wheel would be a few gigabytes never used.
    # PyPI stays reachable as an extra index so that every version the
    # constraints pin resolves, and `2.7.1+cpu` still wins over PyPI's `2.7.1`
    # because a local version label sorts after the public one.
    echo "    torch: CPU-only wheel (this demo does not use a GPU)"
    "$VPY" -m pip install "$TORCH" ${C1:+"$C1"} ${C2:+"$C2"} \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple
  fi

  "$VPY" -m pip install -r requirements.txt ${C1:+"$C1"} ${C2:+"$C2"}
  touch "$VENV/.deps-ok"
fi

# The Poseidon model class, from a pinned source archive. --no-deps because its
# own pins are years old and would tear down everything above; every package it
# actually imports (torch, transformers) is pinned in ours.
#
# NOT fatal. It is the only thing this bundle fetches from outside pip's own
# index, and without it the classical column still runs.
if [ ! -f "$VENV/.scot-ok" ]; then
  echo "==> installing scOT (the Poseidon-T model class) from its pinned commit"
  if "$VPY" -m pip install --no-deps --retries 1 --timeout 30 "$SCOT"; then
    touch "$VENV/.scot-ok"
  else
    echo
    echo "    Could not install scOT from github.com/camlab-ethz/poseidon."
    echo "    The CLASSICAL column does not need it and will still run; the"
    echo "    learned switch will be greyed out, with this reason on the page."
    echo "    The next ./run.sh tries the fetch again."
    echo
  fi
fi

for a in "$@"; do
  if [ "$a" = "--check" ]; then exec "$VPY" run.py --check; fi
done

echo "==> self-test"
set +e
"$VPY" run.py --check
RC=$?
set -e
if [ "$RC" -eq 3 ]; then
  echo
  echo "==> the learned column is NOT available here; starting the classical column"
elif [ "$RC" -ne 0 ]; then
  echo
  echo "The self-test failed (exit $RC), so the server is not started."
  exit "$RC"
fi

echo
echo "==> starting the demo"
exec "$VPY" run.py "$@"

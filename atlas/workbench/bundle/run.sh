#!/usr/bin/env bash
# One command on macOS or Linux: build a virtual environment beside this file,
# install the workbench's packages into it at pinned versions, self-test once,
# and serve the page in your browser.
#
#   ./run.sh                    build, self-test, serve, open a browser
#   ./run.sh --no-open          ... without opening a browser
#   ./run.sh --port 8031        another port (default: the first free from 8020)
#   ./run.sh --check            the self-test alone
#   ./run.sh --reinstall        throw .venv away and build it again
#   ./run.sh --no-torch         skip torch (the learned case), about 200 MB
#   ./run.sh --no-gmsh          skip Gmsh (File > Import geometry from Gmsh)
#
# Nothing is installed outside ./.venv and nothing on your system Python is
# touched. Delete the .venv directory to undo everything this does.
#
# Exit codes of the self-test, which this script acts on:
#   0  every type ran, the solver came from vendor/, and the page answers
#   3  the same, and an optional package (torch, Gmsh) is absent: the server
#      still starts
#   anything else: something is broken, and the server is not started.
#
# Written for the bash macOS ships, 3.2: no arrays (an empty one is an "unbound
# variable" under set -u there), and every "$@" spelled ${1+"$@"}, which is
# empty, not unbound, when no argument was given. It has run on Linux; it has
# NOT yet run on a Mac.
set -euo pipefail
cd "$(dirname "$0")"

TORCH="torch==2.7.1"
GMSH="gmsh==4.15.0"
PIP="pip==24.3.1"

REINSTALL=0; NO_TORCH=0; NO_GMSH=0; CHECK=0
for a in ${1+"$@"}; do
  case "$a" in
    --reinstall) REINSTALL=1 ;;
    --no-torch) NO_TORCH=1 ;;
    --no-gmsh) NO_GMSH=1 ;;
    --check) CHECK=1 ;;
  esac
done

# ---- a Python we can use: 3.10, 3.11 or 3.12 -------------------------------
ok_py() {
  "$1" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' \
    >/dev/null 2>&1
}
PY="${PYTHON:-}"
if [ -n "$PY" ]; then
  if ! ok_py "$PY"; then
    echo "PYTHON=$PY is not Python 3.10, 3.11 or 3.12."
    exit 1
  fi
else
  # PATH first, then where Homebrew (Apple silicon, Intel) and python.org put
  # their interpreters, which a fresh terminal does not always have on PATH
  for c in python3.12 python3.11 python3.10 \
           /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 \
           /opt/homebrew/bin/python3.10 \
           /usr/local/bin/python3.12 /usr/local/bin/python3.11 /usr/local/bin/python3.10 \
           /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
           /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
           /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 \
           python3 python; do
    if command -v "$c" >/dev/null 2>&1 && ok_py "$c"; then PY="$c"; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "No Python 3.10, 3.11 or 3.12 was found. The workbench's packages are"
  echo "pinned to versions published for those three."
  if [ "$(uname -s)" = "Darwin" ]; then
    echo "  macOS:  brew install python@3.12"
    echo "          or the macOS installer from https://www.python.org/downloads/"
  else
    echo "  Debian, Ubuntu:  sudo apt install python3.12 python3.12-venv"
  fi
  echo "Then run ./run.sh again, or point at one:  PYTHON=/path/to/python3.12 ./run.sh"
  exit 1
fi

if [ "$REINSTALL" = 1 ]; then rm -rf .venv; fi

VENV=".venv"
VPY="$VENV/bin/python"
if [ ! -x "$VPY" ]; then
  echo "==> creating $VENV with $("$PY" --version 2>&1)"
  if ! "$PY" -m venv "$VENV"; then
    rm -rf "$VENV"
    PV="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    echo
    echo "Could not create a virtual environment with $PY."
    echo "Debian and Ubuntu ship Python without its venv module:"
    echo "    sudo apt install python${PV}-venv"
    echo "or point this script at another interpreter:"
    echo "    PYTHON=/path/to/python3.12 ./run.sh"
    exit 1
  fi
fi

# `-c constraints.txt` pins what the top-level packages pull in. It is passed as
# two plain variables rather than an array: see the note at the top.
C1=""; C2=""
if [ -f constraints.txt ]; then C1="-c"; C2="constraints.txt"; fi

# ---- the packages, pinned ------------------------------------------------------
if [ ! -f "$VENV/.deps-ok" ] || [ requirements.txt -nt "$VENV/.deps-ok" ] \
   || { [ -f constraints.txt ] && [ constraints.txt -nt "$VENV/.deps-ok" ]; }; then
  echo "==> installing the workbench's packages (a few minutes the first time)"
  "$VPY" -m pip install --upgrade "$PIP" >/dev/null 2>&1 \
    || echo "    (could not install $PIP; continuing with $("$VPY" -m pip --version))"
  if ! "$VPY" -m pip install -r requirements.txt ${C1:+"$C1"} ${C2:+"$C2"}; then
    echo
    echo "The packages could not be installed, so nothing was started. The lines"
    echo "above say which one; run ./run.sh again once that is fixed."
    exit 1
  fi
  touch "$VENV/.deps-ok"
  rm -f "$VENV/.selftest-ok"
fi

# ---- torch, CPU-only and optional ----------------------------------------------
# Only the learned case needs it. If it cannot be installed, everything else
# still runs, and the next ./run.sh tries again (or pass --no-torch).
if [ "$NO_TORCH" = 0 ] && [ ! -f "$VENV/.torch-ok" ]; then
  TORCH_OK=0
  if [ "$(uname -s)" = "Darwin" ]; then
    # torch 2.7.1 publishes macOS wheels for Apple silicon (arm64) only. What
    # matters is the INTERPRETER's architecture: an Intel Python under Rosetta
    # on an Apple-silicon Mac cannot use the arm64 wheel either.
    PARCH="$("$VPY" -c 'import platform; print(platform.machine())')"
    if [ "$PARCH" = "arm64" ]; then
      echo "==> installing torch from PyPI (optional: the learned case)"
      if "$VPY" -m pip install "$TORCH" ${C1:+"$C1"} ${C2:+"$C2"}; then TORCH_OK=1; fi
    else
      echo
      echo "    torch is skipped: $TORCH publishes macOS wheels for Apple silicon"
      echo "    only, and this Python runs as '$PARCH'. Only the learned case needs"
      echo "    it. On an Apple-silicon Mac, a native Python fixes this:"
      echo "        PYTHON=/opt/homebrew/bin/python3.12 ./run.sh --reinstall"
      echo
    fi
  else
    # The CPU-only wheel even when there is a GPU: nothing here runs on one.
    # PyPI stays reachable as an extra index so the constraints resolve, and
    # 2.7.1+cpu still wins over PyPI's 2.7.1 (a local label sorts after it).
    echo "==> installing torch, CPU-only (optional: the learned case; about 200 MB)"
    if "$VPY" -m pip install "$TORCH" ${C1:+"$C1"} ${C2:+"$C2"} \
         --index-url https://download.pytorch.org/whl/cpu \
         --extra-index-url https://pypi.org/simple; then TORCH_OK=1; fi
  fi
  if [ "$TORCH_OK" = 1 ]; then
    touch "$VENV/.torch-ok"
    rm -f "$VENV/.selftest-ok"
  else
    echo "    torch is not installed. Only the learned case needs it; every other"
    echo "    part of the workbench runs without it."
  fi
fi

# ---- Gmsh, from PyPI and optional ------------------------------------------------
# Gmsh is GPL: it is installed on this machine from PyPI, never shipped in this
# folder. Only File > Import geometry from Gmsh uses it.
if [ "$NO_GMSH" = 0 ] && [ ! -f "$VENV/.gmsh-ok" ]; then
  echo "==> installing Gmsh (optional: File > Import geometry from Gmsh)"
  if "$VPY" -m pip install "$GMSH" ${C1:+"$C1"} ${C2:+"$C2"}; then
    touch "$VENV/.gmsh-ok"
    rm -f "$VENV/.selftest-ok"
  else
    echo "    Gmsh could not be installed. Only File > Import geometry from Gmsh"
    echo "    needs it. The next ./run.sh tries again, or pass --no-gmsh."
  fi
fi

# ---- the self-test, once; then the page ------------------------------------------
if [ "$CHECK" = 1 ]; then exec "$VPY" run.py --check; fi

if [ ! -f "$VENV/.selftest-ok" ]; then
  echo "==> self-test (about a minute, once)"
  set +e
  "$VPY" run.py --check
  RC=$?
  set -e
  if [ "$RC" -ne 0 ] && [ "$RC" -ne 3 ]; then
    echo
    echo "The self-test failed (exit $RC), so the server is not started."
    exit "$RC"
  fi
  touch "$VENV/.selftest-ok"
fi

echo
echo "==> starting the workbench (Ctrl-C to stop)"
exec "$VPY" run.py ${1+"$@"}

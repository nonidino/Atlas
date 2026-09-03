@echo off
REM One command on Windows: build a virtual environment beside this file,
REM install everything the demo needs into it at pinned versions, self-test,
REM and start the server.
REM
REM   run.cmd                         then open http://127.0.0.1:8011/
REM   run.cmd --domain large --turbines 25
REM   run.cmd --check                 self-test only, no server
REM   run.cmd --reinstall             throw the venv away and build it again
REM
REM Nothing is installed outside .venv. Delete that folder to undo everything.
REM The Poseidon-T checkpoint is IN this checkout (vendor\hf-cache, 83 MB), so
REM nothing is downloaded at run time. Its weights are CC-BY-NC-4.0: research
REM use only -- see vendor\POSEIDON-T-LICENCE.md.
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "TORCH=torch==2.7.1"
set "SCOT=https://github.com/camlab-ethz/poseidon/archive/b8fa28f59bd7f7673323f28d11a12c6f3a215c61.zip"

set "PY=%PYTHON%"
if "%PY%"=="" (
  for %%v in (3.12 3.11 3.10) do (
    if "!PY!"=="" (
      py -%%v -c "import sys" >nul 2>&1 && set "PY=py -%%v"
    )
  )
)
if "%PY%"=="" (
  where python >nul 2>&1 && set "PY=python"
)
if "%PY%"=="" (
  echo No python found. Install Python 3.12 from python.org/downloads
  echo and tick "Add python.exe to PATH".
  exit /b 1
)

%PY% -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)"
if errorlevel 1 (
  echo This bundle pins its dependencies to versions published for Python
  echo 3.10, 3.11 and 3.12, and the interpreter found is outside that range.
  echo Install Python 3.12 and run again, or set PYTHON to one.
  exit /b 1
)

echo %* | findstr /C:"--reinstall" >nul
if not errorlevel 1 (
  if exist ".venv" rmdir /s /q ".venv"
)

if not exist ".venv" (
  echo ==^> creating .venv
  %PY% -m venv .venv
  if errorlevel 1 exit /b 1
)
set "VPY=.venv\Scripts\python.exe"

if not exist ".venv\.deps-ok" (
  echo ==^> installing dependencies ^(a few minutes the first time; torch is large^)
  "%VPY%" -m pip install --upgrade pip >nul

  REM On Windows the PyPI torch wheel is CPU-only, so a CUDA machine needs the
  REM pytorch index explicitly. Anything else gets the CPU wheel by default.
  where nvidia-smi >nul 2>&1
  if errorlevel 1 (
    echo     torch: no NVIDIA driver, CPU-only wheel
    "%VPY%" -m pip install %TORCH% --index-url https://download.pytorch.org/whl/cpu
  ) else (
    echo     torch: NVIDIA driver detected, CUDA wheel
    "%VPY%" -m pip install %TORCH% --index-url https://download.pytorch.org/whl/cu126
  )
  if errorlevel 1 exit /b 1

  "%VPY%" -m pip install -r requirements.txt
  if errorlevel 1 exit /b 1

  echo ==^> installing scOT ^(the Poseidon-T model class^) from its pinned commit
  "%VPY%" -m pip install --no-deps "%SCOT%"
  if errorlevel 1 (
    echo.
    echo Could not install scOT from github.com/camlab-ethz/poseidon.
    echo That archive is the ONLY thing this bundle fetches at install time;
    echo the checkpoint itself is already in vendor\hf-cache. With no network
    echo you can still run the classical column:
    echo     run.cmd --expert reference_exposed
    exit /b 1
  )
  echo ok> ".venv\.deps-ok"
)

echo %* | findstr /C:"--check" >nul
if not errorlevel 1 (
  "%VPY%" run.py --check
  exit /b %errorlevel%
)

echo ==^> self-test
"%VPY%" run.py --check
if errorlevel 1 exit /b 1

echo.
echo ==^> starting the demo
"%VPY%" run.py %*

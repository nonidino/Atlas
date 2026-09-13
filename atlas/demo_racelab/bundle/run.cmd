@echo off
REM One command on Windows: build a virtual environment beside this file,
REM install everything the demo needs into it at pinned versions, self-test,
REM and start the server.
REM
REM   .\run.cmd                       then open http://127.0.0.1:8013/
REM   .\run.cmd --open                ... and open a browser
REM   .\run.cmd --check               self-test only, no server
REM   .\run.cmd --reinstall           throw the venv away and build it again
REM
REM The leading .\ is for PowerShell, which will not run a program from the
REM current directory without a path; in cmd.exe either form works.
REM
REM Nothing is installed outside .venv. Delete that folder to undo everything.
REM The Poseidon-T checkpoint is IN this checkout (vendor\hf-cache, 83 MB), so
REM nothing is downloaded at run time. Its weights are CC-BY-NC-4.0: research
REM use only -- see vendor\POSEIDON-T-LICENCE.md.
REM
REM Exit codes of the self-test, which this script acts on:
REM   0  both columns marched
REM   3  the CLASSICAL column marched and the learned expert is not available
REM      here (usually: scOT could not be fetched). The server still starts and
REM      the page greys the learned switch out with the reason.
REM   anything else: something is broken, and the server is not started.
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "TORCH=torch==2.7.1"
set "PIPPIN=pip==24.3.1"
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

set "CONS="
if exist "constraints.txt" set "CONS=-c constraints.txt"

if exist ".venv\.deps-ok" goto deps_done
echo ==^> installing dependencies ^(a few minutes the first time; torch is large^)
"%VPY%" -m pip install --upgrade %PIPPIN% >nul 2>&1
if errorlevel 1 echo     ^(could not install %PIPPIN%; continuing with the venv's own pip^)

REM The CPU-only wheel whether or not there is a GPU: nothing in this demo runs
REM on one, so the CUDA wheel would be a few gigabytes never used.
echo     torch: CPU-only wheel ^(this demo does not use a GPU^)
"%VPY%" -m pip install %TORCH% %CONS% --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
if errorlevel 1 exit /b 1

"%VPY%" -m pip install -r requirements.txt %CONS%
if errorlevel 1 exit /b 1
echo ok> ".venv\.deps-ok"
:deps_done

REM scOT, the Poseidon model class, from its pinned commit. NOT fatal: it is
REM the only thing fetched from outside pip's own index, and the classical
REM column does not need it.
if exist ".venv\.scot-ok" goto scot_done
echo ==^> installing scOT ^(the Poseidon-T model class^) from its pinned commit
"%VPY%" -m pip install --no-deps --retries 1 --timeout 30 "%SCOT%"
if errorlevel 1 goto scot_failed
echo ok> ".venv\.scot-ok"
goto scot_done
:scot_failed
echo.
echo     Could not install scOT from github.com/camlab-ethz/poseidon.
echo     The CLASSICAL column does not need it and will still run; the
echo     learned switch will be greyed out, with this reason on the page.
echo     The next .\run.cmd tries the fetch again.
echo.
:scot_done

echo %* | findstr /C:"--check" >nul
if not errorlevel 1 (
  "%VPY%" run.py --check
  exit /b !errorlevel!
)

echo ==^> self-test
"%VPY%" run.py --check
set "RC=%errorlevel%"
if "%RC%"=="0" goto start
if "%RC%"=="3" (
  echo.
  echo ==^> the learned column is NOT available here; starting the classical column
  goto start
)
echo.
echo The self-test failed ^(exit %RC%^), so the server is not started.
exit /b %RC%

:start
echo.
echo ==^> starting the demo
"%VPY%" run.py %*

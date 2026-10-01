@echo off
REM One command on Windows: build a virtual environment beside this file,
REM install the workbench's packages into it at pinned versions, self-test once,
REM and serve the page in your browser.
REM
REM   .\run.cmd                   build, self-test, serve, open a browser
REM   .\run.cmd --no-open         ... without opening a browser
REM   .\run.cmd --port 8031       another port (default: the first free from 8020)
REM   .\run.cmd --check           the self-test alone
REM   .\run.cmd --reinstall       throw .venv away and build it again
REM   .\run.cmd --no-torch        skip torch (the learned case), about 200 MB
REM   .\run.cmd --no-gmsh         skip Gmsh (File > Import geometry from Gmsh)
REM
REM The leading .\ is for PowerShell, which will not run a program from the
REM current directory without a path; in cmd.exe either form works.
REM
REM Nothing is installed outside .venv. Delete that folder to undo everything.
REM
REM Exit codes of the self-test, which this script acts on:
REM   0  every type ran, the solver came from vendor\, and the page answers
REM   3  the same, and an optional package (torch, Gmsh) is absent: the server
REM      still starts
REM   anything else: something is broken, and the server is not started.
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "TORCH=torch==2.7.1"
set "GMSH=gmsh==4.15.0"
set "PIPPIN=pip==24.3.1"

REM ---- a Python we can use: 3.10, 3.11 or 3.12 --------------------------------
REM Each candidate is asked its version, so the Microsoft Store's placeholder
REM python.exe (which only opens the Store) is passed over, not chosen.
set "PY="
if defined PYTHON (
  call :try_py "!PYTHON!"
  if not defined PY (
    echo PYTHON is set to !PYTHON!, which is not Python 3.10, 3.11 or 3.12.
    exit /b 1
  )
)
if not defined PY call :try_py py -3.12
if not defined PY call :try_py py -3.11
if not defined PY call :try_py py -3.10
if not defined PY call :try_py python
if not defined PY (
  echo No Python 3.10, 3.11 or 3.12 was found. The workbench's packages are
  echo pinned to versions published for those three.
  echo   Install Python 3.12 from https://www.python.org/downloads/ and tick
  echo   "Add python.exe to PATH", then run .\run.cmd again.
  echo   Or point at one:  set PYTHON=C:\path\to\python.exe
  exit /b 1
)

echo %* | findstr /C:"--reinstall" >nul
if not errorlevel 1 (
  if exist ".venv" rmdir /s /q ".venv"
)

if not exist ".venv\Scripts\python.exe" (
  echo ==^> creating .venv with !PY!
  %PY% -m venv .venv
  if errorlevel 1 (
    echo Could not create a virtual environment with !PY!.
    exit /b 1
  )
)
set "VPY=.venv\Scripts\python.exe"

set "CONS="
if exist "constraints.txt" set "CONS=-c constraints.txt"

REM ---- the packages, pinned ----------------------------------------------------
if exist ".venv\.deps-ok" goto deps_done
echo ==^> installing the workbench's packages ^(a few minutes the first time^)
"%VPY%" -m pip install --upgrade %PIPPIN% >nul 2>&1
if errorlevel 1 echo     ^(could not install %PIPPIN%; continuing with the venv's own pip^)
"%VPY%" -m pip install -r requirements.txt %CONS%
if errorlevel 1 (
  echo.
  echo The packages could not be installed, so nothing was started. The lines
  echo above say which one; run .\run.cmd again once that is fixed.
  exit /b 1
)
echo ok> ".venv\.deps-ok"
if exist ".venv\.selftest-ok" del ".venv\.selftest-ok"
:deps_done

REM ---- torch, CPU-only and optional ----------------------------------------------
REM Only the learned case needs it. The CPU-only wheel even when there is a GPU:
REM nothing here runs on one, and the CUDA wheel is gigabytes never used. If it
REM cannot be installed, everything else still runs and the next run tries again.
echo %* | findstr /C:"--no-torch" >nul
if not errorlevel 1 goto torch_done
if exist ".venv\.torch-ok" goto torch_done
echo ==^> installing torch, CPU-only ^(optional: the learned case; about 200 MB^)
"%VPY%" -m pip install %TORCH% %CONS% --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
if errorlevel 1 goto torch_failed
echo ok> ".venv\.torch-ok"
if exist ".venv\.selftest-ok" del ".venv\.selftest-ok"
goto torch_done
:torch_failed
echo.
echo     torch could not be installed. Only the learned case needs it; every
echo     other part of the workbench runs without it. The next .\run.cmd tries
echo     again, or pass --no-torch to stop trying.
echo.
:torch_done

REM ---- Gmsh, from PyPI and optional ------------------------------------------------
REM Gmsh is GPL: it is installed on this machine from PyPI, never shipped in this
REM folder. Only File > Import geometry from Gmsh uses it.
echo %* | findstr /C:"--no-gmsh" >nul
if not errorlevel 1 goto gmsh_done
if exist ".venv\.gmsh-ok" goto gmsh_done
echo ==^> installing Gmsh ^(optional: File ^> Import geometry from Gmsh^)
"%VPY%" -m pip install %GMSH% %CONS%
if errorlevel 1 goto gmsh_failed
echo ok> ".venv\.gmsh-ok"
if exist ".venv\.selftest-ok" del ".venv\.selftest-ok"
goto gmsh_done
:gmsh_failed
echo.
echo     Gmsh could not be installed. Only File ^> Import geometry from Gmsh
echo     needs it. The next .\run.cmd tries again, or pass --no-gmsh.
echo.
:gmsh_done

REM ---- the self-test, once; then the page ------------------------------------------
echo %* | findstr /C:"--check" >nul
if not errorlevel 1 (
  "%VPY%" run.py --check
  exit /b !errorlevel!
)

if exist ".venv\.selftest-ok" goto serve
echo ==^> self-test ^(about a minute, once^)
"%VPY%" run.py --check
set "RC=!errorlevel!"
if "!RC!"=="0" goto passed
if "!RC!"=="3" goto passed
echo.
echo The self-test failed ^(exit !RC!^), so the server is not started.
exit /b !RC!
:passed
echo ok> ".venv\.selftest-ok"

:serve
echo.
echo ==^> starting the workbench ^(Ctrl-C to stop^)
"%VPY%" run.py %*
exit /b !errorlevel!

REM ---- %* is a command that starts a Python; keep it if it is 3.10 to 3.12 ----
:try_py
%* -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=%*"
goto :eof

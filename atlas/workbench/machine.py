"""What the machine was doing when a run was timed.

A timing is a fact about one machine in one state, so every run records the
state beside its numbers rather than trusting it:

* **the power source.**  On battery this laptop has slept mid-suite and, awake,
  run 4.4x slower at 1.4 GHz (the vault's memory "Standby suspends a suite"),
  so a ratio measured on battery is recorded as such.  Read from the Win32
  ``GetSystemPowerStatus`` call through ctypes: instant, and no subprocess.
* **other Python processes.**  A stray process from an earlier run has
  contaminated a timing here before.  ``tasklist`` (not ``ps``, which misses
  detached Windows processes) lists them; the command lines come from CIM when
  it answers within a few seconds.  The owner's own phone-remote server may be
  among them; it is recorded, not stopped.
* **keep-awake.**  `keep_awake` asks Windows not to sleep while the calling
  thread runs, exactly as W346's driver does for its own process; the request
  lapses when the thread clears it or exits, and no power setting is changed.
"""

from __future__ import annotations

import csv
import datetime
import io
import os
import platform
import subprocess
import sys

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def power_status() -> dict:
    """``{"ac": True/False/None, "battery_percent": int|None, "source": ...}``."""
    if sys.platform != "win32":
        return {"ac": None, "battery_percent": None, "source": "not Windows: not read"}
    import ctypes

    class _SPS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte),
                    ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong),
                    ("BatteryFullLifeTime", ctypes.c_ulong)]

    s = _SPS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
        return {"ac": None, "battery_percent": None, "source": "GetSystemPowerStatus failed"}
    ac = {0: False, 1: True}.get(int(s.ACLineStatus))
    pct = int(s.BatteryLifePercent)
    return {"ac": ac, "battery_percent": None if pct == 255 else pct,
            "source": "GetSystemPowerStatus"}


def other_python_processes(timeout: float = 6.0) -> list[dict]:
    """Every ``python.exe`` but this one: pid, memory, and the command line when known."""
    me = os.getpid()
    out: list[dict] = []
    if sys.platform != "win32":
        return out
    try:
        txt = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV",
                              "/NH"], capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return [{"pid": None, "note": "tasklist did not answer"}]
    for row in csv.reader(io.StringIO(txt)):
        if len(row) >= 5 and row[0].lower() == "python.exe":
            try:
                pid = int(row[1])
            except ValueError:
                continue
            if pid != me:
                out.append({"pid": pid, "memory": row[4]})
    if out:
        cmd = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
               "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
        try:
            txt = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                                 capture_output=True, text=True, timeout=timeout).stdout
            lines = dict(line.split("\t", 1) for line in txt.splitlines() if "\t" in line)
            for p in out:
                c = lines.get(str(p["pid"]))
                if c is not None:
                    p["command"] = c.strip()
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return out


def machine_record() -> dict:
    """Everything a timing needs beside it, read now."""
    import numpy
    import scipy
    return {
        "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "power": power_status(),
        "other_python_processes": other_python_processes(),
        "pid": os.getpid(),
    }


def keep_awake(on: bool) -> bool:
    """Ask Windows not to sleep while THIS thread runs (W346's call).  Returns
    whether the request was made; elsewhere it does nothing."""
    if sys.platform != "win32":
        return False
    import ctypes
    flags = _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if on else 0)
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))


__all__ = ["power_status", "other_python_processes", "machine_record", "keep_awake"]

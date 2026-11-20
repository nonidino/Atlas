"""Opt this process (or given ones of this user's) out of Windows power throttling (EcoQoS).

Windows schedules windowless background processes with EcoQoS, and on this laptop's Core
Ultra that parks them on the low-power cores: on 2026-11-19 a three-thread training job
started from the Claude app used 0.6 of a core at 1.4 GHz, and 2.4 cores once opted out.
A process may declare that it does not want its execution speed throttled
(``SetProcessInformation`` with ``ProcessPowerThrottling``, ``ControlMask`` EXECUTION_SPEED
and ``StateMask`` 0).  That is a property of the process, like its priority, not a system
setting.  Off Windows, a no-op.

    import winqos; winqos.opt_out()            # at the start of a long job
    python scripts/winqos.py PID [PID ...]     # for jobs already running
"""
from __future__ import annotations

import os
import sys


def opt_out(pid: int | None = None) -> bool:
    """True if the process (this one when ``pid`` is None) is now opted out."""
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class PPT(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k32.OpenProcess.restype = wintypes.HANDLE
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    own = pid is None
    h = k32.GetCurrentProcess() if own else k32.OpenProcess(0x0200, False, int(pid))
    if not h:
        return False
    s = PPT(1, 0x1, 0)
    ok = bool(k32.SetProcessInformation(wintypes.HANDLE(h), 4, ctypes.byref(s), ctypes.sizeof(s)))
    if not own:
        k32.CloseHandle(wintypes.HANDLE(h))
    return ok


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, "opted out" if opt_out(int(p)) else "failed")

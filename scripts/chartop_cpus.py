"""How many CPUs this process may really use, and so how many pool workers (Stage 0, W360).

    python scripts/chartop_cpus.py [--reserve N] [--explain]

``os.cpu_count()`` is the host's logical processors.  Inside a container it can be far
more than the container may use: the rented A100 box reported 192 while
``/sys/fs/cgroup/cpu.max`` held ``2304000 100000``, a quota of 23.04 CPUs, and the
generator, given 191 workers, took 4.2 s per example against 2.0 s with 63 workers on a
64-core box.  The count used here is the smallest of

- the cgroup v2 quota, ``cpu.max`` = ``<quota> <period>`` (``max`` = no quota);
- the cgroup v1 quota, ``cpu.cfs_quota_us`` / ``cpu.cfs_period_us`` (``-1`` = no quota);
- ``len(os.sched_getaffinity(0))``, where the platform has it (Linux; not Windows or macOS);
- ``os.cpu_count()``.

A fractional quota rounds down (23.04 -> 23) with a floor of 1.  A source that is absent,
unreadable or malformed is skipped, so Windows, with no cgroup files and no
``sched_getaffinity``, reads ``os.cpu_count()`` alone.  The files read are the ones at
the cgroup mount, which inside a container are the container's own; a quota set only on
an ancestor outside a cgroup namespace is not seen.

The command prints one integer, the available CPUs less ``--reserve`` (default 1, for the
process that collects the workers' results), at least 1 -- what ``chartop_box_run.sh``
takes as its default ``WORKERS``.  ``--explain`` prints every source as one JSON line.

The module imports nothing outside the standard library: the box's runner calls it before
numpy is installed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

CGROUP_ROOT = "/sys/fs/cgroup"
#: the v1 cpu controller is mounted under one of these names
V1_DIRS = ("cpu", "cpu,cpuacct", "cpuacct,cpu")
DEFAULT_PERIOD_US = 100_000


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="ascii", errors="replace") as fh:
            return fh.read().strip()
    except (OSError, ValueError):
        return None


def _quota_cpus(quota: str, period: str) -> int | None:
    """Whole CPUs of ``quota`` microseconds per ``period``; None when there is no quota
    (``max``, ``-1``, zero, or anything that is not a pair of integers)."""
    try:
        q, p = int(quota), int(period)
    except ValueError:
        return None
    if q <= 0 or p <= 0:
        return None
    return max(1, q // p)


def cgroup_v2_cpus(root: str | None = None) -> int | None:
    text = _read(os.path.join(root or CGROUP_ROOT, "cpu.max"))
    if not text:
        return None
    parts = text.split()
    if len(parts) == 1:                       # the period is optional in the kernel's syntax
        parts.append(str(DEFAULT_PERIOD_US))
    return _quota_cpus(parts[0], parts[1])


def cgroup_v1_cpus(root: str | None = None) -> int | None:
    for sub in V1_DIRS:
        quota = _read(os.path.join(root or CGROUP_ROOT, sub, "cpu.cfs_quota_us"))
        period = _read(os.path.join(root or CGROUP_ROOT, sub, "cpu.cfs_period_us"))
        if quota is not None and period is not None:
            return _quota_cpus(quota, period)
    return None


def affinity_cpus() -> int | None:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    try:
        return len(getter(0)) or None
    except OSError:
        return None


def sources(root: str | None = None) -> dict:
    """Every source's count, ``None`` where the source does not exist or says no limit."""
    return {"cgroup_v2": cgroup_v2_cpus(root), "cgroup_v1": cgroup_v1_cpus(root),
            "affinity": affinity_cpus(), "cpu_count": os.cpu_count()}


def available_cpus(root: str | None = None) -> int:
    known = [v for v in sources(root).values() if v is not None]
    return max(1, min(known)) if known else 1


def default_workers(reserve: int = 1, root: str | None = None) -> int:
    """A pool's size: the available CPUs less ``reserve`` for the process that collects
    the workers' results, and at least 1."""
    return max(1, available_cpus(root) - reserve)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reserve", type=int, default=1,
                    help="CPUs kept back from the pool (default 1)")
    ap.add_argument("--explain", action="store_true", help="print every source as one JSON line")
    args = ap.parse_args(argv)
    if args.explain:
        print(json.dumps({"sources": sources(), "available": available_cpus(),
                          "reserve": args.reserve, "workers": default_workers(args.reserve)}))
    else:
        print(default_workers(args.reserve))
    return 0


if __name__ == "__main__":
    sys.exit(main())

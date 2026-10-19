"""The one evaluation's registry and its checkpoints (Stage 0, W360).

`scripts/chartop_evaluate.py` judges a set of weights once and refuses to judge them twice.
``out/chartop/evaluated.json`` is that memory: ``{"<date>:<arm>": sha256}`` for every weights
file an evaluation has *begun* on.  It is written before the first stage runs, so an
evaluation cannot be started again after its numbers have been seen.  Until W360 that entry
was all there was, so a run that was interrupted (the laptop sleeps, a stage crashes) could
not be continued without editing the registry by hand, which is overriding its own refusal.

An evaluation now also keeps ``evaluation-<stamp>-checkpoints/`` beside its record:

- ``state.json``: the weights' sha256; the *identity* the evaluation began with (the gates
  module's sha256, the test split's size, the number of dual-bracket test pieces); the test
  split's sha256 once the split is complete; every session that worked on it; one entry per
  checkpoint (its sha256 and the session that made it); and, at the end, the record written;
- one file per finished unit of work, each written whole or not at all.

`begin` decides what a run may do with the weights it is given:

=======================================  ===============================================
the weights                              outcome
=======================================  ===============================================
never begun                              start (the state first, then the registry)
begun, checkpointed, not finished        refused; continued with ``resume=True``
begun, finished                          refused, always
begun before W360 (registry only)        refused, always: there is nothing to resume
only some of the arms begun              refused, always
``resume=True`` and never begun          refused: there is nothing to resume
=======================================  ===============================================

A resume continues the same evaluation or none.  The weights are the same by hash, the
identity is the same, the test split on disk is the same by hash (`Evaluation.confirm_test_split`),
and a checkpoint is used only if its file's sha256 is the one the state recorded; otherwise
its unit is computed again.  The finished record says which session made which unit.

One session at a time: nothing here locks the folder.

The module imports nothing outside the standard library.
"""
from __future__ import annotations

import datetime as dt
import glob
import hashlib
import json
import os
import re
import time
import zipfile

FORMAT = 1
STATE = "state.json"
REGISTRY = "evaluated.json"
RETRIES = 20
_NAME = re.compile(r"[A-Za-z0-9._-]+")


class Refused(Exception):
    """The evaluation will not begin, or resume; the message says why."""


# ---------------------------------------------------------------------------
# bytes: hashing, and reading and writing in a folder OneDrive syncs
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _retry(fn, errors=(PermissionError,)):
    """OneDrive holds a just-written file for a moment: reads and replaces of it raise
    PermissionError (the vault's memory "onedrive-holds-fresh-files").  A path that is not
    there will not appear by waiting, so those errors are raised at once."""
    for k in range(RETRIES):
        try:
            return fn()
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
            raise
        except errors:
            if k == RETRIES - 1:
                raise
            time.sleep(0.5 * (k + 1))


def read_bytes(path: str) -> bytes:
    def go():
        with open(path, "rb") as fh:
            return fh.read()
    return _retry(go)


def write_atomic(path: str, data: bytes) -> None:
    """The file is the old one or the new one, never half of either."""
    tmp = path + ".tmp"

    def go():
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    _retry(go, errors=(OSError,))


def npz_content_sha256(paths) -> str:
    """sha256 over the *contents* of .npz files: each file's name, then each member's name,
    size and bytes, in sorted order.  The zip container's timestamps are left out, so drawing
    the same split again gives the same hash.  Streams; holds no array in memory."""
    h = hashlib.sha256()
    for path in sorted(paths, key=os.path.basename):
        h.update(os.path.basename(path).encode("utf-8") + b"\0")
        with _retry(lambda p=path: zipfile.ZipFile(p)) as zf:
            for info in sorted(zf.infolist(), key=lambda i: i.filename):
                h.update(info.filename.encode("utf-8") + b"\0" + str(info.file_size).encode() + b"\0")
                with zf.open(info) as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
    return h.hexdigest()


def _dump(obj) -> bytes:
    return (json.dumps(obj, indent=1) + "\n").encode("utf-8")


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------


def read_registry(out: str) -> dict:
    path = os.path.join(out, REGISTRY)
    if not os.path.exists(path):
        return {}
    return json.loads(read_bytes(path).decode("utf-8"))


def _register(out: str, hashes: dict, date: str) -> None:
    """Append ``<date>:<arm>`` for every arm.  An earlier entry is never overwritten: a second
    evaluation of other weights on the same day gets ``<date>:<arm>#2``.  The file keeps the
    format it always had, so the entries already in it are unchanged."""
    seen = read_registry(out)
    for arm, h in hashes.items():
        key, k = f"{date}:{arm}", 1
        while key in seen:
            k += 1
            key = f"{date}:{arm}#{k}"
        seen[key] = h
    write_atomic(os.path.join(out, REGISTRY), json.dumps(seen, indent=1).encode("utf-8"))


def _states(out: str) -> list:
    found = []
    for path in sorted(glob.glob(os.path.join(glob.escape(out), "evaluation-*-checkpoints", STATE))):
        try:
            found.append((os.path.dirname(path), json.loads(read_bytes(path).decode("utf-8"))))
        except (OSError, ValueError) as e:
            raise Refused(f"{path} cannot be read ({e}); an evaluation may be in progress there") from e
    return found


def _finished(out: str, state: dict) -> bool:
    return bool(state.get("completed")) or os.path.exists(os.path.join(out, f"evaluation-{state['stamp']}.json"))


def begin(out: str, hashes: dict, *, resume: bool = False, identity: dict | None = None,
          session: dict | None = None, stamp: str | None = None) -> "Evaluation":
    """Open the evaluation of ``hashes`` ({arm: sha256}) in ``out``, or raise `Refused`."""
    identity = dict(identity or {})
    registry = read_registry(out)
    states = _states(out)
    seen = set(registry.values()) | {h for _d, st in states for h in st["weights"].values()}
    clash = sorted(a for a, h in hashes.items() if h in seen)
    exact = [(d, st) for d, st in states if st["weights"] == hashes]
    if not clash:
        if resume:
            raise Refused("--resume: none of these weights has been begun, so there is nothing to resume")
        return _start(out, hashes, identity, session or {}, stamp)
    if len(exact) > 1:
        raise Refused("more than one checkpoint folder holds exactly these weights: "
                      + ", ".join(d for d, _st in exact))
    if not exact:
        extra = ""
        if resume:
            near = [(d, st) for d, st in states if set(st["weights"].values()) & set(hashes.values())]
            extra = ("; no checkpoints exist for these weights, so there is nothing to resume" if not near else
                     f"; the checkpoints in {near[0][0]} were made for other weights in arm(s) "
                     + str(sorted(a for a in set(near[0][1]["weights"]) | set(hashes)
                                  if near[0][1]["weights"].get(a) != hashes.get(a))))
        raise Refused(f"weights already evaluated: {clash}; the evaluation runs once{extra}")
    directory, state = exact[0]
    if _finished(out, state):
        raise Refused(f"weights already evaluated: {clash}; the evaluation runs once "
                      f"(evaluation {state['stamp']} finished)")
    if not resume:
        raise Refused(f"weights already evaluated: {clash}: evaluation {state['stamp']}, begun "
                      f"{state['started']}, did not finish.  It cannot be started again; run the same "
                      f"command with --resume to continue it from its {len(state['checkpoints'])} "
                      f"checkpoints in {directory}")
    if state["identity"] != identity:
        diff = sorted(k for k in set(state["identity"]) | set(identity)
                      if state["identity"].get(k) != identity.get(k))
        raise Refused(f"--resume: this is not the evaluation that began {state['started']}; it differs in "
                      + ", ".join(f"{k} ({state['identity'].get(k)!r} then, {identity.get(k)!r} now)"
                                  for k in diff))
    unlisted = {a: h for a, h in hashes.items() if h not in registry.values()}
    if unlisted:                                      # a crash fell between the state and the registry
        _register(out, unlisted, state["started"][:10])
    return _continue(directory, state, session or {})


def _start(out: str, hashes: dict, identity: dict, session: dict, stamp: str | None) -> "Evaluation":
    now = dt.datetime.now().astimezone()
    stamp = stamp or now.strftime("%Y%m%d-%H%M%S")
    directory = os.path.join(out, f"evaluation-{stamp}-checkpoints")
    try:
        os.makedirs(directory)                        # and the output folder, if there is none yet
    except FileExistsError as e:
        raise Refused(f"{directory} already exists") from e
    state = {"format": FORMAT, "stamp": stamp, "started": now.isoformat(timespec="seconds"),
             "weights": dict(hashes), "identity": identity, "test": {"sha256": None},
             "sessions": [dict(session, index=0, began=now.isoformat(timespec="seconds"), resume=False)],
             "checkpoints": {}, "completed": None}
    ev = Evaluation(directory, state, resumed=False)
    ev._save_state()                                  # the state first: a crash between the two
    _register(out, hashes, now.date().isoformat())    # still leaves the weights marked as begun
    return ev


def _continue(directory: str, state: dict, session: dict) -> "Evaluation":
    n = len(state["sessions"])
    state["sessions"].append(dict(session, index=n, began=_now(), resume=True))
    ev = Evaluation(directory, state, resumed=True)
    ev._save_state()
    return ev


# ---------------------------------------------------------------------------
# one evaluation
# ---------------------------------------------------------------------------


class Evaluation:
    """The state and the checkpoints of one evaluation (a folder).  A checkpoint is a named
    file; its name is the unit of work it holds."""

    def __init__(self, directory: str, state: dict, resumed: bool):
        self.dir, self.state, self.resumed = directory, state, resumed
        self.session = len(state["sessions"]) - 1
        self.reused: list[str] = []          # checkpoints read back in this session

    @property
    def stamp(self) -> str:
        return self.state["stamp"]

    def _save_state(self) -> None:
        write_atomic(os.path.join(self.dir, STATE), _dump(self.state))

    def confirm_test_split(self, sha256: str) -> None:
        """Record the test split's hash the first time the split is complete; afterwards it
        must be the same split."""
        have = self.state["test"]["sha256"]
        if have is None:
            self.state["test"]["sha256"] = sha256
            self._save_state()
        elif have != sha256:
            raise Refused(f"the test split on disk (sha256 {sha256[:16]}...) is not the one this evaluation "
                          f"began with ({have[:16]}...); restore the original shards, which are drawn from "
                          f"the registered seeds")

    def load(self, name: str) -> bytes | None:
        """A checkpoint's bytes, or None when there is none, or its file is not the one the
        state recorded (the unit is then computed again)."""
        if not _NAME.fullmatch(name):
            raise ValueError("a checkpoint is named with letters, digits and ._-: " + name)
        info = self.state["checkpoints"].get(name)
        path = os.path.join(self.dir, name)
        if info is None or not os.path.exists(path):
            return None
        data = read_bytes(path)
        if sha256_bytes(data) != info["sha256"]:
            return None
        self.reused.append(name)
        return data

    def save(self, name: str, data: bytes) -> None:
        if not _NAME.fullmatch(name):
            raise ValueError("a checkpoint is named with letters, digits and ._-: " + name)
        write_atomic(os.path.join(self.dir, name), data)
        self.state["checkpoints"][name] = {"sha256": sha256_bytes(data), "bytes": len(data),
                                           "session": self.session}
        self._save_state()

    def json_stage(self, name: str, compute):
        """``compute()`` -> something JSON can hold, once.  A resumed run reads the unit back, a
        first run writes it, and both hand the rest of the script the same thing: what JSON makes
        of the value (tuples as lists, keys as strings)."""
        data = self.load(name)
        if data is None:
            data = _dump(compute())
            self.save(name, data)
        return json.loads(data.decode("utf-8"))

    def complete(self, record: str) -> None:
        self.state["completed"] = {"record": os.path.basename(record), "sha256": sha256_file(record),
                                   "at": _now()}
        self._save_state()

    def provenance(self) -> dict | None:
        """What the finished record says about how it was made; None for an evaluation that
        was never resumed."""
        if len(self.state["sessions"]) < 2:
            return None
        return {"sessions": self.state["sessions"],
                "units": {n: i["session"] for n, i in sorted(self.state["checkpoints"].items())}}

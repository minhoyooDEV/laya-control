"""Finite authorized queue, with a process-lifetime single-host accelerator lock."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .store import ControlError, Store, absolute, atomic_json, canonical, digest


def default_lock_root():
    return Path.home() / ".local" / "state" / "laya-control-locks"


@contextmanager
def lock_file(path, blocking=False):
    path = absolute(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield None
            return
        yield fd
    finally:
        # Do not LOCK_UN: inherited descriptors must retain the lock if this supervisor dies.
        os.close(fd)


def identity(pid):
    """OS birth identity, stable across exec and independent of mutable argv."""
    if sys.platform == "darwin":
        # Darwin proc_bsdinfo from sys/proc_info.h (PROC_PIDTBSDINFO = 3).
        class BsdInfo(ctypes.Structure):
            _fields_ = [
                ("ids", ctypes.c_uint32 * 12),
                ("comm", ctypes.c_char * 16),
                ("name", ctypes.c_char * 32),
                ("details", ctypes.c_uint32 * 6),
                ("seconds", ctypes.c_uint64),
                ("microseconds", ctypes.c_uint64),
            ]

        lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        info = BsdInfo()
        lib.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        size = lib.proc_pidinfo(int(pid), 3, 0, ctypes.byref(info), ctypes.sizeof(info))
        if size != ctypes.sizeof(info) or info.ids[1] == 5:  # SZOMB
            return None
        return f"darwin:{pid}:{info.seconds}:{info.microseconds}"
    if sys.platform.startswith("linux"):
        try:
            stat = Path(f"/proc/{int(pid)}/stat").read_text()
            fields = stat[stat.rfind(")") + 2 :].split()
            if fields[0] == "Z":
                return None
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            return f"linux:{pid}:{boot}:{fields[19]}"
        except (FileNotFoundError, ProcessLookupError):
            return None
    raise ControlError("process identity is only supported on macOS and Linux")


def artifact_hashes(directory):
    directory = Path(directory)
    result = {}
    for p in sorted(directory.rglob("*")):
        if p.is_symlink():
            raise ControlError("artifact symlinks are not allowed")
        if p.is_file():
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            result[str(p.relative_to(directory))] = h.hexdigest()
    if not result:
        raise ControlError("empty model output")
    return result


def enqueue(store: Store, snapshot_id, config, owner):
    snapshot = store.get_snapshot(snapshot_id)
    command = config.get("command")
    if (
        not owner
        or not isinstance(command, list)
        or not command
        or not all(isinstance(x, str) for x in command)
    ):
        raise ControlError("owner and command argv required")
    if not Path(command[0]).is_absolute():
        raise ControlError("executable must be absolute")
    if not any("{run_dir}" in x for x in command):
        raise ControlError("command must reference isolated {run_dir}")
    jid = uuid.uuid4().hex
    with store.tx() as db:
        store.validate_labels(snapshot, db)
        request_id = config.get("request_id")
        payload_hash = digest([snapshot_id, config, owner])
        if request_id:
            old = db.execute("SELECT * FROM job_requests WHERE id=?", (request_id,)).fetchone()
            if old:
                if old["payload_hash"] != payload_hash:
                    raise ControlError("job request ID reused with changed content")
                state = db.execute("SELECT state FROM jobs WHERE id=?", (old["job_id"],)).fetchone()[0]
                return {"id": old["job_id"], "state": state}
            db.execute("INSERT INTO job_requests VALUES(?,?,?)", (request_id, payload_hash, jid))
        db.execute(
            "INSERT INTO jobs(id,snapshot_id,config,state,owner,created) VALUES(?,?,?,?,?,?)",
            (jid, snapshot_id, canonical(config), "queued", owner, time.time()),
        )
        store.event(db, "queued", jid, {"snapshot_id": snapshot_id})
    return {"id": jid, "state": "queued"}


def cancel(store, run_id):
    with store.tx() as db:
        r = db.execute("SELECT * FROM jobs WHERE id=?", (run_id,)).fetchone()
        if not r:
            raise ControlError("unknown run")
        if r["state"] not in ("queued", "running", "evaluating"):
            return {"state": "already_terminal"}
        db.execute(
            "UPDATE jobs SET cancel=1,state=CASE WHEN state='queued' THEN 'cancelled' ELSE state END WHERE id=?",
            (run_id,),
        )
        store.event(db, "cancel_requested", run_id, {})
    # The supervising worker delivers signals after verifying its own child identity.
    return {"state": "cancel_requested"}


def work(store, lock_root=None, drain=False):
    lock_root = absolute(lock_root or default_lock_root())
    completed = []
    with lock_file(lock_root / "training.lock") as fd:
        if fd is None:
            return completed
        with store.connect(readonly=True) as db:
            queue = [
                r[0]
                for r in db.execute(
                    "SELECT id FROM jobs WHERE state='queued' AND cancel=0 ORDER BY created,id"
                )
            ]
        for jid in queue if drain else queue[:1]:
            with store.tx() as db:
                row = db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
                if row["state"] != "queued" or row["cancel"]:
                    continue
                snap = store.get_snapshot(row["snapshot_id"])
                store.validate_labels(snap, db)
                config = json.loads(row["config"])
                run_dir = store.root / "runs" / jid
                run_dir.mkdir(mode=0o700)  # No reuse or overwrite.
                db.execute("UPDATE jobs SET state='running' WHERE id=? AND state='queued'", (jid,))
            command = [
                x.replace("{run_dir}", str(run_dir)).replace(
                    "{snapshot}", str(store.root / "objects" / (snap["snapshot_id"] + ".json"))
                )
                for x in config["command"]
            ]
            result = None
            exit_code = None
            failure = None
            try:
                with (run_dir / "process.log").open("xb") as log:
                    env = dict(
                        os.environ,
                        HF_HUB_OFFLINE="1",
                        HF_HUB_DISABLE_TELEMETRY="1",
                        LAYA_RUN_ID=jid,
                        LAYA_LOCK_FD=str(fd),
                        PYTHONUNBUFFERED="1",
                    )
                    child = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env,
                        start_new_session=True,
                        pass_fds=(fd,),
                    )
                    started = identity(child.pid)
                    with store.tx() as db:
                        db.execute(
                            "UPDATE jobs SET pid=?,start_identity=? WHERE id=?", (child.pid, started, jid)
                        )
                    stop_at = None
                    while child.poll() is None:
                        with store.connect(readonly=True) as db:
                            cancelling = db.execute("SELECT cancel FROM jobs WHERE id=?", (jid,)).fetchone()[
                                0
                            ]
                        if cancelling and started is not None and identity(child.pid) == started:
                            if stop_at is None:
                                os.killpg(child.pid, signal.SIGTERM)
                                stop_at = time.monotonic()
                            elif time.monotonic() - stop_at > 5:
                                os.killpg(child.pid, signal.SIGKILL)
                        time.sleep(0.05)
                    exit_code = child.wait()
                if exit_code != 0:
                    raise ControlError("trainer returned nonzero exit status")
                with store.tx() as db:
                    db.execute("UPDATE jobs SET state='evaluating' WHERE id=?", (jid,))
                result = json.loads((run_dir / "result.json").read_text())
                if result.get("trained") is not True:
                    raise ControlError("training completion evidence absent")
                subdir = result.get("model_subdir", "model")
                if subdir not in ("model", "mlx"):
                    raise ControlError("invalid model subdirectory")
                hashes = artifact_hashes(run_dir / subdir)
                result["model_subdir"] = subdir
                result.update(
                    {
                        "artifacts": hashes,
                        "snapshot_id": snap["snapshot_id"],
                        "train_revisions": snap["splits"]["train"],
                        "domain": snap["domain"],
                        "base_revision": snap["base_revision"],
                    }
                )
                atomic_json(run_dir / "manifest.json", result)
            except Exception as exc:  # noqa: BLE001 - job failure must be recorded without leaking input
                failure = type(exc).__name__  # Raw external error text can contain private inputs.
            with store.tx() as db:
                cancelled = db.execute("SELECT cancel FROM jobs WHERE id=?", (jid,)).fetchone()[0]
                state = "cancelled" if cancelled else "failed" if failure else "succeeded"
                db.execute(
                    "UPDATE jobs SET state=?,exit_code=?,result=? WHERE id=?",
                    (state, exit_code, canonical(result or {"error": failure}), jid),
                )
                store.event(db, state, jid, {"exit_code": exit_code})
            completed.append({"id": jid, "state": state})
    return completed


def recover(store, lock_root=None):
    with lock_file(absolute(lock_root or default_lock_root()) / "training.lock") as fd:
        if fd is None:
            return {"state": "busy"}
        with store.tx() as db:
            rows = db.execute("SELECT * FROM jobs WHERE state IN ('running','evaluating')").fetchall()
            for row in rows:
                if (
                    row["pid"]
                    and row["start_identity"] is not None
                    and identity(row["pid"]) == row["start_identity"]
                ):
                    return {"state": "identity_alive"}
            for row in rows:
                state = "cancelled" if row["cancel"] else "interrupted"
                db.execute("UPDATE jobs SET state=? WHERE id=?", (state, row["id"]))
                store.event(db, state, row["id"], {"reason": "worker lost; output not automatically trusted"})
        return {"state": "recovered", "count": len(rows)}

"""Remote training on the shared Laya server; review, snapshots and the promotion gate stay in the controller."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .store import ControlError, Store, absolute, atomic_json, canonical, digest
from .training import passes_policy, rows_for

REMOTE_TASK = "compact-laya-focus-v2"
REMOTE_OWNER = "compact-laya"
REMOTE_TRAIN_KEYS = ("epochs", "micro_batch", "grad_accum", "freeze_encoder", "seed")
SPLITS = ("train", "calibration", "dev", "test")
JOB_ID = re.compile(r"[0-9a-f]{12}")
CLIENT_TIMEOUT_S = 120.0  # A 50 MB split upload does not fit the library's 10 s default.


@dataclass
class Remote:
    client: Any
    rows_to_jsonl: Callable[[list], str]


def poc_home() -> Path:
    return Path(os.environ.get("LAYA_POC_HOME") or Path.home() / "Documents" / "laya-poc").expanduser()


def load_client_module():
    path = poc_home() / "client" / "laya_poc_client.py"
    if not path.is_file():
        raise ControlError("laya-poc client not found; set LAYA_POC_HOME to a checkout containing client/")
    cached = sys.modules.get("laya_poc_client")
    if cached is not None and Path(getattr(cached, "__file__", "") or "").resolve() == path.resolve():
        return cached
    sys.modules.pop("laya_poc_client", None)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    importlib.invalidate_caches()
    return importlib.import_module("laya_poc_client")


def make_remote() -> Remote:
    module = load_client_module()
    return Remote(module.Client(timeout=CLIENT_TIMEOUT_S), module.rows_to_jsonl)


def call(function, *args, **kwargs):
    """Run one client call; library errors become controlled diagnostics without server text."""
    try:
        return function(*args, **kwargs)
    except ControlError:
        raise
    except Exception as exc:  # only the client library's own errors are translated
        names = {c.__name__ for c in type(exc).__mro__}
        if "ServerError" in names:
            status, code = getattr(exc, "status", "?"), getattr(exc, "code", "?")
            raise ControlError(f"remote server rejected the request: HTTP {status} {code}") from None
        if "LayaPocError" in names:
            raise ControlError("remote server unavailable") from None
        raise




def schema_hash(questions) -> str:
    """The server's schema hash: sha256 of canonical question JSON, first 16 hex characters."""
    return hashlib.sha256(canonical(questions).encode()).hexdigest()[:16]


def check_base(snapshot, status) -> None:
    """Refuse unless the server's base files hash to the snapshot's pinned base_revision."""
    files = (status.get("base") or {}).get("files")
    if (
        not isinstance(files, dict)
        or not files
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items())
    ):
        raise ControlError("server status has no base file hashes")
    if digest(files) != snapshot["base_revision"]:
        raise ControlError("server base checkpoint differs from the snapshot base_revision; not submitting")


def snapshot_questions(snapshot) -> dict:
    """The one question schema of a remote-trainable snapshot; other tasks are refused."""
    examples = list(snapshot["examples"].values())
    if not examples or any(e["task_id"] != REMOTE_TASK for e in examples):
        raise ControlError(f"remote training supports only task {REMOTE_TASK}")
    return examples[0]["schema"]


def build_splits(snapshot, rows_to_jsonl):
    """JSONL text and row counts for the four splits; test is the sealed benchmark."""
    splits, counts = {}, {}
    for name in SPLITS:
        rows = rows_for(snapshot, name)
        if not rows:
            raise ControlError("all four confirmed splits are required")
        splits[name] = rows_to_jsonl(rows)
        counts[name] = len(rows)
    return splits, counts


def map_config(train):
    """Keep the five keys the server accepts; report the rest as ignored."""
    if not isinstance(train, dict):
        raise ControlError("train config must be an object")
    kept = {k: train[k] for k in REMOTE_TRAIN_KEYS if k in train}
    return kept, sorted(set(train) - set(REMOTE_TRAIN_KEYS))




def record_path(root, job_id) -> Path:
    if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
        raise ControlError("remote job id must be 12 hex characters")
    return absolute(root) / "remote" / f"{job_id}.json"


def load_record(root, job_id) -> dict:
    path = record_path(root, job_id)
    if not path.is_file():
        raise ControlError("unknown remote job; it was not submitted from this state root")
    return json.loads(path.read_text())


def submit_remote(store: Store, remote: Remote, snapshot_id, cfg, owner, request_id):
    """Send the snapshot splits to the server. The request marker is written before the call so a crash
    between the call and the record never causes a silent second submission."""
    if not owner or not request_id:
        raise ControlError("owner and request ID required")
    if not isinstance(cfg, dict):
        raise ControlError("config must be an object")
    policy = cfg.get("policy", {})
    if not isinstance(policy, dict):
        raise ControlError("policy must be an object")
    config, ignored = map_config(cfg.get("train", {}))
    snapshot = store.get_snapshot(snapshot_id)
    with store.connect(readonly=True) as db:
        store.validate_labels(snapshot, db)
    questions = snapshot_questions(snapshot)
    payload_hash = digest([snapshot_id, config, policy, owner])
    marker = store.root / "remote" / "requests" / (digest(request_id) + ".json")
    if marker.is_file():
        old = json.loads(marker.read_text())
        if old["payload_hash"] != payload_hash:
            raise ControlError("remote request ID reused with changed content")
        if not old["job_id"]:
            raise ControlError(
                "earlier submission outcome unknown; check the server job list before retrying"
            )
        return {"id": old["job_id"], "state": "submitted", "remote": True}
    check_base(snapshot, call(remote.client.status))
    tasks = call(remote.client.tasks)
    registered = next((t for t in tasks if t.get("task") == REMOTE_TASK), None)
    if registered is None:
        call(remote.client.put_task, REMOTE_TASK, REMOTE_OWNER, questions, "compact-laya focus decisions")
    elif registered.get("schema_hash") != schema_hash(questions):
        raise ControlError("server task schema differs from the snapshot schema; not submitting")
    splits, counts = build_splits(snapshot, remote.rows_to_jsonl)
    atomic_json(marker, {"payload_hash": payload_hash, "job_id": None})
    response = call(remote.client.submit_job, REMOTE_TASK, splits, config)
    job_id = response["job_id"]
    record = {
        "job_id": job_id,
        "task": REMOTE_TASK,
        "snapshot_id": snapshot_id,
        "owner": owner,
        "request_id": request_id,
        "schema_hash": schema_hash(questions),
        "config": config,
        "ignored_train_keys": ignored,
        "policy": policy,
        "split_counts": counts,
        "submitted_at": time.time(),
        "activation": None,
    }
    atomic_json(record_path(store.root, job_id), record)
    atomic_json(marker, {"payload_hash": payload_hash, "job_id": job_id})
    with store.tx() as db:
        store.event(db, "remote_submitted", job_id, {"snapshot_id": snapshot_id})
    return {
        "id": job_id,
        "state": response.get("status", "queued"),
        "remote": True,
        "ignored_train_keys": ignored,
    }

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
from .training import rows_for

REMOTE_TASK = "compact-laya-focus-v2"
REMOTE_OWNER = "compact-laya"
REMOTE_TRAIN_KEYS = ("epochs", "micro_batch", "grad_accum", "freeze_encoder", "seed")
SPLITS = ("train", "calibration", "dev", "test")
JOB_ID = re.compile(r"[0-9a-f]{12}")
CLAMP_ISSUE = re.compile(r"on the \[0\.5, 5\] clamp$")  # laya.calibration_report text for the [TEMP_MIN, TEMP_MAX] clamp
MIN_SUPPORT = 5  # a class with fewer examples than this in a split is too small for a recall comparison
ROUNDING = 6e-5  # the server reports accuracy and recall rounded to 4 decimals
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




def feasible_positives(items, accuracy, true_recall, false_recall) -> set:
    """Positive-class sizes consistent with a binary question's rounded accuracy and recalls.

    The server sends no per-class counts, so support is recovered from integers: with P positives and
    items - P negatives, true_recall * P and false_recall * (items - P) must be whole hit counts whose sum
    over items is the accuracy.
    """
    found = set()
    for positives in range(1, items):
        negatives = items - positives
        true_hits, false_hits = round(true_recall * positives), round(false_recall * negatives)
        if (
            abs(true_hits / positives - true_recall) <= ROUNDING
            and abs(false_hits / negatives - false_recall) <= ROUNDING
            and abs((true_hits + false_hits) / items - accuracy) <= ROUNDING
        ):
            found.add(positives)
    return found


def explicit_positives(entry, items):
    """Positive count from an explicit {"support": {"true": a, "false": b}} field, if it adds up."""
    support = entry.get("support")
    if not isinstance(support, dict):
        return None
    a, b = support.get("true"), support.get("false")
    ints = all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in (a, b))
    return {a} if ints and a + b == items else None


def to_score(metric, questions) -> dict:
    """Map a server metric M to a score dict.

    Accuracy is the mean of per-question accuracies (every question has the same item count, so this
    equals the pooled accuracy). Recall keys follow score_predictions: "<qid>:<expected>", where a noul
    expectation is a Python bool, so the server labels "true"/"false" become "True"/"False". "balanced"
    maps each question to the mean recall of the classes the server reported for it. For noul questions,
    "positives" holds the possible positive-class sizes (None when they cannot be determined) and
    "partial" lists questions whose recall dict lacks a class.
    """
    try:
        items = metric["items"]
        by_question = metric["questions"]
        if isinstance(items, bool) or not isinstance(items, int) or items < 1:
            raise ControlError("metrics report no evaluated items")
        if set(by_question) != set(questions):
            raise ControlError("metrics questions differ from the snapshot schema")
        accuracies, recall, balanced = [], {}, {}
        question_accuracy, positives, partial = {}, {}, set()
        for qid, definition in questions.items():
            kind = definition["type"]
            accuracies.append(float(by_question[qid]["accuracy"]))
            question_accuracy[qid] = accuracies[-1]
            if not by_question[qid]["recall"]:
                raise ControlError("metrics have no per-class recall; the balanced gate would be vacuous")
            for label, value in by_question[qid]["recall"].items():
                if kind == "noul" and label in ("true", "false"):
                    name = f"{qid}:{label == 'true'}"
                elif kind == "choice":
                    name = f"{qid}:{label}"
                else:
                    raise ControlError("remote gate supports choice and noul only")
                recall[name] = float(value)
            if kind == "noul":
                entry = by_question[qid]
                if set(entry["recall"]) != {"true", "false"}:
                    partial.add(qid)
                else:
                    positives[qid] = explicit_positives(entry, items) or feasible_positives(
                        items, accuracies[-1], float(entry["recall"]["true"]), float(entry["recall"]["false"])
                    )
            balanced[qid] = sum(float(v) for v in by_question[qid]["recall"].values()) / len(
                by_question[qid]["recall"]
            )
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ControlError("malformed remote metrics") from None
    if not all(math.isfinite(x) for x in [*accuracies, *recall.values()]):
        raise ControlError("non-finite remote metrics")
    return {
        "accuracy": sum(accuracies) / len(accuracies),
        "count": items * len(questions),
        "per_class_recall": recall,
        "balanced": balanced,
        "question_accuracy": question_accuracy,
        "items": items,
        "positives": positives,
        "partial": partial,
        "confusion": {},
    }


def small_questions(baseline, candidate) -> set:
    """Questions with a class under MIN_SUPPORT examples in the split.

    A recall dict lacking a class counts as small. Otherwise the positive-class size must be certain:
    every size consistent with both the base and the candidate numbers is below MIN_SUPPORT on one side.
    Numbers that fit no integer counts leave the question on the stricter balanced comparison.
    """
    items, small = baseline["items"], set()
    for qid in baseline["balanced"]:
        if qid in baseline["partial"] or qid in candidate["partial"]:
            small.add(qid)
            continue
        sizes = baseline["positives"].get(qid)
        other = candidate["positives"].get(qid)
        common = (sizes & other) if sizes and other else set()
        if common and all(min(p, items - p) < MIN_SUPPORT for p in common):
            small.add(qid)
    return small


def passes_balanced(baseline, candidate, policy, small=frozenset()) -> bool:
    """Pooled accuracy must reach min_accuracy and no question's balanced accuracy (mean per-class recall)
    may fall below the base's by more than max_regression. A base that answers one class almost always
    has true-class recall near 1.0, which a per-class rule would treat as untouchable. Questions in
    `small` (a class too rare for a recall) are compared by plain accuracy instead."""
    if not baseline["count"] or candidate["count"] != baseline["count"]:
        return False
    if candidate["accuracy"] < policy.get("min_accuracy", 0.8):
        return False
    delta = policy.get("max_regression", 0)
    for qid, value in baseline["balanced"].items():
        if qid in small:
            ok = candidate["question_accuracy"][qid] >= baseline["question_accuracy"][qid] - delta
        else:
            ok = candidate["balanced"].get(qid, -1) >= value - delta
        if not ok:
            return False
    return True


def evaluate(job, questions, policy) -> dict:
    """The promotion gate for a finished remote job: candidate vs base on dev, then on test only if dev
    passed (see passes_balanced), plus the calibration and parity rules. Clamp-only temperature issues are
    returned as warnings."""
    metrics = job.get("metrics")
    if job.get("status") != "ready" or not isinstance(metrics, dict):
        return {"passed": False, "reasons": ["job is not ready"], "warnings": [], "splits": {}}
    reasons, warnings = [], []
    # A temperature on the [0.5, 5] clamp is a warning; any other calibration issue still rejects.
    calibration = metrics.get("calibration")
    if not isinstance(calibration, dict) or not calibration:
        reasons.append("calibration report missing")
    else:
        for kind, entry in calibration.items():
            issues = entry.get("issues") or []
            if any(not CLAMP_ISSUE.search(issue) for issue in issues):
                reasons.append(f"calibration issue: {kind}")
            elif issues:
                warnings.append(f"temperature on the clamp: {kind}")
    bound = metrics.get("temperature_at_bound")
    if not isinstance(bound, bool):
        reasons.append("temperature_at_bound missing")
    elif bound:
        warnings.append("temperature at bound")
    parity = metrics.get("parity")
    if not isinstance(parity, dict) or parity.get("passed") is not True:
        reasons.append("parity not passed")
    results = {}
    for split in ("dev", "test"):
        # The sealed test metrics are read only after dev passed, as in the local flow.
        if split == "test" and not results["dev"]["passed"]:
            break
        entry = (metrics.get("splits") or {}).get(split)
        if not isinstance(entry, dict) or not entry.get("candidate") or not entry.get("base"):
            reasons.append(f"{split} metrics missing")
            break
        baseline = to_score(entry["base"], questions)
        candidate = to_score(entry["candidate"], questions)
        small = small_questions(baseline, candidate)
        ok = passes_balanced(baseline, candidate, policy, small)
        results[split] = {
            "baseline_accuracy": baseline["accuracy"],
            "candidate_accuracy": candidate["accuracy"],
            "balanced": {
                q: {
                    "metric": "accuracy" if q in small else "balanced",
                    "base": (baseline["question_accuracy"] if q in small else baseline["balanced"])[q],
                    "candidate": (candidate["question_accuracy"] if q in small else candidate["balanced"])[q],
                }
                for q in baseline["balanced"]
            },
            "passed": ok,
        }
        for qid in baseline["balanced"]:
            note = f"{qid}: small class support (<{MIN_SUPPORT}), compared by accuracy"
            if qid in small and note not in warnings:
                warnings.append(note)
        if not ok:
            reasons.append(f"{split} policy not met")
    return {"passed": not reasons, "reasons": reasons, "warnings": warnings, "splits": results}


def activate_remote(store: Store, remote: Remote, job_id, reason):
    """Gate a finished remote job with the balanced policy, then ask the server to activate its version.
    The server keeps its cumulative dataset; no job data is deleted here."""
    if not reason:
        raise ControlError("explicit activation reason required")
    record = load_record(store.root, job_id)
    job = call(remote.client.job, job_id)
    version = job.get("version")
    if (
        job.get("task") != record["task"]
        or job.get("schema_hash") != record["schema_hash"]
        or job.get("split_counts") != record["split_counts"]
    ):
        raise ControlError("remote job does not match the submitted snapshot")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ControlError("remote job has no trained version")
    snapshot = store.get_snapshot(record["snapshot_id"])
    with store.connect(readonly=True) as db:
        store.validate_labels(snapshot, db)
    verdict = evaluate(job, snapshot_questions(snapshot), record["policy"])
    path = record_path(store.root, job_id)
    decision = {
        "passed": verdict["passed"],
        "reasons": verdict["reasons"],
        "warnings": verdict["warnings"],
        "splits": verdict["splits"],
        "reason": reason,
        "at": time.time(),
    }
    terminal = job.get("status") == "ready"
    if not verdict["passed"]:
        if terminal:
            record["decision"] = decision
            atomic_json(path, record)
            with store.tx() as db:
                store.event(db, "remote_rejected", job_id, {"reasons": verdict["reasons"]})
        raise ControlError("remote evaluation gate not passed: " + "; ".join(verdict["reasons"]))
    response = call(remote.client.activate, record["task"], version)
    record["decision"] = decision
    record["activation"] = {"version": version, "reason": reason, "at": decision["at"]}
    atomic_json(path, record)
    with store.tx() as db:
        store.event(db, "remote_activated", job_id, {"version": version, "reason": reason})
    return {"job_id": job_id, "version": version, "server": response, "evaluation": verdict}




def remote_status(root, factory) -> dict:
    """Remote jobs submitted from this state root, with live server state; creates nothing."""
    directory = absolute(root) / "remote"
    records = sorted(
        (json.loads(p.read_text()) for p in directory.glob("*.json")) if directory.is_dir() else [],
        key=lambda r: r["submitted_at"],
    )
    remote, unavailable = None, None
    if records:
        try:
            remote = factory()
        except ControlError as exc:
            unavailable = str(exc)
    jobs = []
    for record in records:
        entry = {
            "id": record["job_id"],
            "task": record["task"],
            "snapshot_id": record["snapshot_id"],
            "owner": record["owner"],
            "activated": bool(record.get("activation")),
        }
        try:
            if remote is None:
                raise ControlError(unavailable)
            job = call(remote.client.job, record["job_id"])
            entry.update({k: job.get(k) for k in ("step", "total", "wait_reason", "version")})
            entry["state"] = job.get("status")
        except ControlError as exc:
            entry.update(state="unavailable", error=str(exc))
        jobs.append(entry)
    return {"jobs": jobs}

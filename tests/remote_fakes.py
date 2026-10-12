"""Shared fixtures: a fake laya-poc client module surface, fake server metrics and a noul snapshot."""

import json

from laya_control.remote import REMOTE_TASK, Remote
from laya_control.store import digest

QUESTIONS = {
    "edit": {"type": "noul", "instructions": "Are code or configuration files being modified?"},
    "pr": {"type": "noul", "instructions": "Is a GitHub pull request being created, reviewed or merged?"},
}
BASE_FILES = {"model.safetensors": "aa" * 32, "tokenizer/tokenizer.json": "bb" * 32}
BASE_REVISION = digest(BASE_FILES)
JOB_ID = "0123456789ab"


class LayaPocError(Exception):
    pass


class Unavailable(LayaPocError):
    pass


class ServerError(LayaPocError):
    def __init__(self, status, code, message="private detail must not be printed"):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def rows_to_jsonl(rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


class FakeClient:
    def __init__(self, base_files=None, tasks=None, jobs=None, fail=None):
        self.base_files = BASE_FILES if base_files is None else base_files
        self.task_list = [] if tasks is None else tasks
        self.jobs_by_id = {} if jobs is None else jobs
        self.fail = fail
        self.calls = []
        self.on_delete = None

    def _enter(self, name, *args):
        self.calls.append((name, *args))
        if self.fail:
            raise self.fail

    def status(self):
        self._enter("status")
        return {"base": {"path": "/srv/base/torch", "files": self.base_files}, "queue": [], "running": None}

    def tasks(self):
        self._enter("tasks")
        return self.task_list

    def put_task(self, task, owner, questions, description=""):
        self._enter("put_task", task, owner)
        self.task_list.append(
            {"task": task, "owner": owner, "questions": questions, "schema_hash": schema_hash_of(questions)}
        )
        return {"task": task, "active_version": 0}

    def submit_job(self, task, splits, config=None):
        self._enter("submit_job", task)
        self.submitted = {"task": task, "splits": splits, "config": config}
        return {"job_id": JOB_ID, "status": "queued"}

    def job(self, job_id):
        self._enter("job", job_id)
        if job_id not in self.jobs_by_id:
            raise ServerError(404, "unknown_job")
        return self.jobs_by_id[job_id]

    def delete_job_data(self, job_id):
        self._enter("delete_job_data", job_id)
        if self.on_delete:
            self.on_delete(job_id)
        return {"id": job_id, "data_deleted": True}

    def activate(self, task, version):
        self._enter("activate", task, version)
        return {"task": task, "active_version": version, "previous_version": 0}


class SealedSplits(dict):
    """Splits whose test entry must never be read; used to prove the dev gate runs first."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_reads = 0

    def get(self, key, default=None):
        if key == "test":
            self.test_reads += 1
            raise AssertionError("the test split was read")
        return super().get(key, default)

    def __getitem__(self, key):
        if key == "test":
            self.test_reads += 1
            raise AssertionError("the test split was read")
        return super().__getitem__(key)


def schema_hash_of(questions):
    import hashlib

    text = json.dumps(questions, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def fake_remote(**kwargs):
    return Remote(FakeClient(**kwargs), rows_to_jsonl)


def noul_record(i):
    return {
        "task_id": REMOTE_TASK,
        "schema": QUESTIONS,
        "state": f"synthetic state {i}",
        "expected": {"edit": i % 2 == 0, "pr": i % 3 == 0},
        "source_group": f"group-{i}",
        "session_id": "test",
        "origin": "authoritative_source",
    }


def make_snapshot(store, count=80, base_revision=BASE_REVISION):
    ids = [store.submit(noul_record(i), f"req-{i}")["revision_id"] for i in range(count)]
    store.confirm(ids, "human", "review")
    return store.snapshot(store.authorize(ids, "human", "train"), base_revision)


def metric(accuracy, true_recall=None, false_recall=None, items=45):
    recall = {
        "true": accuracy if true_recall is None else true_recall,
        "false": accuracy if false_recall is None else false_recall,
    }
    return {
        "items": items,
        "questions": {q: {"accuracy": accuracy, "ece": 0.05, "recall": dict(recall)} for q in QUESTIONS},
    }


def ready_job(candidate=0.9, base=0.8, version=1, **overrides):
    job = {
        "id": JOB_ID,
        "task": REMOTE_TASK,
        "status": "ready",
        "version": version,
        "schema_hash": schema_hash_of(QUESTIONS),
        "split_counts": None,
        "metrics": {
            "active_version": 0,
            "splits": {
                "dev": {"candidate": metric(candidate), "active": None, "base": metric(base)},
                "test": {"candidate": metric(candidate, items=46), "active": None, "base": metric(base, items=46)},
            },
            "calibration": {"noul": {"items": 60, "temperature": 1.3, "issues": []}},
            "temperature": {"noul": 1.3},
            "temperature_at_bound": False,
            "parity": {"passed": True, "max_probability_difference": 0.01, "tolerance": 0.03, "items": 32},
        },
    }
    job.update(overrides)
    return job

"""Transactional metadata and immutable, content-addressed dataset snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


class ControlError(ValueError):
    """An actionable, fail-closed input or state error."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(canonical(value))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def absolute(path):
    p = Path(path).expanduser()
    if not p.is_absolute():
        raise ControlError("absolute path required")
    return Path(os.path.abspath(p))  # Preserve interpreter symlinks and their virtual environment.


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(version INTEGER NOT NULL);
INSERT INTO meta SELECT 1 WHERE NOT EXISTS(SELECT 1 FROM meta);
CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,kind TEXT,ref TEXT,detail TEXT);
CREATE TABLE IF NOT EXISTS groups(domain TEXT,group_id TEXT,split TEXT,PRIMARY KEY(domain,group_id));
CREATE TABLE IF NOT EXISTS examples(id TEXT PRIMARY KEY,example_key TEXT,domain TEXT,payload TEXT,split TEXT,
 confirmed INTEGER DEFAULT 0,origin TEXT);
CREATE INDEX IF NOT EXISTS example_key_idx ON examples(example_key);
CREATE TABLE IF NOT EXISTS submissions(request_id TEXT PRIMARY KEY,payload_hash TEXT,revision_id TEXT,
 payload TEXT,FOREIGN KEY(revision_id) REFERENCES examples(id));
CREATE TABLE IF NOT EXISTS resolutions(example_key TEXT PRIMARY KEY,chosen TEXT);
CREATE TABLE IF NOT EXISTS authorizations(id TEXT PRIMARY KEY,payload TEXT);
CREATE TABLE IF NOT EXISTS snapshots(id TEXT PRIMARY KEY,payload TEXT);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,snapshot_id TEXT,config TEXT,state TEXT,owner TEXT,
 cancel INTEGER DEFAULT 0,pid INTEGER,start_identity TEXT,exit_code INTEGER,created REAL,result TEXT);
CREATE TABLE IF NOT EXISTS job_requests(id TEXT PRIMARY KEY, payload_hash TEXT, job_id TEXT);
CREATE TABLE IF NOT EXISTS releases(id TEXT PRIMARY KEY,payload TEXT);
CREATE TABLE IF NOT EXISTS active(domain TEXT PRIMARY KEY,generation INTEGER,release_id TEXT);
CREATE TABLE IF NOT EXISTS intents(id TEXT PRIMARY KEY,payload TEXT,state TEXT);
"""


class Store:
    def __init__(self, root):
        self.root = absolute(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        for d in ("objects", "runs", "locks", "releases"):
            (self.root / d).mkdir(exist_ok=True, mode=0o700)
        # Serialize initialization separately; all steady-state writers use SQLite.
        import fcntl

        with open(self.root / "locks" / "init.lock", "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self.connect() as db:
                db.execute("PRAGMA journal_mode=WAL")
                db.executescript(SCHEMA)
                if db.execute("SELECT version FROM meta").fetchone()[0] != 1:
                    raise ControlError("unsupported state schema")
            os.chmod(self.root / "control.sqlite3", 0o600)

    def connect(self, readonly=False):
        target = (
            (self.root / "control.sqlite3").as_uri() + "?mode=ro"
            if readonly
            else str(self.root / "control.sqlite3")
        )
        db = sqlite3.connect(target, uri=readonly, timeout=5, isolation_level=None, factory=ClosingConnection)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        if not readonly:
            db.execute("PRAGMA synchronous=FULL")
        return db

    @contextmanager
    def tx(self):
        db = self.connect()
        try:
            for attempt in range(4):
                try:
                    db.execute("BEGIN IMMEDIATE")
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc) or attempt == 3:
                        raise
                    time.sleep(0.02 * 2**attempt)
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def event(db, kind, ref, detail):
        db.execute("INSERT INTO events(kind,ref,detail) VALUES(?,?,?)", (kind, ref, canonical(detail)))

    @staticmethod
    def conflict(db, key):
        rows = db.execute("SELECT id FROM examples WHERE example_key=?", (key,)).fetchall()
        chosen = db.execute("SELECT chosen FROM resolutions WHERE example_key=?", (key,)).fetchone()
        return len(rows) > 1 and chosen is None

    def submit(self, record, request_id):
        required = ("task_id", "schema", "state", "expected", "source_group", "session_id", "origin")
        if not request_id or any(k not in record for k in required):
            raise ControlError("missing submission fields")
        if not isinstance(record["state"], str) or not record["state"]:
            raise ControlError("state must be nonempty text")
        if record["origin"] not in ("user", "authoritative_source", "rule_batch", "synthetic", "pseudo"):
            raise ControlError("invalid label origin")
        if (
            not isinstance(record["schema"], dict)
            or not record["schema"]
            or not isinstance(record["expected"], dict)
        ):
            raise ControlError("schema and expected must be objects")
        if set(record["schema"]) != set(record["expected"]):
            raise ControlError("expected keys differ from schema")
        for q, definition in record["schema"].items():
            if definition.get("type") not in ("choice", "score", "noul"):
                raise ControlError("unsupported question type")
            if definition["type"] == "choice" and record["expected"][q] not in definition.get("criteria", []):
                raise ControlError("unknown choice label")
        payload_hash = digest(record)
        domain = digest([record["task_id"], record["schema"]])
        key = digest([domain, record["state"]])
        revision = digest([key, record["expected"]])
        response = {"revision_id": revision, "example_key": key, "domain": domain}
        with self.tx() as db:
            old = db.execute(
                "SELECT payload_hash FROM submissions WHERE request_id=?", (request_id,)
            ).fetchone()
            if old:
                if old[0] != payload_hash:
                    raise ControlError("idempotency key reused with different payload")
                return response
            group = str(record["source_group"])
            g = db.execute(
                "SELECT split FROM groups WHERE domain=? AND group_id=?", (domain, group)
            ).fetchone()
            bucket = int(digest([domain, group])[:8], 16) % 100
            split = (
                g[0]
                if g
                else (
                    "train"
                    if bucket < 70
                    else "dev"
                    if bucket < 80
                    else "calibration"
                    if bucket < 90
                    else "test"
                )
            )
            if record["origin"] == "pseudo":
                if g and split != "train":
                    raise ControlError("pseudo label in evaluation group")
                split = "train"
            existing = db.execute("SELECT split,origin FROM examples WHERE example_key=?", (key,)).fetchone()
            if existing and existing[0] != split:
                raise ControlError("duplicate crosses source group splits; resolve grouping first")
            if existing and (existing[1] == "pseudo") != (record["origin"] == "pseudo"):
                raise ControlError(
                    "pseudo provenance requires explicit reviewed resubmission in a new domain"
                )
            db.execute("INSERT OR IGNORE INTO groups VALUES(?,?,?)", (domain, group, split))
            db.execute(
                "INSERT OR IGNORE INTO examples(id,example_key,domain,payload,split,origin) VALUES(?,?,?,?,?,?)",
                (revision, key, domain, canonical(record), split, record["origin"]),
            )
            if not db.execute("SELECT 1 FROM submissions WHERE revision_id=?", (revision,)).fetchone():
                db.execute("DELETE FROM resolutions WHERE example_key=?", (key,))
            db.execute(
                "INSERT INTO submissions VALUES(?,?,?,?)",
                (request_id, payload_hash, revision, canonical(record)),
            )
            self.event(db, "submitted", revision, {"request_id": request_id})
        return response

    def revisions(self):
        with self.connect(readonly=True) as db:
            return [r[0] for r in db.execute("SELECT id FROM examples ORDER BY id")]

    def confirm(self, ids, actor, reason):
        if not actor or not reason:
            raise ControlError("review actor and evidence required")
        with self.tx() as db:
            for rid in sorted(set(ids)):
                if not db.execute("SELECT 1 FROM examples WHERE id=?", (rid,)).fetchone():
                    raise ControlError("unknown revision")
                db.execute("UPDATE examples SET confirmed=1 WHERE id=?", (rid,))
                self.event(db, "confirmed", rid, {"actor": actor, "reason": reason, "time": time.time()})

    def resolve(self, key, chosen, actor, reason):
        if not actor or not reason:
            raise ControlError("resolution evidence required")
        with self.tx() as db:
            if not db.execute(
                "SELECT 1 FROM examples WHERE id=? AND example_key=? AND confirmed=1", (chosen, key)
            ).fetchone():
                raise ControlError("chosen revision must be confirmed")
            db.execute("INSERT OR REPLACE INTO resolutions VALUES(?,?)", (key, chosen))
            self.event(db, "resolved", key, {"chosen": chosen, "actor": actor, "reason": reason})

    def authorize(self, ids, actor, reason):
        if not actor or not reason or not ids:
            raise ControlError("explicit authorization evidence and revisions required")
        with self.tx() as db:
            selected = []
            domains = set()
            for rid in sorted(set(ids)):
                row = db.execute("SELECT * FROM examples WHERE id=?", (rid,)).fetchone()
                if not row or not row["confirmed"] or self.conflict(db, row["example_key"]):
                    raise ControlError("unconfirmed, conflicting, or unknown revision")
                resolution = db.execute(
                    "SELECT chosen FROM resolutions WHERE example_key=?", (row["example_key"],)
                ).fetchone()
                if resolution and resolution[0] != rid:
                    raise ControlError("superseded revision")
                domains.add(row["domain"])
                selected.append(rid)
            if len(domains) != 1:
                raise ControlError("authorization must cover one task/schema domain")
            obj = {
                "revisions": selected,
                "domain": domains.pop(),
                "actor": actor,
                "reason": reason,
                "cutoff_seq": db.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0],
            }
            aid = digest(obj)
            db.execute("INSERT OR IGNORE INTO authorizations VALUES(?,?)", (aid, canonical(obj)))
            self.event(db, "authorized", aid, {"count": len(selected)})
        return aid

    def snapshot(self, authorization_id, base_revision):
        if not base_revision:
            raise ControlError("pinned base revision required")
        with self.tx() as db:
            row = db.execute("SELECT payload FROM authorizations WHERE id=?", (authorization_id,)).fetchone()
            if not row:
                raise ControlError("unknown authorization")
            auth = json.loads(row[0])
            sid = digest([authorization_id, base_revision])
            old = db.execute("SELECT payload FROM snapshots WHERE id=?", (sid,)).fetchone()
            if old:
                return json.loads(old[0])
            rows = []
            for rid in auth["revisions"]:
                r = dict(db.execute("SELECT * FROM examples WHERE id=?", (rid,)).fetchone())
                chosen = db.execute(
                    "SELECT chosen FROM resolutions WHERE example_key=?", (r["example_key"],)
                ).fetchone()
                if self.conflict(db, r["example_key"]) or (chosen and chosen[0] != rid):
                    raise ControlError("labels changed after authorization; review required")
                rows.append(r)
            previous = db.execute("SELECT payload FROM snapshots ORDER BY rowid").fetchall()
            earlier = next(
                (
                    json.loads(x[0])
                    for x in previous
                    if json.loads(x[0])["domain"] == auth["domain"] and json.loads(x[0])["benchmark"]
                ),
                None,
            )
            sealed = (
                earlier["benchmark"] if earlier else sorted(r["id"] for r in rows if r["split"] == "test")
            )
            if not set(sealed) <= set(auth["revisions"]):
                raise ControlError("authorization omits sealed benchmark; preserve all benchmark revisions")
            obj = {
                "snapshot_id": sid,
                "authorization_id": authorization_id,
                "domain": auth["domain"],
                "cutoff_seq": auth["cutoff_seq"],
                "base_revision": base_revision,
                "revisions": auth["revisions"],
                "unique_count": len(rows),
                "splits": {
                    k: [r["id"] for r in rows if r["split"] == k]
                    for k in ("train", "dev", "calibration", "test")
                },
                "benchmark": sealed,
                "examples": {r["id"]: json.loads(r["payload"]) for r in rows},
            }
            # A snapshot is small local metadata. Publishing before commit leaves only an orphan on crash.
            atomic_json(self.root / "objects" / (sid + ".json"), obj)
            db.execute("INSERT INTO snapshots VALUES(?,?)", (sid, canonical(obj)))
            self.event(db, "snapshot", sid, {"count": len(rows)})
        return obj

    def get_snapshot(self, sid):
        with self.connect(readonly=True) as db:
            row = db.execute("SELECT payload FROM snapshots WHERE id=?", (sid,)).fetchone()
            if not row:
                raise ControlError("unknown snapshot")
            obj = json.loads(row[0])
        disk = json.loads((self.root / "objects" / (sid + ".json")).read_text())
        if canonical(obj) != canonical(disk):
            raise ControlError("snapshot integrity failure")
        return obj

    def validate_labels(self, snapshot, db):
        for rid in snapshot["revisions"]:
            row = db.execute("SELECT example_key,confirmed FROM examples WHERE id=?", (rid,)).fetchone()
            if not row or not row["confirmed"] or self.conflict(db, row["example_key"]):
                raise ControlError("snapshot contains labels requiring renewed review")
            choice = db.execute(
                "SELECT chosen FROM resolutions WHERE example_key=?", (row["example_key"],)
            ).fetchone()
            if choice and choice[0] != rid:
                raise ControlError("snapshot contains superseded label")

    def reconcile(self):
        with self.connect(readonly=True) as db:
            ids = {r[0] + ".json" for r in db.execute("SELECT id FROM snapshots")}
        files = {p.name for p in (self.root / "objects").glob("*.json")}
        return {"orphans": sorted(files - ids), "missing": sorted(ids - files)}

    def status(self):
        return self.read_status(self.root)

    @classmethod
    def read_status(cls, root):
        root = absolute(root)
        if not (root / "control.sqlite3").exists():
            return {"state": "not_configured"}
        obj = object.__new__(cls)
        obj.root = root
        with obj.connect(readonly=True) as db:
            db.execute("BEGIN")
            ids = {r[0] for r in db.execute("SELECT id FROM examples")}
            included = set()
            for r in db.execute("SELECT payload FROM snapshots"):
                included.update(json.loads(r[0])["revisions"])
            conflicts = sum(
                cls.conflict(db, r[0]) for r in db.execute("SELECT DISTINCT example_key FROM examples")
            )
            jobs = [dict(r) for r in db.execute("SELECT id,state,owner,created FROM jobs ORDER BY created")]
            active = [dict(r) for r in db.execute("SELECT domain,generation,release_id FROM active")]
            active_ids = set()
            for active_row in active:
                release = db.execute(
                    "SELECT payload FROM releases WHERE id=?", (active_row["release_id"],)
                ).fetchone()
                if release:
                    active_ids.update(json.loads(release[0])["revisions"])
            return {
                "not_active_count": len(ids - active_ids),
                "state": "configured",
                "submitted_count": len(ids),
                "pending_count": len(ids - included),
                "conflicts": conflicts,
                "jobs": jobs,
                "active": active,
                "running_count": sum(j["state"] in ("running", "evaluating") for j in jobs),
                "queued_count": sum(j["state"] == "queued" for j in jobs),
                "observed_at": time.time(),
            }

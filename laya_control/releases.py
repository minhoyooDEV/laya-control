"""Verified release bundles, coverage checks, generation CAS, and recovery journals."""

import json
import shutil
import uuid
from pathlib import Path

from .jobs import artifact_hashes, lock_file
from .store import ControlError, atomic_json, canonical


def _checked(service, pointer):
    health = service.switch(pointer)
    if not health.get("ready") or any(health.get(k) != pointer[k] for k in ("release_id", "generation")):
        raise ControlError("serving health does not match intended release")


def activate(store, run_id, expected_generation, reason, service, *, rollback_of=None):
    if not reason or service is None:
        raise ControlError("explicit activation reason and serving adapter required")
    with lock_file(store.root / "locks" / "activation.lock") as fd:
        if fd is None:
            raise ControlError("activation busy; retry after checking generation")
        with store.connect(readonly=True) as db:
            if db.execute(
                "SELECT 1 FROM intents WHERE state IN ('prepared','switched','recovery_required')"
            ).fetchone():
                raise ControlError("unfinished activation; recover before retry")
            row = db.execute("SELECT * FROM jobs WHERE id=?", (run_id,)).fetchone()
            if not row or row["state"] != "succeeded":
                raise ControlError("successful run required")
            result = json.loads(row["result"])
            if not result.get("evaluation", {}).get("passed"):
                raise ControlError("evaluation gate not passed")
            snap = store.get_snapshot(row["snapshot_id"])
            domain = snap["domain"]
            if not rollback_of:
                store.validate_labels(snap, db)
            active = db.execute("SELECT * FROM active WHERE domain=?", (domain,)).fetchone()
            generation = active["generation"] if active else 0
            if generation != expected_generation:
                raise ControlError("stale activation generation")
            old = None
            if active:
                old = json.loads(
                    db.execute("SELECT payload FROM releases WHERE id=?", (active["release_id"],)).fetchone()[
                        0
                    ]
                )
                if not rollback_of and not set(old["revisions"]) <= set(snap["revisions"]):
                    raise ControlError("candidate loses active data coverage; cumulative retraining required")
                if not rollback_of and old["base_revision"] != snap["base_revision"]:
                    raise ControlError("canonical base changed; use a separate domain or explicit rollback")
        source = store.root / "runs" / run_id / result["model_subdir"]
        if rollback_of:
            with store.connect(readonly=True) as db:
                saved = db.execute("SELECT payload FROM releases WHERE id=?", (rollback_of,)).fetchone()
                if not saved:
                    raise ControlError("unknown rollback release")
                target = json.loads(saved[0])
                if target["run_id"] != run_id:
                    raise ControlError("rollback lineage mismatch")
                source = Path(target["model_path"])
                result = dict(result, artifacts=target["artifacts"], runtime=target["runtime"])
        if artifact_hashes(source) != result["artifacts"]:
            raise ControlError("candidate artifact integrity failure")
        rid = uuid.uuid4().hex
        directory = store.root / "releases" / rid
        directory.mkdir(mode=0o700)
        shutil.copytree(source, directory / "model")
        if artifact_hashes(directory / "model") != result["artifacts"]:
            raise ControlError("release copy integrity failure")
        schema = next(iter(snap["examples"].values()))["schema"]
        release = {
            "release_id": rid,
            "generation": generation + 1,
            "parent_generation": generation,
            "domain": domain,
            "run_id": run_id,
            "snapshot_id": snap["snapshot_id"],
            "base_revision": snap["base_revision"],
            "revisions": snap["revisions"],
            "train_revisions": snap["splits"]["train"],
            "schema": schema,
            "runtime": result.get("runtime", "pytorch"),
            "model_path": str(directory / "model"),
            "artifacts": result["artifacts"],
            "reason": reason,
            "rollback_of": rollback_of,
        }
        atomic_json(directory / "release.json", release)
        pointer_path = store.root / ("active-" + domain + ".json")
        intent = {"release": release, "previous": old}
        with store.tx() as db:
            db.execute("INSERT INTO releases VALUES(?,?)", (rid, canonical(release)))
            db.execute("INSERT INTO intents VALUES(?,?,?)", (rid, canonical(intent), "prepared"))
        try:
            atomic_json(pointer_path, release)
            with store.tx() as db:
                db.execute("UPDATE intents SET state='switched' WHERE id=?", (rid,))
            _checked(service, release)
            with store.tx() as db:
                if not rollback_of:
                    store.validate_labels(snap, db)
                current = db.execute("SELECT generation FROM active WHERE domain=?", (domain,)).fetchone()
                if (current[0] if current else 0) != generation:
                    raise ControlError("concurrent generation update")
                db.execute("INSERT OR REPLACE INTO active VALUES(?,?,?)", (domain, generation + 1, rid))
                db.execute("UPDATE intents SET state='verified' WHERE id=?", (rid,))
                store.event(db, "active", rid, {"generation": generation + 1, "reason": reason})
        except BaseException:
            try:
                if old:
                    atomic_json(pointer_path, old)
                    _checked(service, old)
                else:
                    if hasattr(service, "stop"):
                        service.stop()
                    pointer_path.unlink(missing_ok=True)
                recovery = "failed"
            except Exception:  # noqa: BLE001 - failed recovery must remain explicit
                recovery = "recovery_required"
            with store.tx() as db:
                db.execute("UPDATE intents SET state=? WHERE id=?", (recovery, rid))
            raise
        return release


def rollback(store, release_id, expected_generation, reason, service):
    with store.connect(readonly=True) as db:
        row = db.execute("SELECT payload FROM releases WHERE id=?", (release_id,)).fetchone()
        if not row:
            raise ControlError("unknown release")
        release = json.loads(row[0])
    return activate(store, release["run_id"], expected_generation, reason, service, rollback_of=release_id)


def recover_activation(store, service):
    with lock_file(store.root / "locks" / "activation.lock") as fd:
        if fd is None:
            raise ControlError("activation busy")
        with store.connect(readonly=True) as db:
            rows = db.execute(
                "SELECT * FROM intents WHERE state IN ('prepared','switched','recovery_required')"
            ).fetchall()
        recovered = []
        for row in rows:
            intent = json.loads(row["payload"])
            old = intent["previous"]
            new = intent["release"]
            path = store.root / ("active-" + new["domain"] + ".json")
            if old:
                atomic_json(path, old)
                _checked(service, old)
            else:
                service.stop()
                path.unlink(missing_ok=True)
            with store.tx() as db:
                db.execute("UPDATE intents SET state='failed' WHERE id=?", (row["id"],))
            recovered.append(row["id"])
        return recovered

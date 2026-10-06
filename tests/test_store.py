import multiprocessing as mp
import tempfile
import unittest
from pathlib import Path

from laya_control.store import ControlError, Store


def record(i, label="yes", group=None):
    return {
        "task_id": "demo",
        "schema": {"label": {"type": "choice", "criteria": ["yes", "no"]}},
        "state": f"synthetic example {i}",
        "expected": {"label": label},
        "source_group": group or f"group-{i}",
        "session_id": "test",
        "origin": "user",
    }


def submit_many(root, start, stop):
    s = Store(Path(root))
    for i in range(start, stop):
        s.submit(record(i), f"req-{start}-{i}")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "control"
        self.s = Store(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_idempotent_and_conflict(self):
        a = self.s.submit(record(1), "a")
        self.assertEqual(a, self.s.submit(record(1), "a"))
        with self.assertRaises(ControlError):
            self.s.submit(record(2), "a")
        b = self.s.submit(record(1, "no"), "b")
        self.s.confirm([a["revision_id"], b["revision_id"]], "human", "reviewed")
        self.assertEqual(self.s.status()["conflicts"], 1)
        self.s.resolve(a["example_key"], b["revision_id"], "human", "corrected")
        self.assertEqual(self.s.status()["conflicts"], 0)

    def test_parallel_union_and_late_data(self):
        ctx = mp.get_context("spawn")
        ps = [ctx.Process(target=submit_many, args=(str(self.root), a, b)) for a, b in [(0, 100), (80, 160)]]
        for p in ps:
            p.start()
        for p in ps:
            p.join(20)
            self.assertEqual(p.exitcode, 0)
        ids = self.s.revisions()
        self.assertEqual(len(ids), 160)
        self.s.confirm(ids, "human", "synthetic fixture")
        auth = self.s.authorize(ids, "human", "train this union")
        snap = self.s.snapshot(auth, "base-v1")
        self.assertEqual(snap["unique_count"], 160)
        for i in range(160, 175):
            self.s.submit(record(i), f"late-{i}")
        self.s.confirm(self.s.revisions(), "human", "synthetic fixture")
        self.assertEqual(self.s.snapshot(auth, "base-v1")["snapshot_id"], snap["snapshot_id"])
        self.assertEqual(self.s.status()["pending_count"], 15)
        later = self.s.snapshot(self.s.authorize(self.s.revisions(), "human", "train new union"), "base-v1")
        self.assertEqual(later["unique_count"], 175)
        self.assertTrue(set(snap["revisions"]) <= set(later["revisions"]))

    def test_conflicts_and_unconfirmed_excluded(self):
        a = self.s.submit(record(1), "a")
        b = self.s.submit(record(1, "no"), "b")
        c = self.s.submit(record(2), "c")
        self.s.confirm([a["revision_id"], b["revision_id"]], "human", "review")
        with self.assertRaises(ControlError):
            self.s.authorize([c["revision_id"]], "human", "train")
        with self.assertRaises(ControlError):
            self.s.authorize([a["revision_id"]], "human", "train")

    def test_cross_domain_rejected(self):
        a = self.s.submit(record(1), "a")
        r = record(2)
        r["task_id"] = "other"
        b = self.s.submit(r, "b")
        self.s.confirm([a["revision_id"], b["revision_id"]], "human", "review")
        with self.assertRaises(ControlError):
            self.s.authorize(self.s.revisions(), "human", "train")

    def test_status_no_creation(self):
        absent = self.root / "absent"
        self.assertEqual(Store.read_status(absent)["state"], "not_configured")
        self.assertFalse(absent.exists())

    def test_integrity_and_reconcile(self):
        a = self.s.submit(record(1), "a")
        self.s.confirm([a["revision_id"]], "human", "review")
        auth = self.s.authorize(self.s.revisions(), "human", "train")
        self.s.snapshot(auth, "base")
        (self.root / "objects" / "orphan.json").write_text("{}")
        self.assertIn("orphan.json", self.s.reconcile()["orphans"])
        self.assertEqual(self.root.stat().st_mode & 0o777, 0o700)

import tempfile
import unittest
from pathlib import Path

from test_store import record

from laya_control.jobs import enqueue
from laya_control.store import ControlError, Store


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = Store(Path(self.tmp.name) / "state")

    def tearDown(self):
        self.tmp.cleanup()

    def test_queue_retry_idempotency(self):
        r = self.s.submit(record(1), "a")
        self.s.confirm([r["revision_id"]], "human", "review")
        snap = self.s.snapshot(self.s.authorize([r["revision_id"]], "human", "train"), "base")
        config = {"command": ["/bin/echo", "{run_dir}"], "request_id": "request-1"}
        a = enqueue(self.s, snap["snapshot_id"], config, "owner")
        b = enqueue(self.s, snap["snapshot_id"], config, "owner")
        self.assertEqual(a["id"], b["id"])
        config["command"] = ["/bin/false", "{run_dir}"]
        with self.assertRaises(ControlError):
            enqueue(self.s, snap["snapshot_id"], config, "owner")

    def test_active_pending_is_distinct_from_snapshot_pending(self):
        r = self.s.submit(record(1), "a")
        self.s.confirm([r["revision_id"]], "human", "review")
        self.s.snapshot(self.s.authorize([r["revision_id"]], "human", "train"), "base")
        status = self.s.status()
        self.assertEqual(status["pending_count"], 0)
        self.assertEqual(status["not_active_count"], 1)

    def test_conflict_after_snapshot_blocks_enqueue(self):
        r = self.s.submit(record(1), "a")
        self.s.confirm([r["revision_id"]], "human", "review")
        snap = self.s.snapshot(self.s.authorize([r["revision_id"]], "human", "train"), "base")
        self.s.submit(record(1, "no"), "b")
        with self.assertRaises(ControlError):
            enqueue(self.s, snap["snapshot_id"], {"command": ["/bin/echo", "{run_dir}"]}, "owner")

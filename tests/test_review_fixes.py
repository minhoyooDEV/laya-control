import os
import sys
import unittest
from unittest.mock import patch

import test_jobs
import test_releases
from test_store import record

from laya_control import training
from laya_control.jobs import lock_file, work
from laya_control.releases import activate, rollback
from laya_control.store import ControlError, absolute


class ReviewFixTests(unittest.TestCase):
    setUp = test_jobs.JobTests.setUp
    tearDown = test_jobs.JobTests.tearDown
    queue = test_jobs.JobTests.queue

    def test_interpreter_symlink_preserves_venv(self):
        link = self.root / "python"
        link.symlink_to(sys.executable)
        self.assertEqual(absolute(link), link)

    def test_conversion_child_retains_lock_fd(self):
        with lock_file(self.root / "fd.lock") as fd, patch.dict(os.environ, {"LAYA_LOCK_FD": str(fd)}):
            result = training.run_child(
                [sys.executable, "-c", 'import os;os.fstat(int(os.environ["LAYA_LOCK_FD"]))']
            )
            self.assertEqual(result.returncode, 0)

    def test_late_label_conflict_rolls_back_activation(self):
        j = self.queue()
        work(self.s, self.locks)
        service = test_releases.Service()
        original = service.switch

        def switch(pointer):
            self.s.submit(record(1, "no"), "concurrent-conflict")
            return original(pointer)

        service.switch = switch
        with self.assertRaises(ControlError):
            activate(self.s, j, 0, "explicit", service)
        self.assertEqual(self.s.status()["active"], [])

    def test_rollback_uses_release_not_mutated_run(self):
        j = self.queue()
        work(self.s, self.locks)
        service = test_releases.Service()
        first = activate(self.s, j, 0, "first", service)
        activate(self.s, j, 1, "second", service)
        (self.s.root / "runs" / j / "model" / "model.safetensors").write_bytes(b"corrupt")
        restored = rollback(self.s, first["release_id"], 2, "explicit rollback", service)
        self.assertEqual(restored["generation"], 3)

    def test_sealed_benchmark_cannot_be_subsetted(self):
        ids = []
        for i in range(1, 101):
            ids.append(self.s.submit(record(i), f"bench-{i}")["revision_id"])
        self.s.confirm(ids, "human", "review")
        first = self.s.snapshot(self.s.authorize(ids, "human", "train"), "base")
        self.assertTrue(first["benchmark"])
        subset = [i for i in ids if i != first["benchmark"][0]]
        auth = self.s.authorize(subset, "human", "subset")
        with self.assertRaises(ControlError):
            self.s.snapshot(auth, "base")

    def test_rows_for_rejects_incomplete_benchmark(self):
        with self.assertRaises(ControlError):
            training.rows_for(
                {
                    "splits": {"test": ["a"]},
                    "benchmark": ["a", "b"],
                    "examples": {"a": {"state": "x", "schema": {}, "expected": {}}},
                },
                "test",
            )

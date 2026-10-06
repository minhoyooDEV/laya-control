import multiprocessing as mp
import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path

from test_store import record

from laya_control.jobs import cancel, enqueue, recover, work
from laya_control.store import ControlError, Store

FAKE = """import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); time.sleep(float(sys.argv[2]))
(p/'model').mkdir()
(p/'model'/'model.safetensors').write_bytes(b'synthetic test artifact')
(p/'result.json').write_text(json.dumps({'trained':True,'evaluation':{'passed':True},'runtime':'fixture'}))
"""


def worker(root, locks):
    work(Store(Path(root)), Path(locks), drain=True)


class JobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.s = Store(self.root / "state")
        self.locks = self.root / "locks"
        r = self.s.submit(record(1), "a")
        self.s.confirm([r["revision_id"]], "human", "review")
        self.snap = self.s.snapshot(self.s.authorize([r["revision_id"]], "human", "train"), "base")

    def tearDown(self):
        self.tmp.cleanup()

    def queue(self, code=FAKE, delay="0"):
        return enqueue(
            self.s,
            self.snap["snapshot_id"],
            {"command": [sys.executable, "-c", code, "{run_dir}", delay]},
            "session",
        )["id"]

    def test_output_and_idempotent_request(self):
        j = self.queue()
        work(self.s, self.locks)
        with self.s.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (j,)).fetchone()
            self.assertEqual(row["state"], "succeeded")
            self.assertEqual(row["exit_code"], 0)
        self.assertTrue((self.s.root / "runs" / j / "manifest.json").exists())

    def test_exit_zero_without_output_fails(self):
        j = self.queue("pass")
        work(self.s, self.locks)
        with self.s.connect() as db:
            self.assertEqual(db.execute("SELECT state FROM jobs WHERE id=?", (j,)).fetchone()[0], "failed")

    def test_cancel_queued(self):
        j = self.queue()
        cancel(self.s, j)
        self.assertEqual(work(self.s, self.locks), [])
        with self.s.connect() as db:
            self.assertEqual(db.execute("SELECT state FROM jobs WHERE id=?", (j,)).fetchone()[0], "cancelled")

    def test_two_workers_only_one_claim(self):
        self.queue(delay=".3")
        self.queue(delay=".3")
        ctx = mp.get_context("spawn")
        ps = [ctx.Process(target=worker, args=(str(self.s.root), str(self.locks))) for _ in range(2)]
        for p in ps:
            p.start()
        for p in ps:
            p.join(10)
            self.assertEqual(p.exitcode, 0)
        self.assertEqual([j["state"] for j in self.s.status()["jobs"]], ["succeeded", "succeeded"])

    def test_supervisor_death_retains_child_lock(self):
        j = self.queue(delay="1.2")
        p = mp.get_context("spawn").Process(target=worker, args=(str(self.s.root), str(self.locks)))
        p.start()
        end = time.monotonic() + 5
        while time.monotonic() < end:
            with self.s.connect() as db:
                row = db.execute("SELECT pid FROM jobs WHERE id=?", (j,)).fetchone()
            if row[0]:
                break
            time.sleep(0.02)
        self.assertIsNotNone(row[0])
        os.kill(p.pid, signal.SIGKILL)
        p.join(5)
        self.assertEqual(work(self.s, self.locks), [])
        self.assertEqual(recover(self.s, self.locks)["state"], "busy")
        time.sleep(1.4)
        self.assertEqual(recover(self.s, self.locks)["state"], "recovered")
        self.assertEqual(self.s.status()["jobs"][0]["state"], "interrupted")

    def test_corrupted_snapshot_not_run(self):
        self.queue()
        (self.s.root / "objects" / (self.snap["snapshot_id"] + ".json")).write_text("{}")
        with self.assertRaises(ControlError):
            work(self.s, self.locks)
        self.assertEqual(self.s.status()["jobs"][0]["state"], "queued")

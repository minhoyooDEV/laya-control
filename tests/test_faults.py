"""Crash and race tests against real local processes; every payload is synthetic."""

import multiprocessing as mp
import os
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import test_jobs
import test_releases

from laya_control.jobs import cancel, identity, work
from laya_control.releases import activate, recover_activation
from laya_control.store import Store


def snapshot_crash(root, auth):
    store = Store(root)
    original = store.event

    def event(db, kind, ref, detail):
        if kind == "snapshot":
            os._exit(73)
        original(db, kind, ref, detail)

    store.event = event
    store.snapshot(auth, "crash-base")


def activation_crash(root, jid):
    class Service:
        def switch(self, pointer):
            os._exit(74)

    activate(Store(root), jid, 1, "synthetic crash", Service())


def activate_race(root, jid, barrier, results):
    store = Store(root)
    barrier.wait()
    try:
        activate(store, jid, 0, "synthetic race", test_releases.Service())
        results.put("activated")
    except ValueError:
        results.put("rejected")


class FaultTests(unittest.TestCase):
    setUp = test_jobs.JobTests.setUp
    tearDown = test_jobs.JobTests.tearDown
    queue = test_jobs.JobTests.queue

    def test_snapshot_crash_preserves_orphan_and_retry(self):
        auth = self.snap["authorization_id"]
        p = mp.get_context("spawn").Process(target=snapshot_crash, args=(str(self.s.root), auth))
        p.start()
        p.join(5)
        self.assertEqual(p.exitcode, 73)
        self.assertEqual(len(self.s.reconcile()["orphans"]), 1)
        self.s.snapshot(auth, "crash-base")
        self.assertEqual(self.s.reconcile()["orphans"], [])

    def test_disk_error_does_not_commit_snapshot(self):
        with (
            patch("laya_control.store.atomic_json", side_effect=OSError("synthetic disk full")),
            self.assertRaises(OSError),
        ):
            self.s.snapshot(self.snap["authorization_id"], "disk-full")
        with self.s.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0], 1)

    def test_activation_crash_recovers_prior_release(self):
        jid = self.queue()
        work(self.s, self.locks)
        service = test_releases.Service()
        old = activate(self.s, jid, 0, "first", service)
        p = mp.get_context("spawn").Process(target=activation_crash, args=(str(self.s.root), jid))
        p.start()
        p.join(5)
        self.assertEqual(p.exitcode, 74)
        self.assertEqual(len(recover_activation(self.s, service)), 1)
        self.assertEqual(service.current["release_id"], old["release_id"])
        self.assertEqual(self.s.status()["active"][0]["generation"], 1)

    def test_competing_activations_only_one_wins(self):
        jid = self.queue()
        work(self.s, self.locks)
        ctx = mp.get_context("spawn")
        barrier = ctx.Barrier(2)
        results = ctx.Queue()
        ps = [
            ctx.Process(target=activate_race, args=(str(self.s.root), jid, barrier, results))
            for _ in range(2)
        ]
        for p in ps:
            p.start()
        for p in ps:
            p.join(5)
            self.assertEqual(p.exitcode, 0)
        self.assertEqual(sorted([results.get(timeout=2), results.get(timeout=2)]), ["activated", "rejected"])

    def test_running_cancellation(self):
        jid = self.queue(delay="10")
        p = mp.get_context("spawn").Process(target=test_jobs.worker, args=(str(self.s.root), str(self.locks)))
        p.start()
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with self.s.connect() as db:
                    pid = db.execute("SELECT pid FROM jobs WHERE id=?", (jid,)).fetchone()[0]
                if pid:
                    break
                time.sleep(0.02)
            self.assertIsNotNone(pid)
            cancel(self.s, jid)
            p.join(5)
            self.assertEqual(p.exitcode, 0)
            self.assertEqual(self.s.status()["jobs"][0]["state"], "cancelled")
        finally:
            if p.is_alive():
                p.terminate()
                p.join(5)

    def test_identity_survives_exec(self):
        before = self.root / "before"
        proceed = self.root / "proceed"
        after = self.root / "after"
        second = f"from pathlib import Path; import time; Path({str(after)!r}).touch(); time.sleep(10)"
        first = (
            f"import os,sys,time; from pathlib import Path; Path({str(before)!r}).touch(); "
            f"\nwhile not Path({str(proceed)!r}).exists(): time.sleep(.01)\n"
            f"os.execv(sys.executable,[sys.executable,'-c',{second!r}])"
        )
        child = subprocess.Popen([sys.executable, "-c", first])
        try:
            deadline = time.monotonic() + 5
            while not before.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(before.exists())
            old = identity(child.pid)
            self.assertIsNotNone(old)
            proceed.touch()
            while not after.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(after.exists())
            self.assertEqual(identity(child.pid), old)
        finally:
            child.terminate()
            child.wait(timeout=5)

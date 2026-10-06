import multiprocessing as mp
import tempfile
import time
import unittest
from pathlib import Path

from laya_control.server import rpc, serve


class FakeAgent:
    def predict(self, state, questions):
        return {"answers": {"label": {"choice": "yes", "confidence": 0.9}}}


def fake_loader(release):
    return FakeAgent()


def start(release, socket):
    serve(release, Path(socket), loader=fake_loader)


class ServerTests(unittest.TestCase):
    def test_singleton_and_immutable_release(self):
        with tempfile.TemporaryDirectory(prefix="lc-") as d:
            path = Path(d) / "s.sock"
            release = {
                "release_id": "r1",
                "generation": 1,
                "snapshot_id": "s1",
                "schema": {"label": {}},
                "runtime": "fixture",
            }
            ctx = mp.get_context("spawn")
            a = ctx.Process(target=start, args=(release, str(path)))
            b = ctx.Process(target=start, args=(release, str(path)))
            a.start()
            try:
                for _ in range(100):
                    try:
                        if rpc(path, {"kind": "health"})["ready"]:
                            break
                    except OSError:
                        time.sleep(0.03)
                self.assertEqual(rpc(path, {"kind": "health"})["release_id"], "r1")
                b.start()
                b.join(3)
                self.assertNotEqual(b.exitcode, 0)
                self.assertTrue(a.is_alive())
                self.assertEqual(
                    rpc(path, {"kind": "predict", "state": "synthetic"})["answers"]["label"]["choice"], "yes"
                )
            finally:
                for p in (a, b):
                    if p.pid and p.is_alive():
                        p.terminate()
                    if p.pid:
                        p.join(3)

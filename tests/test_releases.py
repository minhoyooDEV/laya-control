from test_jobs import JobTests
from test_store import record

from laya_control.jobs import work
from laya_control.releases import activate, rollback
from laya_control.store import ControlError


class Service:
    def __init__(self):
        self.current = None
        self.fail = False

    def switch(self, pointer):
        if self.fail:
            self.fail = False
            raise ControlError("load failed")
        self.current = pointer
        return {"ready": True, "release_id": pointer["release_id"], "generation": pointer["generation"]}


class ReleaseTests(JobTests):
    def setUp(self):
        super().setUp()
        self.service = Service()

    def test_activation_cas_and_corruption(self):
        j = self.queue()
        work(self.s, self.locks)
        release = activate(self.s, j, 0, "explicit activation", self.service)
        self.assertEqual(release["generation"], 1)
        with self.assertRaises(ControlError):
            activate(self.s, j, 0, "stale activation", self.service)
        (self.s.root / "runs" / j / "model" / "model.safetensors").write_bytes(b"changed")
        with self.assertRaises(ControlError):
            activate(self.s, j, 1, "corrupted", self.service)

    def test_lost_coverage_rejected(self):
        old = self.queue()
        work(self.s, self.locks)
        a = self.s.submit(record(2), "b")
        self.s.confirm([a["revision_id"]], "human", "review")
        self.snap = self.s.snapshot(self.s.authorize(self.s.revisions(), "human", "union"), "base")
        new = self.queue()
        work(self.s, self.locks)
        activate(self.s, new, 0, "activate union", self.service)
        with self.assertRaises(ControlError):
            activate(self.s, old, 1, "activate subset", self.service)

    def test_failed_switch_restores_old(self):
        j = self.queue()
        work(self.s, self.locks)
        first = activate(self.s, j, 0, "first", self.service)
        self.service.fail = True
        with self.assertRaises(ControlError):
            activate(self.s, j, 1, "second", self.service)
        self.assertEqual(self.service.current["release_id"], first["release_id"])
        self.assertEqual(self.s.status()["active"][0]["generation"], 1)

    def test_no_implicit_activation(self):
        j = self.queue()
        work(self.s, self.locks)
        self.assertEqual(self.s.status()["active"], [])
        with self.assertRaises(ControlError):
            activate(self.s, j, 0, "", self.service)

    def test_rollback_is_audited_new_generation(self):
        j = self.queue()
        work(self.s, self.locks)
        first = activate(self.s, j, 0, "first", self.service)
        activate(self.s, j, 1, "second", self.service)
        restored = rollback(self.s, first["release_id"], 2, "explicit rollback", self.service)
        self.assertEqual(restored["generation"], 3)
        self.assertEqual(restored["rollback_of"], first["release_id"])

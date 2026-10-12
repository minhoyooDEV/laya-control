import tempfile
import unittest
from pathlib import Path

from remote_fakes import JOB_ID, Unavailable, fake_remote, make_snapshot

from laya_control import remote
from laya_control.store import ControlError, Store


class RemoteStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "state"

    def tearDown(self):
        self.tmp.cleanup()

    def submitted(self, r):
        store = Store(self.root)
        snapshot = make_snapshot(store)
        remote.submit_remote(store, r, snapshot["snapshot_id"], {"train": {}}, "compact-laya", "req-1")

    def test_no_records_means_no_client_and_no_state_created(self):
        def factory():
            raise AssertionError("client must not be built")

        self.assertEqual(remote.remote_status(self.root, factory), {"jobs": []})
        self.assertFalse(self.root.exists())

    def test_shows_live_server_state_for_each_local_record(self):
        r = fake_remote()
        self.submitted(r)
        r.client.jobs_by_id[JOB_ID] = {"id": JOB_ID, "status": "running", "step": 12, "total": 2776,
                                       "wait_reason": None, "version": None, "error": "secret text"}
        (job,) = remote.remote_status(self.root, lambda: r)["jobs"]
        self.assertEqual((job["id"], job["state"], job["step"], job["total"]), (JOB_ID, "running", 12, 2776))
        self.assertNotIn("secret", str(job))

    def test_unreachable_server_degrades_to_unavailable_entries(self):
        r = fake_remote()
        self.submitted(r)
        r.client.fail = Unavailable("down")
        (job,) = remote.remote_status(self.root, lambda: r)["jobs"]
        self.assertEqual(job["state"], "unavailable")

    def test_missing_client_checkout_degrades_too(self):
        self.submitted(fake_remote())

        def factory():
            raise ControlError("laya-poc client not found")

        (job,) = remote.remote_status(self.root, factory)["jobs"]
        self.assertEqual((job["state"], job["error"]), ("unavailable", "laya-poc client not found"))


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from remote_fakes import (
    BASE_FILES,
    JOB_ID,
    QUESTIONS,
    ServerError,
    fake_remote,
    make_snapshot,
    schema_hash_of,
)

from laya_control import remote
from laya_control.store import ControlError, Store

CFG = {
    "device": "mps",
    "mlx_python": "/x/python",
    "train": {"epochs": 4, "micro_batch": 2, "grad_accum": 16, "freeze_encoder": False, "seed": 0,
              "gradient_checkpointing": True},
    "policy": {"min_accuracy": 0.8, "max_regression": 0.0},
}


class SubmitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "state")
        self.snapshot = make_snapshot(self.store)
        self.sid = self.snapshot["snapshot_id"]

    def tearDown(self):
        self.tmp.cleanup()

    def submit(self, r, request_id="req-1", cfg=CFG):
        return remote.submit_remote(self.store, r, self.sid, cfg, "compact-laya", request_id)

    def test_sends_four_splits_with_mapped_config_and_registers_missing_task(self):
        r = fake_remote()
        result = self.submit(r)
        self.assertEqual(result["id"], JOB_ID)
        self.assertEqual(result["ignored_train_keys"], ["gradient_checkpointing"])
        sent = r.client.submitted
        self.assertEqual(sent["task"], "compact-laya-focus-v2")
        self.assertEqual(sent["config"], {"epochs": 4, "micro_batch": 2, "grad_accum": 16, "freeze_encoder": False, "seed": 0})
        self.assertEqual(set(sent["splits"]), set(remote.SPLITS))
        first = json.loads(sent["splits"]["test"].splitlines()[0])
        self.assertEqual(set(first), {"state", "questions", "expected"})
        self.assertIn(("put_task", "compact-laya-focus-v2", "compact-laya"), r.client.calls)
        record = json.loads((self.store.root / "remote" / f"{JOB_ID}.json").read_text())
        self.assertEqual(record["split_counts"]["test"], len(self.snapshot["benchmark"]))
        self.assertEqual(record["policy"], CFG["policy"])

    def test_base_mismatch_refuses_before_any_upload(self):
        r = fake_remote(base_files=dict(BASE_FILES, **{"model.safetensors": "cc" * 32}))
        with self.assertRaises(ControlError) as caught:
            self.submit(r)
        self.assertIn("base_revision", str(caught.exception))
        self.assertFalse(hasattr(r.client, "submitted"))
        self.assertFalse((self.store.root / "remote").exists())

    def test_registered_task_with_other_schema_is_refused(self):
        r = fake_remote(tasks=[{"task": "compact-laya-focus-v2", "schema_hash": "0" * 16}])
        with self.assertRaises(ControlError):
            self.submit(r)
        self.assertFalse(hasattr(r.client, "submitted"))

    def test_registered_task_with_same_schema_is_not_overwritten(self):
        r = fake_remote(tasks=[{"task": "compact-laya-focus-v2", "schema_hash": schema_hash_of(QUESTIONS)}])
        self.submit(r)
        self.assertNotIn("put_task", [c[0] for c in r.client.calls])

    def test_replayed_request_returns_the_same_job_without_a_second_upload(self):
        r = fake_remote()
        first = self.submit(r)
        uploads = [c for c in r.client.calls if c[0] == "submit_job"]
        again = self.submit(r)
        self.assertEqual(first["id"], again["id"])
        self.assertEqual(uploads, [c for c in r.client.calls if c[0] == "submit_job"])

    def test_request_id_reuse_with_changed_config_is_refused(self):
        r = fake_remote()
        self.submit(r)
        with self.assertRaises(ControlError):
            self.submit(r, cfg=dict(CFG, train=dict(CFG["train"], epochs=1)))

    def test_failed_upload_leaves_marker_that_blocks_blind_retry(self):
        r = fake_remote()
        r.client.submit_job = lambda *a, **k: (_ for _ in ()).throw(ServerError(500, "internal"))
        with self.assertRaises(ControlError):
            self.submit(r)
        with self.assertRaises(ControlError) as caught:
            self.submit(fake_remote())
        self.assertIn("outcome unknown", str(caught.exception))

    def test_unreviewed_label_change_blocks_submission(self):
        with self.store.tx() as db:
            db.execute("UPDATE examples SET confirmed=0 WHERE id=?", (self.snapshot["revisions"][0],))
        with self.assertRaises(ControlError):
            self.submit(fake_remote())

    def test_unavailable_server_is_a_controlled_error(self):
        from remote_fakes import Unavailable

        with self.assertRaises(ControlError) as caught:
            self.submit(fake_remote(fail=Unavailable("down")))
        self.assertIn("unavailable", str(caught.exception))

    def test_unknown_snapshot_is_refused(self):
        with self.assertRaises(ControlError):
            remote.submit_remote(self.store, fake_remote(), "nope", CFG, "o", "r")


if __name__ == "__main__":
    unittest.main()

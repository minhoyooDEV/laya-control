import json
import tempfile
import unittest
from pathlib import Path

from remote_fakes import JOB_ID, SealedSplits, ServerError, fake_remote, make_snapshot, ready_job

from laya_control import remote
from laya_control.store import ControlError, Store

CFG = {"train": {"epochs": 1}, "policy": {"min_accuracy": 0.8, "max_regression": 0.0}}


class ActivateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "state")
        self.snapshot = make_snapshot(self.store)
        self.r = fake_remote()
        remote.submit_remote(self.store, self.r, self.snapshot["snapshot_id"], CFG, "compact-laya", "req-1")
        self.record = json.loads((self.store.root / "remote" / f"{JOB_ID}.json").read_text())

    def tearDown(self):
        self.tmp.cleanup()

    def put_job(self, **kwargs):
        job = ready_job(split_counts=self.record["split_counts"], **kwargs)
        self.r.client.jobs_by_id[JOB_ID] = job
        return job

    def activate(self, reason="Apply reviewed candidate"):
        return remote.activate_remote(self.store, self.r, JOB_ID, reason)

    def test_passing_job_calls_server_activate_with_its_version(self):
        self.put_job(version=3)
        result = self.activate()
        self.assertIn(("activate", "compact-laya-focus-v2", 3), self.r.client.calls)
        self.assertEqual(result["version"], 3)
        saved = json.loads((self.store.root / "remote" / f"{JOB_ID}.json").read_text())
        self.assertEqual(saved["activation"]["version"], 3)
        with self.store.connect(readonly=True) as db:
            kinds = [row[0] for row in db.execute("SELECT kind FROM events WHERE ref=?", (JOB_ID,))]
        self.assertEqual(kinds, ["remote_submitted", "remote_activated"])

    def saved(self):
        return json.loads((self.store.root / "remote" / f"{JOB_ID}.json").read_text())

    def deleted(self):
        return [c for c in self.r.client.calls if c[0] == "delete_job_data"]

    def test_failing_gate_never_calls_activate_and_records_the_rejection(self):
        job = self.put_job()
        job["metrics"]["parity"]["passed"] = False
        with self.assertRaises(ControlError) as caught:
            self.activate()
        self.assertIn("parity not passed", str(caught.exception))
        self.assertNotIn("activate", [c[0] for c in self.r.client.calls])
        saved = self.saved()
        self.assertFalse(saved["decision"]["passed"])
        self.assertEqual(saved["decision"]["reasons"], ["parity not passed"])
        self.assertIsNone(saved["activation"])

    def test_dev_rejected_job_decision_never_reads_the_test_split(self):
        job = self.put_job(candidate=0.6, base=0.5)
        splits = SealedSplits(job["metrics"]["splits"])
        job["metrics"]["splits"] = splits
        with self.assertRaises(ControlError) as caught:
            self.activate()
        self.assertIn("dev policy not met", str(caught.exception))
        self.assertEqual(splits.test_reads, 0)
        self.assertNotIn("activate", [c[0] for c in self.r.client.calls])

    def test_server_data_is_kept_after_activation_and_after_rejection(self):
        self.put_job(version=2)
        result = self.activate()
        self.assertEqual(self.deleted(), [])
        self.assertNotIn("data_deleted", result)
        self.assertNotIn("data_deleted", self.saved())
        job = self.put_job()
        job["metrics"]["parity"]["passed"] = False
        with self.assertRaises(ControlError):
            self.activate()
        self.assertEqual(self.deleted(), [])
        self.assertNotIn("data_deleted", self.saved())

    def test_decision_record_and_result_carry_balanced_accuracies_and_warnings(self):
        job = self.put_job(version=2)
        job["metrics"]["temperature_at_bound"] = True
        job["metrics"]["calibration"]["noul"]["issues"] = ["the fitted temperature 5 is on the [0.5, 5] clamp"]
        result = self.activate()
        decision = self.saved()["decision"]
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["warnings"], ["temperature on the clamp: noul", "temperature at bound"])
        for split in ("dev", "test"):
            table = decision["splits"][split]["balanced"]
            self.assertEqual(set(table), {"edit", "pr"})
            self.assertAlmostEqual(table["edit"]["candidate"], 0.9)
            self.assertAlmostEqual(table["edit"]["base"], 0.8)
        self.assertEqual(result["evaluation"]["warnings"], decision["warnings"])
        self.assertEqual(result["evaluation"]["splits"]["dev"]["balanced"], decision["splits"]["dev"]["balanced"])

    def test_rejection_record_keeps_the_dev_table_and_no_test_table(self):
        self.put_job(candidate=0.6, base=0.5)
        with self.assertRaises(ControlError):
            self.activate()
        decision = self.saved()["decision"]
        self.assertEqual(set(decision["splits"]), {"dev"})
        self.assertIn("balanced", decision["splits"]["dev"])

    def test_server_activate_failure_records_nothing_as_activated(self):
        self.put_job()
        self.r.client.activate = lambda *a: (_ for _ in ()).throw(ServerError(409, "parity_failed"))
        with self.assertRaises(ControlError):
            self.activate()
        self.assertIsNone(self.saved()["activation"])

    def test_unfinished_job_is_not_decided(self):
        self.put_job(status="evaluating")
        with self.assertRaises(ControlError):
            self.activate()
        self.assertNotIn("decision", self.saved())

    def test_job_that_is_not_the_submitted_one_is_refused(self):
        for field, value in (("schema_hash", "0" * 16), ("task", "other"), ("split_counts", {"train": 1})):
            job = self.put_job()
            job[field] = value
            with self.assertRaises(ControlError, msg=field):
                self.activate()
        self.assertNotIn("activate", [c[0] for c in self.r.client.calls])
        self.assertEqual(self.deleted(), [])

    def test_version_zero_or_missing_is_refused(self):
        for version in (0, None, True):
            self.put_job(version=version)
            with self.assertRaises(ControlError):
                self.activate()

    def test_labels_changed_after_submission_block_activation(self):
        self.put_job()
        with self.store.tx() as db:
            db.execute("UPDATE examples SET confirmed=0 WHERE id=?", (self.snapshot["revisions"][0],))
        with self.assertRaises(ControlError):
            self.activate()

    def test_unknown_or_malformed_job_id_never_touches_the_filesystem_outside_remote(self):
        for bad in ("../control", "short", "ZZZZZZZZZZZZ", "0123456789abc"):
            with self.assertRaises(ControlError):
                remote.activate_remote(self.store, self.r, bad, "reason")
        with self.assertRaises(ControlError):
            remote.activate_remote(self.store, self.r, "ffffffffffff", "reason")

    def test_reason_required_and_server_rejection_is_reported(self):
        self.put_job()
        with self.assertRaises(ControlError):
            self.activate(reason="")
        self.r.client.activate = lambda *a: (_ for _ in ()).throw(ServerError(409, "parity_failed"))
        with self.assertRaises(ControlError) as caught:
            self.activate()
        self.assertIn("409 parity_failed", str(caught.exception))
        saved = json.loads((self.store.root / "remote" / f"{JOB_ID}.json").read_text())
        self.assertIsNone(saved["activation"])


if __name__ == "__main__":
    unittest.main()

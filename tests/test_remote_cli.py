import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from remote_fakes import JOB_ID, fake_remote, make_snapshot, ready_job

from laya_control import cli
from laya_control.store import Store

CFG = {
    "device": "mps",
    "train": {"epochs": 4, "micro_batch": 2, "grad_accum": 16, "freeze_encoder": False, "seed": 0},
    "policy": {"min_accuracy": 0.8, "max_regression": 0.0},
}


class RemoteCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "state"
        self.snapshot = make_snapshot(Store(self.root))
        self.config = Path(self.tmp.name) / "config.json"
        self.config.write_text(json.dumps(CFG))
        self.r = fake_remote()

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with (
            mock.patch("laya_control.remote.make_remote", return_value=self.r),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            code = cli.main(["--state-root", str(self.root), *args])
        return code, out.getvalue(), err.getvalue()

    def train(self, *extra):
        return self.run_cli(
            "train", "--snapshot", self.snapshot["snapshot_id"], "--config", str(self.config),
            "--owner", "compact-laya", "--request-id", "req-1", *extra,
        )

    def test_train_remote_needs_no_python_or_base_and_submits(self):
        code, out, err = self.train("--remote")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["id"], JOB_ID)
        self.assertEqual(self.r.client.submitted["config"]["epochs"], 4)

    def test_local_train_still_requires_python_and_base(self):
        code, _, err = self.train()
        self.assertEqual(code, 2)
        self.assertIn("--python and --base", err)
        self.assertFalse(hasattr(self.r.client, "submitted"))

    def test_remote_failure_exits_2_with_controlled_message(self):
        self.r.client.base_files = {"model.safetensors": "00"}
        code, _, err = self.train("--remote")
        self.assertEqual(code, 2)
        self.assertIn("base_revision", err)

    def test_status_remote_lists_remote_jobs_and_plain_status_does_not_call_server(self):
        self.assertEqual(self.train("--remote")[0], 0)
        self.r.client.jobs_by_id[JOB_ID] = {"id": JOB_ID, "status": "queued", "wait_reason": "memory"}
        before = len(self.r.client.calls)
        code, out, _ = self.run_cli("status", "--remote")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["remote"]["jobs"][0]["state"], "queued")
        calls_after_remote = len(self.r.client.calls)
        self.assertGreater(calls_after_remote, before)
        code, out, _ = self.run_cli("status")
        self.assertNotIn("remote", json.loads(out))
        self.assertEqual(len(self.r.client.calls), calls_after_remote)

    def test_activate_remote_needs_no_local_serving_arguments(self):
        self.assertEqual(self.train("--remote")[0], 0)
        record = json.loads((self.root / "remote" / f"{JOB_ID}.json").read_text())
        self.r.client.jobs_by_id[JOB_ID] = ready_job(split_counts=record["split_counts"], version=2)
        code, out, err = self.run_cli("activate", "--remote", "--run-id", JOB_ID, "--reason", "Apply reviewed candidate")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["version"], 2)
        shown = json.loads(out)["evaluation"]
        self.assertEqual(shown["warnings"], [])
        self.assertIn("edit", shown["splits"]["dev"]["balanced"])
        self.assertIn("edit", shown["splits"]["test"]["balanced"])

    def test_local_activate_without_serving_arguments_is_refused_and_rollback_unchanged(self):
        code, _, err = self.run_cli("activate", "--run-id", JOB_ID, "--reason", "x")
        self.assertEqual(code, 2)
        self.assertIn("unless --remote", err)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            cli.main(["--state-root", str(self.root), "rollback", "--release-id", "r", "--reason", "x"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

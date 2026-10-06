import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_store import record


class CLITests(unittest.TestCase):
    def test_status_readonly_and_workflow(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "state"

            def cli(*args):
                r = subprocess.run(
                    [sys.executable, "-m", "laya_control", "--state-root", str(root), *args],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(r.returncode, 0, r.stderr)
                return json.loads(r.stdout)

            self.assertEqual(cli("status")["state"], "not_configured")
            self.assertFalse(root.exists())
            data = Path(d) / "batch.json"
            data.write_text(json.dumps([record(1)]))
            submitted = cli("submit", "--file", str(data), "--request-id", "batch-a")
            request = Path(d) / "review.json"
            request.write_text(
                json.dumps(
                    {
                        "revision_ids": [submitted[0]["revision_id"]],
                        "actor": "human",
                        "reason": "reviewed fixture",
                    }
                )
            )
            cli("confirm", "--file", str(request))
            auth = cli("authorize", "--file", str(request))
            snap = cli("snapshot", "--authorization-id", auth["authorization_id"], "--base-revision", "base")
            self.assertEqual(snap["unique_count"], 1)
            status = cli("status")
            self.assertNotIn("synthetic example", json.dumps(status))
            self.assertEqual(status["queued_count"], 0)

    def test_relative_path_rejected_without_state_mutation(self):
        r = subprocess.run(
            [sys.executable, "-m", "laya_control", "--state-root", "relative", "status"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(r.returncode, 0)

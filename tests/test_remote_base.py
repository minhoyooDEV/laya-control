import tempfile
import unittest
from pathlib import Path

from remote_fakes import BASE_FILES, BASE_REVISION, QUESTIONS, make_snapshot, noul_record

from laya_control import remote
from laya_control.store import ControlError, Store


class BaseAndSchemaTests(unittest.TestCase):
    def test_matching_base_passes_and_changed_file_is_refused(self):
        snapshot = {"base_revision": BASE_REVISION}
        remote.check_base(snapshot, {"base": {"files": BASE_FILES}})
        changed = dict(BASE_FILES, **{"model.safetensors": "cc" * 32})
        with self.assertRaises(ControlError) as caught:
            remote.check_base(snapshot, {"base": {"files": changed}})
        self.assertIn("base_revision", str(caught.exception))

    def test_extra_server_file_changes_the_identity(self):
        extra = dict(BASE_FILES, **{".DS_Store": "dd" * 32})
        with self.assertRaises(ControlError):
            remote.check_base({"base_revision": BASE_REVISION}, {"base": {"files": extra}})

    def test_missing_or_malformed_hashes_are_refused(self):
        for status in ({}, {"base": {}}, {"base": {"files": {}}}, {"base": {"files": {"a": 1}}}):
            with self.assertRaises(ControlError):
                remote.check_base({"base_revision": BASE_REVISION}, status)

    def test_schema_hash_matches_the_contract_formula(self):
        import hashlib
        import json

        expected = hashlib.sha256(
            json.dumps(QUESTIONS, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        self.assertEqual(remote.schema_hash(QUESTIONS), expected)
        self.assertEqual(len(expected), 16)

    def test_config_keeps_only_the_five_server_keys(self):
        kept, ignored = remote.map_config(
            {"epochs": 4, "micro_batch": 2, "grad_accum": 16, "freeze_encoder": False, "seed": 0,
             "gradient_checkpointing": True, "loss": "rlcd"}
        )
        self.assertEqual(kept, {"epochs": 4, "micro_batch": 2, "grad_accum": 16, "freeze_encoder": False, "seed": 0})
        self.assertEqual(ignored, ["gradient_checkpointing", "loss"])
        with self.assertRaises(ControlError):
            remote.map_config([])


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "state")
        self.snapshot = make_snapshot(self.store)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sealed_benchmark_is_the_test_split_and_rows_have_the_server_shape(self):
        splits, counts = remote.build_splits(self.snapshot, lambda rows: rows)
        self.assertEqual(set(splits), set(remote.SPLITS))
        self.assertEqual(counts["test"], len(self.snapshot["benchmark"]))
        for name, rows in splits.items():
            self.assertEqual(len(rows), counts[name])
            self.assertEqual(set(rows[0]), {"state", "questions", "expected"})
        self.assertIsInstance(splits["train"][0]["expected"]["edit"], bool)

    def test_incomplete_benchmark_is_refused(self):
        broken = dict(self.snapshot, benchmark=self.snapshot["benchmark"] + ["not-in-test"])
        with self.assertRaises(ControlError):
            remote.build_splits(broken, lambda rows: rows)

    def test_empty_split_is_refused(self):
        emptied = dict(self.snapshot, splits=dict(self.snapshot["splits"], dev=[]))
        with self.assertRaises(ControlError):
            remote.build_splits(emptied, lambda rows: rows)

    def test_other_tasks_are_refused(self):
        self.assertEqual(remote.snapshot_questions(self.snapshot), QUESTIONS)
        other = {"examples": {"a": dict(noul_record(1), task_id="demo")}}
        with self.assertRaises(ControlError):
            remote.snapshot_questions(other)


if __name__ == "__main__":
    unittest.main()

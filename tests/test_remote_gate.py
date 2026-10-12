import copy
import unittest

from remote_fakes import QUESTIONS, SealedSplits, metric, ready_job

from laya_control import remote
from laya_control.store import ControlError
from laya_control.training import passes_policy, score_predictions

POLICY = {"min_accuracy": 0.8, "max_regression": 0.0}


class ToScoreTests(unittest.TestCase):
    def test_noul_labels_become_python_bool_keys_like_score_predictions(self):
        score = remote.to_score(metric(0.9, true_recall=0.7, false_recall=0.95, items=10), QUESTIONS)
        self.assertEqual(score["count"], 20)
        self.assertAlmostEqual(score["accuracy"], 0.9)
        self.assertEqual(
            score["per_class_recall"],
            {"edit:True": 0.7, "edit:False": 0.95, "pr:True": 0.7, "pr:False": 0.95},
        )
        rows = [{"state": "s", "questions": {"edit": {"type": "noul"}}, "expected": {"edit": True}}]
        local = score_predictions(rows, [{"edit": {"noul": 0.9}}])
        self.assertIn("edit:True", local["per_class_recall"])

    def test_choice_labels_pass_through(self):
        questions = {"screen": {"type": "choice", "criteria": ["home", "pdp"]}}
        m = {"items": 4, "questions": {"screen": {"accuracy": 0.75, "ece": 0.1, "recall": {"home": 1.0, "pdp": 0.5}}}}
        self.assertEqual(remote.to_score(m, questions)["per_class_recall"], {"screen:home": 1.0, "screen:pdp": 0.5})

    def test_score_questions_and_malformed_metrics_are_refused(self):
        with self.assertRaises(ControlError):
            remote.to_score(metric(0.9), {"edit": {"type": "score"}, "pr": {"type": "noul"}})
        for bad in (
            {"items": 0, "questions": {}},
            {"items": 5, "questions": {"edit": {}}},
            {"items": 5, "questions": {q: {"accuracy": None, "recall": {}} for q in QUESTIONS}},
            {"items": 5, "questions": {q: {"accuracy": float("nan"), "recall": {}} for q in QUESTIONS}},
            {"items": 5, "questions": {"edit": {"accuracy": 1, "recall": {}}}},
            {"items": 5, "questions": {q: {"accuracy": 0.9, "recall": {}} for q in QUESTIONS}},
        ):
            with self.assertRaises(ControlError):
                remote.to_score(bad, QUESTIONS)


class EvaluateTests(unittest.TestCase):
    def verdict(self, job):
        return remote.evaluate(job, QUESTIONS, POLICY)

    def test_good_candidate_passes_dev_and_test(self):
        v = self.verdict(ready_job(candidate=0.9, base=0.8))
        self.assertTrue(v["passed"], v["reasons"])
        self.assertEqual(set(v["splits"]), {"dev", "test"})

    def test_accuracy_below_minimum_fails(self):
        self.assertFalse(self.verdict(ready_job(candidate=0.79, base=0.5))["passed"])

    def test_candidate_worse_than_base_fails(self):
        v = self.verdict(ready_job(candidate=0.85, base=0.9))
        self.assertIn("dev policy not met", v["reasons"])

    def test_single_class_recall_regression_fails_despite_better_accuracy(self):
        job = ready_job(candidate=0.9, base=0.8)
        for split in ("dev", "test"):
            job["metrics"]["splits"][split]["candidate"] = metric(0.95, true_recall=0.7, false_recall=1.0,
                                                                  items=job["metrics"]["splits"][split]["base"]["items"])
            job["metrics"]["splits"][split]["base"] = metric(0.8, true_recall=0.8, false_recall=0.8,
                                                             items=job["metrics"]["splits"][split]["base"]["items"])
        self.assertFalse(self.verdict(job)["passed"])

    def test_dev_rejected_job_never_reads_the_test_split(self):
        job = ready_job(candidate=0.6, base=0.5)
        splits = SealedSplits(job["metrics"]["splits"])
        job["metrics"]["splits"] = splits
        v = self.verdict(job)
        self.assertFalse(v["passed"])
        self.assertEqual(v["reasons"], ["dev policy not met"])
        self.assertEqual(set(v["splits"]), {"dev"})
        self.assertEqual(splits.test_reads, 0)

    def test_missing_dev_metrics_also_stop_before_test(self):
        job = ready_job()
        job["metrics"]["splits"]["dev"]["base"] = None
        splits = SealedSplits(job["metrics"]["splits"])
        job["metrics"]["splits"] = splits
        self.assertIn("dev metrics missing", self.verdict(job)["reasons"])
        self.assertEqual(splits.test_reads, 0)

    def test_test_split_is_read_once_dev_passed(self):
        job = ready_job(candidate=0.9, base=0.8)
        splits = SealedSplits({"dev": job["metrics"]["splits"]["dev"]})
        job["metrics"]["splits"] = splits
        with self.assertRaises(AssertionError):
            self.verdict(job)
        self.assertEqual(splits.test_reads, 1)

    def test_test_split_failure_alone_fails(self):
        job = ready_job(candidate=0.9, base=0.8)
        job["metrics"]["splits"]["test"]["candidate"] = metric(0.6, items=46)
        v = self.verdict(job)
        self.assertEqual(v["reasons"], ["test policy not met"])

    def test_calibration_issue_temperature_bound_and_parity_each_fail(self):
        cases = {}
        job = ready_job()
        job["metrics"]["calibration"]["noul"]["issues"] = ["the fitted temperature 5 is on the [0.5, 5] clamp"]
        cases["calibration issue: noul"] = job
        job = ready_job()
        job["metrics"]["temperature_at_bound"] = True
        cases["temperature at bound"] = job
        job = ready_job()
        job["metrics"]["parity"]["passed"] = False
        cases["parity not passed"] = job
        for reason, job in cases.items():
            self.assertIn(reason, self.verdict(job)["reasons"])

    def test_missing_fields_fail_closed(self):
        for path in (("temperature_at_bound",), ("parity",), ("calibration",)):
            job = ready_job()
            del job["metrics"][path[0]]
            self.assertFalse(self.verdict(job)["passed"], path)
        job = ready_job()
        del job["metrics"]["splits"]["test"]
        self.assertIn("test metrics missing", self.verdict(job)["reasons"])
        job = ready_job()
        job["metrics"]["splits"]["dev"]["base"] = None
        self.assertIn("dev metrics missing", self.verdict(job)["reasons"])

    def test_not_ready_job_fails(self):
        self.assertFalse(self.verdict(ready_job(status="evaluating"))["passed"])
        self.assertFalse(self.verdict(ready_job(metrics=None))["passed"])

    def test_gate_is_identical_to_passes_policy_on_mapped_scores(self):
        job = ready_job(candidate=0.9, base=0.8)
        base = remote.to_score(job["metrics"]["splits"]["dev"]["base"], QUESTIONS)
        cand = remote.to_score(job["metrics"]["splits"]["dev"]["candidate"], QUESTIONS)
        self.assertEqual(passes_policy(base, cand, POLICY), self.verdict(copy.deepcopy(job))["splits"]["dev"]["passed"])


if __name__ == "__main__":
    unittest.main()

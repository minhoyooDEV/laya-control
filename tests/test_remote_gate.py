import unittest

from remote_fakes import QUESTIONS, SealedSplits, metric, ready_job

from laya_control import remote
from laya_control.store import ControlError
from laya_control.training import passes_policy, score_predictions

POLICY = {"min_accuracy": 0.8, "max_regression": 0.0}
CLAMP = "the fitted temperature 5 is on the [0.5, 5] clamp"


def qmetric(items, **per_question):
    """Metric with (accuracy, true_recall, false_recall) per question."""
    return {
        "items": items,
        "questions": {
            q: {"accuracy": a, "ece": 0.05, "recall": {"true": t, "false": f}}
            for q, (a, t, f) in per_question.items()
        },
    }


def counted(items, **per_question):
    """Metric built from integer counts: question -> (positives, true_hits, false_hits), server-rounded."""
    questions = {}
    for q, (positives, true_hits, false_hits) in per_question.items():
        negatives = items - positives
        questions[q] = {
            "accuracy": round((true_hits + false_hits) / items, 4),
            "ece": 0.1,
            "recall": {"true": round(true_hits / positives, 4), "false": round(false_hits / negatives, 4)},
        }
    return {"items": items, "questions": questions}


SMALL = "pr: small class support (<5), compared by accuracy"


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

    def job_with_test(self, base, candidate, dev=None):
        job = ready_job(candidate=0.9, base=0.8)
        job["metrics"]["splits"]["test"]["base"] = base
        job["metrics"]["splits"]["test"]["candidate"] = candidate
        if dev:
            job["metrics"]["splits"]["dev"]["base"], job["metrics"]["splits"]["dev"]["candidate"] = dev
        return job

    def deploy_like(self, positives=1, base_hits=(1, 2), cand_hits=(0, 45)):
        n = 46
        negatives = n - positives
        base = counted(n, edit=(20, 20, 2), pr=(positives, base_hits[0], base_hits[1]))
        cand = counted(n, edit=(20, 17, 24), pr=(positives, cand_hits[0], cand_hits[1]))
        assert negatives >= cand_hits[1]
        return base, cand

    def test_deploy_like_single_positive_missed_passes_by_accuracy_with_a_warning(self):
        base, cand = self.deploy_like()
        sb, sc = remote.to_score(base, QUESTIONS), remote.to_score(cand, QUESTIONS)
        self.assertFalse(remote.passes_balanced(sb, sc, POLICY))  # balanced accuracy alone would reject
        v = self.verdict(self.job_with_test(base, cand))
        self.assertTrue(v["passed"], v["reasons"])
        self.assertEqual(v["warnings"], [SMALL])
        row = v["splits"]["test"]["balanced"]["pr"]
        self.assertEqual(row["metric"], "accuracy")
        self.assertAlmostEqual(row["candidate"], 0.9783, places=4)
        self.assertAlmostEqual(row["base"], 0.0652, places=4)
        self.assertEqual(v["splits"]["test"]["balanced"]["edit"]["metric"], "balanced")

    def test_small_support_question_still_fails_when_its_accuracy_regresses(self):
        base, cand = self.deploy_like(base_hits=(0, 45), cand_hits=(0, 43))
        v = self.verdict(self.job_with_test(base, cand))
        self.assertEqual(v["reasons"], ["test policy not met"])
        self.assertEqual(v["warnings"], [SMALL])

    def test_max_regression_applies_to_the_accuracy_comparison(self):
        base, cand = self.deploy_like(base_hits=(0, 45), cand_hits=(0, 44))
        job = self.job_with_test(base, cand)
        self.assertFalse(self.verdict(job)["passed"])
        loose = {"min_accuracy": 0.8, "max_regression": 0.05}
        self.assertTrue(remote.evaluate(job, QUESTIONS, loose)["passed"])

    def test_support_of_four_is_small_and_five_is_not(self):
        base, cand = self.deploy_like(positives=4, base_hits=(4, 2), cand_hits=(0, 42))
        v = self.verdict(self.job_with_test(base, cand))
        self.assertTrue(v["passed"], v["reasons"])
        self.assertEqual(v["warnings"], [SMALL])
        base, cand = self.deploy_like(positives=5, base_hits=(5, 5), cand_hits=(0, 41))
        v = self.verdict(self.job_with_test(base, cand))
        self.assertEqual(v["reasons"], ["test policy not met"])
        self.assertEqual(v["warnings"], [])
        self.assertEqual(v["splits"]["test"]["balanced"]["pr"]["metric"], "balanced")

    def test_warning_is_listed_once_when_dev_and_test_are_both_small(self):
        base, cand = self.deploy_like()
        v = self.verdict(self.job_with_test(base, cand, dev=(base, cand)))
        self.assertTrue(v["passed"], v["reasons"])
        self.assertEqual(v["warnings"], [SMALL])
        self.assertEqual(v["splits"]["dev"]["balanced"]["pr"]["metric"], "accuracy")

    def test_class_missing_from_the_recall_dict_counts_as_small(self):
        job = ready_job(candidate=0.9, base=0.5)
        for side, acc in (("base", 0.5), ("candidate", 0.95)):
            job["metrics"]["splits"]["test"][side]["questions"]["pr"] = {
                "accuracy": acc, "ece": 0.1, "recall": {"false": acc}
            }
        v = self.verdict(job)
        self.assertTrue(v["passed"], v["reasons"])
        self.assertEqual(v["warnings"], [SMALL])

    def test_explicit_support_from_the_server_is_used_when_present(self):
        job = ready_job(candidate=0.9, base=0.5)
        for side in ("base", "candidate"):
            entry = job["metrics"]["splits"]["test"][side]["questions"]["pr"]
            entry["support"] = {"true": 3, "false": 43}
        v = self.verdict(job)
        self.assertEqual(v["warnings"], [SMALL])
        job["metrics"]["splits"]["test"]["candidate"]["questions"]["pr"]["support"] = {"true": 3, "false": 99}
        self.assertEqual(self.verdict(job)["warnings"], [])  # counts that do not add up are ignored

    def test_numbers_that_fit_no_integer_counts_fall_back_to_balanced(self):
        base = metric(0.6, true_recall=0.9, false_recall=0.3, items=46)
        cand = metric(0.95, true_recall=0.2, false_recall=0.97, items=46)
        v = self.verdict(self.job_with_test(base, cand))
        self.assertEqual(v["warnings"], [])
        self.assertEqual(v["reasons"], ["test policy not met"])

    def test_choice_questions_are_never_treated_as_small(self):
        questions = {"screen": {"type": "choice", "criteria": ["home", "pdp"]}}
        m = {"items": 46, "questions": {"screen": {"accuracy": 0.9783, "ece": 0.1, "recall": {"home": 0.0, "pdp": 1.0}}}}
        self.assertEqual(remote.to_score(m, questions)["positives"], {})

    def test_good_candidate_passes_dev_and_test(self):
        v = self.verdict(ready_job(candidate=0.9, base=0.8))
        self.assertTrue(v["passed"], v["reasons"])
        self.assertEqual(set(v["splits"]), {"dev", "test"})

    def test_accuracy_below_minimum_fails(self):
        self.assertFalse(self.verdict(ready_job(candidate=0.79, base=0.5))["passed"])

    def test_candidate_worse_than_base_fails(self):
        v = self.verdict(ready_job(candidate=0.85, base=0.9))
        self.assertIn("dev policy not met", v["reasons"])

    def test_always_yes_base_no_longer_blocks_a_better_model(self):
        job = ready_job(candidate=0.9, base=0.5)
        for split in ("dev", "test"):
            items = job["metrics"]["splits"][split]["base"]["items"]
            job["metrics"]["splits"][split]["base"] = metric(0.5, true_recall=1.0, false_recall=0.0, items=items)
            job["metrics"]["splits"][split]["candidate"] = metric(0.9, true_recall=0.8, false_recall=0.95, items=items)
        base = remote.to_score(job["metrics"]["splits"]["dev"]["base"], QUESTIONS)
        cand = remote.to_score(job["metrics"]["splits"]["dev"]["candidate"], QUESTIONS)
        self.assertFalse(passes_policy(base, cand, POLICY))  # the per-class recall rule used to reject this
        v = self.verdict(job)
        self.assertTrue(v["passed"], v["reasons"])
        self.assertAlmostEqual(v["splits"]["dev"]["balanced"]["edit"]["base"], 0.5)
        self.assertAlmostEqual(v["splits"]["dev"]["balanced"]["edit"]["candidate"], 0.875)

    def test_one_question_with_lower_balanced_accuracy_fails_despite_better_pooled_accuracy(self):
        job = ready_job(candidate=0.9, base=0.8)
        for split in ("dev", "test"):
            items = job["metrics"]["splits"][split]["base"]["items"]
            job["metrics"]["splits"][split]["base"] = qmetric(items, edit=(0.8, 0.8, 0.8), pr=(0.5, 0.5, 0.5))
            job["metrics"]["splits"][split]["candidate"] = qmetric(items, edit=(0.9, 0.6, 0.9), pr=(0.95, 0.95, 0.95))
        v = self.verdict(job)
        self.assertEqual(v["reasons"], ["dev policy not met"])
        self.assertAlmostEqual(v["splits"]["dev"]["balanced"]["edit"]["candidate"], 0.75)
        self.assertGreater(v["splits"]["dev"]["candidate_accuracy"], v["splits"]["dev"]["baseline_accuracy"])

    def test_max_regression_is_the_balanced_accuracy_tolerance(self):
        job = ready_job(candidate=0.9, base=0.8)
        job["metrics"]["splits"]["dev"]["base"] = qmetric(45, edit=(0.9, 0.9, 0.9), pr=(0.9, 0.9, 0.9))
        job["metrics"]["splits"]["dev"]["candidate"] = qmetric(45, edit=(0.9, 0.87, 0.87), pr=(0.9, 0.9, 0.9))
        self.assertFalse(remote.evaluate(job, QUESTIONS, POLICY)["passed"])
        loose = {"min_accuracy": 0.8, "max_regression": 0.05}
        self.assertTrue(remote.evaluate(job, QUESTIONS, loose)["passed"])

    def test_pooled_accuracy_below_minimum_fails_even_when_balanced_accuracy_improves(self):
        job = ready_job(candidate=0.79, base=0.5)
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

    def test_clamp_only_issue_and_bound_flag_pass_with_warnings(self):
        job = ready_job()
        job["metrics"]["calibration"]["noul"]["issues"] = [CLAMP]
        job["metrics"]["temperature_at_bound"] = True
        v = self.verdict(job)
        self.assertTrue(v["passed"], v["reasons"])
        self.assertEqual(v["reasons"], [])
        self.assertEqual(v["warnings"], ["temperature on the clamp: noul", "temperature at bound"])

    def test_bound_flag_alone_passes_with_a_warning(self):
        job = ready_job()
        job["metrics"]["temperature_at_bound"] = True
        v = self.verdict(job)
        self.assertTrue(v["passed"])
        self.assertEqual(v["warnings"], ["temperature at bound"])

    def test_clean_job_has_no_warnings(self):
        self.assertEqual(self.verdict(ready_job())["warnings"], [])

    def test_any_other_calibration_issue_rejects_even_next_to_a_clamp_message(self):
        for issues in (
            ["fitted on only 40 calibration items"],
            ["the fit came back unchanged at 1.0 (accuracy 0.90)"],
            ["not fitted: 5 calibration items, fewer than 10, so the temperature stays 1.0"],
            [CLAMP, "fitted on only 40 calibration items"],
        ):
            job = ready_job()
            job["metrics"]["calibration"]["noul"]["issues"] = issues
            job["metrics"]["temperature_at_bound"] = True
            v = self.verdict(job)
            self.assertFalse(v["passed"], issues)
            self.assertIn("calibration issue: noul", v["reasons"])

    def test_clamp_message_must_be_the_exact_half_to_five_clamp(self):
        job = ready_job()
        job["metrics"]["calibration"]["noul"]["issues"] = ["the fitted temperature 9 is on the [0.1, 9] clamp"]
        self.assertFalse(self.verdict(job)["passed"])

    def test_parity_failure_still_rejects_even_with_only_warnings_otherwise(self):
        job = ready_job()
        job["metrics"]["calibration"]["noul"]["issues"] = [CLAMP]
        job["metrics"]["temperature_at_bound"] = True
        job["metrics"]["parity"]["passed"] = False
        v = self.verdict(job)
        self.assertFalse(v["passed"])
        self.assertEqual(v["reasons"], ["parity not passed"])
        self.assertEqual(len(v["warnings"]), 2)

    def test_bound_flag_that_is_not_a_boolean_fails_closed(self):
        for value in (None, "true", 1):
            job = ready_job()
            job["metrics"]["temperature_at_bound"] = value
            self.assertIn("temperature_at_bound missing", self.verdict(job)["reasons"])

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

    def test_balanced_gate_matches_a_hand_computed_table(self):
        base = remote.to_score(qmetric(10, edit=(0.5, 1.0, 0.0), pr=(0.6, 0.7, 0.5)), QUESTIONS)
        cand = remote.to_score(qmetric(10, edit=(0.9, 0.8, 1.0), pr=(0.8, 0.8, 0.8)), QUESTIONS)
        self.assertEqual(base["balanced"], {"edit": 0.5, "pr": 0.6})
        self.assertEqual(cand["balanced"], {"edit": 0.9, "pr": 0.8})
        self.assertTrue(remote.passes_balanced(base, cand, POLICY))
        self.assertFalse(remote.passes_balanced(cand, base, POLICY))
        self.assertFalse(remote.passes_balanced(base, dict(cand, count=1), POLICY))
        self.assertFalse(remote.passes_balanced(dict(base, count=0), dict(cand, count=0), POLICY))

if __name__ == "__main__":
    unittest.main()

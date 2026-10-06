import unittest

from laya_control.store import ControlError
from laya_control.training import passes_policy, rows_for, score_predictions


class TrainingTests(unittest.TestCase):
    def test_class_regression_rejected(self):
        base = {"accuracy": 0.9, "per_class_recall": {"a": 1.0, "b": 0.8}, "count": 10}
        worse = {"accuracy": 0.95, "per_class_recall": {"a": 0.7, "b": 1.0}, "count": 10}
        self.assertFalse(passes_policy(base, worse, {"min_accuracy": 0.8, "max_regression": 0.05}))

    def test_empty_eval_cannot_pass(self):
        self.assertFalse(
            passes_policy(
                {"accuracy": 0, "per_class_recall": {}, "count": 0},
                {"accuracy": 0, "per_class_recall": {}, "count": 0},
                {},
            )
        )

    def test_metrics(self):
        rows = [
            {"state": "one", "questions": {"q": {"type": "choice"}}, "expected": {"q": "a"}},
            {"state": "two", "questions": {"q": {"type": "choice"}}, "expected": {"q": "b"}},
        ]
        predictions = [{"q": {"choice": "a", "confidence": 0.9}}, {"q": {"choice": "a", "confidence": 0.9}}]
        result = score_predictions(rows, predictions)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["per_class_recall"]["q:b"], 0)

    def test_unsupported_metric_fail_closed(self):
        with self.assertRaises(ControlError):
            score_predictions([{"questions": {"q": {"type": "other"}}, "expected": {"q": 1}}], [{"q": {}}])

    def test_training_rows_exclude_test(self):
        snapshot = {
            "splits": {"train": ["a"], "test": ["b"]},
            "examples": {
                "a": {"state": "a", "schema": {}, "expected": {}},
                "b": {"state": "b", "schema": {}, "expected": {}},
            },
        }
        self.assertEqual(len(rows_for(snapshot, "train")), 1)
        self.assertEqual(rows_for(snapshot, "train")[0]["state"], "a")

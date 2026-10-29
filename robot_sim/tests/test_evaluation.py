import math
import unittest
from robot_sim.evaluation import score_task, task_request, validate_decision


class EvaluationTests(unittest.TestCase):
    def test_sequence_scored_without_order_insensitivity(self):
        self.assertTrue(score_task(["green", "red"], ["green", "red"]))
        self.assertFalse(score_task(["green", "red"], ["red", "green"]))
        self.assertFalse(score_task(["red"], []))
        self.assertTrue(score_task(["stop"], ["stop"]))
        self.assertIsNone(score_task(None, ["blue"]))

    def test_expected_labels_are_not_sent_to_policy(self):
        request = task_request({"instruction": "Read the task", "expected": ["red"]}, ["blue"])
        self.assertEqual(request, {"instruction": "Read the task", "completed_markers": ["blue"]})
        self.assertNotIn("expected", request)


class PolicyValidationTests(unittest.TestCase):
    def test_non_finite_scores_cannot_move_robot(self):
        for value in (math.nan, math.inf, -1, 2):
            with self.assertRaises(ValueError):
                validate_decision({"probabilities": {"red": value}})
        with self.assertRaises(ValueError):
            validate_decision({"probabilities": {"red": 0.3}})
        validate_decision({"probabilities": {"red": 0.7, "blue": 0.3}})

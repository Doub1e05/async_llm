import math
import unittest

from robot_sim.direct_control import decision_velocity, local_target, target_metrics, discrete_target_features, goal_snapshot_changed


class DirectControlTests(unittest.TestCase):
    def test_world_target_is_relative_to_initial_odom(self):
        self.assertEqual(local_target([-4, -3, 0], [-2.6, -3]), (1.4, 0.0))
        target = local_target([1, 2, math.pi / 2], [1, 3])
        self.assertAlmostEqual(target[0], 1.0)
        self.assertAlmostEqual(target[1], 0.0)

    def test_bearing_signs(self):
        self.assertGreater(target_metrics([0, 0, 0], [1, 1])[1], 0)
        self.assertLess(target_metrics([0, 0, 0], [1, -1])[1], 0)

    def test_no_fallback_action_for_invalid_or_stale_inference(self):
        for response, age, sensor_age in [({"action":"backward"},0,0),({"action":"forward"},2,0),
                                           ({"action":"forward"},0,1),({"action":"forward"},0,math.nan),({"action":"left","error":"bad"},0,0)]:
            self.assertEqual(decision_velocity(response,age,sensor_age),((0.0,0.0),False))
        self.assertEqual(decision_velocity({"action":"forward"},0.2,0.1),((0.16,0.0),True))
        self.assertEqual(decision_velocity({"action":"left"},0.2,0.1),((0.0,0.65),True))
        self.assertEqual(decision_velocity({"action":"right"},0.2,0.1),((0.0,-0.65),True))
        self.assertEqual(decision_velocity({"action":"stop"},0.2,0.1),((0.0,0.0),True))


class DirectFeaturesTests(unittest.TestCase):
    def test_discrete_observation_contains_facts_not_actions(self):
        self.assertEqual(discrete_target_features(1,1), {"goal_reached":False,"target_direction":"left"})
        self.assertEqual(discrete_target_features(1,-1), {"goal_reached":False,"target_direction":"right"})
        self.assertEqual(discrete_target_features(0.1,0), {"goal_reached":True,"target_direction":"ahead"})
        self.assertNotIn("action",discrete_target_features(1,0))


class GoalSnapshotTests(unittest.TestCase):
    def test_new_arrival_requires_new_observation_before_scoring_stop(self):
        self.assertTrue(goal_snapshot_changed(0.14,0.16))
        self.assertFalse(goal_snapshot_changed(0.14,0.149))
        self.assertFalse(goal_snapshot_changed(0.16,0.149))

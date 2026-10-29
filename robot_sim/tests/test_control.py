import math
import unittest

from robot_sim.control import ACTIONS, SafetyGate, target_error, waypoint_velocity


class ControlTests(unittest.TestCase):
    def test_watchdog_and_sensor_loss(self):
        gate = SafetyGate()
        for ages in [(0.61, 0, 0), (0, 0.61, 0), (0, 0, 0.61)]:
            velocity, _ = gate.filter((0.2, 0.5), *ages, 1.0)
            self.assertEqual(velocity, (0.0, 0.0))

    def test_obstacle_and_limits(self):
        gate = SafetyGate()
        self.assertEqual(gate.filter((4, 4), 0, 0, 0, 1)[0], (0.2, 1.0))
        self.assertEqual(gate.filter((0.2, 0.5), 0, 0, 0, 0.2)[0], (0, 0.5))
        self.assertEqual(gate.filter((math.nan, 0), 0, 0, 0, 1)[0], (0, 0))
        self.assertEqual(gate.filter((0.2, 0), 0, 0, 0, math.nan)[0], (0, 0))

    def test_rotate_before_drive(self):
        distance, heading = target_error((0, 0, 0), (0, 1))
        self.assertAlmostEqual(distance, 1)
        self.assertAlmostEqual(heading, math.pi / 2)
        linear, angular = waypoint_velocity((0, 0, 0), (0, 1))
        self.assertEqual(linear, 0)
        self.assertGreater(angular, 0)
        self.assertGreater(waypoint_velocity((0, 0, 0), (1, 0))[0], 0)

    def test_no_reverse_action(self):
        self.assertTrue(all(velocity[0] >= 0 for velocity in ACTIONS.values()))


if __name__ == "__main__":
    unittest.main()

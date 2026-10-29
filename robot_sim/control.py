"""Pure control and safety rules, independent of ROS and inference."""
import math
from dataclasses import dataclass

ACTIONS = {"forward": (0.16, 0.0), "left": (0.0, 0.65),
           "right": (0.0, -0.65), "stop": (0.0, 0.0)}
WAYPOINTS = ((0.6, 0.0), (0.6, 0.45), (0.15, 0.45))


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def target_error(pose, target):
    delta_x, delta_y = target[0] - pose[0], target[1] - pose[1]
    return math.hypot(delta_x, delta_y), wrap_angle(math.atan2(delta_y, delta_x) - pose[2])


def waypoint_velocity(pose, target):
    distance, heading = target_error(pose, target)
    return (min(0.18, 0.75 * distance) if abs(heading) < 0.45 else 0.0,
            max(-1.0, min(1.0, 2.2 * heading)))


@dataclass
class SafetyGate:
    timeout: float = 0.6
    clearance: float = 0.28

    def filter(self, velocity, command_age, odom_age, scan_age, front_range):
        if not all(math.isfinite(value) for value in velocity):
            return (0.0, 0.0), "invalid_command"
        if command_age > self.timeout:
            return (0.0, 0.0), "command_timeout"
        if odom_age > self.timeout or scan_age > self.timeout:
            return (0.0, 0.0), "stale_sensor"
        linear = max(0.0, min(0.20, velocity[0]))
        angular = max(-1.0, min(1.0, velocity[1]))
        if math.isnan(front_range) or front_range < self.clearance:
            return (0.0, angular), "obstacle"
        return (linear, angular), "ok"

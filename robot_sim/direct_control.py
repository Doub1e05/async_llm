"""Observation geometry and bounded actuation; never select steering actions."""
import math

from robot_sim.control import ACTIONS, wrap_angle

DIRECT_PROMPT = """You are the local driving controller of a differential-drive robot.
Choose exactly one action: forward, left, right, stop. No destination names.
forward moves straight at 0.16 m/s; left rotates counterclockwise in place;
right rotates clockwise in place; stop holds position. Each pulse lasts 0.65 simulation seconds.
Use the current RGB camera, lidar ranges in meters, target distance and bearing,
and recent actions. Positive bearing means target is to the left; negative means right.
There is no Nav2, path planner, or scripted steering controller.
Approach the target while avoiding obstacles. Do not drive forward if the front range
is below 0.45 m. Turn toward open space to get around obstacles; going around an
obstacle may require temporarily not pointing at the goal. Avoid futile oscillation.
Stop if instructed to halt, if the target is within goal_tolerance_m, or if unsafe.
Choose an action, not an explanation."""


def local_target(spawn, goal):
    delta_x, delta_y = goal[0] - spawn[0], goal[1] - spawn[1]
    cosine, sine = math.cos(spawn[2]), math.sin(spawn[2])
    return cosine * delta_x + sine * delta_y, -sine * delta_x + cosine * delta_y


def target_metrics(odom, target):
    delta_x, delta_y = target[0] - odom[0], target[1] - odom[1]
    return math.hypot(delta_x, delta_y), wrap_angle(math.atan2(delta_y, delta_x) - odom[2])


def decision_velocity(response, observation_age, sensor_age, max_age=1.5):
    if response.get("action") not in ACTIONS or response.get("error"):
        return (0.0, 0.0), False
    if not all(math.isfinite(value) and value >= 0 for value in (observation_age, sensor_age)) or observation_age > max_age or sensor_age > 0.6:
        return (0.0, 0.0), False
    return ACTIONS[response["action"]], True


DIRECT_STATE_PROMPT = """You directly drive the robot. Output one token: forward, left, right, stop.
There is no Nav2, path planner, or automatic steering. Use RGB, lidar and odometry features.
Action meanings: forward = straight motion; left = turn counterclockwise in place;
right = turn clockwise in place; stop = no motion. Pulses last 0.65 simulation seconds.
Highest priority: if the instruction says halt, or goal_reached is true, output stop.
goal_reached and target_direction are sensor-derived facts, not suggestions from another controller.
If goal_reached is false, target_direction says which side the target is on.
When target_direction is left or right, turn that way to align before moving forward.
When target_direction is ahead and front range is above 0.45 meters, drive forward.
Never drive forward into an obstacle. For an obstacle in front, turn toward the more
open side. To bypass it you may need to move forward with the target off-center;
use RGB and recent actions to avoid oscillating left-right or circling forever.
Choose stop if unsafe and no direction is clear. Positive bearing means left."""


def discrete_target_features(distance, bearing, tolerance=0.15):
    return {"goal_reached": distance <= tolerance,
            "target_direction": "ahead" if abs(bearing) <= 0.25 else "left" if bearing > 0 else "right"}


def goal_snapshot_changed(current_distance, observed_distance, tolerance=0.15):
    return current_distance <= tolerance < observed_distance

"""Wheel kinematics and encoder odometry; never uses ground truth."""
import math


def wheel_speeds(linear, angular, radius, track, maximum):
    if not all(math.isfinite(v) for v in (linear, angular)):
        return (0.0, 0.0)
    left = (linear - angular * track/2) / radius
    right = (linear + angular * track/2) / radius
    scale = max(1.0, abs(left)/maximum, abs(right)/maximum)
    return left/scale, right/scale


def integrate_encoders(pose, left_delta, right_delta, radius, track):
    distance = radius * (left_delta + right_delta) / 2
    turn = radius * (right_delta - left_delta) / track
    # Exact SE(2) exponential, stable near zero curvature.
    sinc = math.sin(turn/2)/(turn/2) if abs(turn) > 1e-8 else 1.0
    x, y, yaw = pose
    return (x + distance*sinc*math.cos(yaw+turn/2),
            y + distance*sinc*math.sin(yaw+turn/2),
            math.atan2(math.sin(yaw+turn), math.cos(yaw+turn)))


def approach(current, target, delta):
    return current + max(-delta, min(delta, target-current))

#!/usr/bin/env python3
"""Small Gazebo Transport waypoint follower (no ROS / no LLM)."""
import json
import math
import os
import subprocess
import sys
import threading
import time

GZ = os.environ.get("GZ_BIN", "gz")
ODOM = "/model/turtlebot3_burger/odometry"
CMD = "/model/turtlebot3_burger/cmd_vel"

latest = None
lock = threading.Lock()


def read_odom():
    global latest
    proc = subprocess.Popen([GZ, "topic", "-e", "-t", ODOM, "--json-output"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, bufsize=1)
    for line in proc.stdout:
        try:
            msg = json.loads(line)
            p = msg["pose"]["position"]
            q = msg["pose"]["orientation"]
            yaw = 2.0 * math.atan2(float(q.get("z", 0.0)), float(q.get("w", 1.0)))
            with lock:
                latest = (float(p["x"]), float(p["y"]), yaw)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue


def send(linear, angular):
    text = f"linear: {{x: {linear:.5f}}}, angular: {{z: {angular:.5f}}}"
    subprocess.run([GZ, "topic", "-t", CMD, "-m", "gz.msgs.Twist", "-p", text],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def angle_error(target, current):
    return math.atan2(math.sin(target - current), math.cos(target - current))


def main():
    waypoints = [(0.60, 0.00), (0.60, 0.45), (0.15, 0.45)]
    thread = threading.Thread(target=read_odom, daemon=True)
    thread.start()
    deadline = time.monotonic() + 40.0
    reached = 0
    while time.monotonic() < deadline and reached < len(waypoints):
        with lock:
            pose = latest
        if pose is None:
            time.sleep(0.05)
            continue
        x, y, yaw = pose
        tx, ty = waypoints[reached]
        dx, dy = tx - x, ty - y
        distance = math.hypot(dx, dy)
        if distance < 0.08:
            reached += 1
            continue
        desired = math.atan2(dy, dx)
        error = angle_error(desired, yaw)
        angular = max(-1.2, min(1.2, 2.2 * error))
        linear = 0.0 if abs(error) > 0.45 else min(0.18, 0.75 * distance)
        send(linear, angular)
        time.sleep(0.25)
    send(0.0, 0.0)
    with lock:
        pose = latest
    print(json.dumps({"waypoints_reached": reached, "total_waypoints": len(waypoints), "final_pose": pose}))
    return 0 if reached == len(waypoints) else 2


if __name__ == "__main__":
    sys.exit(main())

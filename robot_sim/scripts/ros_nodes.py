"""ROS sensor adapter, independent safety gateway and route/LLM driver."""
import argparse
import concurrent.futures
import json
import math
import socket
import sys
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from ros_gz_interfaces.msg import Contacts

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robot_sim.control import ACTIONS, WAYPOINTS, SafetyGate, target_error, waypoint_velocity


class RobotNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.pose = None
        self.odom_time = self.scan_time = -math.inf
        self.front_range = 0.0
        self.sectors = {}
        self.create_subscription(Odometry, "/odom", self.odometry, qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/scan", self.scan, qos_profile_sensor_data)

    def odometry(self, message):
        position, orientation = message.pose.pose.position, message.pose.pose.orientation
        yaw = math.atan2(2 * (orientation.w * orientation.z + orientation.x * orientation.y),
                         1 - 2 * (orientation.y**2 + orientation.z**2))
        self.pose = (position.x, position.y, yaw)
        self.odom_time = time.monotonic()

    def scan(self, message):
        sectors = {"front": [], "left": [], "right": []}
        for index, distance in enumerate(message.ranges):
            if math.isnan(distance) or distance < message.range_min:
                continue
            angle = message.angle_min + index * message.angle_increment
            if abs(angle) < 0.4:
                sectors["front"].append(distance)
            elif 0.4 <= angle < 1.6:
                sectors["left"].append(distance)
            elif -1.6 < angle <= -0.4:
                sectors["right"].append(distance)
        self.sectors = {name: min(values, default=0.0) for name, values in sectors.items()}
        self.front_range = self.sectors["front"]
        self.scan_time = time.monotonic()


def twist(velocity):
    message = Twist()
    message.linear.x, message.angular.z = velocity
    return message


class Gateway(RobotNode):
    def __init__(self, output):
        super().__init__("robot_safety")
        self.gate = SafetyGate()
        self.command = (0.0, 0.0)
        self.command_time = -math.inf
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 1)
        self.create_subscription(Twist, "/cmd_vel_requested", self.request, 1)
        self.contact_latched = False
        self.create_subscription(Contacts, "/contacts", self.contact, qos_profile_sensor_data)
        self.log = output.open("w")
        self.create_timer(0.05, self.tick)

    def contact(self, message):
        if message.contacts:
            self.contact_latched = True

    def request(self, message):
        self.command = (message.linear.x, message.angular.z)
        self.command_time = time.monotonic()

    def tick(self):
        now = time.monotonic()
        velocity, reason = self.gate.filter(
            self.command, now - self.command_time, now - self.odom_time,
            now - self.scan_time, self.front_range)
        if self.contact_latched:
            velocity, reason = (0.0, 0.0), "body_contact"
        self.publisher.publish(twist(velocity))
        self.log.write(json.dumps({"time": now, "velocity": velocity, "reason": reason}) + "\n")
        self.log.flush()


def infer(observation, port):
    with socket.create_connection(("127.0.0.1", port), timeout=2) as connection:
        connection.settimeout(30)
        connection.sendall((json.dumps(observation, allow_nan=False) + "\n").encode())
        with connection.makefile("rb") as stream:
            line = stream.readline(65537)
        if not line or len(line) > 65536:
            raise ValueError("invalid inference response")
        response = json.loads(line)
        if response.get("error"):
            raise RuntimeError(response["error"])
        if response.get("action") not in ACTIONS:
            raise ValueError("action outside allowlist")
        return response


class Driver(RobotNode):
    def __init__(self, args):
        super().__init__("robot_driver")
        self.args = args
        self.publisher = self.create_publisher(Twist, "/cmd_vel_requested", 1)
        self.started = time.monotonic()
        self.reached = 0
        self.samples = 0
        self.path_length = 0.0
        self.last_pose = None
        self.finished = False
        self.failed = None
        self.inference_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.observed_at = 0.0
        self.request_waypoint = -1
        self.action_until = 0.0
        self.velocity = (0.0, 0.0)
        self.decisions = (args.output / "decisions.jsonl").open("w")
        self.trace = (args.output / "odometry.jsonl").open("w")
        self.create_timer(0.05, self.tick)

    def tick(self):
        now = time.monotonic()
        if self.pose is not None:
            self.samples += 1
            if self.last_pose is not None:
                self.path_length += math.hypot(self.pose[0] - self.last_pose[0],
                                               self.pose[1] - self.last_pose[1])
            self.last_pose = self.pose
            self.trace.write(json.dumps({"time": now, "pose": self.pose}) + "\n")
            self.trace.flush()
        if now - self.started > self.args.duration:
            self.finished = True
        fresh = now - self.odom_time < 0.6 and now - self.scan_time < 0.6
        if self.finished or not fresh:
            self.publisher.publish(twist((0.0, 0.0)))
            return
        distance, heading = target_error(self.pose, WAYPOINTS[self.reached])
        if distance < 0.08:
            self.reached += 1
            self.velocity = (0.0, 0.0)
            if self.reached == len(WAYPOINTS):
                self.finished = True
            self.publisher.publish(twist((0.0, 0.0)))
            return
        if self.args.mode == "script":
            self.velocity = waypoint_velocity(self.pose, WAYPOINTS[self.reached])
        else:
            if self.future is not None and self.future.done():
                try:
                    response = self.future.result()
                    accepted = (now - self.observed_at < self.args.max_observation_age
                                and self.request_waypoint == self.reached)
                    self.velocity = ACTIONS[response["action"]] if accepted else (0.0, 0.0)
                    self.action_until = now + 0.35
                    self.decisions.write(json.dumps({
                        **response, "accepted": accepted, "latency": now - self.observed_at,
                        "pose": self.pose, "waypoint": self.reached}) + "\n")
                    self.decisions.flush()
                except Exception as error:
                    self.failed = str(error)
                    self.finished = True
                    self.velocity = (0.0, 0.0)
                self.future = None
            if now >= self.action_until:
                self.velocity = (0.0, 0.0)
            if self.future is None and not self.finished:
                observation = {"pose": [round(value, 3) for value in self.pose],
                               "target": WAYPOINTS[self.reached],
                               "distance": round(distance, 3), "heading_error": round(heading, 3),
                               "target_direction": "ahead" if abs(heading) < 0.25 else
                                   ("left" if heading > 0 else "right"),
                               "ranges": {name: round(min(value, 10.0), 3)
                                          for name, value in self.sectors.items()}}
                self.observed_at = now
                self.request_waypoint = self.reached
                self.future = self.inference_pool.submit(infer, observation, self.args.port)
        self.publisher.publish(twist(self.velocity))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["gateway", "driver"])
    parser.add_argument("--mode", choices=["script", "llm"], default="script")
    parser.add_argument("--duration", type=float, default=50)
    parser.add_argument("--max-observation-age", type=float, default=2.0)
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.duration <= 0 or args.max_observation_age <= 0:
        parser.error("durations must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Gateway(args.output / "safety.jsonl") if args.role == "gateway" else Driver(args)
    try:
        if args.role == "gateway":
            rclpy.spin(node)
        else:
            while rclpy.ok() and not node.finished:
                rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.publisher.publish(twist((0.0, 0.0)))
        if args.role == "driver":
            summary = {"mode": args.mode, "waypoints_reached": node.reached,
                       "total_waypoints": len(WAYPOINTS), "path_length_m": node.path_length,
                       "samples": node.samples, "final_pose": node.pose, "error": node.failed,
                       "route_pass": node.reached == len(WAYPOINTS) and node.path_length > 0.8}
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary), flush=True)
            node.inference_pool.shutdown(wait=False, cancel_futures=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if args.role == "driver":
        return 0 if summary["route_pass"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

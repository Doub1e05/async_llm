"""Short-horizon LLM driving without Nav2, path planning, or steering fallback."""
import argparse
import base64
from collections import Counter
import concurrent.futures
import io
import json
import math
from pathlib import Path
import signal
import socket
import sys
import time

import numpy as np
from PIL import Image as PILImage
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robot_sim.control import ACTIONS
from robot_sim.direct_control import decision_velocity, local_target, target_metrics, discrete_target_features, goal_snapshot_changed
from robot_sim.evaluation import validate_decision
from ros_nodes import RobotNode, twist


def infer(observation, port):
    with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
        connection.settimeout(8)
        connection.sendall((json.dumps(observation, allow_nan=False) + "\n").encode())
        response = json.loads(connection.makefile("rb").readline(65536))
    if response.get("error"):
        raise RuntimeError(response["error"])
    validate_decision(response)
    if response.get("action") not in ACTIONS or not response.get("multimodal"):
        raise ValueError("Direct driving requires an allowlisted action and RGB input")
    return response


class DirectDriver(RobotNode):
    def __init__(self, args):
        super().__init__("direct_llm_driver")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.args = args
        self.task = json.loads(args.task_file.read_text())
        self.target = local_target(self.task["spawn"], self.task["goal"])
        self.publisher = self.create_publisher(type(twist((0.0, 0.0))), "/cmd_vel_requested", 1)
        self.create_subscription(Image, "/camera/image_raw", self.image, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "/ground_truth/pose", self.ground_truth, qos_profile_sensor_data)
        self.create_subscription(Contacts, "/contacts", self.contact, qos_profile_sensor_data)
        self.camera = self.truth = None
        self.camera_time = -math.inf
        self.contacts = 0
        self.distance_path = 0.0
        self.last_truth = None
        self.start_truth = None
        self.started = time.monotonic()
        self.active_started = None
        self.sim_started = None
        self.finished = False
        self.error = None
        self.reached = False
        self.stop_at_goal = False
        self.velocity = (0.0, 0.0)
        self.pulse_sim_until = self.pulse_wall_until = -math.inf
        self.future = None
        self.observed_at = 0.0
        self.pending_observation = None
        self.actions = []
        self.accepted = 0
        self.rejected = 0
        self.rejected_goal_changes = 0
        self.best_distance = math.inf
        self.last_progress = time.monotonic()
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.logs = {name: (args.output / (name + ".jsonl")).open("w")
                     for name in ("trajectory", "frames", "decisions", "observations", "execution")}
        (args.output / "frames").mkdir()
        (args.output / "observations").mkdir()
        (args.output / "task.json").write_text(json.dumps(self.task, indent=2))
        self.frame_count = 0
        self.last_frame_sim = -math.inf
        self.wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.05, self.tick, clock=self.wall_clock)

    def write(self, name, record):
        self.logs[name].write(json.dumps(record, allow_nan=False) + "\n")
        self.logs[name].flush()

    def sim_time(self):
        return self.get_clock().now().nanoseconds / 1e9

    def ground_truth(self, message):
        position, rotation = message.pose.position, message.pose.orientation
        self.truth = [position.x, position.y, position.z,
                      math.atan2(2 * (rotation.w * rotation.z + rotation.x * rotation.y),
                                 1 - 2 * (rotation.y**2 + rotation.z**2))]
        if self.start_truth is None:
            self.start_truth = self.truth[:]
        if self.last_truth:
            self.distance_path += math.dist(self.truth[:2], self.last_truth[:2])
        self.last_truth = self.truth[:]
        self.write("trajectory", {"wall_time": time.monotonic(), "sim_time": self.sim_time(),
                                  "truth": self.truth, "odom": self.pose,
                                  "scan_min": min(self.sectors.values(), default=10.0)})

    def contact(self, message):
        self.contacts += len(message.contacts)

    def image(self, message):
        channels = 3 if message.encoding in {"rgb8", "bgr8"} else 4
        pixels = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        pixels = pixels[:, :message.width * channels].reshape(message.height, message.width, channels)
        pixels = pixels[:, :, [2, 1, 0]] if message.encoding in {"bgr8", "bgra8"} else pixels[:, :, :3]
        self.camera = PILImage.fromarray(pixels.copy())
        self.camera_time = time.monotonic()
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        if stamp - self.last_frame_sim >= 0.19:
            filename = f"frames/{self.frame_count:06d}.jpg"
            self.camera.save(self.args.output / filename, quality=85)
            self.write("frames", {"file": filename, "wall_time": self.camera_time,
                                  "sim_time": stamp, "truth": self.truth})
            self.frame_count += 1
            self.last_frame_sim = stamp

    def finish(self, error=None):
        self.error = error
        self.finished = True
        self.velocity = (0.0, 0.0)
        self.publisher.publish(twist(self.velocity))

    def observation(self, distance, bearing):
        image = io.BytesIO()
        self.camera.save(image, format="PNG")
        index = len(self.actions)
        self.camera.save(self.args.output / f"observations/{index:04d}.png")
        state = {"task": "direct_driving", "instruction": self.task["instruction"],
                 "distance_m": round(distance, 3), "target_bearing_rad": round(bearing, 3),
                 "goal_tolerance_m": 0.15, "pulse_sim_seconds": 0.65,
                 "ranges": {name: round(min(value, 10.0), 3) for name, value in self.sectors.items()},
                 "recent_actions": self.actions[-8:], "image": base64.b64encode(image.getvalue()).decode()}
        if self.task.get("observation_protocol") == "state_v2":
            state.update(control_protocol="state_v2", **discrete_target_features(distance, bearing))
        self.write("observations", {"index": index, "wall_time": time.monotonic(), "sim_time": self.sim_time(),
                                    **{key: value for key, value in state.items() if key != "image"}})
        return state

    def tick(self):
        if self.finished:
            self.publisher.publish(twist((0.0, 0.0)))
            return
        now = time.monotonic()
        if self.contacts:
            self.finish("body_contact")
            return
        fresh = (self.pose is not None and self.camera is not None and self.truth is not None
                 and now - min(self.odom_time, self.scan_time, self.camera_time) < 0.6)
        if not fresh:
            self.velocity = (0.0, 0.0)
            self.publisher.publish(twist(self.velocity))
            if now - self.started > 60 and self.active_started is None:
                self.finish("sensors_not_ready")
            elif self.active_started and now - self.active_started > self.task["wall_limit"]:
                self.finish("sensor_staleness_and_wall_budget")
            return
        if self.active_started is None:
            self.active_started, self.sim_started = now, self.sim_time()
            self.last_progress = now
            (self.args.output / "ros_nodes.txt").write_text("Nav2 is not launched. Processes: Gazebo, bridge, safety gateway, direct LLM driver.\n")
        if now - self.active_started > self.task["wall_limit"] or self.sim_time() - self.sim_started > self.task["sim_limit"]:
            self.finish("experiment_budget_exceeded")
            return
        distance, bearing = target_metrics(self.pose, self.target)
        if distance < self.best_distance - 0.04:
            self.best_distance, self.last_progress = distance, now
        if now - self.last_progress > 60 or len(self.actions) >= 100:
            self.finish("no_progress_or_action_budget")
            return
        at_goal = distance <= 0.15
        if at_goal:
            self.reached = True
        if self.future is not None and self.future.done():
            try:
                response = self.future.result()
                velocity, accepted = decision_velocity(response, now - self.observed_at,
                    now - min(self.odom_time, self.scan_time, self.camera_time))
                changed_goal = goal_snapshot_changed(distance, self.pending_observation["distance_m"])
                if changed_goal:
                    velocity, accepted = (0.0, 0.0), False
                    self.rejected_goal_changes += 1
                action = response["action"]
                self.actions.append(action)
                self.accepted += int(accepted)
                self.rejected += int(not accepted)
                self.write("decisions", {**response, "wall_time": now, "sim_time": self.sim_time(),
                    "latency": now - self.observed_at, "observation_age": now - self.observed_at,
                    "accepted": accepted, "rejection_reason": "goal_state_changed" if changed_goal else None, "distance_before": self.pending_observation["distance_m"],
                    "bearing_before": self.pending_observation["target_bearing_rad"]})
                if self.task.get("halt"):
                    self.stop_at_goal = action == "stop" and accepted
                    self.finish(None if self.stop_at_goal else "ignored_halt_instruction")
                elif changed_goal:
                    self.velocity = (0.0, 0.0)
                    self.pulse_sim_until = self.pulse_wall_until = -math.inf
                elif at_goal:
                    self.stop_at_goal = action == "stop" and accepted
                    self.finish(None if self.stop_at_goal else "did_not_stop_at_goal")
                else:
                    self.velocity = velocity
                    self.pulse_sim_until = self.sim_time() + 0.65
                    self.pulse_wall_until = now + 3.0
                    self.write("execution", {"wall_time": now, "sim_time": self.sim_time(),
                        "action": action, "velocity": velocity, "accepted": accepted})
            except Exception as error:
                self.finish("inference_error: " + str(error))
            self.future = None
        if self.finished:
            self.publisher.publish(twist((0.0, 0.0)))
            return
        active = self.sim_time() < self.pulse_sim_until and now < self.pulse_wall_until and not at_goal
        if not active:
            self.velocity = (0.0, 0.0)
            if self.future is None:
                self.pending_observation = self.observation(distance, bearing)
                self.observed_at = now
                self.future = self.pool.submit(infer, self.pending_observation, self.args.port)
        self.publisher.publish(twist(self.velocity))

    def close(self):
        self.publisher.publish(twist((0.0, 0.0)))
        self.pool.shutdown(wait=True, cancel_futures=True)
        for log in self.logs.values():
            log.close()
        physical_error = math.dist(self.truth[:2], self.task["goal"]) if self.truth else None
        if self.error is None and self.stop_at_goal and not self.task.get("halt") and physical_error is not None and physical_error > 0.20:
            self.error = "physical_pose_threshold"
        passed = self.error is None and self.contacts == 0 and self.stop_at_goal
        if self.task.get("halt"):
            passed = passed and self.distance_path < 0.03
        else:
            passed = passed and self.reached and physical_error is not None and physical_error <= 0.20
        summary = {"passed": passed, "mode": "direct_llm", "scene": self.task["scene"],
                   "task_id": self.task["id"], "goal": self.task["goal"], "spawn": self.task["spawn"],
                   "error": self.error, "reached_odom_tolerance": self.reached, "model_stopped_at_goal": self.stop_at_goal,
                   "truth_error_m": physical_error, "truth_path_m": self.distance_path,
                   "contacts": self.contacts, "actions": dict(Counter(self.actions)),
                   "decisions": len(self.actions), "accepted_decisions": self.accepted,
                   "rejected_stale_decisions": self.rejected - self.rejected_goal_changes,
                   "rejected_goal_changed_decisions": self.rejected_goal_changes, "final_truth": self.truth,
                   "wall_seconds": time.monotonic() - self.started,
                   "simulation_seconds": self.sim_time() - self.sim_started if self.sim_started is not None else 0,
                   "nav2_used": False, "scripted_steering_used": False,
                   "observation_protocol": self.task.get("observation_protocol", "numeric_v1"),
                   "target_source": "known initial pose and wheel odometry; no ground truth input"}
        (self.args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    rclpy.init()
    node = DirectDriver(args)
    signal.signal(signal.SIGTERM, lambda signum, frame: node.finish("interrupted"))
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.finish("interrupted")
    finally:
        status = node.close()
        node.destroy_node()
        rclpy.shutdown()
    return status


if __name__ == "__main__":
    raise SystemExit(main())

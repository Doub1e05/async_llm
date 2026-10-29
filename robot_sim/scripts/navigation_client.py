"""Nav2 action client, camera-based goal selector and physics verification."""
import argparse
import base64
import io
import json
import math
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from lifecycle_msgs.srv import GetState
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from tf2_ros import Buffer, TransformListener
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robot_sim.evaluation import score_task, task_request, validate_decision


class Client(Node):
    def __init__(self, args):
        super().__init__("experiment_client")
        self.args = args
        self.pose = self.truth = self.camera = None
        self.calibration = None
        self.camera_time = self.scan_time = self.odom_time = -math.inf
        self.scan_min = math.inf
        self.contacts = 0
        self.plans = 0
        self.odom_distance = self.truth_distance = 0.0
        self.last_odom = self.last_truth = None
        self.initial_truth = None
        self.trace = (args.output / "trajectory.jsonl").open("w")
        self.decisions = (args.output / "vision.jsonl").open("w")
        self.create_subscription(Odometry, "/odom", self.odometry, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "/ground_truth/pose", self.ground_truth, qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/scan", self.scan, qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/image_raw", self.image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/camera_info", self.camera_info, qos_profile_sensor_data)
        self.create_subscription(Contacts, "/contacts", self.contact, qos_profile_sensor_data)
        self.create_subscription(NavPath, "/plan", self.plan, 1)
        self.command = self.create_publisher(Twist, "/cmd_vel_requested", 1)
        self.navigation = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.started = time.monotonic()
        self.records = []
        self.selected_colors = []
        self.task = json.loads(args.task_file.read_text()) if args.task_file else None
        self.frame_count = 0
        self.last_frame_sim = -math.inf
        self.frame_log = (args.output / "frames.jsonl").open("w") if args.record else None
        if args.record:
            (args.output / "frames").mkdir()
        if self.task:
            (args.output / "task.json").write_text(json.dumps(self.task, ensure_ascii=False, indent=2))

    def odometry(self, message):
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        self.pose = [position.x, position.y, 2 * math.atan2(orientation.z, orientation.w)]
        if self.last_odom is not None:
            self.odom_distance += math.dist(self.pose[:2], self.last_odom[:2])
        self.last_odom = self.pose[:]
        self.odom_time = time.monotonic()

    def ground_truth(self, message):
        position, orientation = message.pose.position, message.pose.orientation
        self.truth = [position.x, position.y, position.z,
                      2 * math.atan2(orientation.z, orientation.w)]
        if self.initial_truth is None:
            self.initial_truth = self.truth[:]
        if self.last_truth is not None:
            self.truth_distance += math.dist(self.truth[:2], self.last_truth[:2])
        self.last_truth = self.truth[:]
        self.trace.write(json.dumps({"wall_time": time.monotonic(), "sim_time":
            message.header.stamp.sec + message.header.stamp.nanosec/1e9,
            "truth": self.truth, "odom": self.pose, "pitch": math.asin(max(-1,min(1,2*(orientation.w*orientation.y-orientation.z*orientation.x)))), "scan_min": self.scan_min}) + "\n")
        self.trace.flush()

    def scan(self, message):
        values = [value for value in message.ranges if math.isfinite(value) and value >= message.range_min]
        self.scan_min = min(values, default=math.inf)
        self.scan_time = time.monotonic()

    def image(self, message):
        channels = 3 if message.encoding in {"rgb8", "bgr8"} else 4
        array = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        array = array[:, :message.width * channels].reshape(message.height, message.width, channels)
        if message.encoding in {"bgr8", "bgra8"}:
            array = array[:, :, [2, 1, 0]]
        else:
            array = array[:, :, :3]
        self.camera = PILImage.fromarray(array.copy())
        self.camera_time = time.monotonic()
        sim_time = message.header.stamp.sec + message.header.stamp.nanosec/1e9
        if self.frame_log is not None and sim_time - self.last_frame_sim >= 0.49:
            filename = f"frames/{self.frame_count:06d}.jpg"
            self.camera.save(self.args.output / filename, quality=85)
            self.frame_log.write(json.dumps({"file": filename, "wall_time": self.camera_time,
                                           "sim_time": sim_time, "truth": self.truth}) + "\n")
            self.frame_log.flush()
            self.frame_count += 1
            self.last_frame_sim = sim_time
        if not (self.args.output / "camera_first.png").exists():
            self.camera.save(self.args.output / "camera_first.png")

    def camera_info(self, message):
        self.calibration = {"width": message.width, "height": message.height,
                            "frame": message.header.frame_id, "K": list(message.k),
                            "D": list(message.d), "P": list(message.p)}
        (self.args.output / "camera_info.json").write_text(json.dumps(self.calibration, indent=2))

    def contact(self, message):
        self.contacts += len(message.contacts)

    def plan(self, message):
        if message.poses:
            self.plans += 1
            (self.args.output / "latest_plan.json").write_text(json.dumps(
                [[pose.pose.position.x, pose.pose.position.y] for pose in message.poses]))

    def spin_until(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if predicate():
                return
        raise TimeoutError("Timed out waiting for " + getattr(predicate, "__name__", "condition"))

    def ready(self):
        now = time.monotonic()
        return (self.truth is not None and self.pose is not None and self.camera is not None
                and now - min(self.scan_time, self.odom_time, self.camera_time) < 2.0)

    def velocity(self, linear, angular):
        message = Twist()
        message.linear.x, message.angular.z = float(linear), float(angular)
        self.command.publish(message)

    def segment(self, linear, angular, duration):
        start = self.get_clock().now().nanoseconds / 1e9
        deadline = time.monotonic() + max(20, duration * 8)
        while self.get_clock().now().nanoseconds / 1e9 - start < duration:
            if time.monotonic() > deadline:
                raise TimeoutError("Simulation clock stopped")
            self.velocity(linear, angular)
            rclpy.spin_once(self, timeout_sec=0.04)
            time.sleep(0.01)

    def physics(self):
        self.segment(0, 0, 1.0)
        start = self.truth[:]
        odom_start = self.pose[:]
        self.segment(0.16, 0, 4.0)
        self.segment(0, 0, 2.0)
        after_drive = self.truth[:]
        distance = math.dist(start[:2], after_drive[:2])
        odom_distance = math.dist(odom_start[:2], self.pose[:2])
        yaw_start = self.truth[3]
        self.segment(0, 0.5, 3.0)
        self.segment(0, 0, 2.0)
        yaw_delta = math.atan2(math.sin(self.truth[3] - yaw_start),
                               math.cos(self.truth[3] - yaw_start))
        metrics = {"straight_distance_m": distance, "odometry_distance_m": odom_distance,
                   "distance_error_m": abs(distance - odom_distance),
                   "cross_track_m": abs(after_drive[1] - start[1]), "yaw_change_rad": yaw_delta,
                   "body_height_m": self.truth[2]}
        passed = (0.45 < distance < 0.85 and abs(distance - odom_distance) < 0.08
                  and metrics["cross_track_m"] < 0.05 and 1.1 < yaw_delta < 1.9
                  and 0.065 < self.truth[2] < 0.09 and self.contacts == 0)
        return {"physics": metrics, "passed": passed}

    def vision_goal(self, target, candidates):
        if time.monotonic() - self.camera_time > 2:
            raise RuntimeError("Stale camera")
        image = io.BytesIO()
        self.camera.save(image, format="PNG")
        request = {"task": "navigation", "instruction": (f"Go to the {target} delivery marker." if target else
                       "Choose the nearest unvisited delivery marker visible in the camera image. "
                       "Identify its color. Select only from goal_candidates."),
                   "goal_candidates": candidates,
                   "image": base64.b64encode(image.getvalue()).decode()}
        if self.task:
            request.update(task_request(self.task, self.selected_colors))
        start = time.monotonic()
        (self.args.output / f"request_{len(self.selected_colors)}.json").write_text(
            json.dumps({key: value for key, value in request.items() if key != "image"}, ensure_ascii=False, indent=2))
        with socket.create_connection(("127.0.0.1", self.args.port), timeout=3) as connection:
            connection.settimeout(120)
            connection.sendall((json.dumps(request) + "\n").encode())
            response = json.loads(connection.makefile("rb").readline(2_000_000))
        self.decisions.write(json.dumps({**response, "requested": target,
                                        "instruction": request["instruction"],
                                        "completed_markers": list(self.selected_colors),
                                        "wall_time": time.monotonic(),
                                        "latency": time.monotonic()-start}) + "\n")
        self.decisions.flush()
        if not response.get("error"):
            validate_decision(response)
        if self.task and response.get("action") == "stop" and not response.get("error"):
            self.selected_colors.append("stop")
            return "stop", None
        if response.get("error") or response.get("action") not in candidates:
            raise RuntimeError("Vision decision rejected: " + str(response))
        if target is not None and response["action"] != target:
            raise RuntimeError("Vision selected wrong delivery marker: " + str(response))
        if not response.get("multimodal"):
            raise RuntimeError("Vision worker did not consume the camera image")
        self.selected_colors.append(response["action"])
        return response["action"], candidates[response["action"]]

    def navigate(self, scene):
        self.spin_until(lambda: self.navigation.server_is_ready(), 90)
        state_client = self.create_client(GetState, "/bt_navigator/get_state")
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if state_client.wait_for_service(timeout_sec=0.2):
                future = state_client.call_async(GetState.Request())
                self.spin_until(future.done, 10)
                if future.result().current_state.id == 3:
                    break
            rclpy.spin_once(self, timeout_sec=0.1)
        else:
            raise TimeoutError("Nav2 lifecycle did not become active")
        self.spin_until(lambda: self.buffer.can_transform("map", "base_footprint", rclpy.time.Time()), 60)
        goals = scene["goals"][:self.args.goals]
        candidates = dict(zip(("red", "blue", "green"), scene["goals"]))
        for index, goal in enumerate(goals):
            self.spin_until(self.ready, 15)
            if self.args.mode == "vision":
                self.camera.save(self.args.output / f"vision_goal_{index}.png")
                selected, goal = self.vision_goal(None, candidates)
                if selected == "stop":
                    break
                candidates.pop(selected)
                self.spin_until(self.ready, 15)
            message = NavigateToPose.Goal()
            message.pose.header.frame_id = "map"
            message.pose.header.stamp = self.get_clock().now().to_msg()
            message.pose.pose.position.x, message.pose.pose.position.y = float(goal[0]), float(goal[1])
            message.pose.pose.orientation.z = math.sin(goal[2] / 2)
            message.pose.pose.orientation.w = math.cos(goal[2] / 2)
            accepted = self.navigation.send_goal_async(message)
            self.spin_until(accepted.done, 15)
            handle = accepted.result()
            if not handle.accepted:
                raise RuntimeError("Nav2 rejected goal")
            result = handle.get_result_async()
            try:
                self.spin_until(result.done, self.args.timeout)
            except TimeoutError:
                cancel = handle.cancel_goal_async()
                self.spin_until(cancel.done, 5)
                raise
            status = result.result().status
            error = math.dist(self.truth[:2], goal[:2])
            self.records.append({"color": self.selected_colors[-1] if self.selected_colors else None, "goal": goal, "status": status, "truth_error_m": error,
                                 "truth": self.truth[:]})
            print("Goal result:", self.records[-1], flush=True)
            if status != 4 or error > 0.25:
                raise RuntimeError("Goal failed or physical position differs from localization")
        semantic = score_task(self.task.get("expected") if self.task else None, self.selected_colors)
        stop_task = self.task is not None and self.task.get("expected") == ["stop"]
        navigation_passed = (len(self.records) == len(goals)) if not stop_task else (
            len(self.records) == 0 and self.truth_distance < 0.05)
        return {"passed": navigation_passed and semantic is not False,
                "navigation_passed": navigation_passed, "semantic_passed": semantic,
                "selected_colors": self.selected_colors, "goals": self.records}

    def close(self):
        self.velocity(0, 0)
        if self.camera is not None:
            self.camera.save(self.args.output / "camera_last.png")
        self.trace.close()
        self.decisions.close()
        if self.frame_log is not None:
            self.frame_log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--goals", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scene = yaml.safe_load((root / f"scenes/{args.scene}.yaml").read_text())
    rclpy.init()
    client = Client(args)
    client.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
    summary = {"passed": False}
    try:
        client.spin_until(client.ready, 90)
        if args.mode == "physics":
            summary = client.physics()
        elif args.mode == "inspect":
            topics = subprocess.run(["gz", "topic", "-l"], capture_output=True, text=True, timeout=5)
            (args.output / "gazebo_topics.txt").write_text(topics.stdout)
            client.spin_until(lambda: client.calibration is not None, 5)
            assert client.calibration["width"] == 320 and client.calibration["K"][0] > 0
            client.segment(0, 0, 3)
            summary = {"passed": True, "image_stddev": float(np.asarray(client.camera).std())}
        else:
            summary = client.navigate(scene)
        if client.contacts:
            summary["passed"] = False
    except Exception as error:
        summary.update(error=str(error), goals=client.records)
    finally:
        summary.update(selected_colors=client.selected_colors,
                       semantic_passed=score_task(client.task.get("expected") if client.task else None, client.selected_colors),
                       task_id=client.task.get("id") if client.task else None,
                       scene=args.scene, mode=args.mode, contacts=client.contacts,
                       odom_path_m=client.odom_distance, truth_path_m=client.truth_distance,
                       final_truth=client.truth, plans_received=client.plans,
                       wall_seconds=time.monotonic()-client.started)
        client.close()
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        client.destroy_node()
        rclpy.shutdown()
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

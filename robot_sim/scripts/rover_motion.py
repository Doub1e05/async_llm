"""Deterministic physics acceptance run, not an LLM controller.

Commands /cmd_vel directly so reverse, braking and contact mechanics are tested.
The navigation safety gateway is not started in this mode.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image as PILImage
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan, Imu, JointState, CameraInfo
from tf2_ros import Buffer, TransformListener
from ros_gz_interfaces.msg import Contacts

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robot_sim.control import wrap_angle

SUITE = [("idle", 0, 0, 2), ("forward", .12, 0, 4), ("brake", 0, 0, 1.5),
         ("reverse", -.08, 0, 4), ("stop_reverse", 0, 0, 1),
         ("left", 0, .65, 4), ("stop_left", 0, 0, 1),
         ("right", 0, -.65, 4), ("stop_right", 0, 0, 1),
         ("arc", .10, .4, 3), ("stop_arc", 0, 0, 1),
         ("watchdog_drive", .10, 0, 2), ("watchdog_dropout", None, None, 3),
         ("final_idle", 0, 0, 1)]


class Motion(Node):
    def __init__(self, args):
        super().__init__("rover_physics_acceptance")
        self.args = args
        self.pose = self.odom_pose = None
        self.roll = self.pitch = 0
        self.contacts = 0
        self.counts = dict.fromkeys(("truth", "odom", "scan", "imu", "joint_states", "front", "overview"), 0)
        self.last_sensor = {}
        self.phase = "startup"
        self.frames = []
        self.samples = []
        self.segments = []
        self.previous = None
        self.total_yaw = 0
        self.last_image_sim = -math.inf
        self.files = {name: (args.output/(name+".jsonl")).open("w") for name in ("trajectory", "frames", "commands")}
        (args.output/"frames").mkdir()
        self.effort_max = {}
        self.image_std = {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.calibration = None
        self.command = self.create_publisher(Twist, "/cmd_vel", 1)
        for typ, topic, callback in [
            (PoseStamped, "/ground_truth/pose", self.truth),
            (Odometry, "/odom", self.odometry),
            (Image, "/camera/image_raw", lambda msg: self.image(msg, False)),
            (Image, "/overview/image_raw", lambda msg: self.image(msg, True)),
            (LaserScan, "/scan", lambda msg: self.sensor("scan", msg)),
            (Imu, "/imu", lambda msg: self.sensor("imu", msg)),
            (JointState, "/joint_states", lambda msg: self.sensor("joint_states", msg)),
            (Contacts, "/contacts", self.contact),
            (CameraInfo, "/camera/camera_info", self.camera_info)]:
            self.create_subscription(typ, topic, callback, qos_profile_sensor_data)

    def sim_time(self):
        return self.get_clock().now().nanoseconds/1e9

    def sensor(self, name, msg):
        self.counts[name] += 1
        self.last_sensor[name] = time.monotonic()
        if name == "joint_states":
            for key, value in zip(msg.name,msg.effort):
                self.effort_max[key] = max(self.effort_max.get(key,0),abs(value))
        if name == "scan":
            values = [float(x) for x in msg.ranges if math.isfinite(x)]
            self.scan_min = min(values, default=None)
        if name == "imu":
            self.imu_acceleration = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]

    def camera_info(self, msg):
        self.calibration = {"width":msg.width,"height":msg.height,"frame":msg.header.frame_id,"K":list(msg.k)}
        (self.args.output/"camera_intrinsics.json").write_text(json.dumps(self.calibration,indent=2))

    def contact(self, msg):
        if msg.contacts:
            self.contacts += 1

    def odometry(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.odom_pose = [p.x, p.y, 2*math.atan2(q.z, q.w)]
        self.sensor("odom", msg)

    def truth(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        self.pitch = math.asin(max(-1, min(1, 2*(q.w*q.y-q.z*q.x))))
        self.roll = math.atan2(2*(q.w*q.x+q.y*q.z), 1-2*(q.x*q.x+q.y*q.y))
        self.pose = [p.x, p.y, p.z, yaw]
        if self.previous is not None:
            self.total_yaw += wrap_angle(yaw-self.previous)
        self.previous = yaw
        self.sensor("truth", msg)
        record = {"wall_time": time.monotonic(), "sim_time": self.sim_time(),
                  "truth": self.pose[:], "roll": self.roll, "pitch": self.pitch,
                  "odom": self.odom_pose, "yaw_unwrapped": self.total_yaw, "phase": self.phase}
        self.samples.append(record)
        self.files["trajectory"].write(json.dumps(record)+"\n")

    def image(self, msg, overview):
        channels = 3 if msg.encoding in {"rgb8", "bgr8"} else 4
        array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        array = array[:, :msg.width*channels].reshape(msg.height, msg.width, channels)
        array = array[:, :, [2,1,0]] if msg.encoding in {"bgr8","bgra8"} else array[:, :, :3]
        image = PILImage.fromarray(array.copy())
        name = "overview" if overview else "front"
        self.sensor(name, msg)
        self.image_std[name] = float(array.std())
        first = self.args.output/(name+"_first.png")
        if not first.exists():
            image.save(first)
        if overview:
            stamp = msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
            if stamp-self.last_image_sim < .32:
                return
            filename = f"frames/{len(self.frames):06d}.jpg"
            image.save(self.args.output/filename, quality=90)
            record = {"file": filename, "sim_time": stamp, "wall_time": time.monotonic(),
                      "phase": self.phase, "truth": self.pose, "odom": self.odom_pose}
            self.frames.append(record)
            self.files["frames"].write(json.dumps(record)+"\n")
            self.last_image_sim = stamp

    def velocity(self, v, w):
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        self.command.publish(msg)

    def wait_ready(self):
        deadline = time.monotonic()+90
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=.04)
            if all(self.counts.values()) and self.sim_time() > 0:
                return
        raise TimeoutError("Sensors missing: "+str(self.counts))

    def segment(self, name, v, w, duration):
        self.phase = name
        start = self.sim_time()
        start_index = len(self.samples)
        deadline = time.monotonic()+max(25, duration*12)
        while self.sim_time()-start < duration:
            if time.monotonic() > deadline:
                raise TimeoutError("Simulation stopped in "+name)
            if v is not None:
                self.velocity(v, w)
            rclpy.spin_once(self, timeout_sec=.02)
            time.sleep(.005)
        points = self.samples[start_index:]
        if len(points) < 2:
            raise RuntimeError("No physical samples in "+name)
        a, b = points[0], points[-1]
        dx = b["truth"][0]-a["truth"][0]
        dy = b["truth"][1]-a["truth"][1]
        record = {"phase": name, "command": [v,w], "duration_sim_s": duration,
                  "distance_m": math.hypot(dx,dy), "forward_m": dx*math.cos(a["truth"][3])+dy*math.sin(a["truth"][3]),
                  "cross_track_m": abs(-dx*math.sin(a["truth"][3])+dy*math.cos(a["truth"][3])),
                  "yaw_rad": b["yaw_unwrapped"]-a["yaw_unwrapped"],
                  "start": a["truth"], "end": b["truth"],
                  "odom_distance_m": math.dist(a["odom"][:2],b["odom"][:2]) if a["odom"] and b["odom"] else None,
                  "tail_distance_m": math.dist(points[max(0, len(points)-15)]["truth"][:2],b["truth"][:2]),
                  "max_tilt_rad": max(max(abs(p["roll"]),abs(p["pitch"])) for p in points)}
        self.segments.append(record)
        self.files["commands"].write(json.dumps(record)+"\n")
        print(json.dumps(record), flush=True)

    def run(self):
        self.wait_ready()
        if self.args.case == "suite":
            phases = SUITE
        elif self.args.case == "watchdog":
            phases = [("idle",0,0,2),("watchdog_drive",.1,0,2),("watchdog_dropout",None,None,3),("final_idle",0,0,1)]
        elif self.args.case == "step":
            phases = [("idle",0,0,2),("step_drive",.08,0,9),("final_idle",0,0,2)]
        elif self.args.case == "contact":
            phases = [("idle",0,0,2),("contact_drive",.12,0,6),("final_idle",0,0,1)]
        else:
            v,w = {"forward":(.12,0),"reverse":(-.08,0),"left":(0,.65),"right":(0,-.65),"arc":(.1,.4)}[self.args.case]
            phases = [("idle",0,0,2),(self.args.case,v,w,4),("brake",0,0,2)]
        for phase in phases:
            self.segment(*phase)
        self.velocity(0,0)
        checks = {}
        for s in self.segments:
            name = s["phase"]
            if name == "idle": checks[name] = s["distance_m"] < .005
            elif name == "forward": checks[name] = .32 < s["forward_m"] < .55 and s["cross_track_m"] < .025
            elif name == "reverse": checks[name] = -.38 < s["forward_m"] < -.20 and s["cross_track_m"] < .025
            elif name == "left": checks[name] = 2.0 < s["yaw_rad"] < 2.9 and s["distance_m"] < .04
            elif name == "right": checks[name] = -2.9 < s["yaw_rad"] < -2.0 and s["distance_m"] < .04
            elif name == "arc": checks[name] = s["distance_m"] > .15 and s["yaw_rad"] > .6
            elif name == "brake": checks[name] = s["distance_m"] < .04 and s["tail_distance_m"] < .003
            elif name in {"watchdog_dropout", "final_idle"}: checks[name] = s["tail_distance_m"] < .003
        settled = self.samples[30:]
        checks["stable_body"] = bool(settled) and all(.015 < p["truth"][2] < .03 and max(abs(p["roll"]),abs(p["pitch"])) < .12 for p in settled)
        checks["sensors"] = all(value > 1 for value in self.counts.values())
        checks["camera_image"] = len(self.image_std)==2 and all(x>5 for x in self.image_std.values())
        checks["camera_calibration"] = self.calibration is not None and self.calibration["K"][0]>0
        checks["sensor_tf"] = all(self.tf_buffer.can_transform("odom", frame, rclpy.time.Time())
                                  for frame in ("base_link","laser","camera_optical","imu_link","overview_optical"))
        if self.args.case == "contact":
            checks["wall_blocks_body"] = max(p["truth"][0] for p in self.samples) < 1.815
        if self.args.case == "step":
            checks["step_crossed"] = self.pose[0] > 1.75
        checks["contacts"] = self.contacts > 0 if self.args.case == "contact" else self.contacts == 0
        checks = {key:bool(value) for key,value in checks.items()}
        summary = {"mode":"rover_physics", "scene":self.args.scene, "case":self.args.case,
                   "robot":"yandex_rover", "passed":all(checks.values()), "checks":checks,
                   "body_contact_messages":self.contacts, "reported_joint_effort_max_Nm":self.effort_max,
                   "image_std":self.image_std, "camera_calibration":self.calibration, "sensors_received":self.counts,
                   "segments":self.segments, "frames":len(self.frames),
                   "note":"Scripted physical acceptance, no LLM or Nav2"}
        (self.args.output/"summary.json").write_text(json.dumps(summary,indent=2))
        return summary

    def close(self):
        self.velocity(0,0)
        for f in self.files.values():f.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--case",default="suite")
    p.add_argument("--scene",default="lab")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    rclpy.init()
    node=Motion(args)
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time",value=True)])
    summary={"passed":False}
    try:
        summary=node.run()
    except Exception as e:
        (args.output/"summary.json").write_text(json.dumps({"passed":False,"error":str(e),"sensors":node.counts},indent=2))
        raise
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps(summary),flush=True)
    return 0 if summary["passed"] else 2

if __name__=="__main__":
    raise SystemExit(main())

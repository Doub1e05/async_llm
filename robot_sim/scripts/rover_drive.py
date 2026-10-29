"""Twist -> bounded wheel targets -> Gazebo force PID; encoder-only odometry."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robot_sim.rover import MODEL_DIR
from robot_sim.rover_control import approach, integrate_encoders, wheel_speeds


class RoverDrive(Node):
    def __init__(self, output, odom_topic="/odom", publish_tf=True):
        super().__init__("rover_wheel_controller")
        config = json.loads((MODEL_DIR/"specification.json").read_text())
        self.config, self.limits = config, config["simulation"]
        self.target = self.current = (0.0, 0.0)
        self.command_wall = self.joints_wall = -math.inf
        self.last_tick = None
        self.previous = None
        self.pose = (0.0, 0.0, 0.0)
        self.transform = TransformBroadcaster(self) if publish_tf else None
        self.odom = self.create_publisher(Odometry, odom_topic, 10)
        self.wheels = [self.create_publisher(Float64, f"/rover/{side}_wheel_velocity", 1)
                       for side in ("left", "right")]
        self.create_subscription(Twist, "/cmd_vel", self.command, 1)
        self.create_subscription(JointState, "/joint_states", self.joints, qos_profile_sensor_data)
        self.log = (output/"drive.jsonl").open("w")
        self.create_timer(.02, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def command(self, msg):
        v, w = msg.linear.x, msg.angular.z
        if not all(math.isfinite(value) for value in (v, w)):
            self.target = (0.0, 0.0)
            self.command_wall = -math.inf
            return
        self.target = (max(-self.limits["reverse_max"], min(self.limits["linear_max"], v)),
                       max(-self.limits["angular_max"], min(self.limits["angular_max"], w)))
        self.command_wall = time.monotonic()

    def joints(self, msg):
        names = ("left_middle_joint", "right_middle_joint")
        if not all(name in msg.name for name in names):
            return
        positions = [msg.position[msg.name.index(name)] for name in names]
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec/1e9
        self.joints_wall = time.monotonic()
        if self.previous is None or stamp <= self.previous[0]:
            self.previous = (stamp, positions)
            return
        dt = stamp - self.previous[0]
        dl, dr = [a-b for a, b in zip(positions, self.previous[1])]
        self.previous = (stamp, positions)
        self.pose = integrate_encoders(self.pose, dl, dr, self.config["wheel_radius_m"], self.config["track_m"])
        odom = Odometry()
        odom.header = msg.header
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"
        odom.pose.pose.position.x, odom.pose.pose.position.y = self.pose[:2]
        odom.pose.pose.orientation.z = math.sin(self.pose[2]/2)
        odom.pose.pose.orientation.w = math.cos(self.pose[2]/2)
        odom.twist.twist.linear.x = self.config["wheel_radius_m"]*(dl+dr)/(2*dt)
        odom.twist.twist.angular.z = self.config["wheel_radius_m"]*(dr-dl)/(self.config["track_m"]*dt)
        for index, value in ((0, .0004), (7, .0004), (14, 1e6), (21, 1e6), (28, 1e6), (35, .01)):
            odom.pose.covariance[index] = value
            odom.twist.covariance[index] = value
        self.odom.publish(odom)
        tf = TransformStamped()
        tf.header = odom.header
        tf.child_frame_id = "base_footprint"
        tf.transform.translation.x, tf.transform.translation.y = self.pose[:2]
        tf.transform.rotation = odom.pose.pose.orientation
        if self.transform is not None:
            self.transform.sendTransform(tf)

    def tick(self):
        now = time.monotonic()
        sim = self.get_clock().now().nanoseconds/1e9
        dt = max(0.0, min(.1, sim - self.last_tick)) if self.last_tick is not None else 0
        self.last_tick = sim
        stale = (now-self.command_wall > self.limits["command_timeout_wall_s"]
                 or now-self.joints_wall > .6)
        if stale:
            self.current = (0.0, 0.0)
        else:
            acceleration = self.limits["linear_acceleration"] if abs(self.target[0]) > abs(self.current[0]) else self.limits["linear_deceleration"]
            self.current = (approach(self.current[0], self.target[0], acceleration*dt),
                            approach(self.current[1], self.target[1], self.limits["angular_acceleration"]*dt))
        speeds = wheel_speeds(*self.current, self.config["wheel_radius_m"], self.config["track_m"],
                              self.limits["wheel_velocity_max"])
        for publisher, speed in zip(self.wheels, speeds):
            publisher.publish(Float64(data=float(speed)))
        self.log.write(json.dumps({"wall_time": now, "sim_time": sim, "target": self.target,
                                   "applied": self.current, "wheel_targets": speeds,
                                   "watchdog": stale, "odom": self.pose})+"\n")

    def close(self):
        for publisher in self.wheels:
            publisher.publish(Float64(data=0.0))
        self.log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--odom-topic",default="/odom")
    parser.add_argument("--no-odom-tf",action="store_true")
    args,ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = RoverDrive(args.output,args.odom_topic,not args.no_odom_tf)
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.close()
        else:
            node.log.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

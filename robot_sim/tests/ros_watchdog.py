"""Run in the ROS environment on an isolated ROS_DOMAIN_ID, without Gazebo."""
import sys
import tempfile
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ros_gz_interfaces.msg import Contacts, Contact

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ros_nodes import Gateway


def main():
    rclpy.init()
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".runtime") as folder:
        gateway = Gateway(Path(folder) / "safety.jsonl")
        probe = Node("watchdog_probe")
        odom = probe.create_publisher(Odometry, "/odom", 1)
        scan = probe.create_publisher(LaserScan, "/scan", 1)
        command = probe.create_publisher(Twist, "/cmd_vel_requested", 1)
        contacts = probe.create_publisher(Contacts, "/contacts", 1)
        received = []
        probe.create_subscription(Twist, "/cmd_vel",
                                  lambda message: received.append((message.linear.x, message.angular.z)), 1)
        executor = SingleThreadedExecutor()
        executor.add_node(gateway)
        executor.add_node(probe)

        def phase(duration, send_command=True, send_sensor=True, front=2.0):
            received.clear()
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                if send_sensor:
                    odom.publish(Odometry())
                    message = LaserScan()
                    message.angle_min = -0.1
                    message.angle_increment = 0.1
                    message.range_min = 0.06
                    message.range_max = 10.0
                    message.ranges = [front, front, front]
                    scan.publish(message)
                if send_command:
                    message = Twist()
                    message.linear.x = 0.16
                    command.publish(message)
                for _ in range(12):
                    executor.spin_once(timeout_sec=0)
                time.sleep(0.02)
            assert len(received) >= 3, "No ROS velocity output"
            return received[-3:]

        try:
            assert all(velocity[0] > 0 for velocity in phase(2.0)), "Drive did not start"
            assert all(velocity == (0, 0) for velocity in phase(1.2, send_command=False)), "Command watchdog failed"
            assert all(velocity == (0, 0) for velocity in phase(1.2, front=0.15)), "Obstacle stop failed"
            assert all(velocity[0] > 0 for velocity in phase(0.5)), "Drive did not resume"
            assert all(velocity == (0, 0) for velocity in phase(1.2, send_sensor=False)), "Sensor watchdog failed"
            collision = Contacts()
            collision.contacts = [Contact()]
            contacts.publish(collision)
            assert all(velocity == (0, 0) for velocity in phase(1.0)), "Contact latch failed"
            print("PASS: ROS drive, command timeout, obstacle stop, recovery, sensor timeout, contact latch")
        finally:
            executor.shutdown()
            gateway.log.close()
            gateway.destroy_node()
            probe.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()

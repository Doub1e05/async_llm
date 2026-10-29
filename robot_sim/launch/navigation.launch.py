from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    root = Path(__file__).resolve().parents[1]
    parameters = LaunchConfiguration("params")
    nodes = [
        ("nav2_map_server", "map_server", "map_server"),
        ("nav2_amcl", "amcl", "amcl"),
        ("nav2_controller", "controller_server", "controller_server"),
        ("nav2_planner", "planner_server", "planner_server"),
        ("nav2_behaviors", "behavior_server", "behavior_server"),
        ("nav2_bt_navigator", "bt_navigator", "bt_navigator"),
    ]
    actions = [DeclareLaunchArgument("params", default_value=str(root / "config/nav2.yaml")),
               DeclareLaunchArgument("publish_burger_tf", default_value="true")]
    for package, executable, name in nodes:
        actions.append(Node(package=package, executable=executable, name=name,
                            parameters=[parameters, {"use_sim_time": True}],
                            remappings=[("cmd_vel", "/cmd_vel_requested")],
                            output="screen"))
    actions.append(Node(
        package="nav2_lifecycle_manager", executable="lifecycle_manager",
        name="lifecycle_manager", output="screen",
        parameters=[{"use_sim_time": True, "autostart": True,
                     "bond_timeout": 15.0, "node_names": [name for _, _, name in nodes]}]))
    for parent, child, translation, rotation in [
        ("base_footprint", "base_link", [0, 0, 0.078], [0, 0, 0]),
        ("base_link", "laser", [0, 0, 0.08], [0, 0, 0]),
        ("base_link", "camera_link", [0.09, 0, 0.10], [0, 0, 0]),
        ("camera_link", "camera_optical", [0, 0, 0], [-1.57079632679, 0, -1.57079632679]),
    ]:
        arguments = ["--frame-id", parent, "--child-frame-id", child]
        for name, value in zip(("x", "y", "z", "roll", "pitch", "yaw"), translation + rotation):
            arguments.extend(["--" + name, str(value)])
        actions.append(Node(package="tf2_ros", executable="static_transform_publisher",
                            condition=IfCondition(LaunchConfiguration("publish_burger_tf")),
                            arguments=arguments, parameters=[{"use_sim_time": True}], output="screen"))
    return LaunchDescription(actions)

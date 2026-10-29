"""Own and supervise each process group in a complete navigation experiment."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["warehouse", "apartment", "courtyard", "lab"], default="warehouse")
    parser.add_argument("--mode", choices=["nav2", "vision", "physics", "inspect", "direct", "motion"], default="nav2")
    parser.add_argument("--robot", choices=["burger", "rover"], default="burger")
    parser.add_argument("--motion-case", choices=["suite", "forward", "reverse", "left", "right", "arc", "watchdog", "contact", "step"], default="suite")
    parser.add_argument("--goals", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    if args.robot == "rover" and args.mode == "physics":
        args.mode = "motion"
    if args.scene == "lab" and (args.robot != "rover" or args.mode not in {"motion", "inspect"}):
        parser.error("lab requires --robot rover and --mode motion or inspect")
    if args.mode == "motion" and args.robot != "rover":
        parser.error("motion mode requires --robot rover")
    if not 1 <= args.goals <= 3 or args.timeout <= 0:
        parser.error("goals must be 1..3 and timeout positive")
    output = args.output_dir or ROOT / "robot_sim/results" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.scene}_{args.mode}_{os.getpid()}")
    if (output / "summary.json").exists() or (output / "trajectory.jsonl").exists():
        parser.error("Output contains an existing experiment; select a new --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    print("Results:", output, flush=True)
    prefix = Path(os.environ["CONDA_PREFIX"])
    if args.robot == "rover":
        from robot_sim.rover import MODEL_DIR, make_world, robot_description
        if not (MODEL_DIR / "model.sdf").exists():
            from robot_sim.rover import build
            build()
    description = ({"spawn": [0, 0, 0]} if args.scene == "lab" else
                   yaml.safe_load((ROOT / f"robot_sim/scenes/{args.scene}.yaml").read_text()))
    model_name = "yandex_rover" if args.robot == "rover" else "turtlebot3_burger"
    parameters = yaml.safe_load((ROOT / "robot_sim/config/nav2.yaml").read_text())
    parameters["bt_navigator"]["ros__parameters"]["default_nav_to_pose_bt_xml"] = str(ROOT / "robot_sim/config/navigate.xml")
    parameters["map_server"]["ros__parameters"]["yaml_filename"] = str(
        ROOT / f"robot_sim/maps/{args.scene}.yaml")
    parameters["amcl"]["ros__parameters"]["initial_pose"].update(
        x=float(description["spawn"][0]), y=float(description["spawn"][1]))
    if args.robot == "rover":
        for section in ("local_costmap", "global_costmap"):
            costmap = parameters[section][section]["ros__parameters"]
            costmap.pop("robot_radius", None)
            costmap["footprint"] = "[[0.085,0.075],[0.085,-0.075],[-0.095,-0.075],[-0.095,0.075]]"
            costmap["footprint_padding"] = 0.005
        controller = parameters["controller_server"]["ros__parameters"]["FollowPath"]
        controller.update(max_vel_x=0.18, max_speed_xy=0.18, acc_lim_x=0.30)
    parameter_file = output / "nav2.yaml"
    parameter_file.write_text(yaml.safe_dump(parameters))
    world_file = ROOT / f"robot_sim/worlds/{args.scene}.sdf"
    if args.robot == "rover":
        spawn = [1.5, 0, 0] if args.scene == "lab" and args.motion_case == "contact" else (
            [1.0, -1.5, 0] if args.scene == "lab" and args.motion_case == "step" else None)
        world_file = make_world(args.scene, output / "world.sdf", spawn=spawn)
    bridge_file = ROOT / "robot_sim/config/navigation_bridge.yaml"
    if args.robot == "rover":
        from robot_sim.rover import bridge_configuration
        bridges = bridge_configuration()
        bridge_file = output / "bridge.yaml"
        bridge_file.write_text(yaml.safe_dump(bridges))
    (output / "run.json").write_text(json.dumps({"robot": args.robot, "model": model_name, "scene": args.scene, "mode": args.mode}, indent=2))
    if args.mode == "direct":
        if not args.task_file:
            parser.error("direct mode requires --task-file")
        task = json.loads(args.task_file.read_text())
        if task["scene"] != args.scene:
            parser.error("task scene differs from --scene")
        document = ET.parse(world_file)
        robot = document.find(f"./world/model[@name='{model_name}']")
        if robot is None:
            raise RuntimeError("Robot model not found")
        pose = robot.find("pose")
        height = float(pose.text.split()[2])
        spawn = task["spawn"]
        pose.text = f"{spawn[0]} {spawn[1]} {height} 0 0 {spawn[2]}"
        world_file = output / "world.sdf"
        document.write(world_file, encoding="unicode", xml_declaration=True)
    processes = []
    logs = []

    def start(name, command):
        log = (output / (name + ".log")).open("w")
        logs.append(log)
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        processes.append((name, process))
        return process

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    status = 1
    try:
        start("gazebo", ["xvfb-run", "-a", "-s", "-screen 0 640x480x24 -nolisten tcp",
                         "gz", "sim", "-s", "-r",
                         str(world_file)])
        start("bridge", [str(prefix / "lib/ros_gz_bridge/parameter_bridge"), "--ros-args",
                         "-p", "config_file:=" + str(bridge_file)])
        if args.robot == "rover":
            start("drive", [sys.executable, "robot_sim/scripts/rover_drive.py", "--output", str(output)])
            tf_parameters = output / "robot_state_publisher.yaml"
            tf_parameters.write_text(yaml.safe_dump({"robot_state_publisher": {"ros__parameters": {
                "use_sim_time": True, "publish_frequency": 30.0,
                "robot_description": robot_description()}}}))
            start("robot_state_publisher", [str(prefix / "lib/robot_state_publisher/robot_state_publisher"),
                                            "--ros-args", "--params-file", str(tf_parameters)])
        if args.mode != "motion":
            start("gateway", [sys.executable, "robot_sim/scripts/ros_nodes.py", "gateway",
                              "--output", str(output)])
        if args.mode in {"nav2", "vision"}:
            start("nav2", ["ros2", "launch", str(ROOT / "robot_sim/launch/navigation.launch.py"),
                           "params:=" + str(parameter_file), "publish_burger_tf:=" + str(args.robot == "burger").lower()])
        client_options = []
        if args.task_file:
            client_options.extend(["--task-file", str(args.task_file.resolve())])
        if args.record:
            client_options.append("--record")
        if args.mode == "motion":
            client = start("client", [sys.executable, "robot_sim/scripts/rover_motion.py",
                                      "--case", args.motion_case, "--scene", args.scene, "--output", str(output)])
        elif args.mode == "direct":
            client = start("client", [sys.executable, "robot_sim/scripts/direct_driver.py",
                           "--task-file", str(args.task_file.resolve()), "--output", str(output),
                           "--port", str(args.port)])
        else:
            client = start("client", [sys.executable, "robot_sim/scripts/navigation_client.py",
                                      "--scene", args.scene, "--mode", args.mode,
                                      "--goals", str(args.goals), "--timeout", str(args.timeout),
                                      "--port", str(args.port), "--output", str(output), *client_options])
        deadline = time.monotonic() + 280 + args.timeout * args.goals
        while client.poll() is None:
            for name, process in processes[:-1]:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited ({process.returncode}); see {output}")
            if time.monotonic() > deadline:
                raise TimeoutError("Experiment deadline exceeded")
            time.sleep(0.2)
        status = client.returncode
        summary = output / "summary.json"
        if summary.exists():
            print(summary.read_text(), flush=True)
    except (KeyboardInterrupt, Exception) as error:
        print(f"Experiment stopped: {error}; logs: {output}", file=sys.stderr, flush=True)
    finally:
        try:
            subprocess.run(["gz", "topic", "-t", f"/model/{model_name}/cmd_vel",
                            "-m", "gz.msgs.Twist", "-p", "linear: {x: 0}, angular: {z: 0}"],
                           timeout=2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if args.robot == "rover":
            for side in ("left", "right"):
                try:
                    subprocess.run(["gz", "topic", "-t", f"/model/yandex_rover/{side}_wheel_velocity",
                                    "-m", "gz.msgs.Double", "-p", "data: 0"],
                                   timeout=2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        for _, process in reversed(processes):
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline and any(process.poll() is None for _, process in processes):
            time.sleep(0.1)
        for _, process in processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        for log in logs:
            log.close()
        (output / "processes.json").write_text(json.dumps(
            [{"name": name, "pid": process.pid, "returncode": process.returncode}
             for name, process in processes], indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())

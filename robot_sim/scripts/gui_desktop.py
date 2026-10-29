"""Local-only Gazebo desktop, forwarded over SSH; no XQuartz dependency."""
import argparse
import atexit
import yaml
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "robot_sim/.runtime/viewer"
SESSION = RUNTIME / "session"
SESSION.mkdir(parents=True, exist_ok=True)
SESSION.chmod(0o700)
parser = argparse.ArgumentParser()
parser.add_argument("scene", nargs="?", default="warehouse", choices=["warehouse","apartment","courtyard","lab"])
parser.add_argument("--robot", choices=["rover","burger"], default="rover")
args = parser.parse_args()
scene = args.scene
sys.path.insert(0, str(ROOT))
if args.robot == "burger" and scene == "lab":
    parser.error("lab is available only for rover")
world_file = ROOT / f"robot_sim/worlds/{scene}.sdf"
if args.robot == "rover":
    from robot_sim.rover import MODEL_DIR, build, make_world, bridge_configuration, robot_description
    if not (MODEL_DIR/"model.sdf").exists():
        build()
    world_file = make_world(scene, SESSION/"world.sdf")
processes = []
logs = []


def cleanup():
    for process in reversed(processes):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and any(process.poll() is None for process in processes):
        time.sleep(0.1)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    for log in logs:
        log.close()


atexit.register(cleanup)
def stop(signum, frame):
    raise SystemExit(0)
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)


def start(name, command, environment):
    log = (SESSION / (name + ".log")).open("w")
    logs.append(log)
    process = subprocess.Popen(command, cwd=ROOT, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    processes.append(process)
    return process


display = next(number for number in range(91, 150)
               if not Path(f"/tmp/.X11-unix/X{number}").exists()
               and not Path(f"/tmp/.X{number}-lock").exists())
authority = SESSION / "Xauthority"
authority.touch(mode=0o600)
subprocess.run(["xauth", "-f", str(authority)],
               input=f"add :{display} MIT-MAGIC-COOKIE-1 {secrets.token_hex(16)}\n",
               text=True, check=True, stdout=subprocess.DEVNULL)
environment = os.environ.copy()
environment.update(DISPLAY=f":{display}", XAUTHORITY=str(authority), QT_QPA_PLATFORM="xcb",
                   QT_X11_NO_MITSHM="1", LIBGL_ALWAYS_SOFTWARE="1", GZ_PARTITION="robot_desktop")
environment.pop("QT_QUICK_BACKEND", None)
start("xvfb", ["Xvfb", f":{display}", "-screen", "0", "1440x900x24",
               "-nolisten", "tcp", "-auth", str(authority)], environment)
deadline = time.monotonic() + 10
while not Path(f"/tmp/.X11-unix/X{display}").exists():
    if time.monotonic() > deadline:
        raise RuntimeError("Xvfb startup failed")
    time.sleep(0.1)
password = secrets.token_hex(4)
password_file = SESSION / "password"
password_file.write_text(password + "\n")
password_file.chmod(0o600)
vnc_environment = environment.copy()
vnc_environment["LD_LIBRARY_PATH"] = str(RUNTIME / "root/usr/lib/x86_64-linux-gnu")
start("vnc", [str(RUNTIME / "root/usr/bin/x11vnc"), "-display", f":{display}",
              "-auth", str(authority), "-rfbport", "15909", "-listen", "127.0.0.1",
              "-passwdfile", str(password_file), "-forever", "-shared", "-noxdamage",
              "-no6", "-quiet"], vnc_environment)
web_environment = environment.copy()
web_environment["PYTHONPATH"] = str(RUNTIME / "root/usr/lib/python3/dist-packages")
start("web", [sys.executable, "-m", "websockify", "--web",
              str(RUNTIME / "root/usr/share/novnc"), "127.0.0.1:17080", "127.0.0.1:15909"],
      web_environment)
if args.robot == "rover":
    bridge = SESSION/"bridge.yaml"
    bridge.write_text(yaml.safe_dump(bridge_configuration()))
    prefix = Path(os.environ["CONDA_PREFIX"])
    start("bridge", [str(prefix/"lib/ros_gz_bridge/parameter_bridge"),"--ros-args",
                     "-p","config_file:="+str(bridge)], environment)
    start("drive", [sys.executable,"robot_sim/scripts/rover_drive.py","--output",str(SESSION)], environment)
    parameters = SESSION/"robot_state_publisher.yaml"
    parameters.write_text(yaml.safe_dump({"robot_state_publisher":{"ros__parameters":{
        "use_sim_time":True,"robot_description":robot_description()}}}))
    start("robot_state_publisher", [str(prefix/"lib/robot_state_publisher/robot_state_publisher"),
                                   "--ros-args","--params-file",str(parameters)], environment)
start("gazebo", ["gz", "sim", "-r", "-v", "3", str(world_file)], environment)
(SESSION / "processes.json").write_text(json.dumps(
    {"pids": [process.pid for process in processes], "supervisor": os.getpid(),
     "display": display, "scene": scene, "robot": args.robot}, indent=2))
print("Viewer listening on 127.0.0.1:17080; SSH tunnel required", flush=True)
while True:
    for process in processes:
        if process.poll() is not None:
            raise RuntimeError(f"Desktop child exited: {process.args}; see {SESSION}")
    time.sleep(0.5)

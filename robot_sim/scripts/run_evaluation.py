"""Sequential simulation-only evaluation; each run has an independent scene."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "robot_sim/results/qwen9b_evaluation_20261026"
TASKS = ["warehouse_red_blue", "apartment_green_red", "courtyard_blue_green",
         "warehouse_nearest_visible", "warehouse_unknown_yellow"]
environment = os.environ.copy()
environment["ROS_DOMAIN_ID"] = "74"
results = []
if any((OUTPUT / name / "trajectory.jsonl").exists() for name in TASKS):
    raise SystemExit("This dated evaluation already contains evidence; run tasks with new --output-dir paths")
for name in TASKS:
    task_file = ROOT / f"robot_sim/evaluation_tasks/{name}.json"
    task = json.loads(task_file.read_text())
    directory = OUTPUT / name
    directory.mkdir(exist_ok=True,parents=True)
    command = ["bash", "robot_sim/scripts/run_navigation.sh", "--scene", task["scene"], "--mode", "vision",
               "--goals", str(task["goals"]), "--timeout", "480", "--task-file", str(task_file),
               "--record", "--output-dir", str(directory)]
    print("RUN",name,flush=True)
    with (directory / "launcher.log").open("w") as log:
        status = subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT).returncode
    summary_file = directory / "summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else {"passed": False,"error":"launcher failed"}
    results.append({"task":name,"returncode":status,**summary})
    (OUTPUT / "results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print("DONE",name,json.dumps(summary),flush=True)
    if summary_file.exists():
        rendering = ["bash","-c",'source robot_sim/scripts/robot_env.sh && exec python -m robot_sim.resource_limits python robot_sim/scripts/render_replay.py "$1"',"bash",str(directory)]
        with (directory / "render.log").open("w") as log:
            subprocess.run(rendering,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
    time.sleep(2)

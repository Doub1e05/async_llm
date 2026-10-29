"""Run the matched Nav2-only baseline after the LLM suite has finished."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "robot_sim/results/qwen9b_evaluation_20261026"
deadline = time.monotonic() + 4000
while time.monotonic() < deadline:
    results = OUTPUT / "results.json"
    if results.exists() and len(json.loads(results.read_text())) == 5 and (OUTPUT / "warehouse_unknown_yellow/render.log").exists():
        time.sleep(8)
        break
    time.sleep(15)
else:
    raise SystemExit("Suite did not finish; baseline not started")
directory = OUTPUT / "warehouse_nav2_baseline"
directory.mkdir(exist_ok=True)
environment = os.environ.copy()
environment["ROS_DOMAIN_ID"] = "75"
with (directory / "launcher.log").open("w") as log:
    result = subprocess.run(["bash", "robot_sim/scripts/run_navigation.sh", "--scene", "warehouse", "--mode", "nav2",
                             "--goals", "2", "--timeout", "480", "--record", "--output-dir", str(directory)],
                            cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
print("BASELINE_EXIT", result.returncode, flush=True)
with (directory / "render.log").open("w") as log:
    subprocess.run(["bash", "-c", 'source robot_sim/scripts/robot_env.sh && exec python -m robot_sim.resource_limits python robot_sim/scripts/render_replay.py "$1"',
                    "bash", str(directory)], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
print("BASELINE_DONE", flush=True)

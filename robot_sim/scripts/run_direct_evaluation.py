"""Sequential direct-driving experiment; do not overwrite existing evidence."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "robot_sim/results/qwen9b_direct_20261026"
environment = os.environ.copy()
environment["ROS_DOMAIN_ID"] = "76"
results=[]
for name in ["straight","left_turn","right_turn","blocked_goal","stop_command"]:
    task = ROOT / f"robot_sim/direct_tasks/{name}.json"
    directory = OUTPUT / name
    summary_file = directory / "summary.json"
    if not summary_file.exists():
        print("RUN",name,flush=True)
        with (OUTPUT / (name+"_launcher.log")).open("w") as log:
            subprocess.run(["bash","robot_sim/scripts/run_navigation.sh","--mode","direct","--scene","warehouse",
                            "--task-file",str(task),"--record","--timeout","300","--output-dir",str(directory)],
                           cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
    summary=json.loads(summary_file.read_text()) if summary_file.exists() else {"passed":False,"error":"launcher failed"}
    results.append({"task":name,**summary})
    (OUTPUT/"results.json").write_text(json.dumps(results,indent=2))
    print("DONE",name,json.dumps(summary),flush=True)
    if summary_file.exists():
        with (directory/"render.log").open("w") as log:
            subprocess.run(["bash","-c",'source robot_sim/scripts/robot_env.sh && exec python -m robot_sim.resource_limits python robot_sim/scripts/render_replay.py "$1"',"bash",str(directory)],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)

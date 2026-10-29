"""Re-evaluate only runs affected by the goal-snapshot scoring race."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/"robot_sim/results/qwen9b_direct_20261026"
environment=os.environ.copy()
environment.update(ROBOT_LLM_GPU="GPU-ec74220d-78d7-cc47-69fc-0de1fb580944,GPU-53eaee12-1803-f8da-7399-ccdbfda91a18",
                   ROBOT_LLM_MAX_GPUS="2",MODEL="Qwen/Qwen3.5-9B",ROS_DOMAIN_ID="76")
log=(OUTPUT/"model_rechecks.log").open("w")
model=subprocess.Popen(["bash","robot_sim/scripts/run_llm.sh","--backend","transformers","--gpu-memory-gib","12",
                       "--vision","--reasoning-tokens","1","--metadata",str(OUTPUT/"model_rechecks_metadata.json")],
                       cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
results=[]
try:
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        if model.poll() is not None:
            raise RuntimeError("Recheck model worker exited")
        if "READY 127.0.0.1" in (OUTPUT/"model_rechecks.log").read_text():
            break
        time.sleep(2)
    else:
        raise TimeoutError("Recheck worker not ready")
    for original in ["right_turn","right_turn_state_v2"]:
        name=original+"_rechecked"
        task=ROOT/f"robot_sim/direct_tasks/{original}.json"
        directory=OUTPUT/name
        summary_file=directory/"summary.json"
        if not summary_file.exists():
            with (OUTPUT/(name+"_launcher.log")).open("w") as run_log:
                subprocess.run(["bash","robot_sim/scripts/run_navigation.sh","--mode","direct","--scene","warehouse",
                                "--task-file",str(task),"--record","--timeout","300","--output-dir",str(directory)],
                               cwd=ROOT,env=environment,stdout=run_log,stderr=subprocess.STDOUT)
        summary=json.loads(summary_file.read_text()) if summary_file.exists() else {"passed":False,"error":"launcher failed"}
        results.append({"task":name,**summary})
        (OUTPUT/"results_rechecks.json").write_text(json.dumps(results,indent=2))
        print("DONE",name,json.dumps(summary),flush=True)
        if summary_file.exists():
            with (directory/"render.log").open("w") as run_log:
                subprocess.run(["bash","-c",'source robot_sim/scripts/robot_env.sh && exec python -m robot_sim.resource_limits python robot_sim/scripts/render_replay.py "$1"',"bash",str(directory)],cwd=ROOT,env=environment,stdout=run_log,stderr=subprocess.STDOUT)
finally:
    try:
        os.killpg(model.pid,signal.SIGTERM)
        model.wait(timeout=10)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(model.pid,signal.SIGKILL)
        model.wait()
    log.close()

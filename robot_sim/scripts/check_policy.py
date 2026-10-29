"""Semantic smoke checks before allowing the two-GPU policy to drive."""
import base64
import json
from pathlib import Path
import socket
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "robot_sim/results/qwen9b_evaluation_20261026"
image = OUTPUT / "invalid_backend_nan/warehouse_red_blue/vision_goal_0.png"
candidates = {"red":[-1.7,-2,0],"blue":[1,0,0],"green":[4,2.3,0]}
checks = [("Go to the red delivery marker.","red"),
          ("Go to the blue delivery marker.","blue"),
          ("Go to the green delivery marker.","green"),
          ("Only visit the yellow marker. If yellow is not in goal_candidates, stop.","stop")]
results = []
for instruction,expected in checks:
    request = {"task":"navigation","instruction":instruction,"goal_candidates":candidates,
               "image":base64.b64encode(image.read_bytes()).decode()}
    started=time.monotonic()
    with socket.create_connection(("127.0.0.1",18765),timeout=120) as connection:
        connection.sendall((json.dumps(request)+"\n").encode())
        response=json.loads(connection.makefile("rb").readline())
    result={"instruction":instruction,"expected":expected,"latency":time.monotonic()-started,**response}
    results.append(result)
    print(json.dumps(result),flush=True)
(OUTPUT/"policy_health.json").write_text(json.dumps(results,indent=2))
if not all(result.get("action")==result["expected"] and not result.get("error") for result in results):
    raise SystemExit("Policy health check failed; do not run driving evaluation")

"""Non-driving control probes, kept separate from physical driving scores."""
import base64
import io
import json
from pathlib import Path
import socket
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "robot_sim/results/qwen9b_direct_20261026"
image = io.BytesIO()
Image.new("RGB", (320, 240), (110, 110, 110)).save(image, format="PNG")
cases = [("ahead", 0.0, 1.0, "Reach the target, then stop.", "forward"),
         ("left", 1.0, 1.0, "Reach the target on your left, then stop.", "left"),
         ("right", -1.0, 1.0, "Reach the target on your right, then stop.", "right"),
         ("at_goal", 0.0, 0.1, "Reach the target, then stop.", "stop"),
         ("halt", 0.0, 1.0, "Halt immediately. Stay still. Return stop.", "stop")]
results = []
for name, bearing, distance, instruction, expected in cases:
    request = {"task":"direct_driving", "instruction":instruction, "distance_m":distance,
               "target_bearing_rad":bearing, "goal_tolerance_m":0.15, "pulse_sim_seconds":0.65,
               "ranges":{"front":3.0,"left":3.0,"right":3.0},"recent_actions":[],
               "image":base64.b64encode(image.getvalue()).decode()}
    started = time.monotonic()
    with socket.create_connection(("127.0.0.1",18765),timeout=15) as connection:
        connection.sendall((json.dumps(request)+"\n").encode())
        response = json.loads(connection.makefile("rb").readline())
    result = {"id":name,"expected":expected,"passed":response.get("action")==expected,
              "latency":time.monotonic()-started,**response}
    results.append(result)
    print(json.dumps(result),flush=True)
(OUTPUT/"control_probes.json").write_text(json.dumps(results,indent=2))

"""Non-driving image/language controls, explicitly separate from route scores."""
import base64
import io
import json
from pathlib import Path
import socket
import time

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "robot_sim/results/qwen9b_evaluation_20261026"
DIRECTORY = OUTPUT / "probes"
DIRECTORY.mkdir(exist_ok=True)
CANDIDATES = {"red":[-1.7,-2,0],"blue":[1,0,0],"green":[4,2.3,0]}


def request(name,image,instruction,expected):
    image.save(DIRECTORY/(name+".png"))
    payload=io.BytesIO()
    image.save(payload,format="PNG")
    observation={"task":"navigation","instruction":instruction,"goal_candidates":CANDIDATES,
                 "image":base64.b64encode(payload.getvalue()).decode()}
    started=time.monotonic()
    with socket.create_connection(("127.0.0.1",18765),timeout=120) as connection:
        connection.sendall((json.dumps(observation)+"\n").encode())
        response=json.loads(connection.makefile("rb").readline())
    return {"id":name,"instruction":instruction,"expected":expected,"latency":time.monotonic()-started,
            "passed": response.get("action")==expected if expected else None,**response}


results=[]
for order in [("green","red","blue"),("blue","red","green")]:
    image=Image.new("RGB",(320,240),(110,110,110))
    draw=ImageDraw.Draw(image)
    for index,color in enumerate(order):
        left=18+index*100
        draw.rectangle((left,55,left+65,160),fill=color)
    for side in ("leftmost","rightmost"):
        expected=order[0 if side=="leftmost" else -1]
        name="synthetic_"+"_".join(order)+"_"+side
        results.append(request(name,image,"Choose the color of the "+side+" colored marker visible in the image. Use the image, not coordinates.",expected))
blank=Image.new("RGB",(320,240),(110,110,110))
results.append(request("synthetic_no_marker",blank,"Choose the color of a visible delivery marker. If no colored marker is visible, stop.","stop"))
actual=OUTPUT/"warehouse_red_blue/vision_goal_0.png"
if actual.exists():
    image=Image.open(actual).convert("RGB")
    instruction="Choose the nearest unvisited delivery marker visible in the camera image. Identify its color. Select only from goal_candidates."
    results.append(request("actual_nearest",image,instruction,None))
    results.append(request("blank_nearest",blank,instruction,None))
(OUTPUT/"probes.json").write_text(json.dumps(results,ensure_ascii=False,indent=2))
for result in results:
    print(json.dumps(result),flush=True)

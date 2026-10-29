"""Recorded camera + telemetry replay, not a synthetic Gazebo screenshot."""
import argparse
import bisect
import json
import math
from pathlib import Path
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont
import yaml


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def render(directory):
    summary = json.loads((directory / "summary.json").read_text())
    direct = summary.get("mode") == "direct_llm"
    task = json.loads((directory / "task.json").read_text()) if (directory / "task.json").exists() else {"instruction": "Nav2 baseline"}
    root = Path(__file__).resolve().parents[1]
    scene = yaml.safe_load((root / f"scenes/{summary['scene']}.yaml").read_text())
    frames = records(directory / "frames.jsonl")
    trajectory = records(directory / "trajectory.jsonl")
    decisions = records(directory / ("decisions.jsonl" if direct else "vision.jsonl"))
    safety = records(directory / "safety.jsonl")
    safety_times = [entry["time"] for entry in safety]
    if not frames:
        return None
    times = [frame["sim_time"] for frame in frames]
    walls = [point["wall_time"] for point in trajectory]
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 18)
    title_font = ImageFont.truetype(font_path, 23)
    small = ImageFont.truetype(font_path, 14)
    fps = 4
    total = max(1, int(math.ceil((times[-1] - times[0]) * fps)))
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "1280x800", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-threads", "1",
               "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(directory / "replay.mp4")]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    transform = lambda position: (int(40 + (position[0]+6)*48), int(570-(position[1]+4)*48))
    initial = None
    try:
        for index in range(total):
            stamp = times[0] + index / fps
            frame = frames[max(0, bisect.bisect_right(times, stamp)-1)]
            image = Image.new("RGB", (1280, 800), "#101923")
            draw = ImageDraw.Draw(image)
            draw.text((28, 16), f"Qwen3.5-9B / {summary['scene']} / simulation time {stamp-times[0]:.1f}s", font=title_font, fill="white")
            draw.text((28, 58), "DIRECT LLM DRIVING / no Nav2" if direct else "Telemetry map replay (not a 3D Gazebo video)", font=font, fill="#a9b7c7")
            draw.text((660, 58), "RECORDED ROBOT CAMERA", font=font, fill="#7de0c3")
            for grid in range(-6,7):
                draw.line([transform((grid,-4)),transform((grid,4))], fill="#233243")
            for grid in range(-4,5):
                draw.line([transform((-6,grid)),transform((6,grid))], fill="#233243")
            for obstacle in scene["obstacles"]:
                position = (obstacle["x"],obstacle["y"])
                corner1 = transform((position[0]-obstacle["width"]/2, position[1]+obstacle["depth"]/2))
                corner2 = transform((position[0]+obstacle["width"]/2, position[1]-obstacle["depth"]/2))
                draw.rectangle([corner1,corner2], fill="#657481", outline="#a1adb8")
            for color,goal in zip(("red","blue","green"),scene["goals"]):
                center = transform(goal)
                draw.ellipse((center[0]-7,center[1]-7,center[0]+7,center[1]+7),fill=color)
                draw.text((center[0]+10,center[1]-9),color,font=small,fill="white")
            if direct and not task.get("halt"):
                goal = transform(task["goal"])
                draw.ellipse((goal[0]-10,goal[1]-10,goal[0]+10,goal[1]+10),outline="#ffae42",width=3)
                draw.text((goal[0]+13,goal[1]),"TARGET",font=small,fill="#ffae42")
            end = bisect.bisect_right(walls,frame["wall_time"])
            points = trajectory[:end]
            if len(points)>1:
                draw.line([transform(point["truth"]) for point in points],fill="#ffe18c",width=3)
            pose = frame.get("truth") or (points[-1]["truth"] if points else None)
            if pose:
                center = transform(pose)
                draw.ellipse((center[0]-6,center[1]-6,center[0]+6,center[1]+6),fill="white")
                yaw = pose[3]
                draw.line([center,(center[0]+17*math.cos(yaw),center[1]-17*math.sin(yaw))],fill="#ffb458",width=4)
            with Image.open(directory / frame["file"]) as camera:
                image.paste(camera.resize((600,450)), (650,100))
            completed = [decision["action"] for decision in decisions if decision.get("wall_time",math.inf) <= frame["wall_time"]]
            if direct:
                visible = [decision for decision in decisions if decision.get("wall_time",math.inf) <= frame["wall_time"]]
                latest = visible[-1] if visible else None
                draw.text((650,565), "LLM action: " + (latest["action"] if latest else "waiting"),font=font,fill="white")
                safe_index = bisect.bisect_right(safety_times,frame["wall_time"])-1
                safe = safety[safe_index] if safe_index>=0 else None
                if safe:
                    draw.text((650,592),f"Applied v,w: {safe['velocity'][0]:.2f}, {safe['velocity'][1]:.2f} / {safe['reason']}",font=small,fill="#ffe18c")
            else:
                draw.text((650,565), "LLM choices: " + (" -> ".join(completed) or "waiting"), font=font, fill="white")
            draw.text((28,606), "Task:",font=font,fill="#7de0c3")
            for line_index,line in enumerate(textwrap.wrap(task["instruction"],width=105)[:4]):
                draw.text((28,635+line_index*26),line,font=font,fill="white")
            draw.text((28,758), "Recorded RGB + telemetry; direct camera sampling ~5 Hz. Final card is not extra motion." if direct else "Camera sampled ~1.7 Hz; playback uses simulation time. Final score is in summary.json.",font=small,fill="#a9b7c7")
            if initial is None:
                image.save(directory / "preview.png")
                initial = True
            process.stdin.write(image.tobytes())
        final_image = image.copy()
        final_draw = ImageDraw.Draw(final_image)
        final_draw.rectangle((18, 90, 1260, 600), fill="#172434")
        outcome = "PASS" if summary.get("passed") else "FAIL"
        color = "#7de0c3" if summary.get("passed") else "#ffaea0"
        final_draw.text((45, 118), "FINAL RESULT: " + outcome, font=title_font, fill=color)
        final_draw.text((45, 175), "Qwen actions: " + str(summary.get("actions", {})) if direct else "Qwen choices: " + (" -> ".join(decision.get("action", "error") for decision in decisions) or "Nav2 baseline"), font=font, fill="white")
        expected = task.get("expected")
        final_draw.text((45, 215), "Expected: stop immediately and stay still" if direct and task.get("halt") else "Expected: reach target and choose stop" if direct else "Expected: " + (" -> ".join(expected) if expected else "not scored / fixed baseline"), font=font, fill="white")
        final_draw.text((45, 255), f"Halt test path: {summary.get('truth_path_m', 0):.3f} m; selected stop: {summary.get('model_stopped_at_goal')}" if direct and task.get("halt") else f"Error: {summary.get('truth_error_m', 0):.3f} m; stopped: {summary.get('model_stopped_at_goal')}; body contacts: {summary.get('contacts', 0)}" if direct else f"Reached Nav2 goals: {len(summary.get('goals', []))}   Body contacts: {summary.get('contacts', 0)}   Path: {summary.get('truth_path_m', 0):.2f} m", font=font, fill="white")
        final_draw.text((45, 310), "Wrong semantic decisions are not corrected by the evaluator.", font=font, fill="#a9b7c7")
        final_draw.text((45, 350), "Qwen controls steering. No Nav2, no path planner, no steering fallback." if direct else "Nav2 controls the wheels. Qwen selects a known destination.", font=font, fill="#a9b7c7")
        final_draw.text((45, 400), "This is a result card, not additional robot motion.", font=font, fill="#a9b7c7")
        for final_index in range(3 * fps):
            process.stdin.write(final_image.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("Video encoding failed")
    return {"file": str(directory / "replay.mp4"), "frames": total, "simulation_seconds": times[-1]-times[0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    print(json.dumps(render(parser.parse_args().directory)))

"""Post-hoc prompt controls on exact recorded failure images, not new routes."""
import base64
import json
from pathlib import Path
import socket
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "robot_sim/results/qwen9b_evaluation_20261026"
CASES = [
    ("apartment_green_red", 1, "red", [
        ("english_history", "The delivery order is green, then red. Green has already been reached. Choose the next unvisited destination from goal_candidates; never repeat a completed marker."),
        ("russian_explicit_history", "Зелёная станция уже посещена, этот этап выполнен. Следующий этап — возврат к красной станции. Выбери следующую цель только из goal_candidates; не повторяй выполненные этапы."),
    ]),
    ("courtyard_blue_green", 0, "blue", [
        ("english_negation", "Do not visit red. First deliver to blue, then to green. Choose the next destination from goal_candidates."),
        ("russian_positive_order", "Порядок доставки: синяя станция, затем зелёная. Выбери следующую ещё не посещённую станцию из этого порядка."),
    ]),
    ("warehouse_unknown_yellow", 0, "stop", [
        ("english_missing_target", "Deliver only to the yellow marker. If yellow is not in goal_candidates, stop. Never substitute another color."),
        ("russian_explicit_missing", "Нужна только жёлтая станция. В goal_candidates доступны красная, синяя и зелёная, жёлтой нет. Поэтому никуда не ехать: выбери stop."),
    ]),
]
results = []
for name, step, expected, variants in CASES:
    directory = OUTPUT / name
    observation = json.loads((directory / f"request_{step}.json").read_text())
    observation["image"] = base64.b64encode((directory / f"vision_goal_{step}.png").read_bytes()).decode()
    for variant, instruction in [("original_repeat", observation["instruction"]), *variants]:
        request = {**observation, "instruction": instruction}
        started = time.monotonic()
        with socket.create_connection(("127.0.0.1", 18765), timeout=120) as connection:
            connection.sendall((json.dumps(request) + "\n").encode())
            response = json.loads(connection.makefile("rb").readline())
        result = {"case": name, "step": step, "variant": variant, "expected": expected,
                  "instruction": instruction, "passed": response.get("action") == expected,
                  "latency": time.monotonic() - started, **response}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
(OUTPUT / "failure_prompt_controls.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

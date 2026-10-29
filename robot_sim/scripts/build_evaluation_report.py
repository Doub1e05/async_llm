"""Build an offline review bundle using completed logs only."""
from collections import Counter
import html
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "robot_sim/results/qwen9b_evaluation_20261026"


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


results = []
latencies = []
cards = []
report = [
    "# Qwen3.5-9B: оценка управления роботом в Gazebo",
    "Дата: 26 октября 2026. Симуляция, НЕ испытание физического робота.",
    "## Методика",
    "- Qwen/Qwen3.5-9B, BF16, без квантизации; GPU 2 и 6, RTX A4000.",
    "- Transformers/Accelerate layer sharding. Веса полностью на GPU. Это не benchmark исходного AsyncLLM, где Qwen3.5 attention пока TP=1.",
    "- Прямое CUDA P2P повреждало данные: gpu_copy_check.json. Применён проверенный CPU staging активаций, не offload весов. invalid_backend_nan/ исключён из оценки.",
    "- Все процессы эксперимента закреплены за одним общим CPU 0; Gazebo использует llvmpipe без GPU. Аудит: cpu_audit.json и gpu_during.csv.",
    "- Вход: RGB 320x240, известные координаты трёх станций, история посещений. Выход: red/blue/green/stop. Nav2 отвечает за скорости, локализацию и обход препятствий.",
    "- Эталонные ответы не отправляются модели. Неправильный ответ не исправляется вручную; выбор вне candidates отклоняется.",
    "- Один запуск на задачу, seed=0: smoke evaluation, не статистическая гарантия.",
    "- MP4: настоящие кадры камеры (~2 Гц simulation time) + реконструкция траектории по телеметрии. Карта слева — НЕ запись 3D-окна Gazebo. Воспроизведение по simulation time, не wall time.",
    "## Результаты",
]
run_order = ["warehouse_red_blue", "apartment_green_red", "courtyard_blue_green",
             "warehouse_nearest_visible", "warehouse_unknown_yellow", "warehouse_nav2_baseline"]
for directory in [OUTPUT / name for name in run_order]:
    if not directory.is_dir() or not (directory / "summary.json").exists():
        continue
    summary = load(directory / "summary.json")
    task = load(directory / "task.json", {"instruction": "Nav2 baseline: fixed red -> blue", "expected": None})
    decisions = lines(directory / "vision.jsonl")
    safety = Counter(record["reason"] for record in lines(directory / "safety.jsonl"))
    trajectory = lines(directory / "trajectory.jsonl")
    scan_min = min((record["scan_min"] for record in trajectory), default=None)
    latencies.extend(record["latency"] for record in decisions if not record.get("error"))
    actual = [record.get("action") for record in decisions]
    expected = task.get("expected")
    correct = sum(index < len(expected) and label == expected[index] for index, label in enumerate(actual)) if expected else None
    video = f"{directory.name}/replay.mp4" if (directory / "replay.mp4").exists() else None
    result = {"id": directory.name, "task": task, "summary": summary, "decisions": decisions,
              "safety_reasons": dict(safety), "min_scan_m": scan_min,
              "correct_steps": correct, "evaluated_steps": len(expected) if expected else None, "video": video}
    results.append(result)
    choices = " → ".join(actual) or "—"
    semantic = summary.get("semantic_passed")
    semantic_label = "да" if semantic is True else "нет" if semantic is False else "не размечено"
    goals = summary.get("goals", [])
    arrived = sum(goal.get("status") == 4 and goal.get("truth_error_m", 1) <= 0.25 for goal in goals)
    errors = [round(goal["truth_error_m"], 3) for goal in goals]
    report.extend([
        f"### {directory.name}", task["instruction"],
        f"- Выбор: {choices}; семантика: {semantic_label}; passed={summary.get('passed')}; достигнуто целей: {arrived}.",
        f"- Путь: {summary.get('truth_path_m', 0):.2f} м; ошибки позы: {errors} м; контакты корпуса: {summary.get('contacts', 0)}.",
        f"- Ошибка исполнения: {summary.get('error', 'нет')}",
        f"- Gateway (число тактов, не отдельных событий): {dict(safety)}.",
        f"- Минимальный lidar range: {scan_min}; это не clearance корпуса.",
        f"- Видео: [{directory.name}/replay.mp4]({directory.name}/replay.mp4). Метрики/решения/траектория: JSON и JSONL в том же каталоге.",
    ])
    player = f'<video controls preload="metadata" poster="{directory.name}/preview.png" src="{video}"></video>' if video else "<p>Видео отсутствует</p>"
    cards.append(
        f'<article><h2>{html.escape(directory.name)}</h2><p>{html.escape(task["instruction"])}</p>'
        f'<div class="badges">Выбор: {html.escape(choices)} · Задание: {semantic_label} · Цели: {arrived} · Путь: {summary.get("truth_path_m", 0):.2f} м · Контакты: {summary.get("contacts", 0)}</div>'
        f'{player}<p class="error">{html.escape("Выбор цели отклонён валидатором; подробности в summary.json" if summary.get("error") else "Станции достигнуты, но смысл задания нарушен" if semantic is False else "")}</p>'
        f'<p><a href="{directory.name}/summary.json">Метрики</a> · <a href="{directory.name}/vision.jsonl">Решения LLM</a> · <a download href="{video}">Скачать MP4</a></p></article>'
    )
probes = load(OUTPUT / "probes.json", [])
checked = [probe for probe in probes if probe.get("expected")]
overview = {"runs": results, "probes": probes, "model": load(OUTPUT / "model_metadata.json", {}),
            "decision_latency_median_s": statistics.median(latencies) if latencies else None,
            "decision_latency_max_s": max(latencies) if latencies else None,
            "synthetic_controls_passed": sum(probe.get("passed") is True for probe in checked),
            "synthetic_controls_total": len(checked)}
(OUTPUT / "evaluation.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2))
report.extend(["## Отдельные зрительные контроли", f"Синтетические контроли: {overview['synthetic_controls_passed']}/{len(checked)}; это не реальные проезды."])
for probe in probes:
    report.append(f"- {probe['id']}: эталон {probe.get('expected')}; ответ {probe.get('action')}; passed={probe.get('passed')}; scores={probe.get('probabilities')}.")
report.extend([
    "## Задержки", f"Медиана: {overview['decision_latency_median_s']} с; максимум: {overview['decision_latency_max_s']} с. Это TCP + preprocess + inference на одном CPU, не чистое время GPU.",
    "## Ограничения интерпретации",
    "- Правильный порядок цветов проверяет инструкцию/историю, не самостоятельное планирование движения по изображению.",
    "- Достижение целей и обход препятствий принадлежат Nav2; Qwen не управляет колёсами напрямую.",
    "- Координаты станций известны. Не проверены SLAM, неизвестные предметы, динамические люди и физический робот.",
    "- У nearest-visible нет размеченного эталона на каждом кадре: accuracy для него не вычисляется.",
    "- Softmax по четырём токенам не является калиброванной уверенностью или вероятностью безопасности.",
    "## Следующие шаги",
    "- Повторы с разными позами/семенами; сравнение 0.8B/9B на одинаковых заданиях и кадрах.",
    "- Разметить видимость/пространственный порядок объектов, проверить grounded goals без готовых координат.",
    "- Отдельно оценивать семантику Qwen, исполнение Nav2, остановки защиты и end-to-end success.",
    "- Перед реальным роботом: калибровка сенсоров, одометрии, аппаратный watchdog и аварийная остановка в ограждённой зоне.",
    "## Источники интерфейса", "https://huggingface.co/Qwen/Qwen3.5-9B", "https://huggingface.co/docs/accelerate/main/concept_guides/big_model_inference",
])
analysis = OUTPUT / "ANALYSIS.md"
if analysis.exists():
    report.extend(["## Подробный разбор", "[Разбор ошибок и рекомендации](ANALYSIS.md)", analysis.read_text()])
(OUTPUT / "REPORT.md").write_text("\n\n".join(report) + "\n")
probe_rows = "".join(f'<tr><td>{html.escape(probe["id"])}</td><td>{probe.get("expected")}</td><td>{probe.get("action")}</td><td>{probe.get("passed")}</td></tr>' for probe in probes)
page = '''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen-9B · Проезды робота</title><style>body{margin:0;background:#0d1520;color:#e8eef7;font:16px/1.55 system-ui,sans-serif}main{max-width:1100px;margin:40px auto;padding:0 24px}h1{font-size:36px;line-height:1.2}p{color:#b8c7d9}a{color:#7de0c3}article{background:#172434;border:1px solid #304354;border-radius:16px;padding:24px;margin:28px 0}video{width:100%;margin-top:18px;border-radius:8px;background:black}.badges{color:#ffe18c}.error{color:#ffaea0}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #304354}.notice{padding:16px;border-left:3px solid #7de0c3;background:#172434}</style><main><h1>Qwen3.5-9B: проезды робота</h1><p>Gazebo · 26 октября 2026 · BF16 · GPU 2 + 6 · один CPU</p><div class="notice">Это симуляция. Qwen выбирает станцию; Nav2 отвечает за движение и обход препятствий. Видео: настоящая камера + карта из телеметрии, не запись окна Gazebo.</div><p><a href="REPORT.md">Полный отчёт</a> · <a href="evaluation.json">Все метрики</a> · <a href="model_metadata.json">Модель</a> · <a href="gpu_copy_check.json">GPU P2P</a></p>'''
page += '<article><h2>Итог оценки</h2><p>Семантический успех: <strong>1 из 4</strong> заданий с эталоном. Правильные решения: <strong>3 из 7</strong>. Все 8 принятых Nav2-целей в LLM-прогонах достигнуты, контактов корпуса нет. Это не равнозначно правильному выполнению миссии.</p><p>Главные ошибки: повтор уже посещённой станции, игнорирование запрета красной станции и подмена отсутствующей жёлтой цели зелёной. На исходном nearest-visible запросе пустой кадр также привёл к догадке вместо stop.</p><p>Постфактум уточнение на английском исправило 3/3 проверенных отказов на тех же кадрах, но это не новые успешные проезды.</p><p><a href="ANALYSIS.md">Подробный разбор ошибок</a> · <a href="failure_prompt_controls.json">Сравнение формулировок</a> · <a download href="review_bundle.tar.gz">Скачать видео и доказательства архивом</a></p></article>'
page += "".join(cards)
page += f'<article><h2>Отдельные зрительные контроли</h2><p>Синтетические тесты — не проезды. Успех: {overview["synthetic_controls_passed"]}/{len(checked)}.</p><table><tr><th>Тест</th><th>Эталон</th><th>Ответ</th><th>Успех</th></tr>{probe_rows}</table></article>'
page += '<p>Невалидные прогоны до исправления обмена GPU исключены из оценки. Один запуск на задание — недостаточная выборка для гарантий.</p></main></html>'
(OUTPUT / "index.html").write_text(page)
print(json.dumps({"runs": len(results), "latency_median": overview["decision_latency_median_s"], "synthetic_pass": overview["synthetic_controls_passed"], "synthetic_total": len(checked)}))

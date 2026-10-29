"""Second report: direct LLM steering, with original and revised observations separate."""
from collections import Counter
import html
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "robot_sim/results/qwen9b_direct_20261026"
NAMES = ["straight", "left_turn", "right_turn", "blocked_goal", "stop_command"]


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


corrections = json.loads((OUTPUT / "corrections.json").read_text()) if (OUTPUT / "corrections.json").exists() else {}
excluded = []
results = []
cards = []
versions = {}
report = [
    "# Отчёт 2: прямое вождение Qwen3.5-9B без Nav2",
    "Дата: 26 октября 2026. Все опыты в Gazebo, не на физическом роботе.",
    "## Что теперь управляет движением",
    "RGB-камера + lidar + направление/дальность цели из колёсной одометрии → Qwen → forward/left/right/stop → фиксированная скорость → независимый safety gateway → Gazebo DiffDrive.",
    "Nav2, A*, DWB, waypoint steering и любой резервный алгоритм выбора направления НЕ запускаются. Нет списка цветных станций. Единственная непрерывно принимающая решения о рулении система — LLM.",
    "Действия: forward=(0.16 м/с,0), left=(0,+0.65 рад/с), right=(0,-0.65 рад/с), stop=(0,0). Импульс 0.65 simulation seconds; затем остановка до следующего ответа. Максимум импульса 3 wall seconds. Это дискретное импульсное управление, не непрерывный высокочастотный контроллер.",
    "Во время инференса робот стоит; ROS и watchdog продолжают работу. Ответ старше 1.5 wall seconds или с просроченными сенсорами не исполняется. Ограничение безопасности не выбирает альтернативное направление, только обнуляет/ограничивает команды.",
    "Цель — известная точка. Используются заданная начальная поза и odometry для distance/bearing; ground truth нужен только для оценки и видео. Это PointGoal RGB+lidar+odometry, НЕ camera-only визуальное обнаружение цели и НЕ SLAM.",
    "Модель: Qwen3.5-9B BF16, Transformers/Accelerate на GPU 2 и 6, CPU staging активаций (из-за установленной в первом отчёте проблемы прямого CUDA P2P). Все потоки на одном общем CPU 0; Gazebo llvmpipe не использует GPU. Исходный AsyncLLM backend сохранён, но двухкарточные опыты выполнены reference backend.",
    "## Два интерфейса наблюдения — не смешивать результаты",
    "numeric_v1: RGB, численные distance/bearing, три lidar-сектора и последние 8 действий. На первых отказах обнаружены плохая остановка по численному порогу и нестабильное выравнивание к цели.",
    "state_v2: те же сенсоры + goal_reached и target_direction, детерминированно вычисленные из одометрии. Переформулирован prompt с явными приоритетами. Это инженерная поддержка представления наблюдения и rule-guided prompting, а НЕ увеличение интеллекта модели или чистый end-to-end vision. Автоматически выбранного action или готовой траектории по-прежнему нет.",
    "state_v2 настроена после просмотра numeric_v1 на тех же задачах. Это development comparison, не независимый тест на новой выборке. Один прогон на случай; статистические выводы недопустимы.",
    "## Критерий успеха",
    "Для проездов: odom distance≤0.15 м, реальная ошибка≤0.20 м, модель сама выбрала stop, нет контактов корпуса/таймаута. Защитная остановка при достижении порога НЕ засчитывается как stop от LLM. Для halt: модель выбрала stop и путь<0.03 м.",
    "Время/число действий ограничены. Прерывание из-за отсутствия прогресса, stale-инференса или контакта — отказ, а не успешный проезд.",
    "Видео — настоящие RGB-кадры (~5 Гц) + карта/траектория из телеметрии; не видео 3D-окна Gazebo. На видео показаны действие Qwen и фактические v,w после safety gateway; последние 3 секунды — итоговая карточка, не дополнительное движение.",
    "## Результаты",
]
for suffix, version in [("", "numeric_v1"), ("_state_v2", "state_v2")]:
    group = []
    report.extend([f"### {version}", "| Задание | Успех | Вблизи цели по odom | Stop от модели | Ошибка, м | Путь, м | Контакты |", "|---|---|---|---|---:|---:|---:|"])
    for original in NAMES:
        original_name = original + suffix
        name = original_name
        if original_name in corrections:
            replacement = corrections[original_name]["replacement"]
            if (OUTPUT / replacement / "summary.json").exists():
                excluded.append({"id": original_name, **corrections[original_name]})
                name = replacement
            else:
                continue
        directory = OUTPUT / name
        summary_file = directory / "summary.json"
        if not summary_file.exists():
            continue
        summary = json.loads(summary_file.read_text())
        task = json.loads((directory / "task.json").read_text())
        decisions = records(directory / "decisions.jsonl")
        safety = records(directory / "safety.jsonl")
        counters = Counter(row["reason"] for row in safety)
        action_counts = Counter(row["action"] for row in decisions)
        obstacle_forwards = sum(row["action"] == "forward" and observation.get("ranges", {}).get("front", 10) < 0.45
                               for row, observation in zip(decisions, records(directory / "observations.jsonl")))
        latencies = [row["latency"] for row in decisions]
        record = {"id": name, "version": version, "summary": summary, "task": task,
                  "actions": dict(action_counts), "latency_median": statistics.median(latencies) if latencies else None,
                  "latency_max": max(latencies) if latencies else None, "safety_ticks": dict(counters),
                  "forward_requests_with_front_below_045": obstacle_forwards}
        group.append(record)
        results.append(record)
        error = summary.get("truth_error_m")
        error_text = "halt: n/a" if task.get("halt") else f"{error:.3f}" if error is not None else "—"
        passed = summary.get("passed", False)
        report.append(f"| {original} | {passed} | {summary.get('reached_odom_tolerance')} | {summary.get('model_stopped_at_goal')} | {error_text} | {summary.get('truth_path_m',0):.2f} | {summary.get('contacts',0)} |")
        report.extend([f"- {name}: actions={dict(action_counts)}, причина={summary.get('error')}, latency median/max={record['latency_median']}/{record['latency_max']} с; safety ticks={dict(counters)}; forward при front<0.45: {obstacle_forwards}.", f"- [Видео {name}]({name}/replay.mp4), подробности в summary.json, decisions.jsonl, observations.jsonl, execution.jsonl, safety.jsonl."])
        video = f'<video controls preload="metadata" poster="{name}/preview.png" src="{name}/replay.mp4"></video>' if (directory / "replay.mp4").exists() else "<p>Видео ещё обрабатывается</p>"
        badge = "PASS" if passed else "FAIL"
        cards.append(f'<article id="{name}"><h2>{html.escape(name)} <span class="{"pass" if passed else "fail"}">{badge}</span></h2><p>{html.escape(task["instruction"])}</p><p>Наблюдение: {version} · Ошибка: {error_text} м · Путь: {summary.get("truth_path_m",0):.2f} м · Stop: {summary.get("model_stopped_at_goal")} · Контакты: {summary.get("contacts",0)}</p><p>Действия: {html.escape(str(dict(action_counts)))}</p>{video}<p class="fail">{html.escape(str(summary.get("error") or ""))}</p><p><a href="{name}/summary.json">Метрики</a> · <a href="{name}/decisions.jsonl">Действия LLM</a> · <a href="{name}/observations.jsonl">Входные наблюдения</a> · <a download href="{name}/replay.mp4">Скачать MP4</a></p></article>')
    total_latency = [row["latency"] for record in group for row in records(OUTPUT / record["id"] / "decisions.jsonl")]
    versions[version] = {"passed": sum(record["summary"].get("passed", False) for record in group),
                         "total": len(group), "driving_passed": sum(record["summary"].get("passed", False) for record in group if not record["task"].get("halt")),
                         "driving_total": sum(not record["task"].get("halt") for record in group),
                         "contacts": sum(record["summary"].get("contacts", 0) for record in group),
                         "decision_latency_median": statistics.median(total_latency) if total_latency else None,
                         "decision_latency_max": max(total_latency) if total_latency else None}
    report.append(f"Итог {version}: {json.dumps(versions[version],ensure_ascii=False)}")
probes_file = OUTPUT / "control_probes.json"
probes = json.loads(probes_file.read_text()) if probes_file.exists() else []
report.extend(["## Диагностические контроли numeric_v1 (без движения)"])
for probe in probes:
    report.append(f"- {probe['id']}: ожидалось {probe['expected']}, получено {probe.get('action')}, passed={probe.get('passed')}.")
report.extend([
    "## Короткий вывод",
    "В numeric_v1 полный успех 1/5 (только halt), маршрутные задачи 0/4: дважды робот приблизился к цели, но не выбрал stop, левый поворот закончился кругами, обход не состоялся.",
    "В state_v2 полный успех 3/5: прямая и цель слева достигнуты со stop, halt выполнен. Маршруты 2/4. При правом повороте модель выбрала stop, но реальная ошибка 0.211 м превысила заранее заданный предел 0.20 м; критерий не ослаблялся. При ящике — 5 forward и затем 33 stop, до цели не доехал.",
    "numeric_v1 у ящика выдал 31 forward, 14 right, 2 left и 3 stop; safety gateway 323 такта блокировал forward по obstacle. state_v2 не столкнулся, но замер перед препятствием. Ноль контактов нельзя приписывать только качеству модели.",
    "Медиана/максимум решения: numeric_v1 0.400/0.850 с, state_v2 0.600/1.045 с. Это замер полного цикла ожидания ответа на данном общем CPU и сервере, не чистое GPU-время.",
    "Реализовано настоящее дискретное LLM-руление, но надёжный локальный навигатор из модели пока не получился. Лучшее простое вождение достигнуто с явными одометрическими признаками и инструкциями-правилами, не только за счёт изображения.",
    "## Проблемы и интерпретация",
    "- numeric_v1 чрезмерно предпочитает forward, плохо использует численные bearing/дистанцию и не останавливается у цели. Это воспроизвелось на отдельном численном контроле distance=0.1 при tolerance=0.15.",
    "- Без AMCL/глобальной локализации накапливается ошибка колёсной одометрии, особенно при многократных разворотах. Поэтому proximity по odom и физическая точность измеряются раздельно.",
    "- Локальный обход препятствия требует временно отвернуться от цели, затем вернуться. Короткая история действий и один кадр не заменяют пространственную память; возможны круги/колебания, повтор forward при недостаточном clearance и остановка watchdog по отсутствию прогресса.",
    "- Даже хорошие простые повороты state_v2 не означают готовность к произвольной карте. Сенсорные категории и prompt задают явные правила; обобщение нужно проверять отдельно.",
    "- Логи safety показывают вмешательства защиты. Отсутствие контактов не означает, что LLM сама всегда выбирала безопасные команды; guardian мог блокировать forward.",
    "- Pulse executor принудительно останавливается при достижении порога, при stale-сенсорах/ответе, таймауте и завершении процесса. Эти остановки не приписываются интеллекту модели.",
    "- CPU=1 и задержка RGB-инференса дают низкочастотную stop-and-go езду. Для реального непрерывного управления нужно отдельно проверить latency tails и sensor freshness.",
    "## Что делать дальше",
    "1. Добавить память локальной геометрии и явное состояние манёвра обхода; сравнить с wall-following/реактивным baseline, не маскируя его под LLM.",
    "2. Отдельно оценить vision-only, RGB+lidar, RGB+lidar+pointgoal на новых картах и начальных позах; повторить seed/переформулировки.",
    "3. Настроить продолжительность действий, сглаживание и асинхронное принятие решений, сохранив жёсткий TTL и остановку при потере наблюдения.",
    "4. Для физического робота — только после калибровки, аппаратного watchdog, emergency stop и ограниченной безопасной зоны. Этот эксперимент не даёт допуска к автономной езде в реальном окружении.",
    "## Исправление гонки оценщика",
    "В двух right_turn-прогонах робот пересёк порог 0.15 м после снимка наблюдения, но до ответа. Оценщик ошибочно требовал stop от модели, ещё видевшей distance>0.15. Теперь такой ответ не исполняется и не оценивается как ошибка: робот стоит и получает новое наблюдение. Исправление проверено unit-тестом.",
    "Два первоначальных right_turn-прогона исключены из основной статистики и сохранены по исходным путям. В таблицах использован один повтор каждого протокола с суффиксом _rechecked, не лучший из нескольких. Corrections.json содержит точные пары и причины.",
    "Результаты первого отчёта с Nav2 нельзя напрямую сравнивать с процентами здесь: другие, более короткие PointGoal-задачи, иное определение успеха и нет выбора цветных станций по свободной инструкции.",
    "## Доказательства",
    "runtime_audit.json: реальные ROS-узлы, издатель cmd_vel_requested=direct_llm_driver, отсутствие plan publishers, CPU-affinity всех потоков=0.",
    "В каталогах проездов сохранены world.sdf с начальными позами (препятствия не удалялись), task.json, изображения каждого запроса, решения, запрошенные и фактические скорости, траектории и кадры.",
])
output = {"versions": versions, "runs": results, "probes": probes, "nav2_used": False, "excluded_scoring_race_runs": excluded}
(OUTPUT / "evaluation.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
(OUTPUT / "REPORT_2.md").write_text("\n\n".join(report) + "\n")
overview = "".join(f'<p><strong>{version}</strong>: общий успех {value["passed"]}/{value["total"]}; проезды {value["driving_passed"]}/{value["driving_total"]}; контакты корпуса {value["contacts"]}; медиана решения {value["decision_latency_median"]:.3f} с.</p>' for version,value in versions.items())
page = '''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Отчёт 2 · Прямое LLM-вождение</title><style>body{margin:0;background:#101923;color:#e7eef7;font:16px/1.5 system-ui}main{max-width:1100px;padding:32px 24px;margin:auto}h1{font-size:34px}article{padding:24px;margin:24px 0;background:#1a293a;border:1px solid #334c63;border-radius:14px}p{color:#bbcbdc}a{color:#7de0c3}video{width:100%;border-radius:8px;background:black}.pass{color:#7de0c3}.fail{color:#ffafa0}h2{font-size:24px}.notice{border-left:3px solid #ffcd72;padding:16px;background:#1a293a}</style><main><h1>Отчёт 2: прямое вождение через Qwen-9B</h1><p>Gazebo · BF16 · GPU 2 + 6 · один CPU</p><div class="notice">Теперь Qwen выдаёт <strong>forward / left / right / stop</strong>. Nav2 и скриптового руления нет. Модель получает камеру, lidar и PointGoal из одометрии. Safety gateway может остановить команду, но не выбирает путь.</div><article><h2>Итоги двух интерфейсов наблюдения</h2>'''
page += overview + '<p>numeric_v1 — численные bearing/distance; state_v2 — добавлены факты «цель достигнута» и «цель слева/справа», уточнены приоритеты prompt. Это доработка на тех же задачах, не независимый тест и не camera-only автономность.</p><p><a href="REPORT_2.md">Полный второй отчёт</a> · <a href="evaluation.json">Метрики</a> · <a href="runtime_audit.json">Аудит: без Nav2, один CPU</a> · <a download href="review_bundle.tar.gz">Скачать видео и логи</a></p></article>'
page += '<p><a href="#left_turn_state_v2">Успешный левый поворот</a> · <a href="#blocked_goal_state_v2">Неуспешный обход ящика</a> · <a href="#straight">Исходная ошибка остановки</a></p>'
page += "".join(cards)
if excluded:
    page += '<article><h2>Исключённые прогоны: ошибка оценщика, не модели</h2><p>Момент достижения цели изменился после отправки снимка. Ошибка проверки stop исправлена; эти записи не входят в проценты выше, вместо них выполнены отдельные повторы.</p>'
    for item in excluded:
        name = item["id"]
        page += f'<p>{name} → {item["replacement"]}: <a href="{name}/summary.json">исходные метрики</a> · <a href="{name}/replay.mp4">исходное видео</a></p>'
    page += '</article>'
page += '<p>Камера записана из симуляции, карта восстановлена из телеметрии; не запись 3D-окна Gazebo. Последние 3 секунды — итоговая карточка. Защитная остановка не засчитывается как stop, выбранный моделью.</p></main></html>'
(OUTPUT / "index.html").write_text(page)
print(json.dumps(versions))

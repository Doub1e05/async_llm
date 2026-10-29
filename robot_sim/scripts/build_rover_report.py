"""Package the measured Gazebo integration checks, including the failed step."""
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from robot_sim.rover import MODEL_DIR, ROOT


def main():
    output=ROOT/"robot_sim/results/rover_20261029_acceptance"
    cases=[("suite_verified","Движение, повороты, торможение и watchdog"),
           ("contact_verified","Контакт корпуса с ящиком"),
           ("step","Порог высотой 5 мм — ограничение текущей модели")]
    data={}
    cards=[]
    for name,title in cases:
        folder=output/name
        summary=json.loads((folder/"summary.json").read_text())
        data[name]=summary
        movie=folder/"replay.mp4"
        if not movie.exists():
            subprocess.run(["python","robot_sim/scripts/render_rover_replay.py",str(folder)],check=True,cwd=ROOT)
        verdict="PASS" if summary["passed"] else "FAIL"
        cards.append(f'<article><h2>{html.escape(title)}</h2><p class="{verdict.lower()}">{verdict}</p>'
                     f'<video controls preload="metadata" poster="{name}/preview.png" src="{name}/replay.mp4"></video>'
                     f'<p><a href="{name}/summary.json">Измерения JSON</a></p></article>')
    for scene in ("warehouse","apartment","courtyard"):
        name="inspect_"+scene
        data[name]=json.loads((output/name/"summary.json").read_text())
    llm_cards=[]
    llm_runs=[]
    previous=ROOT/"robot_sim/results/qwen9b_direct_20261026"
    for name,title in [
        ("straight_state_v2","LLM: движение вперёд и остановка"),
        ("left_turn_state_v2","LLM: поворот налево и движение к цели"),
        ("right_turn_state_v2_rechecked","LLM: поворот направо и движение к цели")]:
        source=previous/name
        target=output/"llm_driving"/name
        target.mkdir(parents=True,exist_ok=True)
        for filename in ("replay.mp4","preview.png","summary.json","task.json","decisions.jsonl"):
            shutil.copy2(source/filename,target/filename)
        summary=json.loads((source/"summary.json").read_text())
        same=hashlib.sha256((source/"replay.mp4").read_bytes()).hexdigest()==hashlib.sha256((target/"replay.mp4").read_bytes()).hexdigest()
        if not same:raise RuntimeError("Video copy differs from original")
        actions=summary["actions"]
        if any(action not in {"forward","left","right","stop"} for action in actions):
            raise RuntimeError("Unexpected recorded action")
        note=("Задача выполнена." if summary["passed"] else
              "Модель остановилась, но ошибка положения 0.211 м превышает допуск 0.20 м; задача не засчитана.")
        rel="llm_driving/"+name
        llm_cards.append(f'<article><h3>{html.escape(title)}</h3><p>TurtleBot · Qwen3.5-9B · 26 октября 2026</p>'
                         f'<p>{html.escape(note)} Команды: {html.escape(str(actions))}</p>'
                         f'<video controls preload="metadata" poster="{rel}/preview.png" src="{rel}/replay.mp4"></video>'
                         f'<p><a href="{rel}/summary.json">Результат</a> · <a href="{rel}/decisions.jsonl">Решения LLM</a></p></article>')
        llm_runs.append({"id":name,"robot":"turtlebot3_burger","model":"Qwen3.5-9B",
                         "source":str(source),"actions":actions,"passed":summary["passed"],
                         "truth_error_m":summary["truth_error_m"],
                         "video_sha256":hashlib.sha256((target/"replay.mp4").read_bytes()).hexdigest()})
    data["previous_llm_driving"]={"date":"2026-10-26","runs":llm_runs,
        "reverse_available":False,"actions":["forward","left","right","stop"],
        "note":"Earlier TurtleBot trials; not the new mini rover. RGB, lidar and odometry target features. Safety gateway active; no Nav2 or scripted steering."}
    segments={s["phase"]:s for s in data["suite_verified"]["segments"]}
    metrics={
        "forward_m":segments["forward"]["distance_m"],
        "reverse_m":segments["reverse"]["forward_m"],
        "left_yaw_deg":segments["left"]["yaw_rad"]*180/3.141592653589793,
        "right_yaw_deg":segments["right"]["yaw_rad"]*180/3.141592653589793,
        "left_pivot_drift_m":segments["left"]["distance_m"],
        "right_pivot_drift_m":segments["right"]["distance_m"],
        "braking_distance_m":segments["brake"]["distance_m"],
        "watchdog_tail_motion_m":segments["watchdog_dropout"]["tail_distance_m"],
        "contact_messages":data["contact_verified"]["body_contact_messages"],
    }
    inspection="; ".join(scene+": "+("PASS" if data["inspect_"+scene]["passed"] else "FAIL")
                         for scene in ("warehouse","apartment","courtyard"))
    lines=[
        "# Мини-ровер в Gazebo — 29 октября 2026",
        "",
        "Модель перенесена на сервер в robot_sim/models/yandex_rover. Источник — присланный rover_urdf;",
        "масштаб сохранён, старая MuJoCo-геометрия с другой колеёй не смешивается с URDF.",
        "",
        "Масса 0.500 кг; корпус 166.786 × 112.859 × 123.307 мм; колея 115.356 мм;",
        "диаметр ведущих колёс 44.248 мм; внешний габарит по колёсам около 131.612 мм.",
        "",
        "Настроены текстуры, нормали мешей, коллизии, контакты, моментный привод двух центральных колёс,",
        "ограничение разгона/торможения, watchdog, энкодерная одометрия и TF.",
        "Добавлены виртуальные RGB, lidar, IMU и обзорная камера для записи.",
        "",
        "## Проверка",
        "",
        f"- Прямой ход: {metrics['forward_m']:.3f} м за 4 с команды 0.12 м/с с ограниченным разгоном.",
        f"- Задний ход: {metrics['reverse_m']:.3f} м за 4 с команды −0.08 м/с.",
        f"- Повороты за 4 с команды ±0.65 рад/с: {metrics['left_yaw_deg']:.1f}° и {metrics['right_yaw_deg']:.1f}°.",
        f"- Смещение центра при поворотах: {1000*metrics['left_pivot_drift_m']:.2f} / {1000*metrics['right_pivot_drift_m']:.2f} мм.",
        f"- Путь после команды торможения: {1000*metrics['braking_distance_m']:.1f} мм.",
        f"- Остаточное движение в конце проверки watchdog: {1000*metrics['watchdog_tail_motion_m']:.3f} мм.",
        f"- Контакт с ящиком: физическое препятствие удерживает корпус; получено {metrics['contact_messages']} сообщений.",
        "- В основном проезде корпус не соприкасался с препятствиями.",
        "- Проверены изображение, калибровка камеры, lidar, IMU, состояния колёс и sensor TF.",
        "- 27 unit/regression tests: PASS.",
        "- Загрузка прежних сцен: "+inspection+".",
        "",
        "## Ограничения",
        "",
        "Порог 5 мм не пройден: ровер остановился, а энкодерная одометрия продолжала накапливаться.",
        "Это наблюдение для текущей жёсткой модели с двумя ведущими колёсами, а не измерение проходимости реального ровера.",
        "Параметры массы/инерции, привода и контактов остаются приближёнными.",
        "Подвеска не добавлялась. Низкое трение пассивных колёс 0.01 и увеличенный радиус центральных",
        "сохранены из присланного пресета и не считаются измеренными характеристиками.",
        "",
        "Видео — реальные кадры Gazebo с виртуальной следящей камеры, 3 Гц, с подписью этапа.",
        "Все проезды скриптовые. Qwen и Nav2 на новом ровере в этой работе не прогонялись.",
        "Два первых варианта проверки контактов с ошибочной маршрутизацией topic не входят в итоговую проверку.",
        "Ошибка исправлена и контакт отдельно подтверждён повторным испытанием.",
        "В промежуточном запуске final была ошибка сериализации отчёта; он заменён полным suite_verified.",
        "",
        "## Запуск",
        "",
        "```bash",
        "cd /path/to/async_llm",
        "bash robot_sim/scripts/run_rover.sh",
        "# Просмотр Gazebo:",
        "bash robot_sim/scripts/run_gui.sh warehouse --robot rover",
        "```",
        "",
        "Подробные команды и допущения: MODEL_NOTES.md (копия robot_sim/ROVER.md).",
        "Все процессы симуляции ограничены одним логическим CPU 0; renderer Mesa llvmpipe, 0 GPU.",
        "Исходные файлы и результаты прежних экспериментов сохранены.",
        ""
    ]
    lines += [
        "## Прежние видео непосредственного LLM-управления",
        "",
        "Добавлены три записи Qwen3.5-9B на TurtleBot от 26 октября 2026.",
        "Модель выбирает forward/left/right/stop по камере, lidar и направлению на цель из одометрии.",
        "Nav2 и скриптового руления нет; действует независимый защитный шлюз.",
        "Записи показывают камеру робота и телеметрию, а не внешнюю 3D-камеру нового ровера.",
        "Заднего хода под LLM нет: backward/reverse в том протоколе отсутствовали.",
        "Прямой проезд и левый поворот засчитаны; правый поворот не засчитан из-за ошибки положения 0.211 м при допуске 0.20 м.",
        "",
    ] + [f"- [{run['id']}](llm_driving/{run['id']}/replay.mp4)" for run in llm_runs]
    report="\n".join(lines)
    (output/"ROVER_REPORT.md").write_text(report)
    shutil.copy2(ROOT/"robot_sim/ROVER.md",output/"MODEL_NOTES.md")
    shutil.copy2(ROOT/"robot_sim/results/rover_unit_tests_20261029.log",output/"unit_tests.log")
    shutil.copy2(MODEL_DIR/"specification.json",output/"model_specification.json")
    manifest={}
    for name,_ in cases:
        movie=output/name/"replay.mp4"
        check=subprocess.run(["ffmpeg","-hide_banner","-v","error","-threads","1","-i",str(movie),
                              "-f","null","-"],capture_output=True,text=True)
        if check.returncode:raise RuntimeError(check.stderr)
        manifest[name]={"sha256":hashlib.sha256(movie.read_bytes()).hexdigest(),"decoded":True}
    data["metrics"]=metrics
    data["video_validation"]=manifest
    data["current_source_hashes"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT/"robot_sim/rover.py",ROOT/"robot_sim/rover_control.py",
                  ROOT/"robot_sim/scripts/rover_drive.py",ROOT/"robot_sim/scripts/rover_motion.py"]}
    (output/"evaluation.json").write_text(json.dumps(data,indent=2))
    screenshots="".join('<figure><img src="inspect_'+scene+'/camera_last.png"><figcaption>'+
                        scene+' — камера мини-ровера</figcaption></figure>'
                        for scene in ("warehouse","apartment","courtyard"))
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Мини-ровер в Gazebo</title><style>
body{margin:0;background:#101923;color:#e8eef4;font:17px/1.6 system-ui,sans-serif}
main{max-width:1050px;margin:auto;padding:30px}h1,h2{line-height:1.2}
h1{font-size:34px}h2{font-size:23px}a{color:#90dbff}
article{background:#182634;border:1px solid #354958;border-radius:12px;margin:26px 0;padding:20px}
video{display:block;width:100%;max-height:650px;background:black}
.pass{color:#8ce2b1}.fail{color:#ffb49c}.muted{color:#b0c2d0}
figure{display:inline-block;width:30%;margin:1%}img{width:100%;border-radius:8px}
pre{white-space:pre-wrap;background:#182634;padding:16px;border-radius:8px}summary{cursor:pointer}
@media(max-width:650px){main{padding:16px}figure{width:95%}}
</style><main><h1>Мини-ровер: Gazebo готов к экспериментам</h1>
<p class="muted">29 октября 2026 · ROS 2 Jazzy / Gazebo Harmonic · 1 CPU / 0 GPU</p>
<p>Исходный масштаб: <b>500 г</b>, колея <b>115 мм</b>, ведущие колёса <b>44 мм</b>.
Сохранены геометрия и текстуры. Приводы ограничены по моменту; одометрия вычисляется из колёс.</p>
<p><a href="ROVER_REPORT.md">Отчёт</a> · <a href="MODEL_NOTES.md">Настройки и запуск</a> ·
<a href="#llm-driving">LLM-вождение</a> · <a href="evaluation.json">Все измерения</a> · <a href="unit_tests.log">27 тестов</a> ·
<a href="review_bundle.tar.gz">Скачать результаты</a></p>
<p>Основной проезд и проверка контакта прошли. Порог 5 мм не пройден — результат сохранён ниже.
Физика приближённая, сенсоры виртуальные. Qwen в этих проездах не участвовал.</p>'''
    page += '<section id="llm-driving"><h2>Предыдущие прогоны: непосредственное LLM-вождение</h2>'
    page += '<p>Эти записи сделаны на <b>TurtleBot</b>, до переноса мини-ровера. Qwen3.5-9B выбирает '
    page += '<code>forward / left / right / stop</code> по RGB, lidar и направлению на цель из одометрии. '
    page += 'Nav2 и скриптового руления нет; защитный шлюз может блокировать команды. Видео — камера робота и телеметрия.</p>'
    page += '<p><b>Записи заднего хода с LLM нет:</b> backward/reverse в прежнем наборе действий отсутствовали.</p>'
    page += "".join(llm_cards)+'</section><h2>Скриптовая проверка нового мини-ровера — без LLM</h2>'
    page += "".join(cards)+"<h2>Проверка трёх прежних сцен</h2>"+screenshots
    page += "<details><summary>Краткие измерения</summary><pre>"+html.escape(json.dumps(metrics,ensure_ascii=False,indent=2))+"</pre></details>"
    page += "<h2>Запустить на сервере</h2><pre>cd /path/to/async_llm\nbash robot_sim/scripts/run_rover.sh</pre></main></html>"
    (output/"index.html").write_text(page)
    with tarfile.open(output/"review_bundle.tar.gz","w:gz") as archive:
        for p in output.rglob("*"):
            if p.is_file() and p.name not in {"review_bundle.tar.gz","http.log","http.pid"} and "frames" not in p.parts:
                archive.add(p,arcname=str(p.relative_to(output)))
    print(json.dumps(metrics,indent=2))


if __name__=="__main__":
    main()

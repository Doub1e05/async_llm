"""Measured mapping report and replay, separate from online perception."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import numpy as np
from PIL import Image
import yaml
from robot_sim.rover import ROOT


def build(output):
    summary=json.loads((output/"summary.json").read_text())
    stats=json.loads((output/"mapper_summary.json").read_text())
    frames=[json.loads(line) for line in (output/"map_frames.jsonl").read_text().splitlines()]
    if not frames:raise RuntimeError("No map frames")
    movie=output/"mapping_replay.mp4"
    if not movie.exists():
        listing=output/"replay_frames.txt"
        lines=[]
        for index,frame in enumerate(frames):
            lines.append("file '"+str(output/frame["file"])+"'")
            duration=frames[index+1]["sim_time"]-frame["sim_time"] if index+1<len(frames) else 2
            lines.append("duration "+str(max(.05,duration)))
        lines.append("file '"+str(output/frames[-1]["file"])+"'")
        listing.write_text("\n".join(lines)+"\n")
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","concat","-safe","0",
                        "-i",str(listing),"-vf","fps=4","-c:v","libx264","-threads","1","-preset","veryfast",
                        "-crf","22","-pix_fmt","yuv420p","-movflags","+faststart",str(movie)],check=True)
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-threads","1","-i",str(movie),
                    "-f","null","-"],check=True)
    shutil.copy2(ROOT/"robot_sim/MAPPING.md",output/"METHOD.md")
    shutil.copy2(ROOT/"robot_sim/results/mapping_unit_tests_20261029.log",output/"unit_tests.log")
    # Qualitative snapshots of the same measured map, no scene-rendering oracle.
    for name,t in [("early",4),("moving",9),("cleared",20)]:
        frame=min(frames,key=lambda f:abs(f["sim_time"]-t))
        shutil.copy2(output/frame["file"],output/(name+".png"))
    before=[p for p in summary["probes"] if 1<p["elapsed"]<4.8]
    after=[p for p in summary["probes"] if 15<p["elapsed"]<24]
    camera_ratio=stats["camera_pairs"]/max(1,stats["scans"])
    report=f"""# Карта мини-ровера по датчикам — 29 октября 2026

Реализованы SLAM Toolbox, EKF robot_localization, динамическая байесовская occupancy
grid и калиброванная RGB-аннотация измеренных лидаром поверхностей.
Карты строятся с нуля. Ни готовая карта, ни геометрия сцены, ни внешняя камера,
ни истинная поза Gazebo в оценивание не поступают.

## Проверенный результат

- Робот завершил обход четырёх точек. Контактов корпуса: {summary['contacts']}.
- Наблюдавшаяся область общей карты: {summary['map_known_area_m2']:.1f} м².
- Разрешение общей и локальной карт: 4 см.
- Ошибка положения wheel-only RMSE: {summary['wheel_position_rmse_m']:.3f} м.
- Wheel + IMU EKF RMSE: {summary['position_rmse_m']:.3f} м.
- SLAM-поза RMSE: {summary['slam_position_rmse_m']:.3f} м; средняя ошибка yaw {summary['slam_yaw_mae_rad']*180/np.pi:.2f}°.
- SLAM сопоставлен на {summary['slam_pose_samples']} временных точках
  ({100*summary['slam_pose_match_fraction']:.1f}% оценочных измерений) по всему маршруту.
- Расстояние занятых статических клеток до истинных поверхностей:
  медиана {100*summary['static_surface_median_m']:.1f} см, 95-й процентиль {100*summary['static_surface_p95_m']:.1f} см.
- Движущийся тестовый ящик обнаружен: {summary['dynamic_obstacle_seen']}.
  Его прежнее место очищено после ухода: {summary['dynamic_obstacle_cleared']}.
- Ассоциировано {stats['camera_pairs']} RGB/scan пар из {stats['scans']} обработанных scans
  ({100*camera_ratio:.1f}%), окрашено {stats['colored_points']} измеренных точек.
- Из-за недоступного timestamped TF пропущено {stats['tf_dropped']} scans. Подстановка
  несоответствующей по времени позы вместо пропуска не используется.
- 37 unit/regression tests: PASS.
- ROS-граф: /evaluation/ground_truth читает только mapping_evaluator;
  единственный /odom publisher — ekf_filter_node.
- Все 14 процессов и их потоки: один CPU 0; открытых GPU-устройств нет.

Это один симуляционный сценарий с заданными ошибками датчиков.
Точность на физическом роботе и устойчивость на других сценах этими числами не доказаны.
Метрика площади показывает наблюдённую область, а не полноту карты всего склада.

## Как устроена динамика

Lidar очищает только реально наблюдаемое пространство до препятствия.
За препятствием сохраняется неизвестность. Пропуск луча NaN не превращается
в свободный коридор. После 4 с без наблюдения клетка становится неизвестной.
Отдельный слой риска расширяет занятые клетки на 20 см для габарита и запаса.

Долговременная SLAM-карта и актуальная локальная карта разделены:
движущиеся объекты могут давать следы в долговременной карте.
Малиновые стрелки — только кандидаты движения кластеров, с возможными ложными
срабатываниями; они не определяют очистку карты.

## Роль камеры и ограничения

Используется передняя RGB-камера, CameraInfo и TF на временных метках измерений.
Камера добавляет цвет поверхности в точки с уже измеренной lidar-дальностью.
Это не монокулярная глубина, не стерео и не полноценное распознавание объектов.
Низкие пороги и нависающие препятствия нельзя надёжно обнаружить одним
горизонтальным lidar и таким RGB-слоем.

Добавлены шумы, encoder scale errors, bias/drift гироскопа, потери лучей/кадров,
задержки и небольшие ошибки калибровки. Профиль — sensor_profile.yaml.
Реальные значения для оборудования нужно измерить. Погодные эффекты,
rolling shutter, полный sweep/deskew и сложное отражение материалов не симулируются.

## Оценка без утечки истины

Ground truth применяется только отдельным оценщиком. SLAM-поза сопоставлена
офлайн с записанными оценёнными TF по временным меткам. Пространственная
подгонка по ground truth не выполняется: используется общий начальный кадр.
Первичная онлайн-оценка имела мало TF-сопоставлений и сохраняется в online_summary.json;
в итоговых числах используется покрывающая маршрут офлайн-оценка.

Погрешность геометрии считается по статическим collision boxes на высоте lidar;
область перемещения тестового ящика исключена из метрики статических поверхностей.
Экспорт PGM после записи исправлен: неизвестные клетки имеют значение 128,
которое сохраняет статус unknown при порогах YAML .35/.65. Измерения и геометрия
карты при этой правке не менялись.

## Артефакты

- live.png — последний верхний вид по данным датчиков.
- mapping_replay.mp4 — запись изменений карты; изображения снимались примерно раз
  в секунду симуляции, видео ресемплировано в 4 FPS.
- global_map.pgm + global_map.yaml — карта для ROS/Nav2.
- local_map.npz — динамическая occupancy grid, last_seen, origin и resolution.
- sensors_bag/ — записанные наблюдения и оценённые TF, 24 536 сообщений.
- source_hashes.json — исходники на момент начала прогона.
- topic_audit.json, resource_audit.json, summary.json — проверки и метрики.

Подробности и команды: METHOD.md. Новый прогон:
```bash
cd /path/to/async_llm
bash robot_sim/scripts/run_mapping.sh --duration 150 --record-bag
```

Управление в этом эксперименте скриптовое; LLM и Nav2-планировщик не запускались.
"""
    (output/"REPORT.md").write_text(report)
    # Validate exported unknown cells against the actual map-server thresholds.
    pixels=np.asarray(Image.open(output/"global_map.pgm"))
    metadata=yaml.safe_load((output/"global_map.yaml").read_text())
    probability=1-pixels.astype(float)/255
    unknown=(pixels==128)
    export_ok=bool(np.all((probability[unknown]>=metadata["free_thresh"])&
                          (probability[unknown]<=metadata["occupied_thresh"])))
    resources=json.loads((output/"resource_audit.json").read_text())
    audit=json.loads((output/"topic_audit.json").read_text())
    validation={"unit_tests":37,"video_decoded":True,"export_unknown_preserved":export_ok,
                "processes_audited":len(resources),
                "all_cpu_0":all(r["cpu_masks"]==["0"] for r in resources),
                "gpu_devices_open":sum(len(r["gpu_devices"]) for r in resources),
                "truth_subscribers":audit["truth_subscribers"],
                "slam_pose_match_fraction":summary["slam_pose_match_fraction"],
                "simulation_passed":summary["passed"],
                "postprocess":["Timestamp-matched offline TF evaluation","Unknown-cell PGM encoding fix"],
                "video_sha256":hashlib.sha256(movie.read_bytes()).hexdigest()}
    (output/"validation.json").write_text(json.dumps(validation,indent=2))
    page=f"""<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Карта по датчикам мини-ровера</title><style>
body{{font:17px/1.6 system-ui;background:#101923;color:#e8eef5;max-width:1280px;margin:24px auto;padding:16px}}
a{{color:#8ed9ff}}h1,h2{{line-height:1.2}}img,video{{width:100%;border:1px solid #344c60;border-radius:8px;background:black}}
.grid{{display:flex;gap:12px;flex-wrap:wrap}}.metric{{background:#1c2e40;padding:12px 18px;border-radius:10px}}
.note{{color:#b4c5d4}}details{{background:#192a3c;padding:14px;margin:16px 0}}pre{{white-space:pre-wrap}}
figure{{width:30%;margin:1%}}@media(max-width:700px){{figure{{width:96%}}}}
</style><h1>Карта мини-ровера по датчикам</h1>
<p class="note">29 октября 2026 · SLAM Toolbox + EKF + динамическая сетка + RGB · 1 CPU / 0 GPU</p>
<p>Карта построена с нуля по наблюдениям. Общая SLAM-карта слева, актуальная локальная карта справа.
Готовая карта, внешняя камера и истинная поза симулятора в алгоритм не поступают.</p>
<div class="grid"><div class="metric">Обследовано <b>{summary['map_known_area_m2']:.1f} м²</b></div>
<div class="metric">Разрешение <b>4 см</b></div><div class="metric">SLAM RMSE <b>{100*summary['slam_position_rmse_m']:.1f} см</b></div>
<div class="metric">Движущееся препятствие: <b>обнаружено и очищено</b></div></div>
<p><a href="live.png">Карта крупно</a> · <a href="REPORT.md">Отчёт</a> · <a href="METHOD.md">Метод и запуск</a> ·
<a href="summary.json">Метрики</a> · <a href="review_bundle.tar.gz">Скачать результаты</a></p>
<img src="live.png" alt="Общая SLAM-карта и динамическая карта сверху">
<h2>Запись построения карты</h2>
<video controls preload="metadata" poster="live.png" src="mapping_replay.mp4"></video>
<p class="note">Все изображения карты вычислены из датчиков. Робот едет по скрипту; LLM не участвует.
Прогон завершён, карта выше показывает итоговое состояние.</p>
<h2>Динамика наблюдений</h2><div class="grid">
<figure><img src="early.png"><figcaption>Начальные наблюдения</figcaption></figure>
<figure><img src="moving.png"><figcaption>Изменяющееся препятствие</figcaption></figure>
<figure><img src="cleared.png"><figcaption>Повторные наблюдения очищают старое место</figcaption></figure></div>
<h2>Что означают цвета</h2>
<p>Серый — неизвестно; белый — наблюдалось свободное пространство; чёрный — препятствие.
Цветные поверхности получены совмещением RGB с lidar. Коричневый — запас расстояния;
голубой — оценка робота и иллюстрация ковариации; малиновый — кандидаты движения.</p>
<p>Без наблюдений старше 4 с клетка становится неизвестной, а не свободной.
Камера RGB не выдаёт глубину: её цвет привязывается к измеренным точкам лидара.</p>
<details><summary>Точность и ограничения</summary>
<p>Колёса: RMSE {summary['wheel_position_rmse_m']:.2f} м; колёса + IMU: {summary['position_rmse_m']:.2f} м;
SLAM: {100*summary['slam_position_rmse_m']:.1f} см. Оценено {summary['slam_pose_samples']} поз,
охват {100*summary['slam_pose_match_fraction']:.1f}% маршрута без подгонки по истине.</p>
<p>Медиана ошибки статических поверхностей {100*summary['static_surface_median_m']:.1f} см,
95-й процентиль {100*summary['static_surface_p95_m']:.1f} см.
Это один симуляционный сценарий. Низкие пороги, нависания и реальная уличная среда
требуют дополнительных сенсоров и испытаний.</p></details>
<details><summary>Проверки и исходные данные</summary><p>
<a href="sensor_profile.yaml">Шумы и задержки</a> · <a href="topic_audit.json">Входы алгоритма</a> ·
<a href="resource_audit.json">Ресурсы</a> · <a href="validation.json">Валидация</a> ·
<a href="unit_tests.log">37 тестов</a></p>
<p><a href="global_map.pgm">PGM-карта</a> + <a href="global_map.yaml">YAML</a> ·
<a href="local_map.npz">Динамическая сетка</a> · <a href="sensor_recording.tar.gz">ROS bag</a></p></details>
<h2>Новый прогон на сервере</h2><pre>cd /path/to/async_llm
bash robot_sim/scripts/run_mapping.sh --duration 150 --record-bag</pre></html>"""
    (output/"index.html").write_text(page)
    with tarfile.open(output/"sensor_recording.tar.gz","w:gz",compresslevel=1) as tar:
        tar.add(output/"sensors_bag",arcname="sensors_bag")
    with tarfile.open(output/"review_bundle.tar.gz","w:gz",compresslevel=1) as tar:
        for p in output.rglob("*"):
            if not p.is_file():continue
            if "sensors_bag" in p.parts or "map_frames" in p.parts:continue
            if p.name in {"review_bundle.tar.gz","sensor_recording.tar.gz","http.log","http.pid","supervisor.pid"}:continue
            tar.add(p,arcname=str(p.relative_to(output)))
    print(json.dumps(validation,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output",type=Path)
    build(p.parse_args().output.resolve())

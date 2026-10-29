# Мобильный робот: физика, Nav2 и визуальное управление

Вся работа выполняется на сервере 185.185.59.19 из:
`<repository-root>`.
Ветка: `robot/ros2-gazebo-llm-control`.

## Ограничения ресурсов

Gazebo, ROS/Nav2 и браузерный рабочий стол закреплены через CPU affinity за
одним логическим CPU, общим для всех дочерних процессов и потоков.
По умолчанию выбирается первый разрешённый CPU; задать другой можно через
`ROBOT_SIM_CPU=0 bash robot_sim/scripts/run_gui.sh warehouse`.
Списки CPU не принимаются. Это лимит одного ядра суммарно, а не по ядру на процесс.
Launcher дополнительно устанавливает наследуемый seccomp-фильтр, запрещающий
смену affinity: библиотеки не смогут расширить маску потоков обратно на все ядра.
Для полного ограничения запускать через run_gui.sh/run_navigation.sh/run_llm.sh.
Нужна системная libseccomp.so.2; при невозможности установить фильтр запуск отменяется.
При ручном `source robot_env.sh` affinity применяется также к текущей оболочке;
для изоляции использовать отдельный SSH-терминал или subshell.

Симулятор использует CPU-рендеринг Mesa llvmpipe, без NVIDIA EGL/GLX-контекстов
и без CUDA. `CUDA_VISIBLE_DEVICES` недостаточно для ограничения OpenGL:
поэтому явно выбраны Mesa EGL/GLX и llvmpipe. Рендеринг сенсоров использует
виртуальный X-сервер, а не аппаратный EGL device. Вычислительные пулы ограничены одним потоком.
Таким образом симулятор использует 0 GPU (не больше одной) и максимум 1 CPU.
На одном CPU рендеринг и физика могут работать медленнее реального времени.

LLM запускается отдельно через `run_llm.sh`, на том же CPU и по умолчанию одной CUDA GPU.
Исключение для явно разрешённого двухкарточного эксперимента описано ниже.
Выбор: `ROBOT_LLM_GPU=0 MODEL=Qwen/Qwen3.5-0.8B bash robot_sim/scripts/run_llm.sh --vision`.
Передача списка нескольких GPU отклоняется. Чужие задачи сервера не изменяются.

## Быстрый запуск

Окружение уже установлено внутри репозитория: `robot_sim/.runtime/ros`.
Новый основной entrypoint:

```bash
bash robot_sim/scripts/run_navigation.sh --scene warehouse --mode nav2 --goals 3
bash robot_sim/scripts/run_navigation.sh --scene apartment --mode nav2 --goals 3
bash robot_sim/scripts/run_navigation.sh --scene courtyard --mode nav2 --goals 3
```

Эквивалентный короткий запуск: `bash robot_sim/scripts/run_demo.sh`.
Старые названия режимов script/llm теперь означают nav2/vision.

Проверка сенсоров / камеры и физики:

```bash
bash robot_sim/scripts/run_navigation.sh --scene warehouse --mode inspect
bash robot_sim/scripts/run_navigation.sh --scene warehouse --mode physics
```

Для визуального управления запустить в первом SSH-терминале:

```bash
MODEL=Qwen/Qwen3.5-0.8B CUDA_VISIBLE_DEVICES=0 \
  bash robot_sim/scripts/run_llm.sh --memory-ratio 0.45 --vision
```

Дождаться `READY 127.0.0.1:18765` после прогрева текстового и vision-входов.
Во втором терминале:

```bash
bash robot_sim/scripts/run_navigation.sh --scene warehouse --mode vision --goals 3
```

Модель получает настоящий RGB-кадр 320×240, задачу выбора видимого
непосещённого маркера и известные карте координаты красной, синей и зелёной станций.
Ответ ограничен red/blue/green/stop; несуществующая или посещённая цель отклоняется.
Выбранная моделью цель передаётся в NavigateToPose. Nav2 локализуется и объезжает
препятствия по lidar; LLM не подменяет локальный контроллер скоростей.

Это визуально-семантический выбор целей по известной карте, а не восстановление
3D-координат неизвестных объектов по одной RGB-камере. На 0.8B выбор ближайшего
маркера ненадёжен: проверенный маршрут вышел 21.04 м против 11.24 м у заданного порядка.
Успех навигации не означает высокое качество визуального распознавания.

При свободной GPU для перехода на большую модель достаточно изменить MODEL:

```bash
MODEL=Qwen/Qwen3.5-9B CUDA_VISIBLE_DEVICES=0 \
  bash robot_sim/scripts/run_llm.sh --memory-ratio 0.85 --vision
```

Модель 9B пока не тестировалась. Значение memory-ratio не гарантирует размещение:
сначала проверить свободную VRAM. Внешние скачанные модели/окружения используются
только для чтения, новые кэши и логи создаются в репозитории.

## Просмотр Gazebo с Mac без XQuartz

Gazebo через XQuartz при проверке завершился с `Unable to create suitable GLXContext`.
Рабочий вариант — рендеринг на сервере и noVNC через SSH, без пересылки OpenGL.
Оба порта слушают только 127.0.0.1; внешнего доступа к рабочему столу нет.

На сервере из корня репозитория, в отдельном SSH-терминале:

```bash
bash robot_sim/scripts/run_gui.sh warehouse
```

На Mac, в другом терминале:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:17080:127.0.0.1:17080 185.185.59.19
```

Открыть `http://127.0.0.1:17080/vnc.html`, нажать Connect.
Пароль меняется при запуске; прочитать его на сервере:

```bash
cat robot_sim/.runtime/viewer/session/password
```

Симуляция первоначально на паузе; кнопка Play снизу слева.
Другие сцены: `apartment`, `courtyard`. Оба терминала оставить открытыми.
Ctrl+C завершает запуск и дочерние процессы. Повторный запуск блокируется flock.
Логи: `robot_sim/.runtime/viewer/session/`.
Это отдельный просмотр сцены (`GZ_PARTITION=robot_desktop`), не подключение к Nav2-опыту.

Зависимости уже распакованы без sudo внутри `.runtime/viewer/root`:
Ubuntu-пакеты x11vnc, novnc, python3-websockify, libvncserver1, libvncclient1.
При пересоздании скачать их через `apt-get download` в `.runtime/viewer/packages`
и распаковать каждый `dpkg-deb -x` в `.runtime/viewer/root`.
Также требуются доступные на сервере Xvfb, xauth и flock.

## Сцены и материалы

| Сцена | Препятствия | Поверхности |
|---|---|---|
| warehouse | ящики, стеллаж, паллет | бетон, дерево, металл, кирпич |
| apartment | диван-блок, перегородка, стол-блок, шкаф | плитка, ткань, штукатурка, дерево |
| courtyard | клумба, болларды, скамья, островок | асфальт, кирпич, дерево, металл |

Все сцены имеют ограждения и три цветных маркера доставки. Это процедурные
испытательные сцены, не фотореалистичные интерьеры и не модели мебели высокой детализации.
Материалы имеют локальные albedo/normal/roughness-карты 512×512 и используются
Gazebo как PBR. Внешние текстуры при запуске не скачиваются.

Генератор создаёт SDF, материалы, описания сцен и occupancy maps из одного layout:

```bash
source robot_sim/scripts/robot_env.sh
python robot_sim/scripts/configure_robot.py
python robot_sim/scripts/build_scenes.py
```

Генерируемые SDF содержат абсолютные пути к текстурам; после переноса репозитория
в другой каталог выполнить генерацию заново. Карты известны заранее, SLAM не используется.
Начальная поза (-4, -2, 0), цели (-1.7, -2), (1, 0), (4, 2.3).
Первая цель расположена за препятствием: прямой проезд невозможен.

## Физическая модель

Каноническая модель: `models/mobile_base/model.sdf`.
Это улучшенный Burger-style суррогат, не откалиброванный цифровой двойник TurtleBot3.

- Общая масса 1.62 кг, корпус 0.178×0.135×0.09 м.
- Радиус колеса 0.033 м, колея 0.19 м, ширина колеса 0.018 м.
- Положительные моменты инерции рассчитаны для коробки, цилиндров и опорного шара.
- Центр масс корпуса смещён на 0.025 м назад и 0.015 м вниз:
  проекция находится внутри треугольника опор. Это устранило клевок при торможении.
- Оси колёс выровнены с цилиндрами; радиус опоры и высота спавна согласованы с полом.
- Трение ведущих колёс 0.9; малая податливость контакта; скользящая шаровая опора
  с трением 0.01 приближает свободный ball caster.
- Привод ограничен 0.22 м/с, 1.2 рад/с, разгоном 0.35 м/с² и торможением 0.5 м/с².
  Nav2 использует более узкие лимиты 0.20 м/с и 1.0 рад/с.
- Lidar 360°, 10 Гц, 0.06–10 м, гауссов шум 3 мм; камера 5 Гц, FOV 100°.
- Body contact sensor отслеживает контакты корпуса. При контакте gateway
  защёлкивает остановку до перезапуска эксперимента.

Gazebo DiffDrive использует идеализированный привод с ограничениями скорости/ускорения.
Заданные joint effort limits не заменяют модель мотора с PID и насыщением тока.
Масса, сцепление и центр масс — явно выбранные допущения, не измеренные параметры
реального робота. Неровный грунт, мягкие шины и подвеска здесь не моделируются.

## Nav2 и ROS

Используются map_server + AMCL + NavFn A* + DWB + BT Navigator.
План перестраивается с частотой 1 Гц, local costmap обновляется 5 Гц.
Footprint задан консервативной окружностью радиусом 0.14 м, inflation radius 0.45 м.
Проверяется lifecycle ACTIVE перед отправкой цели.
Цепочка TF: map → odom → base_footprint → base_link → laser / camera_link → camera_optical.
AMCL использует lidar и одометрию; ground truth служит только для проверки результата.

Мосты: /odom, /scan, /tf, /clock, /camera/image_raw, /camera/camera_info,
/contacts, /ground_truth/pose и ROS → Gazebo /cmd_vel.
Nav2 публикует /cmd_vel_requested, отдельный gateway выдаёт /cmd_vel с 20 Гц:
ограничения скорости, потеря команд/сенсоров через 0.6 с, передняя зона остановки
0.28 м и защёлкивание body contact. Защитные таймеры используют wall time,
Nav2 и физические тесты — simulation time.

Launcher владеет отдельными группами процессов, следит за их состоянием,
при завершении отправляет нулевую скорость и убирает свои дочерние процессы.
По умолчанию ROS_DOMAIN_ID=73. Для параллельных опытов задавать разные domain IDs:

```bash
ROS_DOMAIN_ID=75 bash robot_sim/scripts/run_navigation.sh --mode physics
```

Не запускать два контроллера в одном domain. Аппаратный watchdog моторов
остаётся обязательным условием перед переносом на реальный робот.

## Результаты и проверки

Каждый запуск создаёт отдельный каталог в `robot_sim/results/`:
summary.json, trajectory.jsonl с настоящей позой из физики, safety.jsonl,
camera_first.png/camera_last.png, кадры/решения vision, latest_plan.json,
конфигурацию Nav2 и логи процессов.

Код возврата 0 означает прохождение проверки; 2 — неуспех клиента.
При навигации проверяется статус SUCCEEDED, физическая ошибка цели ≤0.25 м
и отсутствие контактов корпуса. Проверенные физические ошибки: 5.5–14.3 см.

```bash
source robot_sim/scripts/robot_env.sh
python -m unittest discover -s robot_sim/tests -v
ROS_DOMAIN_ID=74 python robot_sim/tests/ros_watchdog.py
```

8 unit-тестов проверяют control, инерцию и опоры, карты/цели, наличие и детализацию
PBR-текстур и согласованность лимитов. ROS integration проверяет потерю команды,
потерю сенсоров, препятствие, восстановление и contact latch.

Для пересоздания окружения: `bash robot_sim/scripts/setup_ros.sh`.
Точные версии Linux x86_64 сохранены в ros-linux-64.lock.
Исторические Transport-only и минимальный ROS smoke test оставлены в
waypoint_controller.py, run_ros_demo.sh и burger_course.sdf;
основные эксперименты используют новую модель и сцены через run_navigation.sh.

См. NAVIGATION_REPORT.md и results/navigation_preview.png.
Основа настройки: [Nav2 Jazzy](https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/)
и [официальная интеграция ROS–Gazebo](https://gazebosim.org/docs/latest/ros2_integration/).


## Эксперимент Qwen3.5-9B от 26 октября 2026

Полные результаты: `results/qwen9b_evaluation_20261026/ANALYSIS.md`.
Видео и метрики: `results/qwen9b_evaluation_20261026/index.html`.
Пять заданий модели + один Nav2-only контроль; сохранены также отказы.
Модель выбирает станцию по RGB и известной карте, а не управляет колёсами.
Семантический успех 1/4 заданий с эталоном (3/7 решений); все 8 принятых
Nav2-целей достигнуты без контактов корпуса. Не размеченный nearest-visible
прогон и baseline в семантическую accuracy не входят.

Для этого эксперимента пользователь отдельно разрешил две GPU. Обычный
`run_llm.sh` по-прежнему разрешает одну; две допускаются только явно:

```bash
ROBOT_LLM_GPU=2,6 ROBOT_LLM_MAX_GPUS=2 MODEL=Qwen/Qwen3.5-9B \
  bash robot_sim/scripts/run_llm.sh --backend transformers --gpu-memory-gib 12 --vision
```

Это BF16 Transformers/Accelerate backend с layer sharding, не tensor parallel
исходного AsyncLLM (его Qwen3.5 attention пока TP=1). Прямой CUDA P2P на
выбранной паре карт/текущем окружении повреждал данные; активации проходят
через RAM, веса остаются на GPU. См. gpu_copy_check.json. Никакие системные
настройки GPU или чужие задачи не изменяются. Зависимости уже установлены
в `async_llm`; новые модели не скачивались.

В новом отдельном каталоге можно повторить конкретное задание:

```bash
ROS_DOMAIN_ID=74 bash robot_sim/scripts/run_navigation.sh --scene warehouse \
  --mode vision --goals 2 --timeout 480 --record \
  --task-file robot_sim/evaluation_tasks/warehouse_red_blue.json \
  --output-dir robot_sim/results/my_new_run
source robot_sim/scripts/robot_env.sh
python robot_sim/scripts/render_replay.py robot_sim/results/my_new_run
```

Существующие результаты не перезаписывать. Raw camera сохраняется примерно
1.7 Гц simulation time, MP4 объединяет настоящую камеру и карту из ground-truth
телеметрии (не видео 3D-окна Gazebo); последние 3 секунды — итоговая карточка.
Метрики считают отдельно выбор Qwen, физическое достижение цели и общий успех.
Постфактум-переформулировки из failure_prompt_controls.json не считаются
успешными проездами и не заменяют исходные неудачные записи.

Для просмотра сохранённых видео без запуска модели:

```bash
python3 -m http.server 17081 --bind 127.0.0.1 \
  --directory robot_sim/results/qwen9b_evaluation_20261026
```

На Mac открыть SSH-туннель `ssh -N -L 127.0.0.1:17081:127.0.0.1:17081 185.185.59.19`,
затем `http://127.0.0.1:17081/`. HTTP не должен слушать публичные интерфейсы.


## Прямое вождение LLM без Nav2 (отчёт 2)

Отдельные результаты: `results/qwen9b_direct_20261026/REPORT_2.md` и index.html.
В этом режиме модель на каждом шаге получает RGB, lidar, PointGoal distance/bearing
из одометрии и последние действия. Выход — только forward/left/right/stop.
Nav2, планировщик пути и скриптовый выбор направления не запускаются.

```bash
ROS_DOMAIN_ID=76 bash robot_sim/scripts/run_navigation.sh --mode direct \
  --scene warehouse --task-file robot_sim/direct_tasks/left_turn_state_v2.json \
  --record --timeout 300 --output-dir robot_sim/results/my_new_direct_run
source robot_sim/scripts/robot_env.sh
python robot_sim/scripts/render_replay.py robot_sim/results/my_new_direct_run
```

Модель запускать отдельно как в предыдущем разделе (для BF16 9B — две явно
разрешённые GPU). numeric_v1 передаёт числа; state_v2 дополнительно передаёт
категориальное направление и факт достижения цели, уточняя приоритеты prompt.
Это инженерные сенсорные признаки, не скрытый action/маршрут. Не camera-only.
Начальная поза известна; ground truth не передаётся модели, только оценщику.

Импульс: 0.65 simulation seconds; forward=0.16 м/с; повороты=±0.65 рад/с.
При ожидании инференса робот стоит. Ответ старше 1.5 с не выполняется;
heartbeat и независимый gateway сохраняют ограничения сенсоров/контактов.
Остановка по приближению к цели или бюджету — защита, не успех LLM:
для успеха модель сама должна выдать stop на наблюдении с достигнутой целью.

Исходные неуспехи сохранены. Два случая гонки оценщика помечены в corrections.json
и перепроверены отдельно; они не маскируются под ошибки модели или успешные опыты.
Для просмотра отчёта 2 использовать HTTP на localhost:17082 с корнем каталога
qwen9b_direct_20261026 и SSH-туннель, аналогично первому отчёту на 17081.

## Переданный мини-ровер

Модель, допущения физики и команды запуска: [ROVER.md](ROVER.md). Быстрый тест: `bash robot_sim/scripts/run_rover.sh`.

## Карта по датчикам

SLAM, EKF и динамический слой: [MAPPING.md](MAPPING.md). Запуск: `bash robot_sim/scripts/run_mapping.sh --record-bag`.

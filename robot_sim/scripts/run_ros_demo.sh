#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/robot_env.sh"
if [[ "${ROBOT_CPU_LOCKED:-0}" != 1 ]]; then
    exec python -m robot_sim.resource_limits bash "$0" "$@"
fi
MODE="${1:-script}"
OUTPUT="$ROOT/robot_sim/results/$(date +%Y%m%d_%H%M%S)_${MODE}_$$"
mkdir -p "$OUTPUT"
echo "Results: $OUTPUT"
PIDS=()
cleanup() {
    trap - EXIT INT TERM
    gz topic -t /model/turtlebot3_burger/cmd_vel -m gz.msgs.Twist -p 'linear: {x: 0}, angular: {z: 0}' >/dev/null 2>&1 || true
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
xvfb-run -a -s "-screen 0 640x480x24 -nolisten tcp" gz sim -s -r robot_sim/worlds/burger_course.sdf >"$OUTPUT/gazebo.log" 2>&1 &
PIDS+=("$!")
"$CONDA_PREFIX/lib/ros_gz_bridge/parameter_bridge" --ros-args -p config_file:="$ROOT/robot_sim/config/bridge.yaml" >"$OUTPUT/bridge.log" 2>&1 &
PIDS+=("$!")
python robot_sim/scripts/ros_nodes.py gateway --output "$OUTPUT" >"$OUTPUT/gateway.log" 2>&1 &
PIDS+=("$!")
sleep 5
for pid in "${PIDS[@]}"; do kill -0 "$pid"; done
python robot_sim/scripts/ros_nodes.py driver --mode "$MODE" --duration "${DURATION:-50}" --port "${LLM_PORT:-18765}" --max-observation-age "${MAX_OBSERVATION_AGE:-2}" --output "$OUTPUT" >"$OUTPUT/driver.log" 2>&1 &
PIDS+=("$!")
DRIVER_PID="${PIDS[-1]}"
while kill -0 "$DRIVER_PID" 2>/dev/null; do
    for pid in "${PIDS[@]:0:3}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Simulation infrastructure exited; see $OUTPUT" >&2
            exit 1
        fi
    done
    sleep 0.2
done
STATUS=0
wait "$DRIVER_PID" || STATUS=$?
cat "$OUTPUT/summary.json"
exit "$STATUS"

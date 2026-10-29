#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source robot_sim/scripts/runtime_config.sh
if [[ "$ROBOT_ROS_ENV" == */* ]]; then
    exec "$ROBOT_CONDA_EXE" env create -y -p "$ROBOT_ROS_ENV" -f robot_sim/environment.yml
else
    exec "$ROBOT_CONDA_EXE" env create -y -n "$ROBOT_ROS_ENV" -f robot_sim/environment.yml
fi

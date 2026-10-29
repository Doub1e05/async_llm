#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/robot_env.sh"
mkdir -p robot_sim/.runtime/viewer
exec flock --nonblock robot_sim/.runtime/viewer/desktop.lock python -m robot_sim.resource_limits python robot_sim/scripts/gui_desktop.py "${@}"

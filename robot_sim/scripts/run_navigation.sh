#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/robot_env.sh"
exec python -m robot_sim.resource_limits python robot_sim/scripts/run_navigation.py "${@}"

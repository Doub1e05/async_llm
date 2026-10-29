#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/robot_env.sh"
exec python -m robot_sim.resource_limits python -m robot_sim.setup_assets

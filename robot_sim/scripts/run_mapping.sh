#!/usr/bin/env bash
set -euo pipefail
export ROS_DOMAIN_ID="${ROBOT_MAPPING_DOMAIN_ID:-83}"
source "$(dirname "${BASH_SOURCE[0]}")/robot_env.sh"
exec python -m robot_sim.resource_limits python -m robot_sim.mapping.run "${@}"

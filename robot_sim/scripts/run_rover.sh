#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_navigation.sh" --robot rover --scene lab --mode motion "${@}"

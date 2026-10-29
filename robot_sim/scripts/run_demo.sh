#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-nav2}"
case "$MODE" in
    script) MODE=nav2 ;;
    llm) MODE=vision ;;
    nav2|vision|physics|inspect) ;;
    *) echo "Usage: $0 [nav2|vision|physics|inspect]" >&2; exit 2 ;;
esac
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_navigation.sh" --mode "$MODE" "${@:2}"

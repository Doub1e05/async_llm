#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/robot_sim/scripts/runtime_config.sh"
source "$ROOT/robot_sim/scripts/resource_limits.sh"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
if [[ ! -d "$MODEL" ]]; then
    ROBOT_LOCAL_MODEL="$ROOT/robot_sim/.runtime/models/${MODEL##*/}"
    CACHE="${HF_HUB_CACHE:-$ROBOT_HF_HOME/hub}/models--${MODEL//\//--}"
    if [[ -d "$ROBOT_LOCAL_MODEL" ]]; then
        MODEL="$ROBOT_LOCAL_MODEL"
    elif [[ -r "$CACHE/refs/main" ]]; then
        MODEL="$CACHE/snapshots/$(cat "$CACHE/refs/main")"
    elif [[ -d "$CACHE/snapshots" ]]; then
        mapfile -t ROBOT_SNAPSHOTS < <(find "$CACHE/snapshots" -mindepth 1 -maxdepth 1 -type d | sort)
        if [[ ${#ROBOT_SNAPSHOTS[@]} == 1 ]]; then MODEL="${ROBOT_SNAPSHOTS[0]}"; fi
    fi
fi
export XDG_CACHE_HOME="$ROOT/robot_sim/.runtime/cache"
export PYTHONPYCACHEPREFIX="$ROOT/robot_sim/.runtime/pycache"
export PYTHONPATH="$ROOT/python:$ROOT"
export HF_HOME="$ROBOT_HF_HOME"
export CUDA_VISIBLE_DEVICES="${ROBOT_LLM_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_GPUS="${ROBOT_LLM_MAX_GPUS:-1}"
if [[ "$MAX_GPUS" != 1 && "$MAX_GPUS" != 2 ]]; then
    echo "ROBOT_LLM_MAX_GPUS must be 1 or 2" >&2; exit 2
fi
IFS=, read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#GPU_IDS[@]} -gt $MAX_GPUS || "$CUDA_VISIBLE_DEVICES" == *[[:space:]]* || "$CUDA_VISIBLE_DEVICES" == *, ]]; then
    echo "Select exactly one GPU by default; explicitly set ROBOT_LLM_MAX_GPUS=2 for two GPUs" >&2; exit 2
fi
if [[ ${#GPU_IDS[@]} == 2 && "${GPU_IDS[0]}" == "${GPU_IDS[1]}" ]]; then
    echo "GPU IDs must be distinct" >&2; exit 2
fi
if [[ ${#GPU_IDS[@]} == 2 && " $* " != *" --backend transformers "* ]]; then
    echo "Two GPU layer sharding requires --backend transformers" >&2; exit 2
fi
if [[ ! -x "$ROBOT_LLM_PYTHON" ]]; then
    echo "LLM Python not found: $ROBOT_LLM_PYTHON; set ROBOT_LLM_PYTHON or ROBOT_CONDA_ENV" >&2
    exit 2
fi
# JIT builders invoke helpers (notably ninja) by name, even without conda activate.
export PATH="$(dirname "$ROBOT_LLM_PYTHON"):$PATH"
if [[ -z "${CUDA_HOME:-}" && -x /usr/local/cuda-13.0/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda-13.0
fi
if [[ -n "${CUDA_HOME:-}" ]]; then export PATH="$CUDA_HOME/bin:$PATH"; fi
export HF_HUB_OFFLINE=1
mkdir -p "$XDG_CACHE_HOME"
exec /usr/bin/python3 -m robot_sim.resource_limits "$ROBOT_LLM_PYTHON" -u robot_sim/scripts/llm_server.py --model "$MODEL" "${@}"

#!/usr/bin/env bash
ROBOT_SIM_CPU="${ROBOT_SIM_CPU:-$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')}"
if [[ ! "$ROBOT_SIM_CPU" =~ ^[0-9]+$ ]]; then
    echo "ROBOT_SIM_CPU must select exactly one logical CPU" >&2
    return 2
fi
if [[ "${ROBOT_CPU_LOCKED:-0}" != 1 ]]; then
    taskset -pc "$ROBOT_SIM_CPU" "$BASHPID" >/dev/null || return 2
fi
export ROBOT_SIM_CPU
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 LP_NUM_THREADS=1
export KMP_AFFINITY=disabled OMP_PROC_BIND=false
unset GOMP_CPU_AFFINITY OMP_PLACES
export MESA_GLTHREAD=false

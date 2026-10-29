#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/robot_sim/scripts/runtime_config.sh" || return $?
source "$ROOT/robot_sim/scripts/resource_limits.sh" || return $?
RUNTIME="$ROOT/robot_sim/.runtime"
export HOME="$RUNTIME/home" XDG_CACHE_HOME="$RUNTIME/cache"
export ROS_LOG_DIR="$RUNTIME/ros_logs" ROS_HOME="$RUNTIME/ros_home"
export PYTHONPYCACHEPREFIX="$RUNTIME/pycache"
export TMPDIR="$RUNTIME/tmp"
export GZ_IP=127.0.0.1 GZ_PARTITION="navigation_${GZ_PARTITION_SUFFIX:-$$}"
export GZ_SIM_RESOURCE_PATH="$ROOT/robot_sim/models:$ROOT/robot_sim/assets"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-73}" ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export LIBGL_ALWAYS_SOFTWARE=1 QT_QPA_PLATFORM=offscreen
export CUDA_VISIBLE_DEVICES=""
export GALLIUM_DRIVER=llvmpipe MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export __GLX_VENDOR_LIBRARY_NAME=mesa
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
if [[ ! -r "$__EGL_VENDOR_LIBRARY_FILENAMES" ]]; then
    echo "Mesa EGL vendor configuration is required for CPU-only simulation" >&2
    return 2
fi
export TZ=Europe/Kirov
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$ROS_LOG_DIR" "$ROS_HOME" "$TMPDIR"
set +u
source "$ROBOT_CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ROBOT_ROS_ENV"
set -u

export PATH="$ROOT/robot_sim/.runtime/bin:$PATH"
if [[ -d /usr/local/cuda-13.0 ]]; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
    export PATH="$CUDA_HOME/bin:$PATH"
fi

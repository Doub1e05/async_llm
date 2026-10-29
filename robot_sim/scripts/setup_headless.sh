#!/usr/bin/env bash
# Install project-local Xvfb/noVNC tools from the configured Ubuntu repositories.
# No sudo and no changes to system packages.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="$ROOT/robot_sim/.runtime"
mkdir -p "$RUNTIME/deb_packages" "$RUNTIME/system" "$RUNTIME/bin" "$RUNTIME/viewer/root"
for TOOL in apt-get dpkg-deb; do command -v "$TOOL" >/dev/null; done
(
    cd "$RUNTIME/deb_packages"
    apt-get download xvfb xserver-common novnc websockify x11vnc libvncserver1 libvncclient1
)
for PACKAGE in "$RUNTIME"/deb_packages/*.deb; do
    dpkg-deb -x "$PACKAGE" "$RUNTIME/system"
    dpkg-deb -x "$PACKAGE" "$RUNTIME/viewer/root"
done
cat > "$RUNTIME/bin/Xvfb" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
ROBOT_SYSTEM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../system" && pwd)"
export LD_LIBRARY_PATH="$ROBOT_SYSTEM_ROOT/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
exec "$ROBOT_SYSTEM_ROOT/usr/bin/Xvfb" "${@}"
WRAPPER
chmod +x "$RUNTIME/bin/Xvfb"
ln -sfn ../system/usr/bin/xvfb-run "$RUNTIME/bin/xvfb-run"
if ldd "$RUNTIME/system/usr/bin/Xvfb" | grep 'not found'; then
    echo "Xvfb has missing shared libraries; inspect ldd output before starting Gazebo" >&2
    exit 2
fi
echo "Project-local headless tools ready in $RUNTIME/bin"

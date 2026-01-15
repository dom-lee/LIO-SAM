#!/usr/bin/env bash
set -eo pipefail
log(){ echo "[entrypoint] $*"; }

safe_source() {
  local f="$1"
  if [ -f "$f" ]; then
    set +u
    # shellcheck disable=SC1090
    source "$f"
    set -u 2>/dev/null || true
  fi
}

ros_distro="${ROS_DISTRO:-humble}"
ros_setup="/opt/ros/${ros_distro}/setup.bash"
log "starting (ROS_DISTRO=${ros_distro})"

if [ -f "$ros_setup" ]; then
  safe_source "$ros_setup"
else
  log "ERROR: ROS setup not found: $ros_setup"
  exec "$@"
fi

ws="/root/ros2_ws"
src="${ws}/src"

if [ ! -d "$src" ]; then
  log "ERROR: ${src} does not exist"
  exec "$@"
fi

if [ -z "$(ls -A "${src}" 2>/dev/null)" ]; then
  log "WARNING: ${src} is empty; mount your repo into ${src}/<pkg>"
  exec "$@"
fi

install_setup="${ws}/install/setup.bash"
need_build=0
if [ ! -f "$install_setup" ]; then
  need_build=1
elif find "${src}" -type f -newer "$install_setup" -print -quit | grep -q .; then
  need_build=1
fi

if [ "$need_build" -eq 1 ]; then
  log "building workspace"
  cd "$ws"
  export CMAKE_PREFIX_PATH="/usr/local:${CMAKE_PREFIX_PATH:-}"
  export GTSAM_DIR="$(find /usr/local -path '*cmake*GTSAM*' -name GTSAMConfig.cmake -print -quit | xargs -r dirname)"
  colcon build --symlink-install \
    --cmake-args -DCMAKE_PREFIX_PATH=/usr/local \
    ${GTSAM_DIR:+-DGTSAM_DIR=$GTSAM_DIR}
  log "build done"
fi

safe_source "$install_setup"

if [ "${USE_TMUX:-0}" = "1" ]; then
  SESSION="${TMUX_SESSION:-liosam}"
  if [ $# -eq 0 ] || { [ "$1" = "bash" ] && [ "${2:-}" = "-l" ]; }; then
    log "starting tmux session: ${SESSION}"
    exec tmux new -A -s "${SESSION}"
  fi
fi

exec "$@"

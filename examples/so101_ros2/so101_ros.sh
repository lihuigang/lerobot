#!/bin/bash
# SO-101 ROS2 (Jazzy) lifecycle helper for the Jetson (.9).
#
#   so101_ros.sh up            start the long-lived ROS2 container
#   so101_ros.sh down          remove the container
#   so101_ros.sh nodes         launch camera_node + robot_node inside the container
#   so101_ros.sh killnodes     stop the node processes
#   so101_ros.sh rosbridge     start rosbridge (WebSocket :9090) + rosapi for ros-mcp
#   so101_ros.sh killrosbridge stop rosbridge
#   so101_ros.sh mcpcheck      verify rosbridge/rosapi are up (for ros-mcp-server)
#   so101_ros.sh cli <args>    run `ros2 <args>` inside the container
#   so101_ros.sh logs [name]   tail node logs (camera | robot | rosbridge)
#   so101_ros.sh ps            show the container status
#   so101_ros.sh shell         interactive shell inside the container
#
# Environment overrides: REPO VENVIK CAL URDF_DIR IMG NAME ROS_DOMAIN_ID ROBOT_ID FOLLOWER_BYID BRIDGE_PORT
set -euo pipefail

REPO=${REPO:-/media/lihuigang/zhitai/github/lerobot}
VENVIK=${VENVIK:-/media/lihuigang/zhitai/venv-ik}
CAL=${CAL:-/home/lihuigang/.cache/huggingface/lerobot/calibration}
URDF_DIR=${URDF_DIR:-/media/lihuigang/zhitai/robot-urdfs}
IMG=${IMG:-lerobot-so101-cuda:jp6.2-ros2}
NAME=${NAME:-so101_ros}
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
ROBOT_ID=${ROBOT_ID:-my_awesome_follower_arm}
FOLLOWER_BYID=${FOLLOWER_BYID:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C127069-if00}
URDF=${URDF:-/urdfs/so101/so101_new_calib.urdf}
BRIDGE_PORT=${BRIDGE_PORT:-9090}
NODES_DIR=/work/lerobot/examples/so101_ros2

cmd=${1:-}
[ $# -gt 0 ] && shift

# Resolve the real device nodes to pass through with --device. The by-id symlinks
# are mounted read-only so the nodes can use the stable paths inside the container.
device_args() {
  echo -n " --device=$(readlink -f "$FOLLOWER_BYID")"
  for link in /dev/v4l/by-id/*-video-index0; do
    [ -e "$link" ] && echo -n " --device=$(readlink -f "$link")"
  done
}

case "$cmd" in
  up)
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    # shellcheck disable=SC2046
    docker run -d --name "$NAME" --restart unless-stopped \
      --runtime nvidia --network host --ipc=host \
      --user "$(id -u):$(id -g)" --group-add 20 --group-add 44 --group-add 104 \
      -e HOME=/tmp -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
      -e PYTHONPATH=/work/lerobot/src \
      -v "$VENVIK":/venv-ik \
      -v "$REPO":/work/lerobot:ro \
      -v "$CAL":/calibration \
      -v "$URDF_DIR":/urdfs:ro \
      -v /tmp:/hosttmp \
      -v /dev/serial/by-id:/dev/serial/by-id:ro \
      -v /dev/v4l:/dev/v4l:ro \
      $(device_args) \
      -w /work/lerobot \
      "$IMG" sleep infinity
    echo "container '$NAME' started (image $IMG, ROS_DOMAIN_ID=$ROS_DOMAIN_ID)"
    ;;
  down)
    docker rm -f "$NAME"
    ;;
  nodes)
    # PYTHONPATH already contains /work/lerobot/src via the container env; setup.bash
    # prepends the Jazzy site-packages, so do not override it here.
    # Stop any previous nodes first; the [_] bracket avoids pkill matching this shell.
    docker exec "$NAME" bash -lc "pkill -f 'so101_ros2/camera[_]node.py' || true; pkill -f 'so101_ros2/robot[_]node.py' || true"
    sleep 1
    docker exec -d "$NAME" bash -lc \
      "source /opt/ros/jazzy/setup.bash && /venv-ik/bin/python $NODES_DIR/camera_node.py > /hosttmp/so101_camera_node.log 2>&1"
    docker exec -d "$NAME" bash -lc \
      "source /opt/ros/jazzy/setup.bash && /venv-ik/bin/python $NODES_DIR/robot_node.py --port $FOLLOWER_BYID --calibration-dir /calibration/robots/so101_follower --id $ROBOT_ID --urdf $URDF > /hosttmp/so101_robot_node.log 2>&1"
    echo "nodes launched; logs: /tmp/so101_camera_node.log  /tmp/so101_robot_node.log"
    ;;
  killnodes)
    # The [_] bracket keeps the pattern from matching this very shell.
    docker exec "$NAME" bash -lc "pkill -f 'so101_ros2/camera[_]node.py' || true; pkill -f 'so101_ros2/robot[_]node.py' || true"
    echo "nodes stopped"
    ;;
  rosbridge)
    # rosbridge_server + rosapi are both required by ros-mcp-server; the launch file
    # starts both (ros2 run would skip rosapi and break every introspection tool).
    docker exec "$NAME" bash -lc "pkill -f 'rosbridge_websocke[t]' || true; pkill -f 'rosapi_no[d]e' || true" || true
    sleep 1
    docker exec -d "$NAME" bash -lc \
      "source /opt/ros/jazzy/setup.bash && ros2 launch rosbridge_server rosbridge_websocket_launch.xml port:=$BRIDGE_PORT > /hosttmp/so101_rosbridge.log 2>&1"
    sleep 3
    echo "rosbridge launched on :$BRIDGE_PORT; log: /tmp/so101_rosbridge.log"
    ;;
  killrosbridge)
    docker exec "$NAME" bash -lc "pkill -f 'rosbridge_websocke[t]' || true; pkill -f 'rosapi_no[d]e' || true"
    echo "rosbridge stopped"
    ;;
  mcpcheck)
    echo "== rosbridge / rosapi services =="
    docker exec "$NAME" bash -lc "source /opt/ros/jazzy/setup.bash && (ros2 service list | grep -E 'rosapi|rosbridge' || echo '(none — run: so101_ros.sh rosbridge)')"
    ;;
  cli)
    # Forward arguments verbatim (quotes and spaces preserved) to ros2.
    docker exec "$NAME" bash -lc 'source /opt/ros/jazzy/setup.bash && exec ros2 "$@"' bash "$@"
    ;;
  logs)
    case "${1:-camera}" in
      rosbridge) f=/hosttmp/so101_rosbridge.log ;;
      *)         f=/hosttmp/so101_${1:-camera}_node.log ;;
    esac
    docker exec "$NAME" bash -lc "tail -n 80 $f"
    ;;
  ps)
    docker ps -a --filter "name=^/${NAME}$" \
      --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
    ;;
  shell)
    docker exec -it "$NAME" bash -lc "source /opt/ros/jazzy/setup.bash && exec bash"
    ;;
  *)
    sed -n '2,/^set -euo/p' "$0" | sed '$d'
    ;;
esac

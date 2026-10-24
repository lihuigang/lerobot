#!/usr/bin/env bash
# Run the SO-101 workflow in short-lived containers on the Jetson host.
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
SO101_IMAGE=${SO101_IMAGE:-lerobot-so101-cuda:jp6.2-io}
SO101_DATASET_NAME=${SO101_DATASET_NAME:-so101_pick_v1}
SO101_OUTPUT_NAME=${SO101_OUTPUT_NAME:-so101_act_pick_v1}
SO101_TASK=${SO101_TASK:-抓起桌上的方块并放入盒子}
SO101_FPS=${SO101_FPS:-15}
SO101_EPISODES=${SO101_EPISODES:-50}
SO101_STEPS=${SO101_STEPS:-5000}
SO101_DURATION=${SO101_DURATION:-20}

FOLLOWER_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C127069-if00
LEADER_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128225-if00
FRONT_PORT=/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0
HANDEYE_PORT=/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0
CAL_HOST=/home/lihuigang/.cache/huggingface/lerobot/calibration
FOLLOWER_ID=my_awesome_follower_arm
LEADER_ID=my_awesome_leader_arm
DATA_HOST_ROOT=/media/lihuigang/zhitai/datasets
OUTPUT_HOST_ROOT=/media/lihuigang/zhitai/outputs
CACHE_HOST=/media/lihuigang/zhitai/cache/lerobot_cuda

DATASET_REPO=local/$SO101_DATASET_NAME
DATASET_HOST=$DATA_HOST_ROOT/$SO101_DATASET_NAME
DATASET_CONTAINER=/data/$SO101_DATASET_NAME
OUTPUT_HOST=$OUTPUT_HOST_ROOT/$SO101_OUTPUT_NAME
OUTPUT_CONTAINER=/outputs/$SO101_OUTPUT_NAME
MODEL_HOST=$OUTPUT_HOST/checkpoints/last/pretrained_model
MODEL_CONTAINER=$OUTPUT_CONTAINER/checkpoints/last/pretrained_model
CAMERAS="{front: {type: opencv, index_or_path: $FRONT_PORT, width: 640, height: 480, fps: $SO101_FPS, fourcc: YUYV}, handeye: {type: opencv, index_or_path: $HANDEYE_PORT, width: 640, height: 480, fps: $SO101_FPS, fourcc: MJPG}}"

usage() {
  cat <<'USAGE'
Usage: scripts/so101_docker.sh COMMAND [--resume]

  build                 Build the Jetson CUDA image
  check                 Test CUDA and hardware visibility (no movement)
  calibrate-leader      Back up and calibrate the leader arm
  calibrate-follower    Back up and calibrate the follower arm
  teleop                Teleoperate without recording
  camera                Preview both cameras at http://192.168.1.9:8765/
  record [--resume]     Record a local dataset
  train                 Train ACT on CUDA
  rollout               Run the trained policy for a short trial

Run on 192.168.1.9 in an interactive SSH terminal. After checking calibration
and clearing the workspace, set SO101_ARM_READY=1 for teleop/record/rollout.
Optional overrides: SO101_DATASET_NAME, SO101_OUTPUT_NAME, SO101_TASK,
SO101_FPS, SO101_EPISODES, SO101_STEPS, SO101_DURATION, SO101_IMAGE.
USAGE
}

die() { printf 'SO-101: %s\n' "$*" >&2; exit 1; }

require_no_args() { (($# == 0)) || die "Unexpected arguments: $*"; }

require_positive_int() {
  local name=$1 value=$2
  [[ $value =~ ^[0-9]+$ ]] && ((10#$value > 0)) || die "$name must be a positive integer"
}

validate_config() {
  [[ $SO101_DATASET_NAME =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || die 'Invalid SO101_DATASET_NAME'
  [[ $SO101_OUTPUT_NAME =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || die 'Invalid SO101_OUTPUT_NAME'
  require_positive_int SO101_FPS "$SO101_FPS"
  require_positive_int SO101_EPISODES "$SO101_EPISODES"
  require_positive_int SO101_STEPS "$SO101_STEPS"
  require_positive_int SO101_DURATION "$SO101_DURATION"
}

init_docker() {
  docker image inspect "$SO101_IMAGE" >/dev/null 2>&1 || die "Image missing: $SO101_IMAGE (run build)"
  mkdir -p "$DATA_HOST_ROOT" "$OUTPUT_HOST_ROOT" "$CACHE_HOST"
  DOCKER_ARGS=(
    --rm --runtime nvidia --ipc=host --network host
    --user "$(id -u):$(id -g)" --group-add 20 --group-add 44 --group-add 104
    -e HOME=/cache -e HF_HOME=/cache/hf -e TORCH_HOME=/cache/torch
    -e PYTHONPATH=/work/lerobot/src
    -v "$REPO_ROOT:/work/lerobot:ro"
    -v "$DATA_HOST_ROOT:/data"
    -v "$OUTPUT_HOST_ROOT:/outputs"
    -v "$CACHE_HOST:/cache"
    -w /work/lerobot
  )
}

resolve_device() {
  local label=$1 path=$2 node
  node=$(readlink -f -- "$path") || die "$label is missing: $path"
  [[ -c $node ]] || die "$label is not a character device: $node"
  printf '%s' "$node"
}

init_calibration() {
  [[ -d $CAL_HOST ]] || die "Calibration directory is missing: $CAL_HOST"
  CAL_ARGS=(-v "$CAL_HOST:/calibration")
}

init_follower() {
  init_calibration
  [[ -d /dev/serial/by-id ]] || die 'Serial by-id directory is missing'
  FOLLOWER_NODE=$(resolve_device follower "$FOLLOWER_PORT")
  FOLLOWER_ARGS=(--device="$FOLLOWER_NODE:$FOLLOWER_NODE")
  SERIAL_ARGS=(-v /dev/serial/by-id:/dev/serial/by-id:ro)
}

init_leader() {
  init_calibration
  [[ -d /dev/serial/by-id ]] || die 'Serial by-id directory is missing'
  LEADER_NODE=$(resolve_device leader "$LEADER_PORT")
  LEADER_ARGS=(--device="$LEADER_NODE:$LEADER_NODE")
  SERIAL_ARGS=(-v /dev/serial/by-id:/dev/serial/by-id:ro)
}

init_cameras() {
  [[ -d /dev/v4l/by-id ]] || die 'Camera by-id directory is missing'
  FRONT_NODE=$(resolve_device front_camera "$FRONT_PORT")
  HANDEYE_NODE=$(resolve_device handeye_camera "$HANDEYE_PORT")
  [[ $FRONT_NODE != "$HANDEYE_NODE" ]] || die 'Both cameras resolve to the same device'
  CAMERA_ARGS=(--device="$FRONT_NODE:$FRONT_NODE" --device="$HANDEYE_NODE:$HANDEYE_NODE")
  CAMERA_LINK_ARGS=(-v /dev/v4l/by-id:/dev/v4l/by-id:ro)
}

require_free() {
  command -v fuser >/dev/null || die 'fuser is required to check device occupancy'
  if fuser "$@" >/dev/null 2>&1; then
    die "Device is in use: $*"
  fi
}

prepare_cameras() {
  if command -v systemctl >/dev/null && systemctl --user is-active --quiet so101-camera-web.service; then
    printf 'Stopping legacy host camera preview service.\n' >&2
    systemctl --user stop so101-camera-web.service || die 'Could not stop legacy camera preview'
  fi
  require_free "$FRONT_NODE" "$HANDEYE_NODE"
}

require_calibration_file() {
  [[ -f $CAL_HOST/$1 ]] || die "Calibration file missing: $CAL_HOST/$1"
}

require_arm_ready() {
  [[ ${SO101_ARM_READY:-0} == 1 ]] || die 'Check the elbow calibration and clear the workspace, then rerun with SO101_ARM_READY=1'
}

backup_calibration() {
  local file=$1
  docker run "${DOCKER_ARGS[@]}" "${CAL_ARGS[@]}" -e SO101_CAL_FILE="$file" "$SO101_IMAGE" \
    sh -eu -c 'if [ -f "$SO101_CAL_FILE" ]; then cp -p "$SO101_CAL_FILE" "$SO101_CAL_FILE.$(date +%Y%m%d_%H%M%S).bak"; fi'
}

do_check() {
  init_follower
  init_leader
  init_cameras
  docker run "${DOCKER_ARGS[@]}" "${CAL_ARGS[@]}" "${SERIAL_ARGS[@]}" "${CAMERA_LINK_ARGS[@]}" \
    "${FOLLOWER_ARGS[@]}" "${LEADER_ARGS[@]}" "${CAMERA_ARGS[@]}" \
    -e SO101_FOLLOWER_PORT="$FOLLOWER_PORT" -e SO101_LEADER_PORT="$LEADER_PORT" \
    -e SO101_FRONT_PORT="$FRONT_PORT" -e SO101_HANDEYE_PORT="$HANDEYE_PORT" \
    "$SO101_IMAGE" sh -eu -c '
      test -c "$SO101_FOLLOWER_PORT" && test -c "$SO101_LEADER_PORT" &&
      test -c "$SO101_FRONT_PORT" && test -c "$SO101_HANDEYE_PORT" &&
      test -w /calibration && echo SO101_DOCKER_IO_OK
    '
  docker run "${DOCKER_ARGS[@]}" "$SO101_IMAGE" python3 examples/so101_cuda_smoke.py
}

do_calibrate_leader() {
  init_leader
  require_free "$LEADER_NODE"
  backup_calibration "/calibration/teleoperators/so101_leader/$LEADER_ID.json"
  docker run -it "${DOCKER_ARGS[@]}" "${CAL_ARGS[@]}" "${SERIAL_ARGS[@]}" "${LEADER_ARGS[@]}" \
    "$SO101_IMAGE" python3 -m lerobot.scripts.lerobot_calibrate \
    --teleop.type=so101_leader --teleop.port="$LEADER_PORT" \
    --teleop.id="$LEADER_ID" --teleop.calibration_dir=/calibration/teleoperators/so101_leader
}

do_calibrate_follower() {
  init_follower
  require_free "$FOLLOWER_NODE"
  backup_calibration "/calibration/robots/so101_follower/$FOLLOWER_ID.json"
  docker run -it "${DOCKER_ARGS[@]}" "${CAL_ARGS[@]}" "${SERIAL_ARGS[@]}" "${FOLLOWER_ARGS[@]}" \
    "$SO101_IMAGE" python3 -m lerobot.scripts.lerobot_calibrate \
    --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" \
    --robot.id="$FOLLOWER_ID" --robot.calibration_dir=/calibration/robots/so101_follower
}

do_teleop() {
  require_arm_ready
  init_follower
  init_leader
  require_calibration_file "robots/so101_follower/$FOLLOWER_ID.json"
  require_calibration_file "teleoperators/so101_leader/$LEADER_ID.json"
  require_free "$FOLLOWER_NODE" "$LEADER_NODE"
  docker run -it "${DOCKER_ARGS[@]}" "${CAL_ARGS[@]}" "${SERIAL_ARGS[@]}" \
    "${FOLLOWER_ARGS[@]}" "${LEADER_ARGS[@]}" "$SO101_IMAGE" \
    python3 -m lerobot.scripts.lerobot_teleoperate \
    --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" \
    --robot.id="$FOLLOWER_ID" --robot.calibration_dir=/calibration/robots/so101_follower \
    --robot.max_relative_target=10.0 \
    --teleop.type=so101_leader --teleop.port="$LEADER_PORT" \
    --teleop.id="$LEADER_ID" --teleop.calibration_dir=/calibration/teleoperators/so101_leader \
    --fps="$SO101_FPS" --display_data=false
}

do_camera() {
  init_cameras
  prepare_cameras
  docker run -it "${DOCKER_ARGS[@]}" "${CAMERA_LINK_ARGS[@]}" "${CAMERA_ARGS[@]}" \
    "$SO101_IMAGE" python3 examples/so101_camera_web.py --host 0.0.0.0 --port 8765 --fps "$SO101_FPS"
}

do_record() {
  local resume=$1
  require_arm_ready
  init_follower
  init_leader
  init_cameras
  require_calibration_file "robots/so101_follower/$FOLLOWER_ID.json"
  require_calibration_file "teleoperators/so101_leader/$LEADER_ID.json"
  if [[ $resume == true ]]; then
    [[ -f $DATASET_HOST/meta/info.json ]] || die "Dataset to resume is missing: $DATASET_HOST"
  else
    [[ ! -e $DATASET_HOST ]] || die "Dataset already exists: $DATASET_HOST (use record --resume)"
  fi
  require_free "$FOLLOWER_NODE" "$LEADER_NODE"
  prepare_cameras
  local resume_arg=()
  [[ $resume == true ]] && resume_arg=(--resume=true)
  docker run -it "${DOCKER_ARGS[@]}" "${CAL_ARGS[@]}" "${SERIAL_ARGS[@]}" "${CAMERA_LINK_ARGS[@]}" \
    "${FOLLOWER_ARGS[@]}" "${LEADER_ARGS[@]}" "${CAMERA_ARGS[@]}" "$SO101_IMAGE" \
    python3 -m lerobot.scripts.lerobot_record \
    --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" \
    --robot.id="$FOLLOWER_ID" --robot.calibration_dir=/calibration/robots/so101_follower \
    --robot.max_relative_target=10.0 --robot.cameras="$CAMERAS" \
    --teleop.type=so101_leader --teleop.port="$LEADER_PORT" \
    --teleop.id="$LEADER_ID" --teleop.calibration_dir=/calibration/teleoperators/so101_leader \
    --dataset.repo_id="$DATASET_REPO" --dataset.root="$DATASET_CONTAINER" \
    --dataset.no_stamp=true --dataset.push_to_hub=false \
    --dataset.single_task="$SO101_TASK" --dataset.num_episodes="$SO101_EPISODES" \
    --dataset.episode_time_s=30 --dataset.reset_time_s=10 \
    --dataset.fps="$SO101_FPS" --display_data=false "${resume_arg[@]}"
}

do_train() {
  [[ -f $DATASET_HOST/meta/info.json ]] || die "Dataset missing: $DATASET_HOST"
  [[ ! -e $OUTPUT_HOST ]] || die "Training output already exists: $OUTPUT_HOST"
  docker run "${DOCKER_ARGS[@]}" "$SO101_IMAGE" python3 -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="$DATASET_REPO" --dataset.root="$DATASET_CONTAINER" \
    --policy.type=act --policy.device=cuda \
    --output_dir="$OUTPUT_CONTAINER" --job_name="$SO101_OUTPUT_NAME" \
    --batch_size=1 --steps="$SO101_STEPS" --save_freq=1000 --num_workers=2 \
    --policy.push_to_hub=false --wandb.enable=false
}

do_rollout() {
  require_arm_ready
  [[ -d $MODEL_HOST ]] || die "Model missing: $MODEL_HOST"
  init_follower
  init_cameras
  require_calibration_file "robots/so101_follower/$FOLLOWER_ID.json"
  require_free "$FOLLOWER_NODE"
  prepare_cameras
  docker run -it "${DOCKER_ARGS[@]}" "${CAL_ARGS[@]}" "${SERIAL_ARGS[@]}" "${CAMERA_LINK_ARGS[@]}" \
    "${FOLLOWER_ARGS[@]}" "${CAMERA_ARGS[@]}" "$SO101_IMAGE" \
    python3 -m lerobot.scripts.lerobot_rollout \
    --strategy.type=base --policy.path="$MODEL_CONTAINER" --device=cuda \
    --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" \
    --robot.id="$FOLLOWER_ID" --robot.calibration_dir=/calibration/robots/so101_follower \
    --robot.max_relative_target=10.0 --robot.cameras="$CAMERAS" \
    --task="$SO101_TASK" --fps="$SO101_FPS" --duration="$SO101_DURATION" \
    --return_to_initial_position=false --play_sounds=false --display_data=false
}

main() {
  local command=${1:-help} resume=false
  (($# == 0)) || shift
  case $command in
    help|-h|--help) require_no_args "$@"; usage; return ;;
    build)
      require_no_args "$@"
      docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
        -f "$REPO_ROOT/docker/Dockerfile.so101_cuda" -t "$SO101_IMAGE" "$REPO_ROOT/docker"
      return
      ;;
  esac
  validate_config
  init_docker
  case $command in
    check) require_no_args "$@"; do_check ;;
    calibrate-leader) require_no_args "$@"; do_calibrate_leader ;;
    calibrate-follower) require_no_args "$@"; do_calibrate_follower ;;
    teleop) require_no_args "$@"; do_teleop ;;
    camera) require_no_args "$@"; do_camera ;;
    record)
      if (($# == 1)) && [[ $1 == --resume ]]; then resume=true
      else require_no_args "$@"
      fi
      do_record "$resume"
      ;;
    train) require_no_args "$@"; do_train ;;
    rollout) require_no_args "$@"; do_rollout ;;
    *) die "Unknown command: $command (run --help)" ;;
  esac
}

main "$@"

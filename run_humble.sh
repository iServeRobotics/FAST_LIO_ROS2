#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_humble.sh — Run FAST_LIO (Humble) with rosbag replay or live LiDAR.
#
# Foxglove Bridge starts by default on ws://localhost:8765.
# Open Foxglove Studio and connect to visualize.
#
# Usage:
#   ./run_humble.sh                          # replay mode, interactive bag selection
#   ./run_humble.sh --gpu                    # replay + CUDA
#   ./run_humble.sh --rviz                   # replay + rviz2 GUI
#   ./run_humble.sh --live                   # live LiDAR
#   ./run_humble.sh --live --gpu             # live + CUDA
#   ./run_humble.sh --live --config avia     # live with specific LiDAR config
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DATA_DIR="${HOME}/Downloads/data"
USE_RVIZ=false
USE_GPU=false
LIVE_MODE=false
CONFIG="mid360"

# ── Parse flags ──────────────────────────────────────────────────────────────
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rviz)    USE_RVIZ=true; shift ;;
        --gpu)     USE_GPU=true; shift ;;
        --live)    LIVE_MODE=true; shift ;;
        --config)  CONFIG="$2"; shift 2 ;;
        *)         POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

# Select image based on GPU flag
if [[ "$USE_GPU" == true ]]; then
    IMAGE="fast_lio:humble-cuda"
else
    IMAGE="fast_lio:humble"
fi

# Validate config name
LAUNCH_FILE="mapping_${CONFIG}.launch.py"

# ── Docker run args (common) ────────────────────────────────────────────────
DOCKER_ARGS=(
    -it --rm
    --net=host
)

# GPU passthrough
if [[ "$USE_GPU" == true ]]; then
    DOCKER_ARGS+=(--gpus all)
fi

# rviz / X11
RVIZ_FLAG="false"
if [[ "$USE_RVIZ" == true ]]; then
    RVIZ_FLAG="true"
    xhost +local:docker 2>/dev/null || true
    DOCKER_ARGS+=(
        -e DISPLAY="${DISPLAY}"
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    )
    if [[ -e /dev/dri ]]; then
        DOCKER_ARGS+=(--device /dev/dri:/dev/dri)
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# LIVE MODE
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$LIVE_MODE" == true ]]; then
    echo ""
    echo "Mode:         LIVE"
    echo "Config:       $CONFIG ($LAUNCH_FILE)"
    echo "GPU:          $( [[ "$USE_GPU" == true ]] && echo "enabled (CUDA)" || echo "disabled" )"
    echo "rviz2:        $( [[ "$USE_RVIZ" == true ]] && echo "enabled" || echo "disabled" )"
    echo "Foxglove:     ws://localhost:8765"
    echo "Image:        $IMAGE"
    echo ""
    echo "Launching FAST_LIO (live) …"
    echo "───────────────────────────────────────────────────────────────────────"

    docker run "${DOCKER_ARGS[@]}" \
        "$IMAGE" \
        bash -c "
            ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765 &
            ros2 launch fast_lio $LAUNCH_FILE rviz:=$RVIZ_FLAG use_sim_time:=false
        "

    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# REPLAY MODE
# ─────────────────────────────────────────────────────────────────────────────
if [[ $# -ge 1 ]]; then
    BAG_DIR="$1"
else
    # Collect valid bag directories (must contain metadata.yaml)
    mapfile -t bags < <(find "$DATA_DIR" -maxdepth 3 -name metadata.yaml -printf '%h\n' | sort)

    if [[ ${#bags[@]} -eq 0 ]]; then
        echo "No ROS2 bag directories found under $DATA_DIR"
        exit 1
    fi

    echo "Available rosbags:"
    echo "─────────────────────────────────────────"
    for i in "${!bags[@]}"; do
        # Show path relative to DATA_DIR for clarity
        relpath="${bags[$i]#"$DATA_DIR/"}"
        printf "  [%d] %s\n" "$((i + 1))" "$relpath"
    done
    echo "─────────────────────────────────────────"

    read -rp "Select a bag (1-${#bags[@]}): " choice
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#bags[@]} )); then
        echo "Invalid selection."
        exit 1
    fi
    BAG_DIR="${bags[$((choice - 1))]}"
fi

if [[ ! -f "$BAG_DIR/metadata.yaml" ]]; then
    echo "Error: $BAG_DIR does not look like a valid ROS2 bag (no metadata.yaml)."
    exit 1
fi

BAG_NAME=$(basename "$BAG_DIR")
echo ""
echo "Mode:         REPLAY"
echo "Bag:          $BAG_NAME"
echo "Path:         $BAG_DIR"

# ── Detect topics from metadata.yaml ────────────────────────────────────────
LIDAR_TOPIC=$(grep -A2 'type: livox_ros_driver2/msg/CustomMsg' "$BAG_DIR/metadata.yaml" \
    | head -1 | sed -n 's/.*name: //p' | tr -d '[:space:]')
IMU_TOPIC=$(grep -B1 'type: sensor_msgs/msg/Imu' "$BAG_DIR/metadata.yaml" \
    | head -1 | sed -n 's/.*name: //p' | tr -d '[:space:]')

echo "Lidar topic:  ${LIDAR_TOPIC:-<not found>}"
echo "IMU topic:    ${IMU_TOPIC:-<not found>}"

# The config expects /livox/lidar and /livox/imu.
# Build remapping args if the bag uses different topic names.
REMAP_ARGS=()
if [[ -n "$LIDAR_TOPIC" && "$LIDAR_TOPIC" != "/livox/lidar" ]]; then
    REMAP_ARGS+=("--ros-args" "-r" "/livox/lidar:=$LIDAR_TOPIC")
fi
if [[ -n "$IMU_TOPIC" && "$IMU_TOPIC" != "/livox/imu" ]]; then
    if [[ ${#REMAP_ARGS[@]} -eq 0 ]]; then
        REMAP_ARGS+=("--ros-args")
    fi
    REMAP_ARGS+=("-r" "/livox/imu:=$IMU_TOPIC")
fi

# Mount bag directory
CONTAINER_BAG="/data/$BAG_NAME"
DOCKER_ARGS+=(-v "$BAG_DIR":"$CONTAINER_BAG":ro)

echo "GPU:          $( [[ "$USE_GPU" == true ]] && echo "enabled (CUDA)" || echo "disabled" )"
echo "rviz2:        $( [[ "$USE_RVIZ" == true ]] && echo "enabled" || echo "disabled (use --rviz to enable)" )"
echo "Foxglove:     ws://localhost:8765"
echo "Image:        $IMAGE"
echo ""
echo "Launching FAST_LIO + rosbag playback …"
echo "───────────────────────────────────────────────────────────────────────"

docker run "${DOCKER_ARGS[@]}" \
    "$IMAGE" \
    bash -c "
        ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765 &
        ros2 launch fast_lio $LAUNCH_FILE rviz:=$RVIZ_FLAG use_sim_time:=true ${REMAP_ARGS[*]:+${REMAP_ARGS[*]}} &
        FAST_LIO_PID=\$!

        # Give FAST_LIO a moment to start up
        sleep 3

        echo '>>> Playing bag: $BAG_NAME'
        ros2 bag play '$CONTAINER_BAG' --clock

        echo '>>> Bag playback finished. Waiting for FAST_LIO to finish processing …'
        sleep 2
        kill \$FAST_LIO_PID 2>/dev/null || true
        wait \$FAST_LIO_PID 2>/dev/null || true
    "

#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_humble.sh — Run FAST_LIO (Humble) on an MCAP rosbag from the
#                 Aggressive dataset.
#
# Usage:
#   ./run_humble.sh                          # interactive, with rviz, CPU
#   ./run_humble.sh --gpu                    # interactive, with rviz, CUDA
#   ./run_humble.sh --no-rviz <bag_dir>      # direct path, headless
#   ./run_humble.sh --gpu --no-rviz          # headless + CUDA
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DATA_DIR="${HOME}/Downloads/data/Aggressive"
USE_RVIZ=true
USE_GPU=false

# ── Parse flags ──────────────────────────────────────────────────────────────
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --no-rviz) USE_RVIZ=false ;;
        --gpu)     USE_GPU=true ;;
        *)         POSITIONAL+=("$arg") ;;
    esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

# Select image based on GPU flag
if [[ "$USE_GPU" == true ]]; then
    IMAGE="fast_lio:humble-cuda"
else
    IMAGE="fast_lio:humble"
fi

# ── Bag selection ────────────────────────────────────────────────────────────
if [[ $# -ge 1 ]]; then
    BAG_DIR="$1"
else
    # Collect valid bag directories (must contain metadata.yaml)
    mapfile -t bags < <(find "$DATA_DIR" -maxdepth 2 -name metadata.yaml -printf '%h\n' | sort)

    if [[ ${#bags[@]} -eq 0 ]]; then
        echo "No ROS2 bag directories found under $DATA_DIR"
        exit 1
    fi

    echo "Available rosbags:"
    echo "─────────────────────────────────────────"
    for i in "${!bags[@]}"; do
        name=$(basename "${bags[$i]}")
        printf "  [%d] %s\n" "$((i + 1))" "$name"
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
echo "Selected bag: $BAG_NAME"
echo "Path:         $BAG_DIR"

# ── Detect topics from metadata.yaml ────────────────────────────────────────
LIDAR_TOPIC=$(grep -A2 'type: livox_ros_driver2/msg/CustomMsg' "$BAG_DIR/metadata.yaml" \
    | head -1 | sed -n 's/.*name: //p' | tr -d '[:space:]')
IMU_TOPIC=$(grep -B1 'type: sensor_msgs/msg/Imu' "$BAG_DIR/metadata.yaml" \
    | head -1 | sed -n 's/.*name: //p' | tr -d '[:space:]')

echo "Lidar topic:  ${LIDAR_TOPIC:-<not found>}"
echo "IMU topic:    ${IMU_TOPIC:-<not found>}"

# The mid360 config expects /livox/lidar and /livox/imu.
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

# ── Docker run args ──────────────────────────────────────────────────────────
CONTAINER_BAG="/data/$BAG_NAME"
DOCKER_ARGS=(
    -it --rm
    --net=host
    -v "$BAG_DIR":"$CONTAINER_BAG":ro
)

# GPU passthrough
if [[ "$USE_GPU" == true ]]; then
    DOCKER_ARGS+=(--gpus all)
    echo "GPU:          enabled (CUDA)"
else
    echo "GPU:          disabled (use --gpu to enable)"
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
    echo "rviz2:        enabled"
else
    echo "rviz2:        disabled (use without --no-rviz to enable)"
fi

echo "Image:        $IMAGE"
echo ""
echo "Launching FAST_LIO + rosbag playback …"
echo "───────────────────────────────────────────────────────────────────────"

docker run "${DOCKER_ARGS[@]}" \
    "$IMAGE" \
    bash -c "
        ros2 launch fast_lio mapping_mid360.launch.py rviz:=$RVIZ_FLAG ${REMAP_ARGS[*]:+${REMAP_ARGS[*]}} &
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

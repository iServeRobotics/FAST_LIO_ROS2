# FAST_LIO ROS2 (with CUDA/Metal GPU Backends)

ROS2 port of the [FAST_LIO](https://github.com/hku-mars/FAST_LIO) `cuda+metal` branch. Supports CPU, CUDA, and Metal (Apple Silicon) backends for accelerated LiDAR-inertial odometry.

## Repository Structure

```
├── run_humble.sh              Run script (bag selection, GPU, visualization)
├── ros2-humble/
│   ├── Dockerfile             CPU-only image (~3 GB)
│   └── Dockerfile.cuda        CUDA GPU image (~7 GB)
├── ros2-jazzy/
│   └── Dockerfile             Jazzy CPU-only image
├── ros2-src/
│   └── FAST_LIO/              ROS2 ported source code
└── ros1/
    └── Dockerfile             ROS1 reference image
```

## Quick Start

### Build

```bash
# CPU-only
docker build -f ros2-humble/Dockerfile -t fast_lio:humble .

# CUDA (requires nvidia-container-toolkit on host)
docker build -f ros2-humble/Dockerfile.cuda -t fast_lio:humble-cuda .
```

### Run with rosbag replay

```bash
# Interactive bag selection (Foxglove visualization)
./run_humble.sh

# With CUDA GPU acceleration
./run_humble.sh --gpu

# With rviz2 GUI (requires X11)
./run_humble.sh --rviz

# Direct path to a bag
./run_humble.sh ~/path/to/bag_directory
./run_humble.sh --gpu ~/path/to/bag_directory
```

### Run with live LiDAR

```bash
# Live with Mid-360 (default)
./run_humble.sh --live

# Live with specific LiDAR config
./run_humble.sh --live --config avia
./run_humble.sh --live --config velodyne

# Live with CUDA
./run_humble.sh --live --gpu
```

### Visualization

**Foxglove Studio** (default, no extra setup):
- Foxglove Bridge starts automatically on `ws://localhost:8765`
- Open [Foxglove Studio](https://foxglove.dev/) and connect to `ws://localhost:8765`

**rviz2** (optional, requires X11):
- Add `--rviz` to enable the rviz2 GUI window

### Flags

| Flag | Description |
|---|---|
| `--gpu` | Use CUDA image (`fast_lio:humble-cuda`) with `--gpus all` |
| `--rviz` | Enable rviz2 GUI (requires X11 display) |
| `--live` | Live LiDAR mode instead of bag replay |
| `--config <name>` | LiDAR config: `mid360` (default), `avia`, `horizon`, `velodyne`, `ouster64`, `marsim` |

The run script automatically:
- Starts Foxglove Bridge for web-based visualization
- Lists available ROS2 bag directories (MCAP format)
- Detects LiDAR/IMU topic names from bag metadata
- Remaps topics if they don't match the default config
- Sets `use_sim_time:=true` for replay, `false` for live

### Run manually

```bash
# CPU
docker run -it --rm --net=host fast_lio:humble \
  ros2 launch fast_lio mapping_mid360.launch.py

# CUDA
docker run -it --rm --net=host --gpus all fast_lio:humble-cuda \
  ros2 launch fast_lio mapping_mid360.launch.py
```

## Available Launch Files

| Launch file | LiDAR |
|---|---|
| `mapping_avia.launch.py` | Livox Avia |
| `mapping_horizon.launch.py` | Livox Horizon |
| `mapping_mid360.launch.py` | Livox Mid-360 |
| `mapping_velodyne.launch.py` | Velodyne VLP-16/32 |
| `mapping_ouster64.launch.py` | Ouster OS2-64 |
| `mapping_marsim.launch.py` | MARSIM simulator |

Each launch file accepts `rviz:=true/false` (default: `true`) and `use_sim_time:=true/false` (default: `false`).

## Performance (CPU vs CUDA)

Tested on a 32-core machine with NVIDIA GPU:

| Metric | CPU | CUDA |
|---|---|---|
| CPU usage | ~570% | ~330% |
| RAM | 1.4 GB | 0.66 GB |
| VRAM | — | 258 MB |

## Prerequisites

- Docker
- For CUDA: NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- For visualization: [Foxglove Studio](https://foxglove.dev/) (recommended) or X11 for rviz2

## Acknowledgements

- [FAST_LIO](https://github.com/hku-mars/FAST_LIO) by HKU-MARS Lab
- [FAST_LIO cuda+metal](https://github.com/mdaiter/FAST_LIO) GPU backend by mdaiter
- [ikd-Tree](https://github.com/hku-mars/ikd-Tree) by HKU-MARS Lab
- [livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2) by Livox

## License

[GPL-2.0](LICENSE) — same as the original FAST_LIO.

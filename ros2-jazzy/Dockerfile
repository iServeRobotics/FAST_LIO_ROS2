###############################################################################
# FAST_LIO ROS2 (Jazzy Jalisco – Ubuntu 24.04)
#
# Uses the local ROS2 port (with CUDA/Metal GPU backends) from
# docker/ros2-src/FAST_LIO.
#
# Build (from the docker/ directory):
#   docker build -f ros2-jazzy/Dockerfile -t fast_lio:jazzy .
#
# Run:
#   docker run -it --rm --net=host fast_lio:jazzy \
#     ros2 launch fast_lio mapping_avia.launch.py
###############################################################################

FROM ros:jazzy-ros-base-noble

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libeigen3-dev \
        libpcl-dev \
        python3-dev \
        python3-pip \
        python3-colcon-common-extensions \
        ros-jazzy-pcl-ros \
        ros-jazzy-pcl-conversions \
        ros-jazzy-tf2-ros \
        ros-jazzy-tf2-eigen \
        ros-jazzy-tf2-geometry-msgs \
        ros-jazzy-rviz2 \
        ros-jazzy-visualization-msgs \
        ros-jazzy-rosbag2-storage-mcap \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Livox-SDK2 (required by livox_ros_driver2)
# ---------------------------------------------------------------------------
RUN cd /tmp \
    && git clone https://github.com/Livox-SDK/Livox-SDK2.git \
    && cd Livox-SDK2 && mkdir build && cd build \
    && cmake .. && make -j$(nproc) && make install \
    && rm -rf /tmp/Livox-SDK2

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
ENV COLCON_WS=/root/colcon_ws
RUN mkdir -p ${COLCON_WS}/src

# ---------------------------------------------------------------------------
# livox_ros_driver2 (ROS2)
# The build.sh script swaps in the ROS2 package.xml and launch files.
# We replicate that logic here for a clean colcon build.
# ---------------------------------------------------------------------------
RUN cd ${COLCON_WS}/src \
    && git clone https://github.com/Livox-SDK/livox_ros_driver2.git \
    && cd livox_ros_driver2 \
    && git checkout master \
    && cp package_ROS2.xml package.xml \
    && cp -r launch_ROS2 launch

# ---------------------------------------------------------------------------
# FAST_LIO (local ROS2 port with CUDA/Metal GPU backends)
# ---------------------------------------------------------------------------
COPY ros2-src/FAST_LIO ${COLCON_WS}/src/FAST_LIO

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
RUN source /opt/ros/jazzy/setup.bash \
    && cd ${COLCON_WS} \
    && colcon build \
         --symlink-install \
         --cmake-args \
           -DCMAKE_BUILD_TYPE=Release \
           -DCMAKE_CXX_STANDARD=17 \
           -DROS_EDITION=ROS2 \
           -DHUMBLE_ROS=humble \
         --parallel-workers $(nproc) \
    && echo "source ${COLCON_WS}/install/setup.bash" >> /root/.bashrc

WORKDIR ${COLCON_WS}

ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && source ${COLCON_WS}/install/setup.bash && exec \"$@\"", "--"]
CMD ["bash"]

###############################################################################
# FAST_LIO ROS1 (Noetic) Build
# Source: https://github.com/mdaiter/FAST_LIO (cuda+metal branch)
#
# Build:
#   docker build -t fast_lio:noetic .
#
# Run:
#   docker run -it --rm --net=host fast_lio:noetic \
#     roslaunch fast_lio mapping_avia.launch
#
# With GPU (CUDA):
#   docker build --build-arg USE_CUDA=ON -t fast_lio:noetic-cuda .
#   docker run -it --rm --net=host --gpus all fast_lio:noetic-cuda \
#     roslaunch fast_lio mapping_avia.launch
###############################################################################

FROM ros:noetic-ros-base-focal

ARG USE_CUDA=OFF
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
        ros-noetic-pcl-ros \
        ros-noetic-pcl-conversions \
        ros-noetic-tf \
        ros-noetic-eigen-conversions \
        ros-noetic-rviz \
        ros-noetic-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# livox_ros_driver (ROS1)
# ---------------------------------------------------------------------------
ENV CATKIN_WS=/root/catkin_ws
RUN mkdir -p ${CATKIN_WS}/src

RUN cd ${CATKIN_WS}/src \
    && git clone https://github.com/Livox-SDK/livox_ros_driver.git \
    && cd livox_ros_driver \
    && git checkout master

# ---------------------------------------------------------------------------
# Livox-SDK (required by livox_ros_driver)
# ---------------------------------------------------------------------------
RUN cd /tmp \
    && git clone https://github.com/Livox-SDK/Livox-SDK.git \
    && cd Livox-SDK/build \
    && cmake .. && make -j$(nproc) && make install \
    && rm -rf /tmp/Livox-SDK

# ---------------------------------------------------------------------------
# FAST_LIO (cuda+metal branch)
# ---------------------------------------------------------------------------
RUN cd ${CATKIN_WS}/src \
    && git clone --branch cuda+metal --recursive \
       https://github.com/mdaiter/FAST_LIO.git

# Ensure the ikd-Tree submodule is populated
RUN cd ${CATKIN_WS}/src/FAST_LIO \
    && git submodule update --init --recursive

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
RUN source /opt/ros/noetic/setup.bash \
    && cd ${CATKIN_WS} \
    && catkin_make -DCMAKE_BUILD_TYPE=Release \
    && echo "source ${CATKIN_WS}/devel/setup.bash" >> /root/.bashrc

WORKDIR ${CATKIN_WS}

ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/noetic/setup.bash && source ${CATKIN_WS}/devel/setup.bash && exec \"$@\"", "--"]
CMD ["bash"]

ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git tmux \
    libeigen3-dev \
    python3-colcon-common-extensions python3-rosdep \
    ros-${ROS_DISTRO}-navigation2 \
    ros-${ROS_DISTRO}-robot-localization \
    ros-${ROS_DISTRO}-robot-state-publisher \
    ros-${ROS_DISTRO}-perception-pcl \
    ros-${ROS_DISTRO}-pcl-msgs \
    ros-${ROS_DISTRO}-vision-opencv \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-rviz2 \
 && rm -rf /var/lib/apt/lists/*

# install GTSAM
ARG GTSAM_TAG=4.2.0
RUN git clone --depth 1 --branch "${GTSAM_TAG}" https://github.com/borglab/gtsam.git /tmp/gtsam \
 && cmake -S /tmp/gtsam -B /tmp/gtsam/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DGTSAM_USE_SYSTEM_EIGEN=ON \
      -DGTSAM_BUILD_TESTS=OFF \
      -DGTSAM_BUILD_EXAMPLES=OFF \
      -DGTSAM_BUILD_WRAP=OFF \
      -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
 && cmake --build /tmp/gtsam/build -j"$(nproc)" \
 && cmake --install /tmp/gtsam/build \
 && ldconfig \
 && (find /usr/local -name 'GTSAMConfig.cmake' -print -quit | grep -q .) \
 && rm -rf /tmp/gtsam

# create and build workspace
WORKDIR /root/ros2_ws

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Convenience sourcing for interactive shells
RUN tee /etc/profile.d/ros2_ws.sh > /dev/null <<'EOF'
if [ -f /opt/ros/${ROS_DISTRO}/setup.bash ]; then
  source /opt/ros/${ROS_DISTRO}/setup.bash
fi
if [ -f /root/ros2_ws/install/setup.bash ]; then
  source /root/ros2_ws/install/setup.bash
fi
EOF

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash", "-l"]
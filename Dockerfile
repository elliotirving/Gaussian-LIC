FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    ROS_DISTRO=noetic \
    CMAKE_PREFIX_PATH=/usr/local

SHELL ["/bin/bash", "-c"]

# Build tools and Coco-LIC prerequisites (README steps 2–3)
# NOTE: cmake >3.23 is needed as per: https://github.com/APRIL-ZJU/Gaussian-LIC/issues/12
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl gnupg2 lsb-release \
        build-essential git wget unzip pkg-config nano \
        python3-dev python3-pip python3-numpy \
        libeigen3-dev libyaml-cpp-dev libceres-dev libboost-all-dev libflann-dev libglm-dev \
        libavcodec-dev libavformat-dev libswscale-dev \
        libjpeg-dev libpng-dev libtiff-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install "cmake==3.28.4"

# PCL 1.13.0 from source — Coco-LIC requires >= 1.13; Ubuntu 20.04 apt only ships 1.10
RUN wget -q https://github.com/PointCloudLibrary/pcl/archive/refs/tags/pcl-1.13.0.tar.gz \
        -O /tmp/pcl.tar.gz \
    && tar -xzf /tmp/pcl.tar.gz -C /tmp \
    && cmake -S /tmp/pcl-pcl-1.13.0 -B /tmp/pcl-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_visualization=OFF \
        -DBUILD_examples=OFF \
        -DBUILD_tests=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_OPENGL=OFF \
    && cmake --build /tmp/pcl-build -j"$(nproc)" \
    && cmake --install /tmp/pcl-build \
    && rm -rf /tmp/pcl.tar.gz /tmp/pcl-pcl-1.13.0 /tmp/pcl-build

# ROS Noetic (installed after PCL 1.13 so cmake prefers /usr/local over apt's PCL 1.10)
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
         > /etc/apt/sources.list.d/ros1.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-noetic-desktop-full \
        python3-catkin-tools python3-rosdep \
        ros-noetic-cv-bridge ros-noetic-image-transport \
        ros-noetic-pcl-ros ros-noetic-pcl-conversions \
        ros-noetic-tf ros-noetic-eigen-conversions \
        ros-noetic-nav-msgs ros-noetic-sensor-msgs \
        ros-noetic-geometry-msgs ros-noetic-rosbag \
    && rosdep init || true \
    && rm -rf /var/lib/apt/lists/*

# OpenCV 4.7.0 with contrib + CUDA (README step 4)
RUN mkdir -p /tmp/cv && cd /tmp/cv \
    && wget -q https://github.com/opencv/opencv/archive/refs/tags/4.7.0.tar.gz -O opencv.tar.gz \
    && wget -q https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.tar.gz -O contrib.tar.gz \
    && tar -xzf opencv.tar.gz && tar -xzf contrib.tar.gz \
    && cmake -S opencv-4.7.0 -B build \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_INSTALL_PREFIX=/opt/opencv-4.7.0 \
        -DOPENCV_EXTRA_MODULES_PATH=/tmp/cv/opencv_contrib-4.7.0/modules \
        -DWITH_CUDA=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.7 \
        -DWITH_FFMPEG=ON \
        -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    && cmake --build build -j"$(nproc)" \
    && cmake --install build \
    && rm -rf /tmp/cv

# LibTorch 2.0.1+cu117 (README step 5)
RUN wget -q "https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu117.zip" \
        -O /tmp/libtorch.zip \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm /tmp/libtorch.zip

# TensorRT 8.6.1.6 (README step 6)
RUN wget -q https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz \
        -O /tmp/trt.tar.gz \
    && tar -xzf /tmp/trt.tar.gz -C /opt \
    && rm /tmp/trt.tar.gz

# Coco-LIC workspace (README step 7)
RUN source /opt/ros/noetic/setup.bash \
    && mkdir -p ~/catkin_coco/src \
    && git clone https://github.com/Livox-SDK/livox_ros_driver.git ~/catkin_coco/src/livox_ros_driver \
    && cd ~/catkin_coco && catkin_make -j"$(nproc)" \
    && git clone https://github.com/APRIL-ZJU/Coco-LIC.git ~/catkin_coco/src/Coco-LIC \
    && sed -i 's|<node pkg="rviz" type="rviz" name="rviz_odom".*/>|<!-- & -->|' ~/catkin_coco/src/Coco-LIC/launch/odometry.launch \
    && mkdir -p ~/catkin_coco/src/Coco-LIC/data \
    && source ~/catkin_coco/devel/setup.bash \
    && cd ~/catkin_coco && catkin_make -j"$(nproc)"

# Miniconda (for TensorRT deployment scripts in ckpt/)
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda init bash \
    && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

ENV PATH=/opt/conda/bin:${PATH}

# Symlink TensorRT to path expected by build_trt.sh
RUN mkdir -p ~/Software \
    && ln -s /opt/TensorRT-8.6.1.6 ~/Software/TensorRT-8.6.1.6

# Gaussian-LIC workspace skeleton (README step 8 — source bind-mounted at runtime)
RUN mkdir -p ~/catkin_gaussian/src

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

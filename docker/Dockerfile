# =============================================================================
# Gaussian-LIC on Jetson AGX Orin (JetPack 6.2 / L4T r36.4 / CUDA 12.6 / TRT 10)
# -----------------------------------------------------------------------------
# This is a PORT of the original x86 desktop instructions, not a wrapper around
# them. Every "no compilation required" download in the README is x86_64 and
# pinned to CUDA 11.7 / TensorRT 8.6, none of which exist for the Orin, so the
# whole native stack is rebuilt here for aarch64 / sm_87 against JetPack's
# CUDA 12.6 + TensorRT 10.
#
# Build (ON the Orin, with the nvidia container runtime as the default runtime):
#   docker build -f docker/Dockerfile -t gaussian-lic:orin .
# The OpenCV-with-CUDA stage takes a long time (tens of minutes on AGX Orin);
# it is isolated in its own stage so it is cached across rebuilds.
# =============================================================================

# JetPack 6.2 == L4T r36.4.x. Verify your tag with `cat /etc/nv_tegra_release`
# and adjust if needed (e.g. r36.4.3).
ARG L4T_BASE=nvcr.io/nvidia/l4t-jetpack:r36.4.0

# -----------------------------------------------------------------------------
# Stage 1: OpenCV 4.10.0 + opencv_contrib 4.10.0, built WITH CUDA for sm_87.
# Required because src/tensor_utils.h uses cv::cuda::GpuMat / cudaimgproc.
# -----------------------------------------------------------------------------
FROM ${L4T_BASE} AS opencv-builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget unzip pkg-config \
        libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
        libtbb-dev libjpeg-dev libpng-dev libtiff-dev libeigen3-dev \
        python3-dev python3-numpy \
    && rm -rf /var/lib/apt/lists/*

ARG OPENCV_VERSION=4.10.0
WORKDIR /opt/opencv_src
RUN wget -q https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz -O opencv.tar.gz \
    && wget -q https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.tar.gz -O contrib.tar.gz \
    && tar xf opencv.tar.gz && tar xf contrib.tar.gz && rm -f *.tar.gz

# CUDA_ARCH_BIN=8.7 is the Orin (Ampere) compute capability. Building only that
# arch keeps the build as short as possible.
RUN mkdir -p /opt/opencv_src/opencv-${OPENCV_VERSION}/build \
    && cd /opt/opencv_src/opencv-${OPENCV_VERSION}/build \
    && cmake -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_INSTALL_PREFIX=/opt/opencv \
        -DWITH_CUDA=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON \
        -DCUDA_ARCH_BIN=8.7 -DCUDA_ARCH_PTX=8.7 \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
        -DENABLE_FAST_MATH=ON -DCUDA_FAST_MATH=ON \
        -DOPENCV_EXTRA_MODULES_PATH="/opt/opencv_src/opencv_contrib-${OPENCV_VERSION}/modules" \
        -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_JASPER=ON -DBUILD_CCALIB=ON \
        -DBUILD_JPEG=ON -DWITH_FFMPEG=ON \
        -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
        .. \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /opt/opencv_src

# -----------------------------------------------------------------------------
# Stage 2: runtime image — ROS Noetic (RoboStack/conda) + LibTorch-for-Jetson +
# the catkin workspace with Gaussian-LIC.
# -----------------------------------------------------------------------------
FROM ${L4T_BASE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# System build deps + libs Gaussian-LIC's CMake resolves directly.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget curl ca-certificates bzip2 unzip \
        pkg-config libffi-dev libyaml-cpp-dev libeigen3-dev \
        python3-dev python3-numpy python3-pip \
        libgl1 libglib2.0-0 libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Bring in the CUDA-enabled OpenCV 4.10 built in stage 1.
COPY --from=opencv-builder /opt/opencv /opt/opencv
ENV OpenCV_DIR=/opt/opencv/lib/cmake/opencv4
ENV LD_LIBRARY_PATH=/opt/opencv/lib:${LD_LIBRARY_PATH}

# --- ROS Noetic via RoboStack (conda-forge) -------------------------------
# RoboStack ships a Noetic that runs on Ubuntu 22.04 / aarch64, which official
# Noetic (20.04-only) does not. Installed via micromamba.
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV ROS_ENV=/opt/conda/envs/ros
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest \
        | tar -xvj -C /usr/local bin/micromamba

# Channels: robostack-staging provides ros-noetic-*; conda-forge for the rest.
# Split into two solves (base, then message/utility packages) rather than one
# ~30min+ create: aarch64 conda-forge has thin sharded-repodata coverage so
# libmamba falls back to slow full-repodata fetches per package, and a single
# giant RUN means any transient failure re-solves everything from scratch.
# cmake/make/pkg-config are deliberately left out here — apt already provides
# them above (build-essential + cmake + pkg-config), so including them in the
# conda solve would only add solver work for no benefit.
#
# Python is pinned to 3.11: robostack-staging/linux-aarch64 builds ros-noetic-*
# only for python 3.9 / 3.11 / 3.12 (NOT 3.10), so pinning 3.10 here is
# unsolvable. LibTorch is decoupled from this Python (see below), so the cp310
# Jetson torch wheel no longer constrains the ROS env.
RUN micromamba create -y -n ros -c robostack-staging -c conda-forge \
        python=3.11 \
        ros-noetic-ros-base \
        ros-noetic-catkin catkin_tools \
    && micromamba clean -ya

RUN micromamba install -y -n ros -c robostack-staging -c conda-forge \
        ros-noetic-roscpp ros-noetic-roslib ros-noetic-rospy \
        ros-noetic-std-msgs ros-noetic-sensor-msgs ros-noetic-nav-msgs \
        ros-noetic-geometry-msgs ros-noetic-tf ros-noetic-eigen-conversions \
        ros-noetic-image-transport ros-noetic-pcl-conversions ros-noetic-pcl-ros \
        ros-noetic-rosbag ros-noetic-roslaunch \
    && micromamba clean -ya
# NOTE: cv_bridge is intentionally NOT installed from conda — its conda build
# pulls a second, CUDA-less OpenCV that clashes with our /opt/opencv at link &
# runtime. cv_bridge is built from source below against /opt/opencv instead.

# --- LibTorch + torch for Jetson (CUDA 12.6 aarch64) ----------------------
# Installed into the SYSTEM Python 3.10 (NOT the conda ROS env): the only Jetson
# torch wheels are cp310, and the ROS env must be 3.11 (RoboStack noetic/aarch64
# isn't built for 3.10). System python3 == 3.10, so the cp310 wheels fit there,
# fully decoupled from the ROS env. This one install serves both consumers:
#   * the C++ gs_mapping build links LibTorch via the bundled share/cmake/Torch
#   * the SPNet ONNX export imports the torch + torchvision Python modules
# pip (vs a bare unzip) is required so torch's Python deps (sympy/networkx/...)
# are present for `import torch` during export.
#
# Pinned to exact wheel URLs (the index publishes cp310 only, so --extra-index
# risks pulling PyPI's CPU-only aarch64 wheel). Verified live: torch 2.11.0 /
# torchvision 0.26.0 are the cp310 aarch64 builds; torchvision must pair with
# torch (2.8<->0.23, 2.9<->0.24, 2.10<->0.25, 2.11<->0.26). Other torch versions
# available: 2.8.0 / 2.9.1 / 2.10.0. Browse: https://pypi.jetson-ai-lab.io/jp6/cu126
#
# NOTE: the code targets libtorch 2.0.1; 2.8+ is the floor on Jetson, so a few
# C++ ATen API call sites may need adjustment when the workspace builds.
ARG TORCH_WHL=https://pypi.jetson-ai-lab.io/jp6/cu126/+f/46b/b8b13f844b211/torch-2.11.0-cp310-cp310-linux_aarch64.whl
ARG TORCHVISION_WHL=https://pypi.jetson-ai-lab.io/jp6/cu126/+f/d11/6f08d3d62417d/torchvision-0.26.0-cp310-cp310-linux_aarch64.whl
RUN python3 -m pip install --no-cache-dir "${TORCH_WHL}" "${TORCHVISION_WHL}" \
    && test -f /usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch/TorchConfig.cmake
ENV Torch_DIR=/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}

# --- ROS2 .mcap -> ROS1 .bag conversion --------------------------------------
# rosbags is pure Python (no ROS2 runtime / no ros1_bridge needed); it reads
# ROS2 mcap and writes a ROS1 v2.0 bag, mapping standard message types.
RUN micromamba run -n ros pip install --no-cache-dir rosbags

# cv_bridge (built from source) links Boost.Python. conda-forge splits Boost
# into components and ros-noetic-ros-base pulls only the core, so the Python
# component (libboost_python*.so + boost_pythonConfig.cmake) is missing. Add it
# explicitly — placed after torch/rosbags so those cached layers are preserved.
RUN micromamba install -y -n ros -c conda-forge libboost-python-devel \
    && micromamba clean -ya

# A few more deps that surface at compile or import time, added here (late) so
# the expensive conda-create / torch / OpenCV layers above stay cached on rebuild:
#   * libglm-dev — the CUDA rasterizer includes <glm/glm.hpp> with no vendored
#     copy and no glm path in CMake, so it expects glm as a SYSTEM header. The
#     original x86 desktop had it (apt install libglm-dev); the minimal image
#     did not. Header-only, arch:all — available for arm64 on jammy.
#   * ros-noetic-tf-conversions — mapping.h / gaussian.cpp #include
#     <tf_conversions/tf_eigen.h>. The monolithic apt ROS install always shipped
#     it; our minimal RoboStack env installed only packages actually linked. The
#     header is effectively unused by the code, but the #include must still
#     resolve. Its own headers land on the shared conda include path, so no
#     CMake/COMPONENTS change is needed.
#   * libopenblas0 — the Jetson torch wheel links libopenblas.so.0 as its BLAS
#     backend. `import torch` (SPNet ONNX export, via system Python 3.10) needs
#     it on the loader path; the C++ node happened to find OpenBLAS through the
#     conda env, but system Python doesn't see the conda lib dir. Installing it
#     at system level makes torch self-sufficient. (Cleanup: belongs in the main
#     apt block.)
RUN apt-get update && apt-get install -y --no-install-recommends libglm-dev libopenblas0 \
    && rm -rf /var/lib/apt/lists/* \
    && micromamba install -y -n ros -c robostack-staging -c conda-forge \
        ros-noetic-tf-conversions \
    && micromamba clean -ya

# --- cuDSS (torch 2.11 hard dependency) --------------------------------------
# Jetson's torch 2.11 lists libcudss.so.0 (NVIDIA direct sparse solver) as a
# DT_NEEDED of libtorch_cuda.so, so it's required at BOTH link and runtime. The
# wheel we installed by direct URL didn't pull it (Jetson torch wheels don't
# declare the nvidia-*-cu12 pip deps). Install it into the same system Python
# 3.10 as torch, then symlink the libs next to libtorch_cuda.so so torch's own
# $ORIGIN rpath finds them at runtime and build_ws.sh's -rpath-link resolves the
# transitive NEEDED at link time. The core cuDSS API (cudssCreate/Execute/
# Matrix*/Config*/Data*) is stable across the 0.x line, so the exact pin is
# low-risk; 0.7.1.6 matches the torch-2.11 timeframe.
ARG CUDSS_VERSION=0.7.1.6
RUN python3 -m pip install --no-cache-dir "nvidia-cudss-cu12==${CUDSS_VERSION}" \
    && CUDSS_LIB_DIR="$(dirname "$(find /usr/local/lib/python3.10/dist-packages/nvidia -name 'libcudss.so.0' | head -n1)")" \
    && test -n "${CUDSS_LIB_DIR}" \
    && ln -sf "${CUDSS_LIB_DIR}"/libcudss*.so* /usr/local/lib/python3.10/dist-packages/torch/lib/ \
    && test -e /usr/local/lib/python3.10/dist-packages/torch/lib/libcudss.so.0

# --- TensorRT 10 + CUDA come from the l4t-jetpack base (system multiarch) ---
ENV TENSORRT_INCLUDE_DIR=/usr/include/aarch64-linux-gnu
ENV TENSORRT_LIB_DIR=/usr/lib/aarch64-linux-gnu
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
# Compile every CUDA kernel (rasterizer, simple-knn, fused-ssim) for the Orin.
ENV TORCH_CUDA_ARCH_LIST="8.7"

# --- Build the catkin workspace -------------------------------------------
WORKDIR /root/catkin_gaussian
COPY . /root/catkin_gaussian/src/Gaussian-LIC

# Build cv_bridge from source against the CUDA OpenCV so there is exactly one
# OpenCV in the process. (vision_opencv contains cv_bridge + image_geometry.)
RUN git clone --branch noetic https://github.com/ros-perception/vision_opencv.git \
        /root/catkin_gaussian/src/vision_opencv

COPY docker/build_ws.sh /usr/local/bin/build_ws.sh
RUN chmod +x /usr/local/bin/build_ws.sh && /usr/local/bin/build_ws.sh

COPY docker/build_engine.sh /usr/local/bin/build_engine.sh
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/build_engine.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

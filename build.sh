#!/usr/bin/env bash
set -e

source /opt/ros/noetic/setup.bash
source ~/catkin_coco/devel/setup.bash

export LD_LIBRARY_PATH=/opt/libtorch/lib:/opt/opencv-4.7.0/lib:/opt/TensorRT-8.6.1.6/lib:${LD_LIBRARY_PATH}
export PATH=/usr/local/cuda-11.7/bin:${PATH}
export CUDA_HOME=/usr/local/cuda-11.7

echo "[build] Building catkin_gaussian..."
cd ~/catkin_gaussian
catkin_make -j"$(nproc)"
echo "[build] Done. Source devel/setup.bash to use."

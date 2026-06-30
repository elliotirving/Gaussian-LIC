#!/usr/bin/env bash
# Activate the ROS env + the built workspace, then run whatever was passed
# (defaults to an interactive shell).
set -e

eval "$(micromamba shell hook --shell bash)"
micromamba activate ros
source "${ROS_ENV}/setup.bash"

if [ -f /root/catkin_gaussian/devel/setup.bash ]; then
  source /root/catkin_gaussian/devel/setup.bash
fi

# Make the LibTorch, CUDA OpenCV, and Jetson TensorRT libs discoverable at
# runtime. LibTorch first so its bundled libs win where they must.
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/opt/opencv/lib:${TENSORRT_LIB_DIR:-/usr/lib/aarch64-linux-gnu}:${CUDA_TOOLKIT_ROOT_DIR:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"

exec "$@"

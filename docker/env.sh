# Sourceable environment setup for the Gaussian-LIC container.
#
# Sourced by:
#   * entrypoint.sh (before exec-ing the container command), and
#   * /root/.bashrc (entrypoint appends a source line), so every interactive
#     `docker compose exec gaussian-lic bash` gets a ready-to-use ROS env with
#     no wrapper needed.
#
# No `set -e` / no `exec` here — it must be safe to source into any shell.

eval "$(micromamba shell hook --shell bash)"
micromamba activate ros
source "${ROS_ENV}/setup.bash"

# The built catkin workspace (present after build_ws.sh; lives in a named volume).
if [ -f /root/catkin_gaussian/devel/setup.bash ]; then
  source /root/catkin_gaussian/devel/setup.bash
fi

# Make LibTorch, CUDA OpenCV, the conda ros libs, and Jetson TensorRT/CUDA
# discoverable at runtime. Order matters:
#   * LibTorch first so its bundled libs win where they must.
#   * ${CONDA_PREFIX}/lib (the ros env: libopenblas.so.0, libroscpp, libpcl, ...)
#     must precede the system /usr/lib/aarch64-linux-gnu — the ros libs were
#     built with a newer GCC and need the conda libstdc++ (GLIBCXX_3.4.32 /
#     CXXABI_1.3.15), which the system libstdc++ (max GLIBCXX_3.4.30) lacks. The
#     newer conda libstdc++ is backward-compatible, so torch/opencv still work.
# Without this, gs_mapping dies with "libopenblas.so.0: cannot open shared
# object file" or "version GLIBCXX_3.4.32 not found".
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/opt/opencv/lib:${CONDA_PREFIX:-/opt/conda/envs/ros}/lib:${TENSORRT_LIB_DIR:-/usr/lib/aarch64-linux-gnu}:${CUDA_TOOLKIT_ROOT_DIR:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"

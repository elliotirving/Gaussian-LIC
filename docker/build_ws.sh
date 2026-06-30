#!/usr/bin/env bash
# Build the catkin workspace inside the RoboStack ROS env, wiring Gaussian-LIC's
# overridable CMake variables to the Jetson/JetPack locations.
#
# NOTE: no `set -u` — RoboStack's conda activation scripts (e.g.
# ros-noetic-catkin_activate.sh) reference unset vars like $CONDA_BUILD, which
# would abort under nounset.
set -eo pipefail

# Torch_DIR, OpenCV_DIR, TENSORRT_* and CUDA_TOOLKIT_ROOT_DIR are baked in as
# image ENV vars (see Dockerfile), so they are already in the environment here.

eval "$(micromamba shell hook --shell bash)"
micromamba activate ros
source "${ROS_ENV}/setup.bash"

cd /root/catkin_gaussian

TORCH_LIB=/usr/local/lib/python3.10/dist-packages/torch/lib

# Link seam between the two toolchains in this image:
#   * The conda ROS/Boost libs (libroscpp, libimage_transport,
#     libboost_filesystem, ...) were built with conda-forge GCC 13, so they
#     reference GLIBCXX_3.4.32 / CXXABI_1.3.15. We compile with the base image's
#     system GCC 11, whose libstdc++ only provides up to 3.4.30 -> undefined
#     references at link. The conda env ships the newer libstdc++.so.6 (a
#     superset, backward-compatible with our GCC-11 objects), so we force the
#     link (and, via -rpath, the runtime) to use it. -l:libstdc++.so.6 is added
#     as a standard lib (end of the link line) so --as-needed keeps it to
#     satisfy the conda libs that precede it, and it needs no libstdc++.so
#     devel symlink (which the runtime-only conda env lacks).
#   * libtorch_cuda.so has a DT_NEEDED on libcudss.so.0 (symlinked next to it in
#     ${TORCH_LIB} by the Dockerfile); -rpath-link ${TORCH_LIB} lets ld resolve
#     that transitive dependency at link time.
#
# CMake 4.x removed modules that ROS1 Noetic packages still rely on. Force the
# old behaviour workspace-wide rather than patching each upstream CMakeLists:
#   CMP0167 -> keep FindBoost      (cv_bridge: find_package(Boost ... python))
#   CMP0148 -> keep FindPythonInterp/FindPythonLibs (catkin + cv_bridge)
# Plus the <3.5 minimum-version escape hatch. (If 4.x keeps throwing removals,
# the durable alternative is pinning 'cmake<4' in the conda env.)
catkin_make \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_POLICY_DEFAULT_CMP0167=OLD \
  -DCMAKE_POLICY_DEFAULT_CMP0148=OLD \
  -DOpenCV_DIR="${OpenCV_DIR}" \
  -DTorch_DIR="${Torch_DIR}" \
  -DTENSORRT_INCLUDE_DIR="${TENSORRT_INCLUDE_DIR}" \
  -DTENSORRT_LIB_DIR="${TENSORRT_LIB_DIR}" \
  -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLKIT_ROOT_DIR}" \
  -DGAUSSIAN_LIC_CUDA_ARCH=87 \
  -DCMAKE_EXE_LINKER_FLAGS="-L${ROS_ENV}/lib -Wl,-rpath-link,${ROS_ENV}/lib -Wl,-rpath-link,${TORCH_LIB} -Wl,-rpath,${ROS_ENV}/lib" \
  -DCMAKE_CXX_STANDARD_LIBRARIES="-l:libstdc++.so.6"

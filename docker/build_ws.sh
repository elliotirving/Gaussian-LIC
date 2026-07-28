#!/usr/bin/env bash
# Build the catkin workspace. IDENTICAL on both platforms — every platform
# specific is an image ENV var (set in the Dockerfile), wired into the
# already-overridable variables in CMakeLists.txt.
#
# No `set -u`: RoboStack's conda activation scripts reference unset vars.
set -eo pipefail

# Activates the ROS env + sets Torch_DIR/OpenCV_DIR/TENSORRT_*/CUDA paths etc.
source /usr/local/bin/gaussian_lic_env.sh

cd /root/catkin_gaussian

# Assemble flags as an array so values containing spaces (linker flags) survive.
FLAGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DOpenCV_DIR="${OpenCV_DIR}"
  -DTorch_DIR="${Torch_DIR}"
  -DTENSORRT_INCLUDE_DIR="${TENSORRT_INCLUDE_DIR}"
  -DTENSORRT_LIB_DIR="${TENSORRT_LIB_DIR}"
  -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLKIT_ROOT_DIR}"
)

# x86 installs Miniconda only for SPNet export, but an interactive shell may
# auto-activate conda base before this script runs. Catkin/Noetic must use the
# ROS Python; otherwise it looks for empy in /opt/conda instead of apt Python.
CMAKE_PYTHON_EXECUTABLE="${GLIC_CMAKE_PYTHON_EXECUTABLE:-}"
if [ -z "${CMAKE_PYTHON_EXECUTABLE}" ] && [ -z "${GLIC_CONDA_ENV:-}" ] && [ -x /usr/bin/python3 ]; then
  CMAKE_PYTHON_EXECUTABLE=/usr/bin/python3
fi
if [ -n "${CMAKE_PYTHON_EXECUTABLE}" ]; then
  FLAGS+=(
    -DPYTHON_EXECUTABLE="${CMAKE_PYTHON_EXECUTABLE}"
    -DPython3_EXECUTABLE="${CMAKE_PYTHON_EXECUTABLE}"
  )
fi

# Keep the x86 catkin build from accidentally consuming Miniconda base packages.
# Conda is present there only for SPNet export; C++ dependencies should come from
# apt, /usr/local, or the explicit vendor paths above. Orin's ROS lives in conda,
# so this is disabled when GLIC_CONDA_ENV is set.
CMAKE_IGNORE_PREFIX_PATH_VALUE="${GLIC_CMAKE_IGNORE_PREFIX_PATH:-}"
YAML_CPP_DIR_VALUE="${GLIC_YAML_CPP_DIR:-}"
if [ -z "${GLIC_CONDA_ENV:-}" ]; then
  if [ -d /opt/conda ]; then
    case ";${CMAKE_IGNORE_PREFIX_PATH_VALUE};" in
      *";/opt/conda;"*) ;;
      *) CMAKE_IGNORE_PREFIX_PATH_VALUE="/opt/conda${CMAKE_IGNORE_PREFIX_PATH_VALUE:+;${CMAKE_IGNORE_PREFIX_PATH_VALUE}}" ;;
    esac
  fi
  if [ -z "${YAML_CPP_DIR_VALUE}" ] && [ -f /usr/lib/x86_64-linux-gnu/cmake/yaml-cpp/yaml-cpp-config.cmake ]; then
    YAML_CPP_DIR_VALUE=/usr/lib/x86_64-linux-gnu/cmake/yaml-cpp
  fi
fi
if [ -n "${CMAKE_IGNORE_PREFIX_PATH_VALUE}" ]; then
  FLAGS+=( -DCMAKE_IGNORE_PREFIX_PATH="${CMAKE_IGNORE_PREFIX_PATH_VALUE}" )
fi
if [ -n "${YAML_CPP_DIR_VALUE}" ]; then
  FLAGS+=( -Dyaml-cpp_DIR="${YAML_CPP_DIR_VALUE}" )
fi

# GPU compute target. Unset on x86 (nvcc default + JIT, matches master); 87 on Orin.
[ -n "${GAUSSIAN_LIC_CUDA_ARCH:-}" ] && \
  FLAGS+=( -DGAUSSIAN_LIC_CUDA_ARCH="${GAUSSIAN_LIC_CUDA_ARCH}" )

# Optional runtime eval instrumentation. The explicit OFF keeps a cached
# catkin build from accidentally preserving a previous metrics-enabled build.
FLAGS+=(
  -DGAUSSIAN_LIC_ENABLE_ONLINE_METRICS="${GAUSSIAN_LIC_ENABLE_ONLINE_METRICS:-OFF}"
)

# CMake 4.x / conda-forge compatibility shims (orin only). CMP0167/CMP0148 don't
# exist on x86's older CMake, so they are gated rather than always passed.
if [ "${GLIC_CMAKE_LEGACY_POLICIES:-0}" = "1" ]; then
  FLAGS+=(
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DCMAKE_POLICY_DEFAULT_CMP0167=OLD
    -DCMAKE_POLICY_DEFAULT_CMP0148=OLD
  )
fi

# Toolchain link seam (orin: force conda libstdc++.so.6 + cudss rpath-link).
# Empty on x86.
[ -n "${GLIC_LINKER_FLAGS:-}" ] && \
  FLAGS+=( -DCMAKE_EXE_LINKER_FLAGS="${GLIC_LINKER_FLAGS}" )
[ -n "${GLIC_CXX_STDLIB:-}" ] && \
  FLAGS+=( -DCMAKE_CXX_STANDARD_LIBRARIES="${GLIC_CXX_STDLIB}" )

echo "[build_ws] catkin_make ${FLAGS[*]}"
catkin_make "${FLAGS[@]}"
echo "[build_ws] Done. New shells auto-source devel/setup.bash (see env.sh)."

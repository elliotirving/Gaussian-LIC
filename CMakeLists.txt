cmake_minimum_required(VERSION 3.0.2)
project(gaussian_lic)

SET(CMAKE_BUILD_TYPE "Release")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -Wno-sign-compare -Wno-unused -Wno-comment -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
option(GAUSSIAN_LIC_ENABLE_ONLINE_METRICS
       "Write JetFast-compatible online mapping metrics.csv at shutdown"
       OFF)
# -msse4.2 is x86-only and breaks the aarch64 (Jetson Orin) build. Apply SIMD
# flags per host architecture instead.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  std_msgs
  cv_bridge
  image_transport
  pcl_conversions
  pcl_ros
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  tf
  eigen_conversions
)

# Dependency locations. Defaults match the original x86 desktop setup but can be
# overridden from the command line / Docker build, e.g. on the Jetson Orin:
#   catkin_make -DOpenCV_DIR=... -DTorch_DIR=... -DTENSORRT_ROOT=/usr/lib/aarch64-linux-gnu \
#               -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
if(NOT OpenCV_DIR)
  set(OpenCV_DIR ~/Software/opencv/opencv-4.7.0/build)
endif()
find_package(OpenCV 4 REQUIRED)
if(NOT Torch_DIR)
  set(Torch_DIR ~/Software/libtorch/share/cmake/Torch)
endif()
find_package(Torch REQUIRED)

if(NOT TENSORRT_ROOT)
  set(TENSORRT_ROOT "~/Software/TensorRT-8.6.1.6")
endif()
# On JetPack the headers/libs live in the system multiarch dirs; allow explicit
# include/lib overrides, otherwise derive them from TENSORRT_ROOT.
if(NOT TENSORRT_INCLUDE_DIR)
  set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
endif()
if(NOT TENSORRT_LIB_DIR)
  set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib")
endif()

if(NOT CUDA_TOOLKIT_ROOT_DIR)
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.7")
endif()
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
find_package(CUDA REQUIRED)

# GPU compute target. Leave empty to keep nvcc defaults (x86). On the Jetson
# Orin pass -DGAUSSIAN_LIC_CUDA_ARCH=87 (Ampere, sm_87).
set(GAUSSIAN_LIC_CUDA_ARCH "" CACHE STRING "CUDA arch, e.g. 87 for Orin")
if(GAUSSIAN_LIC_CUDA_ARCH)
  set(CMAKE_CUDA_ARCHITECTURES ${GAUSSIAN_LIC_CUDA_ARCH})
  list(APPEND CUDA_NVCC_FLAGS
       -gencode arch=compute_${GAUSSIAN_LIC_CUDA_ARCH},code=sm_${GAUSSIAN_LIC_CUDA_ARCH})
endif()
find_library(FFI_LIB ffi)

# yaml-cpp: prefer CMake config mode (robust across the conda/system split, and
# works even where Ubuntu's libyaml-cpp-dev 0.7 ships no usable .pc file). Fall
# back to pkg-config for setups that only provide that. Either path leaves
# YAML_CPP_LIBRARIES set for target_link_libraries below.
find_package(yaml-cpp QUIET)
if(yaml-cpp_FOUND)
  if(NOT YAML_CPP_LIBRARIES)
    set(YAML_CPP_LIBRARIES yaml-cpp)
  endif()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(YAML_CPP REQUIRED yaml-cpp>=0.5)
endif()

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES gaussian_lic
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  src
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
)

add_executable(gs_mapping
              ${PROJECT_SOURCE_DIR}/src/mapping.cpp
              ${PROJECT_SOURCE_DIR}/src/gaussian.cpp
              ${PROJECT_SOURCE_DIR}/src/tinyply.cpp
              ${PROJECT_SOURCE_DIR}/src/depth_completer.cpp
              ${PROJECT_SOURCE_DIR}/src/simple-knn/spatial.cu
              ${PROJECT_SOURCE_DIR}/src/simple-knn/simple_knn.cu
              ${PROJECT_SOURCE_DIR}/src/fused-ssim/ssim.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/renderer.cpp
              ${PROJECT_SOURCE_DIR}/src/rasterizer/rasterizer.cpp
              ${PROJECT_SOURCE_DIR}/src/rasterizer/rasterize_points.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/adam.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/forward.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/backward.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/rasterizer_impl.cu)
if(GAUSSIAN_LIC_ENABLE_ONLINE_METRICS)
  target_compile_definitions(gs_mapping PRIVATE GAUSSIAN_LIC_ENABLE_ONLINE_METRICS=1)
else()
  target_compile_definitions(gs_mapping PRIVATE GAUSSIAN_LIC_ENABLE_ONLINE_METRICS=0)
endif()
# Resolve TensorRT libs by name so we pick up either the desktop tarball layout
# or JetPack's multiarch system libs. libnvparsers was REMOVED in TensorRT 10
# (JetPack 6.x), so it is linked only when actually present.
find_library(TRT_NVINFER        NAMES nvinfer        PATHS ${TENSORRT_LIB_DIR})
find_library(TRT_NVONNXPARSER   NAMES nvonnxparser   PATHS ${TENSORRT_LIB_DIR})
find_library(TRT_NVINFER_PLUGIN NAMES nvinfer_plugin PATHS ${TENSORRT_LIB_DIR})
find_library(TRT_NVPARSERS      NAMES nvparsers      PATHS ${TENSORRT_LIB_DIR})

target_link_libraries(gs_mapping
                      ${catkin_LIBRARIES}
                      ${OpenCV_LIBS}
                      ${TORCH_LIBRARIES}
                      ${YAML_CPP_LIBRARIES}
                      ${Python3_LIBRARIES}
                      ${FFI_LIB}
                      ${CUDA_LIB_DIR}/libcudart.so
                      ${TRT_NVINFER}
                      ${TRT_NVONNXPARSER}
                      ${TRT_NVINFER_PLUGIN})
if(TRT_NVPARSERS)
  target_link_libraries(gs_mapping ${TRT_NVPARSERS})
endif()

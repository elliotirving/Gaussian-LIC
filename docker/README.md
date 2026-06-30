# Gaussian-LIC on Jetson AGX Orin (JetPack 6.2)

A Docker build that ports Gaussian-LIC from its x86 desktop instructions
(CUDA 11.7 / TensorRT 8.6 / x86_64 prebuilt LibTorch+TRT) to the Jetson AGX Orin
(aarch64 / sm_87 / **CUDA 12.6 / TensorRT 10** from JetPack 6.2), driven by
**pre-posed rosbags** (no Coco-LIC).

## Why a straight containerization of the README does not work

Every "no compilation required" download in the root README is **x86_64** and
pinned to **CUDA 11.7 / TensorRT 8.6**. None of those run on the Orin:

* The prebuilt **LibTorch (cu117)** and **TensorRT 8.6** tarballs are x86 binaries
  — they cannot execute on the Orin's ARM CPU, regardless of CUDA.
* There is **no CUDA 11.7 toolkit for the Jetson/Tegra** target; JetPack 6.2
  provides CUDA 12.6 and TensorRT 10.
* The Orin GPU is **sm_87** (Ampere), not sm_86/sm_89.

So the stack is rebuilt for aarch64 against JetPack 6.2, which forces two small
source changes (already applied in this repo):

| Change | File | Reason |
|---|---|---|
| `-msse4.2` → arch-guarded SIMD flag | `CMakeLists.txt` | `-msse4.2` is x86-only; aarch64 gcc rejects it |
| `libnvparsers` linked only if present + named lib lookup | `CMakeLists.txt` | `libnvparsers` was **removed in TensorRT 10** |
| Overridable dep paths + `GAUSSIAN_LIC_CUDA_ARCH` | `CMakeLists.txt` | point at JetPack CUDA/TRT, build for sm_87 |
| TensorRT 8 binding API → TRT-10 named-tensor API | `src/depth_completer.{h,cpp}` | `getNbBindings`/`getBindingDimensions`/`executeV2` removed in TRT 10 |

The TRT change is guarded on `NV_TENSORRT_MAJOR`, so the same source still
builds on x86 / TRT 8.6.

## Prerequisites

* Jetson **AGX Orin**, flashed with **JetPack 6.2** (L4T r36.4.3). The base
  image stays at `l4t-jetpack:r36.4.0` — NVIDIA never published an `r36.4.3`
  l4t-jetpack image (NGC tops out at r36.4.0 = JP6.1), and r36.4.0 runs fine on
  a JP6.2 host: both are CUDA 12.6, and the container runtime mounts your host's
  r36.4.3 driver. (`l4t-jetpack` is the full SDK image — it includes CUDA 12.6,
  cuDNN, and TensorRT 10 with `trtexec`, which the build/engine steps rely on.)
* Docker with the **NVIDIA Container Runtime** (default on JetPack).
* `Large_300.pth` SPNet checkpoint (Google Drive link in the root README),
  placed in `../ckpt/`.
* A **pre-posed rosbag** (see topic contract below).

## Build

```bash
cd <repo root>
docker compose -f docker/docker-compose.yml build
```

The OpenCV-with-CUDA stage is the long one (tens of minutes on AGX Orin) and is
cached in its own stage. The torch + torchvision wheels are pinned to exact
cp310 aarch64 URLs from the Jetson AI Lab index. To use a different (matched)
pair — e.g. torch 2.8.0 / torchvision 0.23.0, closer to the code's original
2.0.1 — override both, copying the URLs from
<https://pypi.jetson-ai-lab.io/jp6/cu126>:

```bash
docker build -f docker/Dockerfile \
  --build-arg TORCH_WHL=https://pypi.jetson-ai-lab.io/jp6/cu126/+f/.../torch-2.8.0-cp310-cp310-linux_aarch64.whl \
  --build-arg TORCHVISION_WHL=https://pypi.jetson-ai-lab.io/jp6/cu126/+f/.../torchvision-0.23.0-cp310-cp310-linux_aarch64.whl \
  -t gaussian-lic:orin .
```

## Build the SPNet TensorRT engine (on the Orin, once)

Engines are GPU-arch + TRT-version specific, so they must be built on-device:

```bash
# place ../ckpt/Large_300.pth first
docker compose -f docker/docker-compose.yml run --rm \
  gaussian-lic /usr/local/bin/build_engine.sh
```

This writes `ckpt/spnet_512_640.engine` and `ckpt/spnet_480_640.engine` (the
`ckpt/` dir is a host volume, so they persist).

## Run (bag-driven)

`gs_mapping` is fully decoupled from Coco-LIC — it only subscribes to four
standard topics. Your bag must publish them (remap if names differ):

| Topic | Type |
|---|---|
| `/pose_for_gs` | `geometry_msgs/PoseStamped` |
| `/points_for_gs` | `sensor_msgs/PointCloud2` |
| `/image_for_gs` | `sensor_msgs/Image` |
| `/depth_for_gs` | `sensor_msgs/Image` (sparse depth; SPNet densifies it) |

The Orin host has no ROS1, so **`rosbag play` runs inside the container too**
(`network_mode: host` lets a second shell share the ROS master).

```bash
xhost +local:root                                   # allow GUI windows

# Terminal 1 — ROS master + mapping node (waits for "😋 Gaussian-LIC Ready!"):
docker compose -f docker/docker-compose.yml up gaussian-lic

# Terminal 2 — exec into the SAME container and play the bag:
docker compose -f docker/docker-compose.yml exec gaussian-lic bash
rosbag play /data/bags/your.bag
#   …or with remapping (renames topics only, not types):
# rosbag play /data/bags/your.bag /your_pose:=/pose_for_gs /your_cloud:=/points_for_gs \
#                                 /your_img:=/image_for_gs /your_depth:=/depth_for_gs
```

Results are saved under `result/` on the host.

### Choosing the launch / config

`command:` in compose is only the **default** for `up`; the entrypoint runs
whatever args you give it. Three ways to pick a launch + config (`config/` and
`launch/` are host-mounted, so edits/new files need no rebuild):

```bash
# 1) Override the default for `up` via the LAUNCH env var:
LAUNCH=r3live.launch docker compose -f docker/docker-compose.yml up gaussian-lic

# 2) Same, but also pass extra roslaunch args (e.g. a custom config):
LAUNCH="fastlivo2.launch config_path:=config/mine.yaml" \
  docker compose -f docker/docker-compose.yml up gaussian-lic

# 3) Ad-hoc one-off, ignoring the default entirely:
docker compose -f docker/docker-compose.yml run --rm gaussian-lic \
  roslaunch gaussian_lic r3live.launch config_path:=config/mine.yaml
```

Each `launch/*.launch` defaults `config_path` to its matching `config/*.yaml`
(camera intrinsics, topics, etc.). For a custom dataset, drop your YAML in
`config/` on the host and point `config_path:=config/yourfile.yaml` at it.

## Known integration risks (expect on-device iteration here)

This scaffold is correct on the parts that are fully determinable (base image,
patches, arch flags, engine build, topic contract). These areas commonly need a
round or two of debugging on the actual device:

0. **LibTorch version gap (most likely to bite the build).** The code targets
   libtorch **2.0.1**; the Jetson index only offers **2.8.0–2.11.0**. The ATen
   C++ API is mostly stable across 2.x, but `gaussian.cpp` uses a lot of it
   (~168 call sites), incl. a custom `torch::autograd::Function` and a custom
   `torch::optim` optimizer (`param_groups()[i].options()`) — the kind of API
   that can shift between 2.0 and 2.11. If the workspace build throws ATen
   errors, they'll be concentrated in `gaussian.cpp` / `loss_utils.h` /
   `optim_utils.h`; try the lowest available torch (2.8.0 + torchvision 0.23.0)
   to stay closest to 2.0.1.


1. **RoboStack + system LibTorch/TensorRT ABI.** RoboStack's conda toolchain and
   the JetPack system libraries each carry their own `libstdc++`. If you hit
   `undefined symbol` / `GLIBCXX` errors at link or load time, force the newer
   `libstdc++` first on `LD_LIBRARY_PATH`, or rebuild with a single consistent
   compiler. An alternative to RoboStack is the
   [`dusty-nv/jetson-containers`](https://github.com/dusty-nv/jetson-containers)
   ROS-Noetic image (Noetic built from source on L4T).
2. **One OpenCV in the process.** `cv_bridge` is built from source against the
   CUDA `/opt/opencv` (not installed from conda) precisely to avoid two OpenCVs.
   If other ROS packages drag in a second OpenCV, expect crashes in image
   callbacks — keep `/opt/opencv` first on the library path.

## Performance note

AGX Orin 64GB will run this, but expect throughput well below a 3090/4090; it is
not guaranteed to keep up with the paper's real-time numbers. Play bags at a
reduced rate (`rosbag play -r 0.5 …`) if mapping falls behind.

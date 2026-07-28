<p align="center">
  <h1 align="center">Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion</h1>
  <p align="center">
    ICRA 2025
  </p>
  <p align="center">
    <a href="https://arxiv.org/pdf/2404.06926">
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://xingxingzuo.github.io/gaussian_lic/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>
</p>

Gaussian-LIC is a photo-realistic LiDAR-Inertial-Camera Gaussian Splatting SLAM system, which simultaneously performs robust, accurate pose estimation and constructs a photo-realistic 3D Gaussian map in real time.

<p align="center">
    <img src="figure/r1_compressed.gif" alt="Logo" width="32%">
    <img src="figure/r0_compressed.gif" alt="Logo" width="32%">
    <img src="figure/f2_compressed.gif" alt="Logo" width="32%">
</p>

### 📢 News

- [2026-02-21] Gaussian-LIC2 is released! 🚀 (stay tuned for updates)
- [2025-07-08] Gaussian-LIC2 is unveiled! 🎉 [[`Paper`](https://arxiv.org/pdf/2507.04004)] [[`Page`](https://xingxingzuo.github.io/gaussian_lic2/)] [[`YouTube`](https://www.youtube.com/watch?v=SkPnpuCfh88)] [[`bilibili`](https://www.bilibili.com/video/BV1fJ3kzfEYv/?spm_id_from=333.337.search-card.all.click&vd_source=99ac6409fc9373f3960feff31c28a189)] 
- [2025-07-07] The enhanced version of the Gaussian-LIC code is released!
- [2025-01-28] Gaussian-LIC is accepted to ICRA 2025! 🎉
- [2024-09-26] The second version of the paper is available on arXiv.
- [2024-04-10] The first version of the paper is available on arXiv.

### 💌 Contact

Questions? Please don't hesitate to reach out to Xiaolei Lang (Jerry) at jerry_locker@zju.edu.cn.

## Docker

Two images share one source tree; pick the service for your hardware:

| Service | Hardware | Stack | COCOLIC |
|---|---|---|---|
| `glic-x86` | x86 desktop (3060/3090) | CUDA 11.7 · TRT 8.6 · apt ROS Noetic | yes |
| `glic-orin` | Jetson AGX Orin (JetPack 6.2) | CUDA 12.6 · TRT 10 · RoboStack ROS | no (pre-posed bags) |

Below, `<svc>` is `glic-x86` or `glic-orin`. Set `DATA_DIR` in [`docker/.env`](docker/.env) to your host data folder (mounted read-write at `/data`).

```bash
# 1. Build the image
docker compose -f docker/docker-compose.yml build <svc>

# 2. One-time on the device: build the catkin workspace (persists in a volume)
docker compose -f docker/docker-compose.yml run --rm --name glic <svc> /usr/local/bin/build_ws.sh

# 3. One-time: build the SPNet TensorRT engine(s). First place the SPNet repo and
#    Large_300.pth in ckpt/ (see "Install" below). Engines are arch+TRT-specific,
#    so they must be built on the target machine.
docker compose -f docker/docker-compose.yml run --rm --name glic <svc> /usr/local/bin/build_engine.sh
```

Online metrics logging is off by default. From inside the container, build with
metrics enabled using:

```bash
GAUSSIAN_LIC_ENABLE_ONLINE_METRICS=ON ./docker/build_ws.sh
```

Running `build_ws.sh` without that variable builds with metrics off.

**Run** — one terminal owns the container, others attach (they share one ROS master via `network_mode: host`):

```bash
xhost +local:root                                                        # once, for GUI windows

# Terminal 1 — owns the container; launch the mapping node (waits for "😋 Ready!")
docker compose -f docker/docker-compose.yml run --rm --name glic <svc> bash
roslaunch gaussian_lic r3live.launch          # config_path defaults to config/r3live.yaml

# Terminal 2 — attach and feed poses (see below)
docker exec -it glic bash
```

**Feeding poses** (the node subscribes to exactly these topics):

| Topic | Type |
|---|---|
| `/pose_for_gs` | `geometry_msgs/PoseStamped` |
| `/points_for_gs` | `sensor_msgs/PointCloud2` |
| `/image_for_gs` | `sensor_msgs/Image` |
| `/depth_for_gs` | `sensor_msgs/Image` (sparse; SPNet densifies it) |

- **Pre-posed bag (both images).** Convert a ROS2 mcap once, then play it:
  ```bash
  python3 scripts/mcap_to_glic2_bag.py /data/in.mcap /data/out_glic.bag \
      --pose-topic /odin1/odometry_camera --pcd-topic /odin1/cloud_slam \
      --image-topic /odin1/image/undistorted --depth-topic /odin1/depth_img_competetion
  rosbag play /data/out_glic.bag            # in terminal 2
  ```
- **Live COCOLIC (`glic-x86` only).** In terminal 2, run Coco-LIC to publish the topics live:
  ```bash
  source ~/catkin_coco/devel/setup.bash
  roslaunch cocolic odometry.launch config_path:=config/ct_odometry_r3live.yaml
  ```

Results are saved under `result/` on the host. Each `launch/*.launch` defaults `config_path` to its matching `config/*.yaml`; drop a YAML in `config/` and pass `config_path:=config/yours.yaml` for a custom dataset. The Orin's `config/odin1.yaml` shows the 0.5× (`crop_y`/`width`/`height`) and `point_stride` options — both dormant at full resolution, so `glic-x86` matches the native build.

## Install

We test on ubuntu 20.04 with an NVIDIA RTX 3090 / 4090.

1. Exit Conda environment.

2. Prepare third-party libraries according to [Coco-LIC](https://github.com/APRIL-ZJU/Coco-LIC). 

3. Install [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local) with [cuDNN v8.9.7](https://developer.nvidia.com/rdp/cudnn-archive).

4. Build [OpenCV 4.7.0](https://github.com/opencv/opencv/archive/refs/tags/4.7.0.tar.gz).（must be built with [opencv_contrib 4.7.0](https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.tar.gz) and CUDA, no installation required）

   ```shell
   mkdir -p ~/Software/opencv
   cd ~/Software/opencv
   wget https://github.com/opencv/opencv/archive/refs/tags/4.7.0.tar.gz && tar -zxvf 4.7.0.tar.gz && rm -rf 4.7.0.tar.gz
   wget https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.tar.gz && tar -zxvf 4.7.0.tar.gz && rm -rf 4.7.0.tar.gz
   
   cd ~/Software/opencv/opencv-4.7.0
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=RELEASE -DWITH_CUDA=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON -DWITH_NVCUVID=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.7 -DOPENCV_EXTRA_MODULES_PATH="../../opencv_contrib-4.7.0/modules" -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_JASPER=ON -DBUILD_CCALIB=ON -DBUILD_JPEG=ON -DWITH_FFMPEG=ON ..
   make -j$(nproc)
   ```


5. Prepare [LibTorch](https://pytorch.org/get-started/locally/).（no compilation or installation required）

   ```shell
   cd ~/Software
   wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu117.zip
   unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cu117.zip && rm -rf libtorch-cxx11-abi-shared-with-deps-2.0.1+cu117.zip
   ```


6. Prepare [TensorRT](https://developer.nvidia.com/tensorrt/download).（no compilation or installation required）

   ```shell
   cd ~/Software
   wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
   tar -zxvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz && rm -rf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
   ```
   
7. Install Coco-LIC.

   ```shell
   mkdir -p ~/catkin_coco/src
   cd ~/catkin_coco/src
   git clone https://github.com/Livox-SDK/livox_ros_driver.git
   cd ~/catkin_coco && catkin_make
   cd ~/catkin_coco/src
   git clone https://github.com/APRIL-ZJU/Coco-LIC.git
   cd ~/catkin_coco && catkin_make
   ```

8. Install Gaussian-LIC.

   ```shell
   mkdir -p ~/catkin_gaussian/src
   cd ~/catkin_gaussian/src
   git clone https://github.com/APRIL-ZJU/Gaussian-LIC.git
   cd ~/catkin_gaussian && catkin_make
   ```

   To build the native workspace with online metrics enabled:

   ```shell
   cd ~/catkin_gaussian
   catkin_make -DGAUSSIAN_LIC_ENABLE_ONLINE_METRICS=ON
   ```

9. TensorRT Deployment.

   download and save [Large_300.pth](https://drive.google.com/file/d/11dujPviL4pKLEXytXK0mEmPBNQDqgEak/view?pli=1) to `~/catkin_gaussian/src/Gaussian-LIC/ckpt`.

   ```shell
   cd ~/catkin_gaussian/src/Gaussian-LIC/ckpt
   
   chmod +x setup_spnet.sh
   ./setup_spnet.sh
   
   chmod +x export_onnx.sh
   ./export_onnx.sh
   
   chmod +x build_trt.sh
   ./build_trt.sh
   ```

## Run

Quick start on the sequence CBD_Building_01 in the FAST-LIVO2 dataset.

- Download [FAST-LIVO Dataset](https://connecthkuhk-my.sharepoint.com/personal/zhengcr_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzhengcr%5Fconnect%5Fhku%5Fhk%2FDocuments%2FFAST%2DLIVO%2DDatasets&ga=1) or [FAST-LIVO2 Dataset](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/zhengcr_connect_hku_hk/ErdFNQtjMxZOorYKDTtK4ugBkogXfq1OfDm90GECouuIQA?e=KngY9Z) or [R3LIVE Dataset](https://github.com/ziv-lin/r3live_dataset) or [MCD Dataset](https://mcdviral.github.io/) or [M2DGR Dataset](https://github.com/SJTU-ViSYS/M2DGR).

+ Modify `bag_path` in the `config/ct_odometry_fastlivo2.yaml` file of Coco-LIC.

+ Launch Gaussian-LIC.

  ```shell
  cd ~/catkin_gaussian
  source devel/setup.bash
  roslaunch gaussian_lic fastlivo2.launch  // The terminal will print "😋 Gaussian-LIC Ready!".
  ```


+ Launch Coco-LIC.

  Note：For real-time use and runtime analysis, please turn off the rviz in Coco-LIC by commenting the sentence `<node pkg="rviz" type="rviz" name="rviz_odom" output="log" required = "true" args="-d $(find cocolic)/config/coco.rviz" />`  in `odometry.launch`.

  ```shell
  cd ~/catkin_coco
  source devel/setup.bash
  roslaunch cocolic odometry.launch config_path:=config/ct_odometry_fastlivo2.yaml
  ```


+ The mapping and rendering results will be saved in  `~/catkin_gaussian/src/Gaussian-LIC/result`.

## Checklist

- [ ] Support fast post-optimization
- [ ] Release the optimized Coco-LIC
- [ ] Provide the dockerfile
- [ ] Release the meshing tools
- [ ] Release our Gaussian-LIC2 dataset

## Citation

If you find our work helpful, please consider citing 🌟:

```bibtex
@inproceedings{lang2025gaussian,
  title={Gaussian-LIC: Real-time photo-realistic SLAM with Gaussian splatting and LiDAR-inertial-camera fusion},
  author={Lang, Xiaolei and Li, Laijian and Wu, Chenming and Zhao, Chen and Liu, Lina and Liu, Yong and Lv, Jiajun and Zuo, Xingxing},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={8500--8507},
  year={2025},
  organization={IEEE}
}
```

```bibtex
@article{lang2025gaussian2,
  title={Gaussian-LIC2: LiDAR-Inertial-Camera Gaussian Splatting SLAM}, 
  author={Lang, Xiaolei and Lv, Jiajun and Tang, Kai and Li, Laijian and Huang, Jianxin and Liu, Lina and Liu, Yong and Zuo, Xingxing},
  journal={arXiv}, 
  year={2025}
}
```

## Acknowledgement

Thanks for [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [Taming-3DGS](https://github.com/humansensinglab/taming-3dgs), [StopThePop](https://github.com/r4dl/StopThePop), [Photo-SLAM](https://github.com/HuajianUP/Photo-SLAM) and [SPNet](https://github.com/Wang-xjtu/SPNet).

## LICENSE

The code is released under the [GNU General Public License v3 (GPL-3)](https://www.gnu.org/licenses/gpl-3.0.txt).

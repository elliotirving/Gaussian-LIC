# Build/run docker
cd Gaussian-LIC
docker compose build
docker compose run --rm gaussian-lic bash

## Run once to setup the model
cd ~/catkin_gaussian/src/Gaussian-LIC/ckpt
./setup_spnet.sh
./export_onnx.sh
./build_trt.sh

Build program:
./build.sh

## Inside container
nano ~/catkin_coco/src/Coco-LIC/config/ct_odometry_r3live.yaml
<!-- set: bag_path: /data/datasets/hku_campus_seq_00.bag -->

Turn off cocolic rviz:
nano ~/catkin_coco/src/Coco-LIC/launch/odometry.launch
<!-- Comment rviz line -->

# Run
You need two terminals, both exec'd into the same container.

## Terminal 1 — start the container:
docker compose run --rm gaussian-lic bash

cd ~/catkin_gaussian && source devel/setup.bash

roslaunch gaussian_lic r3live.launch 

## Terminal 2 — attach to the running container:
docker exec -it gaussian-lic bash

cd ~/catkin_coco && source devel/setup.bash

roslaunch cocolic odometry.launch config_path:=config/ct_odometry_r3live.yaml

# Record new posed rosbag
rosbag record  /pose_for_gs /image_for_gs /points_for_gs /depth_for_gs -o /data/datasets/out
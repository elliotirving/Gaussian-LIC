#!/usr/bin/env python3
"""
Convert an odin1 ROS1 bag to the topic names and types that Gaussian-LIC expects.

Changes made:
  - nav_msgs/Odometry  (POSE_SRC)  -> geometry_msgs/PoseStamped  on /pose_for_gs
    The Odometry.pose.pose field is extracted; the header is preserved as-is.
  - sensor_msgs/PointCloud2 (PCD_SRC)   -> /points_for_gs  (type unchanged)
  - sensor_msgs/Image       (IMAGE_SRC) -> /image_for_gs   (type unchanged)
  - sensor_msgs/Image       (DEPTH_SRC) -> /depth_for_gs   (type unchanged)

Edit the four SRC variables below to match your bag's actual topic names
(run `rosbag info <bag>` to find them).

Usage (inside container, conda ros env):
  micromamba run -n ros python3 /root/catkin_gaussian/src/Gaussian-LIC/docker/convert_bag_odin1.py \
      /data/bags/input.bag /data/bags/input_glic.bag
"""

import sys
import rosbag
from geometry_msgs.msg import PoseStamped

# ---- edit these to match your bag ----------------------------------------
POSE_SRC  = "/odin1/odometry_camera"   # nav_msgs/Odometry  -> PoseStamped
PCD_SRC   = "/odin1/points_for_gs"     # sensor_msgs/PointCloud2
IMAGE_SRC = "/odin1/image_for_gs"      # sensor_msgs/Image
DEPTH_SRC = "/odin1/depth_for_gs"      # sensor_msgs/Image (sparse depth)
# --------------------------------------------------------------------------

DST_TOPICS = {
    POSE_SRC:  "/pose_for_gs",
    PCD_SRC:   "/points_for_gs",
    IMAGE_SRC: "/image_for_gs",
    DEPTH_SRC: "/depth_for_gs",
}

def convert(src_path, dst_path):
    src_topics = list(DST_TOPICS.keys())
    written = 0

    with rosbag.Bag(dst_path, "w") as out_bag:
        with rosbag.Bag(src_path, "r") as in_bag:
            total = in_bag.get_message_count(topic_filters=src_topics)
            print(f"Converting {total} messages from {src_path} ...")

            for topic, msg, t in in_bag.read_messages(topics=src_topics):
                dst_topic = DST_TOPICS[topic]

                if topic == POSE_SRC:
                    # nav_msgs/Odometry -> geometry_msgs/PoseStamped
                    ps = PoseStamped()
                    ps.header = msg.header
                    ps.pose   = msg.pose.pose   # unwrap PoseWithCovariance
                    out_bag.write(dst_topic, ps, t)
                else:
                    out_bag.write(dst_topic, msg, t)

                written += 1
                if written % 500 == 0:
                    print(f"  {written}/{total}", end="\r", flush=True)

    print(f"\nDone. Wrote {written} messages to {dst_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.bag> <output.bag>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])

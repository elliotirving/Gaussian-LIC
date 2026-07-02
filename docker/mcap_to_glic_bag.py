#!/usr/bin/env python3
"""
Single-pass MCAP (ROS2) -> ROS1 bag converter for Gaussian-LIC / odin1.

Reads the MCAP with the rosbags ROS2 reader (the only pure-Python MCAP reader),
then writes with the rosbag Python API so that MD5 sums in the bag match the
canonical ROS1 values that `rosbag play` enforces.  rosbags computes its own
MD5 hashes for message definitions and they differ from the official ROS1 sums,
causing "wrong md5sum" connection drops when replaying.

Operations performed in one pass:
  1. nav_msgs/Odometry  -> geometry_msgs/PoseStamped  (/pose_for_gs)
     (extracts Odometry.pose.pose, keeps header)
  2. sensor_msgs/PointCloud2 -> /points_for_gs  (type unchanged)
  3. sensor_msgs/Image       -> /image_for_gs   (type unchanged, use undistorted
     topic — config intrinsics assume no distortion)
  4. sensor_msgs/Image       -> /depth_for_gs   (type unchanged)

Edit the four SRC variables below to match your MCAP topic names.

Usage (inside container, conda ros env):
  micromamba run -n ros python3 \
      /root/catkin_gaussian/src/Gaussian-LIC/docker/mcap_to_glic_bag.py \
      /data/bags/input.mcap /data/bags/output_glic.bag
"""

import sys
from pathlib import Path

# ---- source topic names (from rosbag info / rosbags-info on your MCAP) ------
POSE_SRC  = "/odin1/odometry_camera"         # nav_msgs/Odometry  -> PoseStamped
PCD_SRC   = "/odin1/cloud_slam"              # sensor_msgs/PointCloud2
IMAGE_SRC = "/odin1/image/undistorted"       # sensor_msgs/Image (undistorted —
                                              #   config intrinsics assume no distortion)
DEPTH_SRC = "/odin1/depth_img_competetion"   # sensor_msgs/Image (sparse LiDAR depth;
                                              #   typo in topic name is in the bag itself)
# -----------------------------------------------------------------------------

TOPIC_MAP = {
    POSE_SRC:  "/pose_for_gs",
    PCD_SRC:   "/points_for_gs",
    IMAGE_SRC: "/image_for_gs",
    DEPTH_SRC: "/depth_for_gs",
}


def convert(src: str, dst: str) -> None:
    # rosbags: read MCAP / deserialize CDR
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import get_typestore, Stores

    # rosbag Python API: write with correct ROS1 MD5 sums
    import rosbag
    import rospy
    from sensor_msgs.msg import Image, PointCloud2, PointField
    from geometry_msgs.msg import PoseStamped, Point, Quaternion
    from std_msgs.msg import Header

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    def _header(h):
        hdr = Header()
        hdr.stamp = rospy.Time(h.stamp.sec, h.stamp.nanosec)
        hdr.frame_id = h.frame_id
        return hdr

    def _image(rb):
        msg = Image()
        msg.header = _header(rb.header)
        msg.height = rb.height
        msg.width = rb.width
        msg.encoding = rb.encoding
        msg.is_bigendian = rb.is_bigendian
        msg.step = rb.step
        msg.data = bytes(rb.data)
        return msg

    def _pointcloud2(rb):
        msg = PointCloud2()
        msg.header = _header(rb.header)
        msg.height = rb.height
        msg.width = rb.width
        for rbf in rb.fields:
            f = PointField()
            f.name = rbf.name
            f.offset = rbf.offset
            f.datatype = rbf.datatype
            f.count = rbf.count
            msg.fields.append(f)
        msg.is_bigendian = rb.is_bigendian
        msg.point_step = rb.point_step
        msg.row_step = rb.row_step
        msg.data = bytes(rb.data)
        msg.is_dense = rb.is_dense
        return msg

    def _odom_to_posestamped(rb):
        msg = PoseStamped()
        msg.header = _header(rb.header)
        p = rb.pose.pose.position
        q = rb.pose.pose.orientation
        msg.pose.position = Point(p.x, p.y, p.z)
        msg.pose.orientation = Quaternion(q.x, q.y, q.z, q.w)
        return msg

    with Reader(src) as reader, rosbag.Bag(dst, "w") as writer:
        active = {c.id: c for c in reader.connections if c.topic in TOPIC_MAP}

        if not active:
            print(
                "ERROR: none of the SRC topics were found in the MCAP.\n"
                "Check topic names with `rosbags-info` and update the SRC "
                "variables at the top of this script."
            )
            sys.exit(1)

        total = sum(c.msgcount for c in active.values())
        written = 0

        for conn, timestamp, rawdata in reader.messages():
            if conn.id not in active:
                continue

            dst_topic = TOPIC_MAP[conn.topic]
            t = rospy.Time(timestamp // 10**9, timestamp % 10**9)

            rb = typestore.deserialize_cdr(rawdata, conn.msgtype)

            if conn.topic == POSE_SRC:
                msg = _odom_to_posestamped(rb)
            elif conn.topic == PCD_SRC:
                msg = _pointcloud2(rb)
            else:  # IMAGE_SRC or DEPTH_SRC
                msg = _image(rb)

            writer.write(dst_topic, msg, t)
            written += 1
            if written % 200 == 0:
                pct = 100 * written // total if total else 0
                print(f"  {written}/{total}  ({pct}%)", end="\r", flush=True)

    print(f"\nDone. Wrote {written} messages -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.mcap> <output_glic.bag>")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    if not Path(src).exists():
        print(f"ERROR: {src} not found")
        sys.exit(1)
    if Path(dst).exists():
        print(f"ERROR: {dst} already exists — delete it first")
        sys.exit(1)
    convert(src, dst)

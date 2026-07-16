#!/usr/bin/env python3
"""
Single-pass MCAP (ROS2) -> ROS1 bag converter for Gaussian-LIC / Gaussian-LIC2.

Produces a *pre-posed* ROS1 bag that Gaussian-LIC ingests directly (no COCOLIC).
Reads the MCAP with the rosbags ROS2 reader (the only pure-Python MCAP reader),
then writes with the ROS1 `rosbag` API so the message-definition MD5 sums match
the canonical ROS1 values `rosbag play` enforces (rosbags' own MD5s differ and
cause "wrong md5sum" connection drops on replay).

Operations, in one pass:
  1. nav_msgs/Odometry  -> geometry_msgs/PoseStamped  (/pose_for_gs)
     (extracts Odometry.pose.pose, keeps header)
  2. sensor_msgs/PointCloud2 -> /points_for_gs  (type unchanged)
  3. sensor_msgs/Image       -> /image_for_gs   (type unchanged; use the
     UNDISTORTED image topic — config intrinsics assume no distortion)
  4. sensor_msgs/Image       -> /depth_for_gs   (sparse LiDAR depth; unchanged)

Source topic names are given as CLI flags (defaults are the odin1 layout). Run
inside the container shell (the ROS env is already active, so plain `python3`
has rosbag/rospy on the path):

  python3 scripts/mcap_to_glic2_bag.py IN.mcap OUT_glic.bag \\
      --pose-topic  /odin1/odometry_camera \\
      --pcd-topic   /odin1/cloud_slam \\
      --image-topic /odin1/image/undistorted \\
      --depth-topic /odin1/depth_img_competetion
"""

import argparse
import sys
from pathlib import Path

# ---- default source topic names (odin1 layout) ------------------------------
DEFAULT_POSE_SRC  = "/odin1/odometry_camera"         # nav_msgs/Odometry -> PoseStamped
DEFAULT_PCD_SRC   = "/odin1/cloud_slam"              # sensor_msgs/PointCloud2
DEFAULT_IMAGE_SRC = "/odin1/image/undistorted"       # sensor_msgs/Image (undistorted)
DEFAULT_DEPTH_SRC = "/odin1/depth_img_competetion"   # sensor_msgs/Image (sparse depth;
                                                     #   the typo is in the bag itself)


def convert(src: str, dst: str, topic_map: dict) -> None:
    # rosbags: read MCAP / deserialize CDR
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import get_typestore, Stores

    # rosbag Python API: write with correct ROS1 MD5 sums
    import rosbag
    import rospy
    from sensor_msgs.msg import Image, PointCloud2, PointField
    from geometry_msgs.msg import PoseStamped, Point, Quaternion
    from std_msgs.msg import Header

    pose_src = next(k for k, v in topic_map.items() if v == "/pose_for_gs")
    pcd_src = next(k for k, v in topic_map.items() if v == "/points_for_gs")

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
        active = {c.id: c for c in reader.connections if c.topic in topic_map}

        if not active:
            print(
                "ERROR: none of the source topics were found in the MCAP.\n"
                "Check topic names with `rosbags-info` / `ros2 bag info` and pass\n"
                "them with --pose/--pcd/--image/--depth-topic."
            )
            sys.exit(1)

        total = sum(c.msgcount for c in active.values())
        written = 0

        for conn, timestamp, rawdata in reader.messages():
            if conn.id not in active:
                continue

            dst_topic = topic_map[conn.topic]
            t = rospy.Time(timestamp // 10**9, timestamp % 10**9)

            rb = typestore.deserialize_cdr(rawdata, conn.msgtype)

            if conn.topic == pose_src:
                msg = _odom_to_posestamped(rb)
            elif conn.topic == pcd_src:
                msg = _pointcloud2(rb)
            else:  # image or depth
                msg = _image(rb)

            writer.write(dst_topic, msg, t)
            written += 1
            if written % 200 == 0:
                pct = 100 * written // total if total else 0
                print(f"  {written}/{total}  ({pct}%)", end="\r", flush=True)

    print(f"\nDone. Wrote {written} messages -> {dst}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert a ROS2 MCAP into a pre-posed ROS1 bag for Gaussian-LIC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="input .mcap (ROS2)")
    p.add_argument("output", help="output .bag (ROS1, pre-posed)")
    p.add_argument("--pose-topic", default=DEFAULT_POSE_SRC,
                   help="nav_msgs/Odometry source -> /pose_for_gs")
    p.add_argument("--pcd-topic", default=DEFAULT_PCD_SRC,
                   help="sensor_msgs/PointCloud2 source -> /points_for_gs")
    p.add_argument("--image-topic", default=DEFAULT_IMAGE_SRC,
                   help="sensor_msgs/Image (undistorted) source -> /image_for_gs")
    p.add_argument("--depth-topic", default=DEFAULT_DEPTH_SRC,
                   help="sensor_msgs/Image (sparse depth) source -> /depth_for_gs")
    args = p.parse_args()

    topic_map = {
        args.pose_topic:  "/pose_for_gs",
        args.pcd_topic:   "/points_for_gs",
        args.image_topic: "/image_for_gs",
        args.depth_topic: "/depth_for_gs",
    }
    if len(topic_map) != 4:
        print("ERROR: the four source topics must be distinct.")
        sys.exit(1)

    if not Path(args.input).exists():
        print(f"ERROR: {args.input} not found")
        sys.exit(1)
    if Path(args.output).exists():
        print(f"ERROR: {args.output} already exists — delete it first")
        sys.exit(1)

    convert(args.input, args.output, topic_map)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MCAP (ROS2, Oxford-Spires layout) -> ROS1 bag for Gaussian-LIC2, synthesising the
streams COCOLIC would normally publish live.

The Oxford-Spires bags do NOT ship the four /*_for_gs topics. They carry:
    /cam0/image_rect    sensor_msgs/Image        (rectified RGB or mono)
    /cam0/camera_info   sensor_msgs/CameraInfo   (rectified intrinsics)
    /points_deskewed    sensor_msgs/PointCloud2   (LiDAR, in the `lidar` frame)
    /tf                 map -> base   (dynamic trajectory, ~10 Hz)
    /tf_static          base -> cam0, base -> lidar  (sensor extrinsics)

so all four Gaussian-LIC inputs must be *derived*, exactly the way COCOLIC would:

  /pose_for_gs   PoseStamped   T_map_cam0(t_i) = T_map_base(t_i) . T_base_cam0
                               (TF chain composed, sampled at each image time,
                                LERP on translation + SLERP on rotation)
  /depth_for_gs  Image 32FC1   the deskewed scan projected into cam0 at the image
                               time (dual-time: accounts for the small lidar/cam
                               time offset via the interpolated trajectory),
                               z-buffered, at native image resolution
  /points_for_gs PointCloud2   the same in-FOV points expressed in the `map`
                               (world) frame and colourised from the image pixel
                               -> PointXYZRGB (what GLIC's pcl::fromROSMsg reads)
  /image_for_gs  Image         the rectified image, passed through

The four messages of a keyframe are all stamped with the image time so GLIC's
±0.01 s point-master sync (mapping.cpp getAlignedData) accepts them. Written with
the ROS1 `rosbag` API so message-definition MD5 sums match what `rosbag play`
enforces. Run inside the container shell (ROS env active, rosbag/rospy on path).

  python3 scripts/mcap_oxspires_to_glic2_bag.py IN.mcap OUT_glic.bag
  python3 scripts/mcap_oxspires_to_glic2_bag.py IN.mcap OUT_glic.bag --max-frames 50

Frame names, topics, association tolerance and range clip are all CLI flags; the
defaults are the Oxford-Spires layout above.
"""

import argparse
import bisect
import sys
from collections import deque
from pathlib import Path

import numpy as np


# ---------------------------- small SO3/SE3 helpers --------------------------
def quat_to_R(x, y, z, w):
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    n = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def R_to_quat(R):
    """3x3 rotation matrix -> quaternion (x,y,z,w)."""
    t = np.trace(R)
    if t > 0:
        s = (t + 1.0) ** 0.5 * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5 * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5 * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5 * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def slerp(q0, q1, u):
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    dot = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1
    if dot < 0.0:
        x1, y1, z1, w1, dot = -x1, -y1, -z1, -w1, -dot
    if dot > 0.9995:
        x, y, z, w = (x0 + u * (x1 - x0), y0 + u * (y1 - y0),
                      z0 + u * (z1 - z0), w0 + u * (w1 - w0))
    else:
        import math
        th0 = math.acos(dot)
        s0 = math.sin(th0 * (1.0 - u)) / math.sin(th0)
        s1 = math.sin(th0 * u) / math.sin(th0)
        x, y, z, w = (s0 * x0 + s1 * x1, s0 * y0 + s1 * y1,
                      s0 * z0 + s1 * z1, s0 * w0 + s1 * w1)
    n = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
    return (x / n, y / n, z / n, w / n)


class Trajectory:
    """Interpolatable map->base pose stream, keyed by integer nanoseconds."""

    def __init__(self):
        self.t = []          # sorted ns
        self.p = []          # (x,y,z)
        self.q = []          # (x,y,z,w)
        self.clamped = 0

    def add(self, t_ns, p, q):
        self.t.append(t_ns)
        self.p.append(p)
        self.q.append(q)

    def T_at(self, t_ns):
        i = bisect.bisect_left(self.t, t_ns)
        if i == 0:
            self.clamped += 1
            return make_T(quat_to_R(*self.q[0]), np.array(self.p[0]))
        if i >= len(self.t):
            self.clamped += 1
            return make_T(quat_to_R(*self.q[-1]), np.array(self.p[-1]))
        t0, t1 = self.t[i - 1], self.t[i]
        if t1 == t0 or t_ns == t0:
            return make_T(quat_to_R(*self.q[i - 1]), np.array(self.p[i - 1]))
        u = (t_ns - t0) / (t1 - t0)
        p0, p1 = np.array(self.p[i - 1]), np.array(self.p[i])
        p = p0 + u * (p1 - p0)
        q = slerp(self.q[i - 1], self.q[i], u)
        return make_T(quat_to_R(*q), p)


# ------------------------------ message decoders -----------------------------
_PF_F32 = 7  # sensor_msgs PointField FLOAT32 datatype id


def cloud_xyz(msg):
    """Extract Nx3 float32 XYZ from a ROS2 PointCloud2 (dropping NaN/inf)."""
    offs = {f.name: (f.offset, f.datatype) for f in msg.fields}
    for k in ("x", "y", "z"):
        if k not in offs or offs[k][1] != _PF_F32:
            raise ValueError(f"cloud field '{k}' missing or not FLOAT32")
    ps = msg.point_step
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, ps)

    def f32(name):
        o = offs[name][0]
        return raw[:, o:o + 4].copy().view("<f4").ravel()

    xyz = np.stack([f32("x"), f32("y"), f32("z")], axis=1).astype(np.float64)
    good = np.isfinite(xyz).all(axis=1)
    return xyz[good]


def image_rgb(msg):
    """Decode a ROS2 sensor_msgs/Image to an HxWx3 uint8 RGB array."""
    enc = msg.encoding
    chan = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(enc)
    if chan is None:
        raise ValueError(f"unsupported image encoding '{enc}' (expected "
                         "rgb8/bgr8/rgba8/bgra8/mono8)")
    rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    img = rows[:, : msg.width * chan].reshape(msg.height, msg.width, chan)
    if enc == "rgb8":
        return np.ascontiguousarray(img)
    if enc == "bgr8":
        return np.ascontiguousarray(img[..., ::-1])
    if enc == "rgba8":
        return np.ascontiguousarray(img[..., :3])
    if enc == "bgra8":
        return np.ascontiguousarray(img[..., [2, 1, 0]])
    return np.repeat(img, 3, axis=2)  # mono8


def intrinsics_from_camera_info(ci):
    """Return (fx, fy, cx, cy, width, height); prefer the rectified P matrix."""
    P = list(ci.p) if hasattr(ci, "p") else list(ci.P)
    K = list(ci.k) if hasattr(ci, "k") else list(ci.K)
    if P and (P[0] != 0.0 and P[5] != 0.0):
        fx, cx, fy, cy = P[0], P[2], P[5], P[6]
    else:
        fx, cx, fy, cy = K[0], K[2], K[4], K[5]
    return fx, fy, cx, cy, ci.width, ci.height


# ---------------------------------- convert ----------------------------------
def convert(args) -> None:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import get_typestore, Stores

    import rosbag
    import rospy
    from sensor_msgs.msg import Image, PointCloud2, PointField
    from geometry_msgs.msg import PoseStamped, Point, Quaternion
    from std_msgs.msg import Header

    ts = get_typestore(Stores.ROS2_HUMBLE)

    # Intrinsic override: some Oxford-Spires bags publish a mis-scaled
    # /cam0/camera_info (e.g. 720x540 fields for a 1440x1080 image). Passing
    # --fx/--fy/--cx/--cy (with --cam-width/--cam-height as their reference
    # resolution) bypasses camera_info entirely; emit() rescales to the actual
    # image size, so the reference resolution need not equal the image size.
    override_intr = args.fx is not None

    want_topics = {args.tf_topic, args.tf_static_topic,
                   args.image_topic, args.cloud_topic}
    if not override_intr:
        want_topics.add(args.camera_info_topic)

    # ---- collected state -----------------------------------------------------
    traj = Trajectory()
    T_base_cam = None       # static
    T_base_lidar = None     # static
    T_cam_base = None       # inv(T_base_cam), static
    intr = None             # (fx, fy, cx, cy, ref_w, ref_h)
    if override_intr:
        intr = (args.fx, args.fy, args.cx, args.cy,
                args.cam_width, args.cam_height)

    def ros_time(ns):
        return rospy.Time(ns // 10**9, ns % 10**9)

    def stamp_ns(h):
        return h.stamp.sec * 10**9 + h.stamp.nanosec

    def build_ros1_image(rb, stamp, frame_id=None):
        m = Image()
        m.header = Header()
        m.header.stamp = stamp
        m.header.frame_id = frame_id if frame_id is not None else rb.header.frame_id
        m.height, m.width = rb.height, rb.width
        m.encoding = rb.encoding
        m.is_bigendian = rb.is_bigendian
        m.step = rb.step
        m.data = bytes(rb.data)
        return m

    def depth_to_msg(depth, stamp, frame_id):
        m = Image()
        m.header = Header()
        m.header.stamp = stamp
        m.header.frame_id = frame_id
        m.height, m.width = depth.shape
        m.encoding = "32FC1"
        m.is_bigendian = 0
        m.step = m.width * 4
        m.data = depth.astype("<f4").tobytes()
        return m

    def points_to_msg(xyz_map, rgb, stamp, frame_id):
        n = xyz_map.shape[0]
        pts = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"),
                                 ("z", "<f4"), ("rgb", "<f4")])
        pts["x"], pts["y"], pts["z"] = xyz_map[:, 0], xyz_map[:, 1], xyz_map[:, 2]
        rgb_u32 = ((rgb[:, 0].astype(np.uint32) << 16)
                   | (rgb[:, 1].astype(np.uint32) << 8)
                   | rgb[:, 2].astype(np.uint32))
        pts["rgb"] = rgb_u32.view("<f4")
        m = PointCloud2()
        m.header = Header()
        m.header.stamp = stamp
        m.header.frame_id = frame_id
        m.height, m.width = 1, n
        for name, off in (("x", 0), ("y", 4), ("z", 8), ("rgb", 12)):
            f = PointField()
            f.name, f.offset, f.datatype, f.count = name, off, PointField.FLOAT32, 1
            m.fields.append(f)
        m.is_bigendian = False
        m.point_step = 16
        m.row_step = 16 * n
        m.data = pts.tobytes()
        m.is_dense = True
        return m

    def pose_to_msg(T, stamp, frame_id):
        m = PoseStamped()
        m.header = Header()
        m.header.stamp = stamp
        m.header.frame_id = frame_id
        m.pose.position = Point(float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
        qx, qy, qz, qw = R_to_quat(T[:3, :3])
        m.pose.orientation = Quaternion(float(qx), float(qy), float(qz), float(qw))
        return m

    compression = {
        "none": rosbag.Compression.NONE,
        "lz4": rosbag.Compression.LZ4,
        "bz2": rosbag.Compression.BZ2,
    }[args.compression]

    written = 0
    emitted = 0
    skipped_no_cloud = 0
    max_frames = args.max_frames if args.max_frames > 0 else None

    with Reader(args.input) as reader, \
            rosbag.Bag(args.output, "w", compression=compression) as writer:

        conns = [c for c in reader.connections if c.topic in want_topics]
        present = {c.topic for c in conns}
        missing = want_topics - present
        if missing:
            print(f"ERROR: topics not found in MCAP: {sorted(missing)}")
            print("Available:")
            for c in reader.connections:
                print(f"  {c.topic}  ({c.msgtype})")
            sys.exit(1)

        # emit(): build + write one keyframe quad for (image rb @ t_i, cloud @ t_l)
        def emit(img_rb, t_i_ns, img_record_ns, rgb_img, cloud_lidar, t_l_ns):
            nonlocal written, emitted
            fx, fy, cx, cy, ci_w, ci_h = intr
            H, W = rgb_img.shape[:2]
            # scale intrinsics if camera_info res differs from the actual image
            if (ci_w, ci_h) != (W, H) and ci_w and ci_h:
                sx, sy = W / ci_w, H / ci_h
                fxs, fys, cxs, cys = fx * sx, fy * sy, cx * sx, cy * sy
            else:
                fxs, fys, cxs, cys = fx, fy, cx, cy

            stamp = ros_time(t_i_ns)
            rec = ros_time(img_record_ns)

            # trajectory at cloud time (t_l) and image time (t_i)
            T_mb_l = traj.T_at(t_l_ns)
            T_mb_i = traj.T_at(t_i_ns)

            # lidar -> map (world) at scan time
            P_base = (T_base_lidar[:3, :3] @ cloud_lidar.T).T + T_base_lidar[:3, 3]
            P_map = (T_mb_l[:3, :3] @ P_base.T).T + T_mb_l[:3, 3]

            # map -> cam0 at image time:  T_cam_base . inv(T_map_base(t_i))
            T_cam_map = T_cam_base @ inv_T(T_mb_i)
            P_cam = (T_cam_map[:3, :3] @ P_map.T).T + T_cam_map[:3, 3]

            Z = P_cam[:, 2]
            valid = Z > 1e-6
            if args.max_range > 0:
                valid &= Z <= args.max_range
            u = np.zeros_like(Z)
            v = np.zeros_like(Z)
            u[valid] = fxs * P_cam[valid, 0] / Z[valid] + cxs
            v[valid] = fys * P_cam[valid, 1] / Z[valid] + cys
            ui = np.round(u).astype(np.int64)
            vi = np.round(v).astype(np.int64)
            valid &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

            if not valid.any():
                return  # nothing projects into this camera; drop the frame

            ui, vi, Zv = ui[valid], vi[valid], Z[valid]
            P_map_v = P_map[valid]

            # depth image (z-buffer: nearest wins), 0 = no measurement
            flat = vi * W + ui
            zbuf = np.full(H * W, np.inf, dtype=np.float64)
            np.minimum.at(zbuf, flat, Zv)
            depth = np.where(np.isfinite(zbuf), zbuf, 0.0).reshape(H, W).astype(np.float32)

            # colourise the in-FOV world points from the image pixel
            colors = rgb_img[vi, ui]  # Nx3 uint8 RGB

            writer.write("/pose_for_gs",
                         pose_to_msg(T_mb_i @ T_base_cam, stamp, args.map_frame), rec)
            writer.write("/points_for_gs",
                         points_to_msg(P_map_v, colors, stamp, args.map_frame), rec)
            writer.write("/depth_for_gs",
                         depth_to_msg(depth, stamp, args.cam_frame), rec)
            writer.write("/image_for_gs",
                         build_ros1_image(img_rb, stamp), rec)
            written += 4
            emitted += 1

        # ---- streaming pass: buffer TF/extrinsics/intrinsics; merge-join -----
        # image is master, matched to nearest cloud within tol (1-cloud lookahead)
        tol_ns = int(args.tol * 1e9)
        prev_cloud = None   # (t_l_ns, xyz)
        img_pending = deque()  # (t_i_ns, record_ns, img_rb, rgb_img)

        def resolve(cur_cloud):
            nonlocal skipped_no_cloud
            # match every pending image whose stamp <= cur cloud stamp
            while img_pending and (cur_cloud is None
                                   or img_pending[0][0] <= cur_cloud[0]):
                t_i, rec_ns, img_rb, rgb = img_pending.popleft()
                cands = [c for c in (prev_cloud, cur_cloud) if c is not None]
                if not cands:
                    skipped_no_cloud += 1
                    continue
                cl = min(cands, key=lambda c: abs(c[0] - t_i))
                if abs(cl[0] - t_i) > tol_ns:
                    skipped_no_cloud += 1
                    continue
                emit(img_rb, t_i, rec_ns, rgb, cl[1], cl[0])
                if max_frames and emitted >= max_frames:
                    return True
            return False

        stop = False
        for conn, timestamp, rawdata in reader.messages(connections=conns):
            topic = conn.topic
            if topic == args.tf_topic:
                m = ts.deserialize_cdr(rawdata, conn.msgtype)
                for tr in m.transforms:
                    if (tr.header.frame_id == args.map_frame
                            and tr.child_frame_id == args.base_frame):
                        q = tr.transform.rotation
                        tl = tr.transform.translation
                        traj.add(stamp_ns(tr.header),
                                 (tl.x, tl.y, tl.z), (q.x, q.y, q.z, q.w))
            elif topic == args.tf_static_topic:
                m = ts.deserialize_cdr(rawdata, conn.msgtype)
                for tr in m.transforms:
                    q = tr.transform.rotation
                    tl = tr.transform.translation
                    T = make_T(quat_to_R(q.x, q.y, q.z, q.w),
                               np.array([tl.x, tl.y, tl.z]))
                    if (tr.header.frame_id == args.base_frame
                            and tr.child_frame_id == args.cam_frame):
                        T_base_cam = T
                        T_cam_base = inv_T(T)
                    elif (tr.header.frame_id == args.base_frame
                          and tr.child_frame_id == args.lidar_frame):
                        T_base_lidar = T
            elif topic == args.camera_info_topic:
                if intr is None:
                    ci = ts.deserialize_cdr(rawdata, conn.msgtype)
                    intr = intrinsics_from_camera_info(ci)
            elif topic == args.cloud_topic:
                # need static extrinsics + intrinsics before we can project;
                # the trajectory may still be partial (T_at clamps early times)
                if T_base_cam is None or T_base_lidar is None or intr is None:
                    continue
                m = ts.deserialize_cdr(rawdata, conn.msgtype)
                cur = (stamp_ns(m.header), cloud_xyz(m))
                stop = resolve(cur)
                prev_cloud = cur
                if stop:
                    break
            elif topic == args.image_topic:
                if T_base_cam is None or T_base_lidar is None or intr is None:
                    continue  # cannot use images before extrinsics/intrinsics
                m = ts.deserialize_cdr(rawdata, conn.msgtype)
                img_pending.append((stamp_ns(m.header), timestamp, m, image_rgb(m)))

            if emitted and emitted % 100 == 0:
                print(f"  emitted {emitted} keyframes "
                      f"(skipped {skipped_no_cloud} unmatched images)", end="\r",
                      flush=True)

        if not stop:
            resolve(None)  # flush trailing images against the last cloud

    if T_base_cam is None or T_base_lidar is None:
        print("ERROR: required static transforms not found "
              f"({args.base_frame}->{args.cam_frame}, {args.base_frame}->{args.lidar_frame}).")
        sys.exit(1)

    print(f"\nDone. {emitted} keyframes -> {written} messages -> {args.output}")
    if skipped_no_cloud:
        print(f"  {skipped_no_cloud} images had no cloud within {args.tol}s and were skipped.")
    if traj.clamped:
        print(f"  {traj.clamped} pose lookups fell outside the /tf range (clamped).")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Synthesise Gaussian-LIC2 pre-posed streams from an "
                    "Oxford-Spires MCAP (TF chain + LiDAR->camera projection).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="input .mcap (ROS2)")
    p.add_argument("output", help="output .bag (ROS1, pre-posed)")
    p.add_argument("--image-topic", default="/cam0/image_rect")
    p.add_argument("--cloud-topic", default="/points_deskewed")
    p.add_argument("--camera-info-topic", default="/cam0/camera_info")
    p.add_argument("--tf-topic", default="/tf")
    p.add_argument("--tf-static-topic", default="/tf_static")
    p.add_argument("--map-frame", default="map")
    p.add_argument("--base-frame", default="base")
    p.add_argument("--cam-frame", default="cam0")
    p.add_argument("--lidar-frame", default="lidar")
    p.add_argument("--tol", type=float, default=0.05,
                   help="max |t_image - t_cloud| (s) to pair a scan with an image")
    p.add_argument("--max-range", type=float, default=100.0,
                   help="drop LiDAR points beyond this range (m); 0 = no limit")
    p.add_argument("--compression", choices=["none", "lz4", "bz2"], default="lz4",
                   help="output bag compression (depth is sparse -> compresses well)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after N keyframes (0 = all); use for a quick test")
    # Intrinsic override (bypass /cam0/camera_info, which can be mis-scaled).
    # Give all four of fx/fy/cx/cy; --cam-width/--cam-height are their reference
    # resolution (defaults to the native rectified 1440x1080). emit() rescales
    # to the real image size, so these need not match the image resolution.
    p.add_argument("--fx", type=float, help="override intrinsic fx (skip camera_info)")
    p.add_argument("--fy", type=float, help="override intrinsic fy")
    p.add_argument("--cx", type=float, help="override intrinsic cx")
    p.add_argument("--cy", type=float, help="override intrinsic cy")
    p.add_argument("--cam-width", type=int, default=1440,
                   help="reference width the override fx/cx are given at")
    p.add_argument("--cam-height", type=int, default=1080,
                   help="reference height the override fy/cy are given at")
    args = p.parse_args()

    over = [args.fx, args.fy, args.cx, args.cy]
    if any(v is not None for v in over) and any(v is None for v in over):
        print("ERROR: --fx/--fy/--cx/--cy must be given together (all four).")
        sys.exit(1)

    if not Path(args.input).exists():
        print(f"ERROR: {args.input} not found")
        sys.exit(1)
    if Path(args.output).exists():
        print(f"ERROR: {args.output} already exists — delete it first")
        sys.exit(1)

    convert(args)


if __name__ == "__main__":
    main()

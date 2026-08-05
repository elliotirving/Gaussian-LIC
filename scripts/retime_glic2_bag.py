#!/usr/bin/env python3
"""
Rewrite a pre-posed Gaussian-LIC/GLIC2 ROS1 bag onto a fixed frame cadence.

This is for bags that already contain the four direct GLIC inputs:

  /pose_for_gs    geometry_msgs/PoseStamped
  /points_for_gs  sensor_msgs/PointCloud2
  /image_for_gs   sensor_msgs/Image
  /depth_for_gs   sensor_msgs/Image

The script groups one message from each topic by their original header stamps,
then rewrites the output timeline. In the default `fixed-rate` mode, complete
frames are written at a uniform timestamp, defaulting to 10 Hz. In `header` mode,
the original header-stamp timeline is preserved and only the bag record time is
made equal to the header time.

Run inside the ROS/Gaussian-LIC container with ROS/system Python:

  /usr/bin/python3 scripts/retime_glic2_bag.py IN.bag OUT_10hz.bag --fps 10
  /usr/bin/python3 scripts/retime_glic2_bag.py IN.bag OUT_header.bag --timeline header
"""

import argparse
import sys


DEFAULT_POSE_TOPIC = "/pose_for_gs"
DEFAULT_POINTS_TOPIC = "/points_for_gs"
DEFAULT_IMAGE_TOPIC = "/image_for_gs"
DEFAULT_DEPTH_TOPIC = "/depth_for_gs"


def stamp_to_sec(msg, topic):
    if not hasattr(msg, "header") or not hasattr(msg.header, "stamp"):
        raise ValueError(f"{topic} message has no header.stamp")
    return msg.header.stamp.to_sec()


def find_group(pending, topic, stamp, tolerance):
    best = None
    best_delta = None
    for idx, group in enumerate(pending):
        if topic in group["msgs"]:
            continue
        delta = abs(group["stamp"] - stamp)
        if delta <= tolerance and (best_delta is None or delta < best_delta):
            best = idx
            best_delta = delta
    return best


def output_stamp_for_group(group, rospy, args, state):
    if state["base_stamp"] is None:
        state["first_original_stamp"] = group["stamp"]
        state["base_stamp"] = (
            rospy.Time.from_sec(args.start_time)
            if args.start_time is not None
            else rospy.Time.from_sec(group["stamp"])
        )

    if args.timeline == "fixed-rate":
        offset_seconds = state["frames_written"] / args.fps
    else:
        offset_seconds = group["stamp"] - state["first_original_stamp"]

    return state["base_stamp"] + rospy.Duration.from_sec(offset_seconds)


def write_ready_frames(outbag, pending, topics, rospy, args, state):
    while pending and all(topic in pending[0]["msgs"] for topic in topics):
        group = pending.pop(0)
        new_stamp = output_stamp_for_group(group, rospy, args, state)

        for topic in topics:
            msg = group["msgs"][topic]
            msg.header.stamp = new_stamp
            outbag.write(topic, msg, new_stamp)

        if state["last_original_stamp"] is not None:
            state["max_original_inter_frame_gap"] = max(
                state["max_original_inter_frame_gap"],
                group["stamp"] - state["last_original_stamp"],
            )
        state["last_original_stamp"] = group["stamp"]
        state["last_output_stamp"] = new_stamp.to_sec()
        state["frames_written"] += 1
        state["max_original_span"] = max(
            state["max_original_span"],
            group["max_stamp"] - group["min_stamp"],
        )


def retime_bag(args):
    try:
        import rosbag
        import rospy
    except ModuleNotFoundError as exc:
        if exc.name == "yaml":
            raise RuntimeError(
                "ROS rosbag could not import PyYAML. This usually means Conda's "
                f"Python is active ({sys.executable}) while ROS Noetic expects "
                "the system Python. Re-run with `/usr/bin/python3 "
                "scripts/retime_glic2_bag.py ...`, or install PyYAML into the "
                "active interpreter."
            ) from exc
        raise

    topics = [
        args.pose_topic,
        args.points_topic,
        args.image_topic,
        args.depth_topic,
    ]
    topic_set = set(topics)
    counts = {topic: 0 for topic in topics}
    pending = []
    state = {
        "frames_written": 0,
        "max_original_span": 0.0,
        "max_pending": 0,
        "base_stamp": None,
        "first_original_stamp": None,
        "last_original_stamp": None,
        "last_output_stamp": None,
        "max_original_inter_frame_gap": 0.0,
    }

    with rosbag.Bag(args.output_bag, "w") as outbag:
        with rosbag.Bag(args.input_bag, "r") as inbag:
            for topic, msg, _record_time in inbag.read_messages(topics=topics):
                if topic not in topic_set:
                    continue

                stamp = stamp_to_sec(msg, topic)
                group_idx = find_group(pending, topic, stamp, args.sync_tolerance)

                if group_idx is None:
                    pending.append({
                        "stamp": stamp,
                        "min_stamp": stamp,
                        "max_stamp": stamp,
                        "msgs": {topic: msg},
                    })
                else:
                    group = pending[group_idx]
                    group["msgs"][topic] = msg
                    group["min_stamp"] = min(group["min_stamp"], stamp)
                    group["max_stamp"] = max(group["max_stamp"], stamp)

                counts[topic] += 1
                pending.sort(key=lambda group: group["stamp"])
                state["max_pending"] = max(state["max_pending"], len(pending))

                if len(pending) > args.max_pending_frames:
                    oldest = pending[0]
                    missing = [name for name in topics if name not in oldest["msgs"]]
                    raise RuntimeError(
                        "too many incomplete pending frames; oldest frame is "
                        f"missing {missing}. Increase --sync-tolerance only if "
                        "the four messages really should be associated."
                    )

                write_ready_frames(outbag, pending, topics, rospy, args, state)

        complete_pending = [
            group for group in pending if all(topic in group["msgs"] for topic in topics)
        ]
        incomplete_pending = [
            group for group in pending if not all(topic in group["msgs"] for topic in topics)
        ]

        if incomplete_pending and not args.drop_incomplete:
            examples = []
            for group in incomplete_pending[:5]:
                missing = [topic for topic in topics if topic not in group["msgs"]]
                examples.append(f"stamp={group['stamp']:.6f} missing={missing}")
            raise RuntimeError(
                "input ended with incomplete synchronized frames. "
                "Use --drop-incomplete to discard them if this is expected. "
                + "; ".join(examples)
            )

        if complete_pending:
            pending[:] = sorted(complete_pending, key=lambda group: group["stamp"])
            write_ready_frames(outbag, pending, topics, rospy, args, state)

        if incomplete_pending:
            print(
                f"dropped {len(incomplete_pending)} incomplete frame group(s)",
                file=sys.stderr,
            )

    output_duration = 0.0
    original_header_span = 0.0
    if state["frames_written"] > 1:
        output_duration = state["last_output_stamp"] - state["base_stamp"].to_sec()
        original_header_span = (
            state["last_original_stamp"] - state["first_original_stamp"]
        )

    nominal_slots = int(round(original_header_span * args.fps)) + 1
    estimated_missing_slots = max(0, nominal_slots - state["frames_written"])

    print(f"timeline mode: {args.timeline}")
    print(f"wrote {state['frames_written']} complete frame groups")
    print(f"nominal fps: {args.fps:g}")
    print(f"output duration: {output_duration:.3f} s")
    print(f"original header span: {original_header_span:.3f} s")
    print(f"largest original header gap: {state['max_original_inter_frame_gap']:.6f} s")
    print(f"estimated missing nominal frame slots: {estimated_missing_slots}")
    print(f"max original within-frame stamp span: {state['max_original_span']:.6f} s")
    print(f"max pending synchronized frames: {state['max_pending']}")
    print("input topic counts:")
    for topic in topics:
        print(f"  {topic}: {counts[topic]}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Retimestamp a pre-posed Gaussian-LIC ROS1 bag to fixed FPS."
    )
    parser.add_argument("input_bag")
    parser.add_argument("output_bag")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--timeline", choices=("fixed-rate", "header"),
                        default="header",
                        help="fixed-rate compacts retained frames to --fps; "
                             "header preserves original header-stamp intervals")
    parser.add_argument("--sync-tolerance", type=float, default=0.01)
    parser.add_argument("--start-time", type=float, default=None,
                        help="new first timestamp in seconds; default keeps the first original frame stamp")
    parser.add_argument("--max-pending-frames", type=int, default=200)
    parser.add_argument("--drop-incomplete", action="store_true",
                        help="discard incomplete trailing frames instead of failing")
    parser.add_argument("--pose-topic", default=DEFAULT_POSE_TOPIC)
    parser.add_argument("--points-topic", default=DEFAULT_POINTS_TOPIC)
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC)
    args = parser.parse_args(argv)

    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.sync_tolerance < 0:
        parser.error("--sync-tolerance must be non-negative")
    if args.max_pending_frames <= 0:
        parser.error("--max-pending-frames must be positive")
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        retime_bag(args)
    except Exception as exc:
        print(f"retime_glic2_bag.py: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

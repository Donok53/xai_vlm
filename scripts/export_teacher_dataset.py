#!/usr/bin/env python3
import argparse
import json
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as point_cloud2
from cv_bridge import CvBridge

from export_camera_only_teacher_dataset import summarize_flow


ALLOWED_LABELS_KO = [
    "사람",
    "자전거",
    "자동차",
    "오토바이",
    "기차",
    "트럭",
    "신호등",
    "소화전",
    "정지 표지판",
    "주차 미터기",
    "벤치",
    "고양이",
    "개",
    "가방",
    "우산",
    "손가방",
    "캐리어",
    "공",
    "병",
    "컵",
    "와인잔",
    "책",
    "시계",
    "로봇",
    "쓰레기통",
    "나무",
    "안전봉",
    "문",
    "벽",
]

INDOOR_LABELS_KO = [
    "사람",
    "우산",
    "가방",
    "손가방",
    "캐리어",
    "병",
    "컵",
    "책",
    "시계",
    "벤치",
    "로봇",
    "쓰레기통",
    "문",
    "벽",
]

OUTDOOR_LABELS_KO = [
    "사람",
    "자전거",
    "자동차",
    "오토바이",
    "기차",
    "트럭",
    "신호등",
    "소화전",
    "정지 표지판",
    "주차 미터기",
    "벤치",
    "고양이",
    "개",
    "가방",
    "우산",
    "손가방",
    "캐리어",
    "병",
    "컵",
    "나무",
    "안전봉",
    "벽",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export offline VLM teacher dataset from ROS bag."
    )
    parser.add_argument("--bag", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--planner-topic", default="/xai/planner_snapshot")
    parser.add_argument("--event-topic", default="/xai/event_log")
    parser.add_argument(
        "--point-cloud-topic",
        default="/planning/linefit_ground/non_ground_cloud",
    )
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--max-image-age-s", type=float, default=0.25)
    parser.add_argument("--max-planner-age-s", type=float, default=0.75)
    parser.add_argument("--max-pointcloud-age-s", type=float, default=0.40)
    parser.add_argument("--max-pointcloud-points", type=int, default=2500)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--flow-image-side-px", type=int, default=320)
    parser.add_argument("--flow-motion-threshold", type=float, default=1.5)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def safe_json_loads(raw_text):
    try:
        return json.loads(raw_text)
    except Exception:
        return {}


def stamp_to_float(value, fallback=None):
    if value is None:
        return fallback
    try:
        return float(value)
    except Exception:
        return fallback


def quaternion_to_yaw(orientation):
    x = float(getattr(orientation, "x", 0.0))
    y = float(getattr(orientation, "y", 0.0))
    z = float(getattr(orientation, "z", 0.0))
    w = float(getattr(orientation, "w", 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    value = float(angle)
    while value > math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


def summarize_actual_motion(odom_window):
    if len(odom_window) < 2:
        return {
            "received": False,
            "motion_ko": "실제 이동 정보 없음",
            "linear_speed_mps": 0.0,
            "yaw_rate_radps": 0.0,
            "distance_m": 0.0,
        }

    first = odom_window[0]
    last = odom_window[-1]
    dt = max(1e-3, float(last["stamp"] - first["stamp"]))
    dx = float(last["x"] - first["x"])
    dy = float(last["y"] - first["y"])
    distance = math.hypot(dx, dy)
    linear_speed = distance / dt
    yaw_delta = normalize_angle(float(last["yaw"] - first["yaw"]))
    yaw_rate = yaw_delta / dt

    stopped = linear_speed < 0.008 and abs(yaw_rate) < 0.03
    if stopped:
        motion_ko = "정지"
    else:
        forward_component = dx
        if forward_component > 0.015:
            if yaw_rate > 0.05:
                motion_ko = "전진 좌회전"
            elif yaw_rate < -0.05:
                motion_ko = "전진 우회전"
            else:
                motion_ko = "전진"
        elif forward_component < -0.015:
            if yaw_rate > 0.05:
                motion_ko = "후진 좌회전"
            elif yaw_rate < -0.05:
                motion_ko = "후진 우회전"
            else:
                motion_ko = "후진"
        else:
            if yaw_rate > 0.05:
                motion_ko = "좌회전"
            elif yaw_rate < -0.05:
                motion_ko = "우회전"
            else:
                motion_ko = "정지"

    return {
        "received": True,
        "motion_ko": motion_ko,
        "linear_speed_mps": linear_speed,
        "yaw_rate_radps": yaw_rate,
        "distance_m": distance,
    }


def sample_point_cloud(msg, max_points):
    points = []
    width = int(getattr(msg, "width", 0) or 0)
    height = int(getattr(msg, "height", 0) or 0)
    estimated = max(1, width * max(1, height))
    stride = max(1, int(math.ceil(float(estimated) / float(max(1, max_points)))))
    for index, point_xyz in enumerate(
        point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )
    ):
        if (index % stride) != 0:
            continue
        points.append((float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])))
    return np.asarray(points, dtype=np.float32)


def obstacle_summary(event_data):
    obstacle = (event_data or {}).get("obstacle_evidence") or {}
    near_raw = obstacle.get("near_field_raw_overlay_hits") or {}
    centroid = near_raw.get("sample_centroid") or {}
    summary = {
        "near_raw_points": int(near_raw.get("reported_points") or 0),
        "near_raw_min_range_m": near_raw.get("min_range_m"),
        "near_raw_min_x_m": near_raw.get("min_x_m"),
        "near_raw_centroid_xyz": {
            "x": centroid.get("x"),
            "y": centroid.get("y"),
            "z": centroid.get("z"),
        },
    }
    return summary


def planner_reason(planner_snapshot, event_data):
    decision = ((event_data or {}).get("decision") or {})
    behavior = decision.get("behavior") or {}
    reason = behavior.get("reason")
    if reason:
        return str(reason)
    decision = ((planner_snapshot or {}).get("decision") or {})
    behavior = decision.get("behavior") or {}
    reason = behavior.get("reason")
    if reason:
        return str(reason)
    return "unknown"


def motion_state(planner_snapshot, event_data):
    control = ((event_data or {}).get("control") or {})
    state = control.get("motion_state")
    if state:
        return str(state)
    control = ((planner_snapshot or {}).get("control") or {})
    state = control.get("motion_state")
    if state:
        return str(state)
    return "unknown"


def control_summary(planner_snapshot, event_data):
    control = ((event_data or {}).get("control") or {})
    if not control:
        control = ((planner_snapshot or {}).get("control") or {})
    return {
        "linear_x_mps": float(control.get("linear_x_mps") or 0.0),
        "angular_z_radps": float(control.get("angular_z_radps") or 0.0),
        "motion_state": str(control.get("motion_state") or "unknown"),
        "steering_direction": str(control.get("steering_direction") or "unknown"),
        "received": bool(control.get("received")),
        "topic": str(control.get("topic") or "/cmd_vel"),
    }


def planning_summary(planner_snapshot, event_data):
    source = event_data or {}
    planning = source.get("planning") or {}
    decision = source.get("decision") or {}
    if not planning and planner_snapshot:
        planning = (planner_snapshot.get("planning") or {})
        decision = (planner_snapshot.get("decision") or decision)

    path_change = planning.get("path_change") or {}
    latest = path_change.get("latest") or {}
    global_path = planning.get("global_path") or {}
    behavior = decision.get("behavior") or {}
    path_blocked = (decision.get("path_blocked") or {}).get("value")

    return {
        "behavior_reason": str(behavior.get("reason") or "unknown"),
        "behavior_stop": bool(behavior.get("stop")),
        "speed_limit_mps": float(behavior.get("speed_limit_mps") or 0.0),
        "path_blocked": bool(path_blocked),
        "path_change_seq": int(path_change.get("seq") or 0),
        "path_change_changed": bool(latest.get("changed")),
        "path_change_direction": str(latest.get("direction") or "unknown"),
        "path_change_lateral_shift_m": float(latest.get("lateral_shift_m") or 0.0),
        "global_path_length_m": float(global_path.get("length_m") or 0.0),
        "global_path_points": int(global_path.get("points") or 0),
    }


def build_teacher_prompt(sample):
    event_label = sample.get("event_label", "unknown")
    planner_reason_text = sample.get("planner_reason", "unknown")
    motion_state_text = sample.get("motion_state", "unknown")
    obstacle = sample.get("obstacle_summary") or {}
    control = sample.get("control_summary") or {}
    planning = sample.get("planning_summary") or {}
    motion = sample.get("motion_summary") or {}
    actual_motion = sample.get("actual_motion_summary") or {}
    near_range = obstacle.get("near_raw_min_range_m")
    near_centroid = obstacle.get("near_raw_centroid_xyz") or {}
    near_centroid_text = json.dumps(near_centroid, ensure_ascii=False)
    command_vs_actual = (
        "명령 대비 실제 이동 없음"
        if (
            (float(control.get("linear_x_mps") or 0.0) > 0.02 or abs(float(control.get("angular_z_radps") or 0.0)) > 0.05)
            and actual_motion.get("received")
            and str(actual_motion.get("motion_ko") or "") == "정지"
        )
        else "일치 또는 정보 부족"
    )

    prompt_lines = [
        "너는 자율주행 XAI teacher 모델이다.",
        "세 장의 연속 카메라 프레임(prev, current, next)과 planner/LiDAR/cmd_vel/odom 문맥을 함께 보고, 현재 장면에서 왜 그런 주행 판단이 나왔는지 설명하라.",
        "목표는 planner/LiDAR 맥락과 가장 관련 있는 대표 시각 객체를 1개 고르고, 현재 주행 판단의 시각적 이유를 짧게 정리하는 것이다.",
        "설명은 카메라에 실제로 보이는 것과 제공된 planner/LiDAR/cmd_vel 문맥을 함께 사용하되, planner의 reason 문장을 그대로 복사하지 마라.",
        "",
        "주행 문맥:",
        "- event_label: {}".format(event_label),
        "- planner_reason: {}".format(planner_reason_text),
        "- motion_state: {}".format(motion_state_text),
        "- cmd_vel.linear_x_mps: {:.3f}".format(float(control.get("linear_x_mps") or 0.0)),
        "- cmd_vel.angular_z_radps: {:.3f}".format(float(control.get("angular_z_radps") or 0.0)),
        "- actual_motion.motion_ko: {}".format(actual_motion.get("motion_ko") or "실제 이동 정보 없음"),
        "- actual_motion.linear_speed_mps: {:.3f}".format(float(actual_motion.get("linear_speed_mps") or 0.0)),
        "- actual_motion.yaw_rate_radps: {:.3f}".format(float(actual_motion.get("yaw_rate_radps") or 0.0)),
        "- command_vs_actual: {}".format(command_vs_actual),
        "- steering_direction: {}".format(control.get("steering_direction") or "unknown"),
        "- path_blocked: {}".format(bool(planning.get("path_blocked"))),
        "- path_change_changed: {}".format(bool(planning.get("path_change_changed"))),
        "- path_change_direction: {}".format(planning.get("path_change_direction") or "unknown"),
        "- path_change_lateral_shift_m: {:.3f}".format(float(planning.get("path_change_lateral_shift_m") or 0.0)),
        "- global_path_length_m: {:.3f}".format(float(planning.get("global_path_length_m") or 0.0)),
        "- global_path_points: {}".format(int(planning.get("global_path_points") or 0)),
        "- near_raw_min_range_m: {}".format(near_range),
        "- near_raw_centroid_xyz: {}".format(near_centroid_text),
        "- camera_motion_ego: {}".format(motion.get("ego_motion_ko") or "정지"),
        "- camera_motion_scene_state: {}".format(motion.get("scene_state_ko") or "정적/동적 혼합"),
        "",
        "허용 라벨 후보:",
        "- {}".format(", ".join(ALLOWED_LABELS_KO)),
        "- 실내 후보: {}".format(", ".join(INDOOR_LABELS_KO)),
        "- 실외 후보: {}".format(", ".join(OUTDOOR_LABELS_KO)),
        "",
        "규칙:",
        "- 먼저 현재 장면이 실내인지 실외인지 판단한다.",
        "- 실내면 실내 후보 위주로, 실외면 실외 후보 위주로 대표 객체를 고른다.",
        "- 허용 라벨에 없으면 primary_object_ko는 반드시 '벽'으로 답한다.",
        "- current 이미지에서 가장 관련 있는 대표 객체 1개만 고른다.",
        "- driving_reason_ko는 왜 감속/조향/우회/정지/직진 같은 주행이 나왔는지 짧게 설명한다.",
        "- dynamic은 static, dynamic, unknown 중 하나로 답한다.",
        "- 반드시 JSON만 출력한다.",
        "",
        "출력 형식:",
        '{'
        '"scene_domain_ko":"실내|실외|불명",'
        '"primary_object_ko":"",'
        '"dynamic":"static|dynamic|unknown",'
        '"scene_summary_ko":"",'
        '"driving_reason_ko":"",'
        '"confidence":0.0'
        '}',
    ]
    return "\n".join(prompt_lines)


def ensure_dirs(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "pointcloud").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)
    (output_dir / "annotations").mkdir(exist_ok=True)


def main():
    args = parse_args()
    bag_paths = [Path(raw).expanduser().resolve() for raw in args.bag]
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dirs(output_dir)

    bridge = CvBridge()
    exported = 0
    skipped = 0

    metadata_path = output_dir / "metadata" / "teacher_dataset.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for bag_path in bag_paths:
            latest_image = None
            recent_images = deque(maxlen=3)
            latest_planner = None
            latest_cloud = None
            odom_window = deque(maxlen=20)
            bag_stem = bag_path.stem

            with rosbag.Bag(str(bag_path)) as bag:
                for topic, msg, bag_stamp in bag.read_messages(
                    topics=[
                        args.image_topic,
                        args.planner_topic,
                        args.event_topic,
                        args.point_cloud_topic,
                        args.cmd_vel_topic,
                        args.odom_topic,
                    ]
                ):
                    if topic == args.image_topic:
                        latest_image = {
                            "stamp": float(msg.header.stamp.to_sec()),
                            "msg": msg,
                        }
                        recent_images.append(latest_image)
                        continue

                    if topic == args.planner_topic:
                        snapshot = safe_json_loads(msg.data)
                        latest_planner = {
                            "stamp": stamp_to_float(snapshot.get("stamp"), float(bag_stamp.to_sec())),
                            "data": snapshot,
                        }
                        continue

                    if topic == args.point_cloud_topic:
                        latest_cloud = {
                            "stamp": float(msg.header.stamp.to_sec()),
                            "frame_id": str(msg.header.frame_id or ""),
                            "msg": msg,
                        }
                        continue

                    if topic == args.odom_topic:
                        odom_window.append(
                            {
                                "stamp": float(msg.header.stamp.to_sec()),
                                "x": float(msg.pose.pose.position.x),
                                "y": float(msg.pose.pose.position.y),
                                "yaw": quaternion_to_yaw(msg.pose.pose.orientation),
                            }
                        )
                        continue

                    if topic == args.cmd_vel_topic:
                        continue

                    if topic != args.event_topic:
                        continue

                    event_data = safe_json_loads(msg.data)
                    event_stamp = stamp_to_float(event_data.get("stamp"), float(bag_stamp.to_sec()))

                    if latest_image is None:
                        skipped += 1
                        continue
                    if abs(event_stamp - latest_image["stamp"]) > float(args.max_image_age_s):
                        skipped += 1
                        continue
                    if latest_planner is None or abs(event_stamp - latest_planner["stamp"]) > float(args.max_planner_age_s):
                        skipped += 1
                        continue

                    sample_id = "sample_{:05d}".format(exported)
                    image_rel_paths = []
                    if len(recent_images) >= 3:
                        triplet = [recent_images[0], recent_images[1], recent_images[2]]
                    else:
                        triplet = [latest_image, latest_image, latest_image]
                    for suffix, image_item in zip(["prev", "current", "next"], triplet):
                        rel_path = Path("images") / "{}_{}.jpg".format(sample_id, suffix)
                        abs_path = output_dir / rel_path
                        image_item_bgr = bridge.imgmsg_to_cv2(image_item["msg"], desired_encoding="bgr8")
                        cv2.imwrite(
                            str(abs_path),
                            image_item_bgr,
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
                        )
                        image_rel_paths.append(str(rel_path))

                    motion_summary = summarize_flow(
                        bridge.imgmsg_to_cv2(triplet[0]["msg"], desired_encoding="bgr8"),
                        bridge.imgmsg_to_cv2(triplet[1]["msg"], desired_encoding="bgr8"),
                        bridge.imgmsg_to_cv2(triplet[2]["msg"], desired_encoding="bgr8"),
                        args.flow_image_side_px,
                        args.flow_motion_threshold,
                    )
                    actual_motion_summary = summarize_actual_motion(odom_window)

                    pointcloud_relpath = None
                    pointcloud_summary = {
                        "frame_id": None,
                        "stamp": None,
                        "point_count": 0,
                    }
                    if latest_cloud is not None and abs(event_stamp - latest_cloud["stamp"]) <= float(args.max_pointcloud_age_s):
                        points_xyz = sample_point_cloud(latest_cloud["msg"], args.max_pointcloud_points)
                        pointcloud_path = output_dir / "pointcloud" / "{}.npz".format(sample_id)
                        np.savez_compressed(pointcloud_path, points_xyz=points_xyz)
                        pointcloud_relpath = str(pointcloud_path.relative_to(output_dir))
                        pointcloud_summary = {
                            "frame_id": latest_cloud["frame_id"],
                            "stamp": latest_cloud["stamp"],
                            "point_count": int(points_xyz.shape[0]),
                        }

                    planner_snapshot = latest_planner["data"]
                    event_label = str(event_data.get("event_label") or "unknown")
                    sample = {
                        "sample_id": sample_id,
                        "stamp": event_stamp,
                        "event_label": event_label,
                        "event_type": str(event_data.get("event_type") or "unknown"),
                        "planner_reason": planner_reason(planner_snapshot, event_data),
                        "motion_state": motion_state(planner_snapshot, event_data),
                        "path_blocked": bool((((event_data.get("decision") or {}).get("path_blocked") or {}).get("value"))),
                        "image_path": image_rel_paths[1],
                        "temporal_image_paths": image_rel_paths,
                        "motion_summary": motion_summary,
                        "actual_motion_summary": actual_motion_summary,
                        "pointcloud_path": pointcloud_relpath,
                        "pointcloud_summary": pointcloud_summary,
                        "obstacle_summary": obstacle_summary(event_data),
                        "control_summary": control_summary(planner_snapshot, event_data),
                        "planning_summary": planning_summary(planner_snapshot, event_data),
                        "teacher_label_candidates_ko": list(ALLOWED_LABELS_KO),
                        "planner_snapshot": planner_snapshot,
                        "event_log": event_data,
                        "source_bag": str(bag_path),
                        "source_bag_stem": bag_stem,
                    }
                    sample["teacher_prompt_ko"] = build_teacher_prompt(sample)
                    metadata_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

                    exported += 1
                    if args.limit > 0 and exported >= int(args.limit):
                        break

                if args.limit > 0 and exported >= int(args.limit):
                    break

    print("exported_samples={}".format(exported))
    print("skipped_events={}".format(skipped))
    print("metadata={}".format(metadata_path))


if __name__ == "__main__":
    main()

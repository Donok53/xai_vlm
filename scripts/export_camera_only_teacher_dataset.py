#!/usr/bin/env python3
import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge


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
    "벽",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export camera-only temporal teacher dataset from ROS bag."
    )
    parser.add_argument("--bag", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--sample-every-n", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--flow-image-side-px", type=int, default=320)
    parser.add_argument("--flow-motion-threshold", type=float, default=1.5)
    return parser.parse_args()


def ensure_dirs(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)
    (output_dir / "annotations").mkdir(exist_ok=True)


def resize_for_flow(image_bgr, max_side):
    max_side = max(32, int(max_side))
    h, w = image_bgr.shape[:2]
    scale = float(max_side) / float(max(h, w))
    if scale >= 1.0:
        resized = image_bgr
    else:
        resized = cv2.resize(
            image_bgr,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray


def infer_raw_screen_motion_ko(prev_stats, next_stats):
    direction_ko = "정지 또는 미미한 움직임"
    dx = 0.5 * (float(prev_stats.get("mean_dx") or 0.0) + float(next_stats.get("mean_dx") or 0.0))
    dy = 0.5 * (float(prev_stats.get("mean_dy") or 0.0) + float(next_stats.get("mean_dy") or 0.0))
    center_ratio = max(
        float(prev_stats.get("center_moving_ratio") or 0.0),
        float(next_stats.get("center_moving_ratio") or 0.0),
    )
    if center_ratio > 0.02:
        if abs(dx) >= abs(dy):
            direction_ko = "화면 기준 좌우 방향 움직임"
            if dx > 0.0:
                direction_ko = "화면 기준 우측으로 이동하는 움직임"
            elif dx < 0.0:
                direction_ko = "화면 기준 좌측으로 이동하는 움직임"
        else:
            direction_ko = "화면 기준 상하 방향 움직임"
            if dy > 0.0:
                direction_ko = "화면 기준 아래쪽으로 이동하는 움직임"
            elif dy < 0.0:
                direction_ko = "화면 기준 위쪽으로 이동하는 움직임"
    return direction_ko


def infer_ego_motion_ko(prev_stats, next_stats):
    mean_dx = 0.5 * (float(prev_stats.get("mean_dx") or 0.0) + float(next_stats.get("mean_dx") or 0.0))
    mean_dy = 0.5 * (float(prev_stats.get("mean_dy") or 0.0) + float(next_stats.get("mean_dy") or 0.0))
    mean_mag = 0.5 * (
        float(prev_stats.get("mean_magnitude") or 0.0) + float(next_stats.get("mean_magnitude") or 0.0)
    )
    center_ratio = max(
        float(prev_stats.get("center_moving_ratio") or 0.0),
        float(next_stats.get("center_moving_ratio") or 0.0),
    )
    moving_ratio = max(
        float(prev_stats.get("moving_ratio") or 0.0),
        float(next_stats.get("moving_ratio") or 0.0),
    )

    if center_ratio < 0.03 and moving_ratio < 0.10 and mean_mag < 1.0:
        return "정지"

    dx_threshold = 0.35
    dy_threshold = 0.35

    if mean_dy >= dy_threshold:
        if mean_dx >= dx_threshold:
            return "전진 좌회전"
        if mean_dx <= -dx_threshold:
            return "전진 우회전"
        return "전진"

    if mean_dy <= -dy_threshold:
        if mean_dx >= dx_threshold:
            return "후진 좌회전"
        if mean_dx <= -dx_threshold:
            return "후진 우회전"
        return "후진"

    if mean_dx >= dx_threshold:
        return "좌회전"
    if mean_dx <= -dx_threshold:
        return "우회전"
    return "정지"


def infer_scene_state_ko(prev_stats, next_stats, ego_motion_ko):
    center_ratio = max(
        float(prev_stats.get("center_moving_ratio") or 0.0),
        float(next_stats.get("center_moving_ratio") or 0.0),
    )
    moving_ratio = max(
        float(prev_stats.get("moving_ratio") or 0.0),
        float(next_stats.get("moving_ratio") or 0.0),
    )
    if ego_motion_ko == "정지":
        if center_ratio >= 0.20 or moving_ratio >= 0.25:
            return "동적 객체 영향 큼"
        return "정적 구조 우세"
    if center_ratio >= 0.12 or moving_ratio >= 0.20:
        return "자차 이동 영향 큼"
    return "정적/동적 혼합"


def summarize_flow(prev_bgr, curr_bgr, next_bgr, max_side, motion_threshold):
    prev_gray = resize_for_flow(prev_bgr, max_side)
    curr_gray = resize_for_flow(curr_bgr, max_side)
    next_gray = resize_for_flow(next_bgr, max_side)

    flow_prev = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    flow_next = cv2.calcOpticalFlowFarneback(
        curr_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    def _stats(flow):
        dx = flow[..., 0]
        dy = flow[..., 1]
        mag = np.sqrt(dx * dx + dy * dy)
        moving = mag > float(motion_threshold)
        moving_ratio = float(np.mean(moving))
        mean_mag = float(np.mean(mag))
        center = mag[
            mag.shape[0] // 4 : (mag.shape[0] * 3) // 4,
            mag.shape[1] // 4 : (mag.shape[1] * 3) // 4,
        ]
        center_moving_ratio = float(np.mean(center > float(motion_threshold)))
        if np.any(moving):
            mean_dx = float(np.mean(dx[moving]))
            mean_dy = float(np.mean(dy[moving]))
        else:
            mean_dx = 0.0
            mean_dy = 0.0
        return {
            "mean_magnitude": mean_mag,
            "moving_ratio": moving_ratio,
            "center_moving_ratio": center_moving_ratio,
            "mean_dx": mean_dx,
            "mean_dy": mean_dy,
        }

    prev_stats = _stats(flow_prev)
    next_stats = _stats(flow_next)

    raw_motion_ko = infer_raw_screen_motion_ko(prev_stats, next_stats)
    ego_motion_ko = infer_ego_motion_ko(prev_stats, next_stats)
    scene_state_ko = infer_scene_state_ko(prev_stats, next_stats, ego_motion_ko)

    return {
        "prev_to_curr": prev_stats,
        "curr_to_next": next_stats,
        "dominant_motion_ko": raw_motion_ko,
        "raw_screen_motion_ko": raw_motion_ko,
        "ego_motion_ko": ego_motion_ko,
        "scene_state_ko": scene_state_ko,
    }


def build_camera_only_prompt(sample):
    motion = sample.get("motion_summary") or {}
    dominant_motion_ko = motion.get("ego_motion_ko") or motion.get("dominant_motion_ko", "정지")
    scene_state_ko = motion.get("scene_state_ko", "정적/동적 혼합")
    prev_to_curr = motion.get("prev_to_curr") or {}
    curr_to_next = motion.get("curr_to_next") or {}

    return "\n".join(
        [
            "너는 자율주행용 오프라인 camera-only teacher 모델이다.",
            "세 장의 연속 프레임(prev, current, next)을 보고, 현재 장면에서 왜 이런 주행 판단이 나왔을지 시각적으로 설명하라.",
            "Planner나 LiDAR 정보는 사용하지 말고, 오직 카메라와 프레임 간 움직임 단서만 사용하라.",
            "",
            "입력 이미지 순서:",
            "1) prev",
            "2) current",
            "3) next",
            "",
            "움직임 요약:",
            "- ego_motion_ko: {}".format(dominant_motion_ko),
            "- scene_state_ko: {}".format(scene_state_ko),
            "- prev_to_curr.moving_ratio: {:.4f}".format(float(prev_to_curr.get("moving_ratio") or 0.0)),
            "- prev_to_curr.center_moving_ratio: {:.4f}".format(float(prev_to_curr.get("center_moving_ratio") or 0.0)),
            "- curr_to_next.moving_ratio: {:.4f}".format(float(curr_to_next.get("moving_ratio") or 0.0)),
            "- curr_to_next.center_moving_ratio: {:.4f}".format(float(curr_to_next.get("center_moving_ratio") or 0.0)),
            "",
            "허용 대표 라벨:",
            ", ".join(ALLOWED_LABELS_KO),
            "",
            "규칙:",
            "- 대표 객체는 허용 라벨 중 하나만 고른다.",
            "- 확실하지 않으면 primary_object_ko는 '벽'으로 둔다.",
            "- driving_reason_ko는 카메라에서 보이는 통로, 장애물, 사람 움직임, 접근/이격 단서만으로 쓴다.",
            "- 반드시 JSON만 출력한다.",
            '- 형식: {"primary_object_ko":"", "dynamic":"static|dynamic|unknown", "scene_summary_ko":"", "driving_reason_ko":"", "confidence":0.0}',
        ]
    )


def main():
    args = parse_args()
    bag_paths = [Path(bag).expanduser().resolve() for bag in args.bag]
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dirs(output_dir)

    bridge = CvBridge()
    exported = 0

    metadata_path = output_dir / "metadata" / "teacher_dataset.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for bag_path in bag_paths:
            frames = deque(maxlen=3)
            frame_index = -1
            bag_stem = bag_path.stem

            with rosbag.Bag(str(bag_path)) as bag:
                for topic, msg, _ in bag.read_messages(topics=[args.image_topic]):
                    if topic != args.image_topic:
                        continue
                    frame_index += 1
                    image_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    frames.append(
                        {
                            "stamp": float(msg.header.stamp.to_sec()),
                            "frame_index": frame_index,
                            "image_bgr": image_bgr,
                        }
                    )
                    if len(frames) < 3:
                        continue
                    center = frames[1]
                    if (int(center["frame_index"]) % max(1, int(args.sample_every_n))) != 0:
                        continue

                    prev_frame = frames[0]
                    curr_frame = frames[1]
                    next_frame = frames[2]

                    sample_id = "{}_sample_{:05d}".format(bag_stem, exported)
                    image_rel_paths = []
                    for suffix, frame in [("prev", prev_frame), ("current", curr_frame), ("next", next_frame)]:
                        rel_path = Path("images") / "{}_{}.jpg".format(sample_id, suffix)
                        abs_path = output_dir / rel_path
                        cv2.imwrite(
                            str(abs_path),
                            frame["image_bgr"],
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
                        )
                        image_rel_paths.append(str(rel_path))

                    motion_summary = summarize_flow(
                        prev_frame["image_bgr"],
                        curr_frame["image_bgr"],
                        next_frame["image_bgr"],
                        args.flow_image_side_px,
                        args.flow_motion_threshold,
                    )
                    row = {
                        "sample_id": sample_id,
                        "source_bag": str(bag_path),
                        "source_bag_stem": bag_stem,
                        "stamp": curr_frame["stamp"],
                        "frame_index": int(curr_frame["frame_index"]),
                        "image_path": image_rel_paths[1],
                        "temporal_image_paths": image_rel_paths,
                        "motion_summary": motion_summary,
                    }
                    row["teacher_prompt_camera_only_ko"] = build_camera_only_prompt(row)
                    metadata_file.write(json.dumps(row, ensure_ascii=False) + "\n")

                    exported += 1
                    if args.max_samples > 0 and exported >= int(args.max_samples):
                        break
                if args.max_samples > 0 and exported >= int(args.max_samples):
                    break

    print("exported_samples={}".format(exported))
    print("metadata={}".format(metadata_path))


if __name__ == "__main__":
    main()

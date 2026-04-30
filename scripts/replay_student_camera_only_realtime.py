#!/usr/bin/env python3
import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np
import rosbag
from cv_bridge import CvBridge
from PIL import Image, ImageDraw, ImageFont

from export_camera_only_teacher_dataset import summarize_flow
from student_baseline_common import build_context_feature, load_image_feature_from_bgr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a camera bag with student inference visualization in real time."
    )
    parser.add_argument("--bag", required=True)
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--output-video", default="")
    parser.add_argument("--sample-every-n", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--playback-rate", type=float, default=1.0)
    parser.add_argument("--flow-image-side-px", type=int, default=320)
    parser.add_argument("--flow-motion-threshold", type=float, default=1.5)
    parser.add_argument("--display-window", action="store_true")
    parser.add_argument("--no-display-window", dest="display_window", action="store_false")
    parser.set_defaults(display_window=True)
    return parser.parse_args()


def find_font(size):
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def wrap_text(text, width=18):
    text = str(text or "")
    if len(text) <= width:
        return [text]
    lines = []
    start = 0
    while start < len(text):
        lines.append(text[start : start + width])
        start += width
    return lines


def infer_dynamic_from_motion(row):
    motion = row.get("motion_summary") or {}
    prev = motion.get("prev_to_curr") or {}
    nxt = motion.get("curr_to_next") or {}
    center_ratio = max(
        float(prev.get("center_moving_ratio") or 0.0),
        float(nxt.get("center_moving_ratio") or 0.0),
    )
    if center_ratio >= 0.25:
        return "dynamic"
    if center_ratio <= 0.05:
        return "static"
    return "unknown"


def build_camera_thought(pred_label, row):
    motion = row.get("motion_summary") or {}
    dominant_motion = motion.get("dominant_motion_ko") or "정지 또는 미미한 움직임"
    dynamic_state = infer_dynamic_from_motion(row)

    if pred_label == "사람":
        if dynamic_state == "dynamic":
            reason = "사람 움직임이 보여 감속이나 회피를 준비한다고 본다."
        else:
            reason = "사람이 보여 주변을 경계하며 천천히 본다고 해석한다."
    elif pred_label == "자동차":
        if dynamic_state == "dynamic":
            reason = "차량 움직임이 보여 간격을 두고 지나가려 한다고 본다."
        else:
            reason = "차량이 보여 통과 가능 공간을 살핀다고 해석한다."
    elif pred_label == "벽":
        if dynamic_state == "dynamic":
            reason = "시점 변화가 커서 통로 구조를 다시 확인한다고 본다."
        else:
            reason = "정적인 통로 구조가 보여 즉시 회피할 대상은 약하다고 본다."
    else:
        reason = "{}이 보여 보수적으로 진행한다고 해석한다.".format(pred_label)

    scene = "대표 객체 {} / {}".format(pred_label, dominant_motion)
    return scene, reason


def render_panel(curr_bgr, bag_stem, frame_index, pred_label, confidence, motion_summary, infer_ms):
    h, w = curr_bgr.shape[:2]
    panel_w = 460
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = curr_bgr
    canvas[:, w:] = (18, 18, 18)

    row = {"motion_summary": motion_summary}
    scene_summary, camera_reason = build_camera_thought(pred_label, row)
    dominant_motion = motion_summary.get("dominant_motion_ko") or "정지 또는 미미한 움직임"
    dynamic_state = infer_dynamic_from_motion(row)

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    title_font = find_font(28)
    body_font = find_font(22)
    small_font = find_font(18)

    x0 = w + 20
    y = 24
    draw.text((x0, y), "student camera thought", fill=(255, 255, 255), font=title_font)
    y += 46

    lines = [
        "bag: {}".format(bag_stem),
        "frame: {}".format(frame_index),
        "대표 객체: {} ({:.2f})".format(pred_label, confidence),
        "motion: {}".format(dominant_motion),
        "dynamic: {}".format(dynamic_state),
        "scene: {}".format(scene_summary),
        "reason: {}".format(camera_reason),
        "infer: {:.1f} ms".format(infer_ms),
        "q: quit",
    ]
    for idx, line in enumerate(lines):
        font = body_font if idx < 5 else small_font
        for subline in wrap_text(line, width=22):
            draw.text((x0, y), subline, fill=(240, 240, 240), font=font)
            y += 28 if font == body_font else 24
        y += 4

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    args = parse_args()
    bag_path = Path(args.bag).expanduser().resolve()
    project_dir = Path(__file__).resolve().parent.parent
    model_path = (
        Path(args.model_path).expanduser().resolve()
        if args.model_path
        else project_dir / "data" / "made_map_camera_lr_full" / "student_baseline" / "student_baseline.joblib"
    )
    output_video = (
        Path(args.output_video).expanduser().resolve()
        if args.output_video
        else project_dir / "data" / "made_map_camera_lr_full" / "student_inference" / (bag_path.stem + "_realtime.mp4")
    )
    output_video.parent.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(model_path)
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    label_encoder = bundle["label_encoder"]
    image_size = int(bundle["image_size"])

    bridge = CvBridge()
    frames = deque(maxlen=3)
    frame_index = -1
    shown = 0
    bag_stem = bag_path.stem
    writer = None
    last_stamp = None

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

            t0 = time.perf_counter()
            prev_bgr = frames[0]["image_bgr"]
            curr_bgr = frames[1]["image_bgr"]
            next_bgr = frames[2]["image_bgr"]
            motion_summary = summarize_flow(
                prev_bgr,
                curr_bgr,
                next_bgr,
                args.flow_image_side_px,
                args.flow_motion_threshold,
            )
            row = {
                "source_bag_stem": bag_stem,
                "motion_summary": motion_summary,
                "event_label": None,
                "planner_reason": None,
                "motion_state": None,
                "path_blocked": False,
                "obstacle_summary": {},
            }
            image_feature = load_image_feature_from_bgr(curr_bgr, image_size)
            context_feature = build_context_feature(row)
            context_matrix = vectorizer.transform([context_feature]).astype(np.float32)
            x = np.concatenate([image_feature.reshape(1, -1), context_matrix], axis=1)
            probs = model.predict_proba(x)[0]
            pred_index = int(np.argmax(probs))
            pred_label = str(label_encoder.inverse_transform([pred_index])[0])
            confidence = float(probs[pred_index])
            infer_ms = (time.perf_counter() - t0) * 1000.0

            frame = render_panel(
                curr_bgr=curr_bgr,
                bag_stem=bag_stem,
                frame_index=int(center["frame_index"]),
                pred_label=pred_label,
                confidence=confidence,
                motion_summary=motion_summary,
                infer_ms=infer_ms,
            )

            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10.0,
                    (w, h),
                )
                if not writer.isOpened():
                    raise RuntimeError("video writer를 열지 못했습니다: {}".format(output_video))

            if last_stamp is not None:
                dt = max(0.0, float(center["stamp"]) - float(last_stamp))
                target_sleep = dt / max(1e-6, float(args.playback_rate))
                if target_sleep > 0.0 and args.display_window:
                    time.sleep(min(target_sleep, 0.2))
            last_stamp = float(center["stamp"])

            writer.write(frame)
            shown += 1
            if args.display_window:
                cv2.imshow("student_camera_only_realtime", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if int(args.max_samples) > 0 and shown >= int(args.max_samples):
                break

    if writer is not None:
        writer.release()
    if args.display_window:
        cv2.destroyAllWindows()

    print("shown_frames={}".format(shown))
    print("output_video={}".format(output_video))


if __name__ == "__main__":
    main()

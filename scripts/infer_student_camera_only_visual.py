#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from export_camera_only_teacher_dataset import infer_ego_motion_ko, infer_scene_state_ko
from student_baseline_common import build_context_feature, load_image_feature, read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run student inference on camera-only dataset and render visual explanation."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--prepared-path", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-video", default="")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--source-bag-stem", default="")
    parser.add_argument("--display-window", action="store_true")
    parser.add_argument("--show-teacher", action="store_true")
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


def load_bgr_or_blank(path, shape):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        h, w = shape
        return np.zeros((h, w, 3), dtype=np.uint8)
    return image


def describe_motion(row):
    motion = row.get("motion_summary") or {}
    prev = motion.get("prev_to_curr") or {}
    nxt = motion.get("curr_to_next") or {}
    raw_motion = motion.get("raw_screen_motion_ko") or motion.get("dominant_motion_ko") or "정지 또는 미미한 움직임"
    ego_motion = motion.get("ego_motion_ko") or infer_ego_motion_ko(prev, nxt)
    scene_state = motion.get("scene_state_ko") or infer_scene_state_ko(prev, nxt, ego_motion)
    return raw_motion, ego_motion, scene_state


def build_camera_thought(pred_label, row):
    _, ego_motion, scene_state = describe_motion(row)

    if pred_label == "사람":
        if scene_state == "동적 객체 영향 큼":
            reason = "사람 움직임이 보여 감속하거나 경로를 조심스럽게 본다."
        elif ego_motion != "정지":
            reason = "{} 중 사람과의 간격을 살피며 지나가려 한다고 본다.".format(ego_motion)
        else:
            reason = "사람이 보여 주변을 경계하며 진행한다고 본다."
    elif pred_label == "자동차":
        if scene_state == "동적 객체 영향 큼":
            reason = "차량 움직임이 보여 간격을 두고 진행한다고 본다."
        elif ego_motion != "정지":
            reason = "{} 중 차량과의 간격을 확인한다고 본다.".format(ego_motion)
        else:
            reason = "차량이 보여 통로와 간격을 확인한다고 본다."
    elif pred_label == "자전거":
        reason = "자전거가 보여 진행 방향과 접근을 조심한다고 본다."
    elif pred_label == "벽":
        if ego_motion != "정지":
            reason = "{} 중 통로 구조와 진행 공간을 확인한다고 본다.".format(ego_motion)
        else:
            reason = "정적인 통로 구조가 보여 즉시 회피할 대상은 뚜렷하지 않다고 본다."
    else:
        reason = "{}이 보여 주변 상황을 보수적으로 해석한다고 본다.".format(pred_label)

    scene = "대표 객체: {} / 로봇: {}".format(pred_label, ego_motion)
    return scene, reason


def wrap_text(text, width=30):
    text = str(text or "")
    if len(text) <= width:
        return [text]
    lines = []
    start = 0
    while start < len(text):
        lines.append(text[start : start + width])
        start += width
    return lines


def format_top_candidates(top_candidates):
    if not top_candidates:
        return "후보 없음"
    return ", ".join("{} {:.2f}".format(item["label_ko"], float(item["confidence"])) for item in top_candidates)


def render_frame(prev_bgr, curr_bgr, next_bgr, row, pred_label, confidence, show_teacher, top_candidates):
    h, w = curr_bgr.shape[:2]
    panel_h = 220
    canvas = np.zeros((h + panel_h, w * 3, 3), dtype=np.uint8)
    canvas[:h, 0:w] = prev_bgr
    canvas[:h, w : 2 * w] = curr_bgr
    canvas[:h, 2 * w : 3 * w] = next_bgr
    canvas[h:, :] = (18, 18, 18)

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    title_font = find_font(26)
    body_font = find_font(22)
    small_font = find_font(18)

    draw.text((20, 16), "prev", fill=(255, 255, 255), font=title_font)
    draw.text((w + 20, 16), "current", fill=(255, 255, 255), font=title_font)
    draw.text((2 * w + 20, 16), "next", fill=(255, 255, 255), font=title_font)

    sample_id = row.get("sample_id") or "unknown"
    source_bag = row.get("source_bag_stem") or "unknown"
    teacher_label = row.get("label_ko") or row.get("label_ko_raw") or "n/a"
    raw_motion, ego_motion, scene_state = describe_motion(row)
    scene_summary, camera_reason = build_camera_thought(pred_label, row)

    x0 = 24
    y0 = h + 18
    lines = [
        "sample: {} | bag: {}".format(sample_id, source_bag),
        "student: {} ({:.2f})".format(pred_label, confidence),
        "robot motion: {}".format(ego_motion),
        "scene state: {}".format(scene_state),
        "raw flow: {}".format(raw_motion),
        "candidates: {}".format(format_top_candidates(top_candidates)),
        "scene: {}".format(scene_summary),
        "reason: {}".format(camera_reason),
    ]
    if show_teacher:
        lines.insert(2, "teacher: {}".format(teacher_label))

    y = y0
    for index, line in enumerate(lines):
        font = body_font if index < 2 else small_font
        wrapped = wrap_text(line, width=70 if index >= 3 else 90)
        for subline in wrapped:
            draw.text((x0, y), subline, fill=(240, 240, 240), font=font)
            y += 28 if font == body_font else 24

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    prepared_path = (
        Path(args.prepared_path).expanduser().resolve()
        if args.prepared_path
        else dataset_dir / "metadata" / "prepared_teacher_labels.jsonl"
    )
    model_path = (
        Path(args.model_path).expanduser().resolve()
        if args.model_path
        else dataset_dir / "student_baseline" / "student_baseline.joblib"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else dataset_dir / "student_inference"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = (
        Path(args.output_video).expanduser().resolve()
        if args.output_video
        else output_dir / "student_camera_reason.mp4"
    )

    rows = read_jsonl(prepared_path)
    if args.source_bag_stem:
        rows = [row for row in rows if str(row.get("source_bag_stem") or "") == str(args.source_bag_stem)]
    if int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]
    if not rows:
        raise RuntimeError("시각화할 prepared rows가 없습니다.")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    label_encoder = bundle["label_encoder"]
    image_size = int(bundle["image_size"])

    first_curr = load_bgr_or_blank(dataset_dir / rows[0]["image_path"], (480, 640))
    h, w = first_curr.shape[:2]
    frame_size = (w * 3, h + 220)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError("video writer를 열지 못했습니다: {}".format(output_video))

    try:
        for row in rows:
            temporal = row.get("temporal_image_paths") or []
            if len(temporal) >= 3:
                prev_path = dataset_dir / temporal[0]
                curr_path = dataset_dir / temporal[1]
                next_path = dataset_dir / temporal[2]
            else:
                curr_path = dataset_dir / row["image_path"]
                prev_path = curr_path
                next_path = curr_path

            prev_bgr = load_bgr_or_blank(prev_path, (h, w))
            curr_bgr = load_bgr_or_blank(curr_path, (h, w))
            next_bgr = load_bgr_or_blank(next_path, (h, w))

            image_feature = load_image_feature(curr_path, image_size)
            context_feature = build_context_feature(row)
            context_matrix = vectorizer.transform([context_feature]).astype(np.float32)
            x = np.concatenate([image_feature.reshape(1, -1), context_matrix], axis=1)
            probs = model.predict_proba(x)[0]
            top_indices = np.argsort(probs)[::-1][: min(3, len(probs))]
            pred_index = int(top_indices[0])
            pred_label = str(label_encoder.inverse_transform([pred_index])[0])
            confidence = float(probs[pred_index])
            top_candidates = [
                {
                    "label_ko": str(label_encoder.inverse_transform([int(idx)])[0]),
                    "confidence": float(probs[int(idx)]),
                }
                for idx in top_indices
            ]

            frame = render_frame(
                prev_bgr=prev_bgr,
                curr_bgr=curr_bgr,
                next_bgr=next_bgr,
                row=row,
                pred_label=pred_label,
                confidence=confidence,
                show_teacher=bool(args.show_teacher),
                top_candidates=top_candidates,
            )
            writer.write(frame)
            if args.display_window:
                cv2.imshow("student_camera_reason", frame)
                key = cv2.waitKey(max(1, int(round(1000.0 / max(1.0, float(args.fps)))))) & 0xFF
                if key == ord("q"):
                    break
    finally:
        writer.release()
        if args.display_window:
            cv2.destroyAllWindows()

    print("written_video={}".format(output_video))
    print("num_frames={}".format(len(rows)))


if __name__ == "__main__":
    main()

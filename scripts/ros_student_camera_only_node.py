#!/usr/bin/env python3
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np
import rospy
from cv_bridge import CvBridge
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from sensor_msgs.msg import Image
from std_msgs.msg import String

from export_camera_only_teacher_dataset import summarize_flow
from student_baseline_common import build_context_feature, load_image_feature_from_bgr


def _parse_explicit_private_args():
    explicit = {}
    for arg in sys.argv[1:]:
        if not arg.startswith("_") or ":=" not in arg:
            continue
        name, value = arg[1:].split(":=", 1)
        explicit[name] = value
    return explicit


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in ("1", "true", "yes", "on")


def get_private_param(name, default, explicit_args):
    resolved_name = rospy.resolve_name("~" + name)
    if name in explicit_args:
        return rospy.get_param(resolved_name, default)
    rospy.set_param(resolved_name, default)
    return default


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


def render_panel(curr_bgr, pred_label, confidence, motion_summary, infer_ms, frame_index):
    h, w = curr_bgr.shape[:2]
    panel_w = 460
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = curr_bgr
    canvas[:, w:] = (18, 18, 18)

    row = {"motion_summary": motion_summary}
    scene_summary, camera_reason = build_camera_thought(pred_label, row)
    dominant_motion = motion_summary.get("dominant_motion_ko") or "정지 또는 미미한 움직임"
    dynamic_state = infer_dynamic_from_motion(row)

    pil = PILImage.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    title_font = find_font(28)
    body_font = find_font(22)
    small_font = find_font(18)

    x0 = w + 20
    y = 24
    draw.text((x0, y), "student camera thought", fill=(255, 255, 255), font=title_font)
    y += 46

    lines = [
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
        font = body_font if idx < 4 else small_font
        for subline in wrap_text(line, width=22):
            draw.text((x0, y), subline, fill=(240, 240, 240), font=font)
            y += 28 if font == body_font else 24
        y += 4

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


class StudentCameraOnlyNode(object):
    def __init__(self):
        explicit_args = _parse_explicit_private_args()

        self.image_topic = get_private_param("image_topic", "/camera/color/image_raw", explicit_args)
        self.output_topic = get_private_param("output_topic", "/student_xai/camera_reason", explicit_args)
        self.overlay_topic = get_private_param("overlay_topic", "/student_xai/overlay", explicit_args)
        self.model_path = Path(
            get_private_param(
                "model_path",
                str(
                    Path(__file__).resolve().parent.parent
                    / "data"
                    / "made_map_camera_lr_full"
                    / "student_baseline"
                    / "student_baseline.joblib"
                ),
                explicit_args,
            )
        ).expanduser().resolve()
        self.sample_every_n = int(get_private_param("sample_every_n", 8, explicit_args))
        self.flow_image_side_px = int(get_private_param("flow_image_side_px", 320, explicit_args))
        self.flow_motion_threshold = float(get_private_param("flow_motion_threshold", 1.5, explicit_args))
        self.display_window = _coerce_bool(get_private_param("display_window", False, explicit_args))
        self.launch_rviz = _coerce_bool(get_private_param("launch_rviz", True, explicit_args))
        self.rviz_config_path = Path(
            get_private_param(
                "rviz_config_path",
                str(Path(__file__).resolve().parent.parent / "rviz" / "student_camera_only.rviz"),
                explicit_args,
            )
        ).expanduser().resolve()

        bundle = joblib.load(str(self.model_path))
        self.model = bundle["model"]
        self.vectorizer = bundle["vectorizer"]
        self.label_encoder = bundle["label_encoder"]
        self.image_size = int(bundle["image_size"])

        self.bridge = CvBridge()
        self.frames = deque(maxlen=3)
        self.frame_index = -1
        self.rviz_process = None

        self.message_pub = rospy.Publisher(self.output_topic, String, queue_size=10)
        self.overlay_pub = rospy.Publisher(self.overlay_topic, Image, queue_size=2)
        self.sub = rospy.Subscriber(self.image_topic, Image, self._image_callback, queue_size=1)

        rospy.loginfo(
            "student_camera_only_node started | image=%s output=%s overlay=%s model=%s display_window=%s launch_rviz=%s",
            self.image_topic,
            self.output_topic,
            self.overlay_topic,
            self.model_path,
            self.display_window,
            self.launch_rviz,
        )
        if self.launch_rviz:
            self._start_rviz()
        rospy.on_shutdown(self._on_shutdown)

    def _start_rviz(self):
        if not self.rviz_config_path.exists():
            rospy.logwarn("rviz config가 없습니다: %s", self.rviz_config_path)
            return
        try:
            self.rviz_process = subprocess.Popen(
                ["rviz", "-d", str(self.rviz_config_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            rospy.loginfo("rviz started | config=%s", self.rviz_config_path)
        except Exception as exc:
            rospy.logwarn("rviz 실행에 실패했습니다: %s", exc)

    def _on_shutdown(self):
        if self.rviz_process is not None and self.rviz_process.poll() is None:
            try:
                self.rviz_process.terminate()
            except Exception:
                pass

    def _image_callback(self, msg):
        self.frame_index += 1
        image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.frames.append(
            {
                "stamp": float(msg.header.stamp.to_sec()),
                "frame_index": self.frame_index,
                "image_bgr": image_bgr,
            }
        )
        if len(self.frames) < 3:
            return
        center = self.frames[1]
        if (int(center["frame_index"]) % max(1, self.sample_every_n)) != 0:
            return

        t0 = time.perf_counter()
        prev_bgr = self.frames[0]["image_bgr"]
        curr_bgr = self.frames[1]["image_bgr"]
        next_bgr = self.frames[2]["image_bgr"]
        motion_summary = summarize_flow(
            prev_bgr,
            curr_bgr,
            next_bgr,
            self.flow_image_side_px,
            self.flow_motion_threshold,
        )
        row = {
            "motion_summary": motion_summary,
            "event_label": None,
            "planner_reason": None,
            "motion_state": None,
            "path_blocked": False,
            "obstacle_summary": {},
            "source_bag_stem": "realtime",
        }
        image_feature = load_image_feature_from_bgr(curr_bgr, self.image_size)
        context_feature = build_context_feature(row)
        context_matrix = self.vectorizer.transform([context_feature]).astype(np.float32)
        x = np.concatenate([image_feature.reshape(1, -1), context_matrix], axis=1)
        probs = self.model.predict_proba(x)[0]
        pred_index = int(np.argmax(probs))
        pred_label = str(self.label_encoder.inverse_transform([pred_index])[0])
        confidence = float(probs[pred_index])
        infer_ms = (time.perf_counter() - t0) * 1000.0

        scene_summary, camera_reason = build_camera_thought(pred_label, row)
        payload = {
            "frame_index": int(center["frame_index"]),
            "stamp": center["stamp"],
            "primary_object_ko": pred_label,
            "confidence": confidence,
            "dynamic": infer_dynamic_from_motion(row),
            "motion_summary": motion_summary,
            "scene_summary_ko": scene_summary,
            "driving_reason_ko": camera_reason,
            "infer_ms": infer_ms,
        }
        self.message_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        rospy.loginfo(
            "[STUDENT-XAI] frame=%d | object=%s | confidence=%.2f | motion=%s | final=%s",
            int(center["frame_index"]),
            pred_label,
            confidence,
            motion_summary.get("dominant_motion_ko") or "정지 또는 미미한 움직임",
            camera_reason,
        )

        overlay_bgr = render_panel(
            curr_bgr=curr_bgr,
            pred_label=pred_label,
            confidence=confidence,
            motion_summary=motion_summary,
            infer_ms=infer_ms,
            frame_index=int(center["frame_index"]),
        )
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay_bgr, encoding="bgr8")
        overlay_msg.header = msg.header
        self.overlay_pub.publish(overlay_msg)

        if self.display_window:
            cv2.imshow("student_camera_only_realtime", overlay_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                rospy.signal_shutdown("user requested quit")


def main():
    rospy.init_node("student_camera_only_node")
    StudentCameraOnlyNode()
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

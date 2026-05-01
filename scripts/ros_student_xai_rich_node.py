#!/usr/bin/env python3
import json
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
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String

from export_camera_only_teacher_dataset import summarize_flow
from export_teacher_dataset import (
    control_summary,
    motion_state,
    obstacle_summary,
    planner_reason,
    planning_summary,
    safe_json_loads,
    stamp_to_float,
)
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


def wrap_text(text, width=24):
    text = str(text or "")
    if len(text) <= width:
        return [text]
    lines = []
    start = 0
    while start < len(text):
        lines.append(text[start : start + width])
        start += width
    return lines


def infer_scene_and_reason(pred_label, row):
    obstacle = row.get("obstacle_summary") or {}
    planning = row.get("planning_summary") or {}
    control = row.get("control_summary") or {}
    near_range = obstacle.get("near_raw_min_range_m")
    centroid = obstacle.get("near_raw_centroid_xyz") or {}
    cy = float(centroid.get("y") or 0.0)
    side = "중앙"
    if cy <= -0.25:
        side = "우측"
    elif cy >= 0.25:
        side = "좌측"

    steering = str(control.get("steering_direction") or "unknown")
    motion_state_text = str(control.get("motion_state") or row.get("motion_state") or "unknown")
    path_blocked = bool(planning.get("path_blocked"))
    path_change_changed = bool(planning.get("path_change_changed"))
    path_change_direction = str(planning.get("path_change_direction") or "unknown")
    behavior_reason = str(planning.get("behavior_reason") or row.get("planner_reason") or "unknown")

    if pred_label == "사람":
        scene = "전방 {} 사람 영향".format(side)
        if path_blocked:
            reason = "전방 사람 때문에 멈춤이나 우회를 검토한다고 본다."
        elif path_change_changed:
            reason = "{} 사람을 피해 경로를 조정한다고 본다.".format(side)
        else:
            reason = "사람과의 간격을 보며 조심스럽게 진행한다고 본다."
    else:
        if path_blocked:
            scene = "전방 {} 근거리 장애물".format(side)
            reason = "가까운 장애물 때문에 경로가 막혔다고 본다."
        elif path_change_changed:
            scene = "경로 {} 조정".format(path_change_direction)
            reason = "장애물 위치 때문에 {} 방향으로 경로를 바꾼다고 본다.".format(path_change_direction)
        elif steering in ("left", "right"):
            turn_ko = "좌측" if steering == "left" else "우측"
            scene = "전방 {} 구조물 영향".format(turn_ko)
            reason = "{} 구조나 장애물을 보며 조향을 보정한다고 본다.".format(turn_ko)
        elif near_range is not None and float(near_range) < 1.0:
            scene = "전방 {} 근거리 구조".format(side)
            reason = "가까운 구조물을 보며 속도와 진행 공간을 확인한다고 본다."
        else:
            scene = "통로 유지"
            reason = "정적인 구조를 보며 현재 통로를 유지한다고 본다."

    return scene, reason, {
        "steering_direction": steering,
        "motion_state": motion_state_text,
        "path_change_direction": path_change_direction,
        "behavior_reason": behavior_reason,
    }


def format_top_candidates(top_candidates):
    if not top_candidates:
        return "후보 없음"
    return ", ".join("{} {:.2f}".format(item["label_ko"], float(item["confidence"])) for item in top_candidates)


def render_panel(curr_bgr, pred_label, confidence, event_label, scene_summary, reason, infer_ms, top_candidates, row):
    h, w = curr_bgr.shape[:2]
    panel_w = 520
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = curr_bgr
    canvas[:, w:] = (18, 18, 18)

    planning = row.get("planning_summary") or {}
    control = row.get("control_summary") or {}
    obstacle = row.get("obstacle_summary") or {}
    near_range = obstacle.get("near_raw_min_range_m")

    pil = PILImage.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    title_font = find_font(28)
    body_font = find_font(22)
    small_font = find_font(18)

    x0 = w + 20
    y = 24
    draw.text((x0, y), "student xai thought", fill=(255, 255, 255), font=title_font)
    y += 46

    lines = [
        "event: {}".format(event_label),
        "대표 객체: {} ({:.2f})".format(pred_label, confidence),
        "motion: {}".format(control.get("motion_state") or "unknown"),
        "steer: {}".format(control.get("steering_direction") or "unknown"),
        "path_change: {}".format(planning.get("path_change_direction") or "unknown"),
        "near_range: {}".format("n/a" if near_range is None else "{:.2f} m".format(float(near_range))),
        "후보: {}".format(format_top_candidates(top_candidates)),
        "scene: {}".format(scene_summary),
        "reason: {}".format(reason),
        "planner: {}".format(planning.get("behavior_reason") or row.get("planner_reason") or "unknown"),
        "infer: {:.1f} ms".format(infer_ms),
        "q: quit",
    ]
    for idx, line in enumerate(lines):
        font = body_font if idx < 5 else small_font
        for subline in wrap_text(line, width=26):
            draw.text((x0, y), subline, fill=(240, 240, 240), font=font)
            y += 28 if font == body_font else 24
        y += 4

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


class StudentXAIRichNode(object):
    def __init__(self):
        explicit_args = _parse_explicit_private_args()

        self.image_topic = get_private_param("image_topic", "/camera/color/image_raw", explicit_args)
        self.event_topic = get_private_param("event_topic", "/xai/event_log", explicit_args)
        self.planner_topic = get_private_param("planner_topic", "/xai/planner_snapshot", explicit_args)
        self.point_cloud_topic = get_private_param(
            "point_cloud_topic", "/planning/linefit_ground/non_ground_cloud", explicit_args
        )
        self.output_topic = get_private_param("output_topic", "/student_xai/rich_reason", explicit_args)
        self.overlay_topic = get_private_param("overlay_topic", "/student_xai/rich_overlay", explicit_args)
        self.model_path = Path(
            get_private_param(
                "model_path",
                str(
                    Path(__file__).resolve().parent.parent
                    / "data"
                    / "record_real_rich_full"
                    / "student_baseline"
                    / "student_baseline.joblib"
                ),
                explicit_args,
            )
        ).expanduser().resolve()
        self.flow_image_side_px = int(get_private_param("flow_image_side_px", 320, explicit_args))
        self.flow_motion_threshold = float(get_private_param("flow_motion_threshold", 1.5, explicit_args))
        self.display_window = _coerce_bool(get_private_param("display_window", True, explicit_args))
        self.max_image_age_s = float(get_private_param("max_image_age_s", 0.30, explicit_args))
        self.max_planner_age_s = float(get_private_param("max_planner_age_s", 0.80, explicit_args))
        self.max_pointcloud_age_s = float(get_private_param("max_pointcloud_age_s", 0.60, explicit_args))

        bundle = joblib.load(str(self.model_path))
        self.model = bundle["model"]
        self.vectorizer = bundle["vectorizer"]
        self.label_encoder = bundle["label_encoder"]
        self.image_size = int(bundle["image_size"])

        self.bridge = CvBridge()
        self.frames = deque(maxlen=3)
        self.latest_planner = None
        self.latest_cloud = None
        self.overlay_bgr = None
        self.frame_index = -1

        self.message_pub = rospy.Publisher(self.output_topic, String, queue_size=20)
        self.overlay_pub = rospy.Publisher(self.overlay_topic, Image, queue_size=2)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self._on_image, queue_size=1)
        self.event_sub = rospy.Subscriber(self.event_topic, String, self._on_event, queue_size=20)
        self.planner_sub = rospy.Subscriber(self.planner_topic, String, self._on_planner, queue_size=20)
        self.point_cloud_sub = rospy.Subscriber(self.point_cloud_topic, PointCloud2, self._on_point_cloud, queue_size=2)

        rospy.loginfo(
            "student_xai_rich_node started | image=%s event=%s planner=%s point_cloud=%s output=%s overlay=%s model=%s display_window=%s",
            self.image_topic,
            self.event_topic,
            self.planner_topic,
            self.point_cloud_topic,
            self.output_topic,
            self.overlay_topic,
            self.model_path,
            self.display_window,
        )

    def _on_image(self, msg):
        self.frame_index += 1
        image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.frames.append(
            {
                "stamp": float(msg.header.stamp.to_sec()),
                "frame_index": self.frame_index,
                "image_bgr": image_bgr,
                "header": msg.header,
            }
        )
        if self.overlay_bgr is not None:
            overlay_msg = self.bridge.cv2_to_imgmsg(self.overlay_bgr, encoding="bgr8")
            overlay_msg.header = msg.header
            self.overlay_pub.publish(overlay_msg)

    def _on_planner(self, msg):
        data = safe_json_loads(msg.data)
        self.latest_planner = {
            "stamp": stamp_to_float(data.get("stamp"), rospy.Time.now().to_sec()),
            "data": data,
        }

    def _on_point_cloud(self, msg):
        self.latest_cloud = {
            "stamp": float(msg.header.stamp.to_sec()),
            "frame_id": str(msg.header.frame_id or ""),
            "point_count": int((getattr(msg, "width", 0) or 0) * max(1, int(getattr(msg, "height", 1) or 1))),
        }

    def _on_event(self, msg):
        if len(self.frames) < 3 or self.latest_planner is None:
            return

        event_data = safe_json_loads(msg.data)
        event_stamp = stamp_to_float(event_data.get("stamp"), rospy.Time.now().to_sec())
        center = self.frames[1]
        if abs(float(center["stamp"]) - float(event_stamp)) > self.max_image_age_s:
            return
        if abs(float(self.latest_planner["stamp"]) - float(event_stamp)) > self.max_planner_age_s:
            return
        if self.latest_cloud is not None and abs(float(self.latest_cloud["stamp"]) - float(event_stamp)) > self.max_pointcloud_age_s:
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
        planner_snapshot = self.latest_planner["data"]
        pointcloud_summary = {
            "frame_id": None,
            "stamp": None,
            "point_count": 0,
        }
        if self.latest_cloud is not None:
            pointcloud_summary = {
                "frame_id": self.latest_cloud["frame_id"],
                "stamp": self.latest_cloud["stamp"],
                "point_count": int(self.latest_cloud["point_count"]),
            }

        row = {
            "source_bag_stem": "record_real_realtime",
            "event_label": str(event_data.get("event_label") or "unknown"),
            "planner_reason": planner_reason(planner_snapshot, event_data),
            "motion_state": motion_state(planner_snapshot, event_data),
            "path_blocked": bool((((event_data.get("decision") or {}).get("path_blocked") or {}).get("value"))),
            "motion_summary": motion_summary,
            "obstacle_summary": obstacle_summary(event_data),
            "control_summary": control_summary(planner_snapshot, event_data),
            "planning_summary": planning_summary(planner_snapshot, event_data),
            "pointcloud_summary": pointcloud_summary,
        }

        image_feature = load_image_feature_from_bgr(curr_bgr, self.image_size)
        context_feature = build_context_feature(row)
        context_matrix = self.vectorizer.transform([context_feature]).astype(np.float32)
        x = np.concatenate([image_feature.reshape(1, -1), context_matrix], axis=1)
        probs = self.model.predict_proba(x)[0]
        top_indices = np.argsort(probs)[::-1][: min(3, len(probs))]
        pred_index = int(top_indices[0])
        pred_label = str(self.label_encoder.inverse_transform([pred_index])[0])
        confidence = float(probs[pred_index])
        top_candidates = [
            {
                "label_ko": str(self.label_encoder.inverse_transform([int(idx)])[0]),
                "confidence": float(probs[int(idx)]),
            }
            for idx in top_indices
        ]
        infer_ms = (time.perf_counter() - t0) * 1000.0

        scene_summary, reason, aux = infer_scene_and_reason(pred_label, row)
        payload = {
            "frame_index": int(center["frame_index"]),
            "stamp": center["stamp"],
            "event_label": row["event_label"],
            "primary_object_ko": pred_label,
            "confidence": confidence,
            "top_candidates": top_candidates,
            "motion_state": row["motion_state"],
            "steering_direction": aux["steering_direction"],
            "path_change_direction": aux["path_change_direction"],
            "planner_reason": row["planner_reason"],
            "motion_summary": motion_summary,
            "scene_summary_ko": scene_summary,
            "driving_reason_ko": reason,
            "infer_ms": infer_ms,
        }
        self.message_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        rospy.loginfo(
            "[STUDENT-XAI-RICH] frame=%d | event=%s | object=%s | confidence=%.2f | scene=%s | final=%s",
            int(center["frame_index"]),
            row["event_label"],
            pred_label,
            confidence,
            scene_summary,
            reason,
        )

        self.overlay_bgr = render_panel(
            curr_bgr=curr_bgr,
            pred_label=pred_label,
            confidence=confidence,
            event_label=row["event_label"],
            scene_summary=scene_summary,
            reason=reason,
            infer_ms=infer_ms,
            top_candidates=top_candidates,
            row=row,
        )
        overlay_msg = self.bridge.cv2_to_imgmsg(self.overlay_bgr, encoding="bgr8")
        overlay_msg.header = center["header"]
        self.overlay_pub.publish(overlay_msg)

        if self.display_window:
            cv2.imshow("student_xai_rich_realtime", self.overlay_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                rospy.signal_shutdown("user requested quit")


def main():
    rospy.init_node("student_xai_rich_node")
    StudentXAIRichNode()
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

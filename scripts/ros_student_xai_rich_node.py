#!/usr/bin/env python3
import json
import math
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
from nav_msgs.msg import Odometry, Path as NavPath
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool, String

from export_camera_only_teacher_dataset import summarize_flow
from export_teacher_dataset import (
    control_summary,
    motion_state,
    obstacle_summary,
    planner_reason,
    planning_summary,
    safe_json_loads,
    summarize_stop_hits_cloud,
    stamp_to_float,
)
from model_runtime_common import default_model_path, load_model_info, model_log_fields
from mjpeg_streamer import MjpegStreamer
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


def quaternion_to_yaw(quat):
    x = float(getattr(quat, "x", 0.0))
    y = float(getattr(quat, "y", 0.0))
    z = float(getattr(quat, "z", 0.0))
    w = float(getattr(quat, "w", 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def summarize_actual_motion(odom_window):
    if len(odom_window) < 2:
        return {
            "received": False,
            "linear_speed_mps": 0.0,
            "yaw_rate_radps": 0.0,
            "motion_state": "unknown",
            "motion_ko": "실제 이동 정보 없음",
        }

    first = odom_window[0]
    last = odom_window[-1]
    dt = max(1e-3, float(last["stamp"]) - float(first["stamp"]))
    dx = float(last["x"]) - float(first["x"])
    dy = float(last["y"]) - float(first["y"])
    distance = math.hypot(dx, dy)
    linear_speed = distance / dt
    dyaw = normalize_angle(float(last["yaw"]) - float(first["yaw"]))
    yaw_rate = dyaw / dt

    abs_yaw_rate = abs(yaw_rate)
    if linear_speed < 0.008 and abs_yaw_rate < 0.03:
        motion_state = "stopped"
        motion_ko = "정지"
    elif linear_speed < 0.008 and abs_yaw_rate >= 0.03:
        motion_state = "in_place_left" if yaw_rate > 0.0 else "in_place_right"
        motion_ko = "제자리 좌회전" if yaw_rate > 0.0 else "제자리 우회전"
    else:
        if abs_yaw_rate < 0.06:
            motion_state = "forward_straight"
            motion_ko = "전진"
        elif yaw_rate > 0.0:
            motion_state = "forward_left"
            motion_ko = "전진 좌회전"
        else:
            motion_state = "forward_right"
            motion_ko = "전진 우회전"

    return {
        "received": True,
        "linear_speed_mps": linear_speed,
        "yaw_rate_radps": yaw_rate,
        "motion_state": motion_state,
        "motion_ko": motion_ko,
    }


def canonical_obstacle_label(pred_label):
    text = str(pred_label or "").strip()
    if not text or text == "벽":
        return "장애물"
    return text


def choose_josa(word, with_batchim, without_batchim):
    text = str(word or "").strip()
    if not text:
        return without_batchim
    last = ord(text[-1])
    if 0xAC00 <= last <= 0xD7A3:
        has_batchim = ((last - 0xAC00) % 28) != 0
        return with_batchim if has_batchim else without_batchim
    return without_batchim


def direction_from_control(control):
    motion_state_text = str((control or {}).get("motion_state") or "").lower()
    if "left" in motion_state_text:
        return "left"
    if "right" in motion_state_text:
        return "right"
    angular_z = float((control or {}).get("angular_z_radps") or 0.0)
    if angular_z > 0.08:
        return "left"
    if angular_z < -0.08:
        return "right"
    return ""


def direction_from_lateral_shift(lateral_shift_m):
    shift = float(lateral_shift_m or 0.0)
    if shift >= 0.15:
        return "left"
    if shift <= -0.15:
        return "right"
    return ""


def direction_from_obstacle_side(obstacle):
    centroid = (obstacle or {}).get("near_raw_centroid_xyz") or {}
    cy = centroid.get("y")
    if cy is None:
        return ""
    cy = float(cy)
    if cy <= -0.25:
        return "left"
    if cy >= 0.25:
        return "right"
    return ""


def choose_avoidance_direction(
    planning,
    control,
    obstacle=None,
    allow_control_fallback=False,
    allow_path_fallback=False,
):
    obstacle_direction = direction_from_obstacle_side(obstacle)
    if obstacle_direction in ("left", "right"):
        return obstacle_direction

    explicit = str((planning or {}).get("avoid_direction") or "")
    if explicit in ("left", "right"):
        return explicit

    path_direction = str((planning or {}).get("path_change_direction") or "")
    command_direction = direction_from_control(control)
    shift_direction = direction_from_lateral_shift((planning or {}).get("path_change_lateral_shift_m"))

    candidates = [
        direction
        for direction in (
            command_direction if allow_control_fallback else "",
            path_direction if allow_path_fallback else "",
            shift_direction if allow_path_fallback else "",
        )
        if direction in ("left", "right")
    ]
    if not candidates:
        return ""

    left_count = candidates.count("left")
    right_count = candidates.count("right")
    if left_count > right_count:
        return "left"
    if right_count > left_count:
        return "right"
    if command_direction in ("left", "right"):
        return command_direction
    return path_direction if path_direction in ("left", "right") else candidates[0]


def infer_driving_message(pred_label, row, actual_motion):
    obstacle = row.get("obstacle_summary") or {}
    planning = row.get("planning_summary") or {}
    control = row.get("control_summary") or {}
    emergency = row.get("emergency_summary") or {}
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
    resolved_avoidance_direction = str(planning.get("resolved_avoidance_direction") or "")
    avoid_direction = str(planning.get("avoid_direction") or "")
    planner_action_taken = str(planning.get("planner_action_taken") or "")
    planner_trigger_reason = str(planning.get("planner_trigger_reason") or "")
    if resolved_avoidance_direction in ("left", "right"):
        avoidance_direction = resolved_avoidance_direction
    elif avoid_direction in ("left", "right"):
        avoidance_direction = avoid_direction
    elif not planner_action_taken:
        avoidance_direction = path_change_direction
    else:
        avoidance_direction = ""
    path_change_lateral_shift_m = float(planning.get("path_change_lateral_shift_m") or 0.0)
    behavior_reason = str(planning.get("behavior_reason") or row.get("planner_reason") or "unknown")
    distance_to_goal_m = planning.get("distance_to_goal_m")
    if distance_to_goal_m is not None:
        distance_to_goal_m = float(distance_to_goal_m)
    raw_global_path_length_m = planning.get("global_path_length_m")
    raw_global_path_points = planning.get("global_path_points")
    raw_local_path_length_m = planning.get("local_path_length_m")
    raw_local_path_points = planning.get("local_path_points")
    local_path_available = (
        raw_local_path_length_m is not None
        and raw_local_path_points is not None
        and int(raw_local_path_points or 0) > 0
    )
    global_path_available = (
        raw_global_path_length_m is not None
        and raw_global_path_points is not None
        and int(raw_global_path_points or 0) > 0
    )
    global_path_length_m = float(raw_global_path_length_m or 0.0)
    global_path_points = int(raw_global_path_points or 0)
    local_path_length_m = float(raw_local_path_length_m or 0.0)
    local_path_points = int(raw_local_path_points or 0)
    remaining_path_length_m = local_path_length_m if local_path_available else global_path_length_m
    remaining_path_points = local_path_points if local_path_available else global_path_points
    path_summary_available = local_path_available or global_path_available
    path_length_reached = (
        path_summary_available
        and not path_blocked
        and remaining_path_length_m <= 0.35
        and 0 < remaining_path_points <= 3
    )
    path_length_hold = (
        path_summary_available
        and not path_blocked
        and remaining_path_length_m <= 0.50
        and 0 < remaining_path_points <= 4
    )
    active_route = (
        path_summary_available
        and remaining_path_length_m > 0.50
        and remaining_path_points > 2
    )
    cmd_linear = float(control.get("linear_x_mps") or 0.0)
    cmd_angular = float(control.get("angular_z_radps") or 0.0)
    actual_motion_ko = str(actual_motion.get("motion_ko") or "실제 이동 정보 없음")
    actual_speed = float(actual_motion.get("linear_speed_mps") or 0.0)
    actual_yaw_rate = float(actual_motion.get("yaw_rate_radps") or 0.0)
    actual_motion_received = bool(actual_motion.get("received"))
    commanded_move = abs(cmd_linear) > 0.03 or abs(cmd_angular) > 0.08
    command_stop = not commanded_move
    actual_move = actual_speed > 0.015 or abs(actual_yaw_rate) > 0.03
    safety_stop_mismatch = actual_motion_received and commanded_move and not actual_move
    control_stop = actual_motion_received and not commanded_move and not actual_move
    goal_distance_reached = (
        distance_to_goal_m is not None
        and distance_to_goal_m <= 0.6
        and (command_stop or (actual_motion_received and not actual_move))
    )
    destination_reached = path_length_reached or goal_distance_reached
    destination_hold = path_length_hold or goal_distance_reached
    emergency_stop_active = bool(emergency.get("emergency_stop_active"))
    safety_mode_latched = bool(emergency.get("safety_mode_latched"))
    astar_path_blocked = bool(emergency.get("astar_path_blocked"))
    behavior_stop = bool(planning.get("behavior_stop"))
    planner_reports_avoidance = planner_action_taken == "follow_avoidance"
    planner_reports_clear = planner_action_taken in ("follow_local", "stop")
    has_avoidance_direction = avoidance_direction in ("left", "right")
    legacy_avoidance_signal = (
        not planner_action_taken
        and path_change_changed
        and (path_change_direction in ("left", "right") or abs(path_change_lateral_shift_m) >= 0.15)
    )
    obstacle_label = canonical_obstacle_label(pred_label)
    obj_object = obstacle_label + choose_josa(obstacle_label, "을", "를")
    obj_subject = obstacle_label + choose_josa(obstacle_label, "이", "가")
    strong_avoidance_change = (
        not emergency_stop_active
        and not behavior_stop
        and not safety_mode_latched
        and not planner_reports_clear
        and (
            (planner_reports_avoidance and has_avoidance_direction)
            or legacy_avoidance_signal
        )
    )
    generic_avoidance_change = (
        not emergency_stop_active
        and not behavior_stop
        and not safety_mode_latched
        and planner_reports_avoidance
        and not has_avoidance_direction
    )

    if destination_reached or (destination_hold and command_stop):
        scene = "목적지 도착"
        reason = "목적지에 도착해 정지한 상태로 본다."
        driving_mode = "목적지 도착"
    elif emergency_stop_active or behavior_stop:
        scene = "안전모드 정지"
        reason = "안전모드가 활성화되어 정지를 우선하는 상태로 본다."
        driving_mode = "안전모드 정지"
    elif safety_stop_mismatch:
        scene = "안전모드 정지"
        reason = "주행 명령은 남아 있지만 실제 이동이 거의 없어 안전모드 정지 상태로 본다."
        driving_mode = "안전모드 정지"
    elif control_stop and (emergency_stop_active or behavior_stop or safety_mode_latched):
        scene = "안전모드 정지"
        reason = "안전모드가 유지되어 제어 속도 명령과 실제 이동이 모두 0에 가까운 정지 상태로 본다."
        driving_mode = "안전모드 정지"
    elif control_stop and path_blocked:
        scene = "경로 차단에 따른 제어 정지"
        reason = "경로 차단을 반영해 제어 속도 명령이 0에 가까워졌고 실제 이동도 멈춘 상태로 본다."
        driving_mode = "제어 정지"
    elif control_stop and active_route:
        scene = "출발 대기 정지"
        reason = "목적지 경로는 생성됐지만 제어 속도 명령과 실제 이동이 모두 0에 가까워 아직 출발하지 않은 정지 상태로 본다."
        driving_mode = "출발 대기"
    elif control_stop:
        scene = "제어 정지"
        reason = "제어 속도 명령과 실제 이동이 모두 0에 가까워 정지 상태로 본다."
        driving_mode = "제어 정지"
    elif path_blocked:
        scene = "전방 {} 장애물로 경로 차단".format(side)
        reason = "전방 {}에 {} 인지되어 경로가 막혀 정지 또는 재계획을 진행하고 있다고 본다.".format(
            side, obj_subject
        )
        driving_mode = "경로 차단"
    elif strong_avoidance_change:
        if avoidance_direction == "left":
            scene = "좌측 회피 경로 주행"
            reason = "전방 {}의 {} 피해 좌측 회피 경로로 주행을 진행하고 있다고 본다.".format(
                side, obj_object
            )
            driving_mode = "좌측 회피"
        elif avoidance_direction == "right":
            scene = "우측 회피 경로 주행"
            reason = "전방 {}의 {} 피해 우측 회피 경로로 주행을 진행하고 있다고 본다.".format(
                side, obj_object
            )
            driving_mode = "우측 회피"
        else:
            turn_hint = "좌회전" if steering == "left" else "우회전" if steering == "right" else "회피"
            scene = "회피 경로 재계획"
            reason = "전방 {}의 {} 영향으로 {} 기반 회피 경로를 따라 주행하고 있다고 본다.".format(
                side, obj_subject, turn_hint
            )
            driving_mode = "회피 주행"
    elif generic_avoidance_change:
        scene = "회피 경로 주행"
        reason = "전방 장애물 영향으로 회피 경로를 따라 주행하고 있다고 본다."
        driving_mode = "회피 주행"
    else:
        if steering in ("left", "right") and actual_move:
            steer_ko = "좌측 조향" if steering == "left" else "우측 조향"
            scene = "정상 경로 조향 보정"
            reason = "현재 정상 경로를 따라가며 {}으로 주행을 보정하고 있다고 본다.".format(steer_ko)
        else:
            scene = "정상 경로 주행"
            reason = "현재 정상 경로를 따라 목적지까지 주행을 진행하고 있다고 본다."
        driving_mode = "정상 경로"

    return scene, reason, {
        "steering_direction": steering,
        "motion_state": motion_state_text,
        "path_change_direction": path_change_direction,
        "avoidance_direction": avoidance_direction,
        "distance_to_goal_m": distance_to_goal_m,
        "planner_action_taken": planner_action_taken,
        "planner_trigger_reason": planner_trigger_reason,
        "behavior_reason": behavior_reason,
        "actual_motion_ko": actual_motion_ko,
        "actual_speed_mps": actual_speed,
        "actual_yaw_rate_radps": actual_yaw_rate,
        "command_actual_mismatch": safety_stop_mismatch,
        "control_stop": control_stop,
        "emergency_stop_active": emergency_stop_active,
        "safety_mode_latched": safety_mode_latched,
        "astar_path_blocked": astar_path_blocked,
        "driving_mode": driving_mode,
        "obstacle_label": obstacle_label,
    }


def format_top_candidates(top_candidates):
    if not top_candidates:
        return "후보 없음"
    return ", ".join("{} {:.2f}".format(item["label_ko"], float(item["confidence"])) for item in top_candidates)


def driving_mode_color(mode):
    mapping = {
        "안전모드 정지": (230, 205, 45),
        "출발 대기": (210, 190, 70),
        "제어 정지": (180, 180, 180),
        "목적지 도착": (70, 180, 220),
        "경로 차단": (220, 70, 70),
        "좌측 회피": (245, 166, 35),
        "우측 회피": (245, 166, 35),
        "회피 주행": (245, 166, 35),
        "실제 정지": (210, 190, 60),
        "정상 경로": (80, 170, 95),
    }
    return mapping.get(mode, (110, 110, 110))


def driving_mode_family(mode):
    text = str(mode or "")
    if "회피" in text:
        return "avoid"
    if "차단" in text:
        return "blocked"
    if "도착" in text:
        return "arrival"
    if "안전모드" in text or "출발 대기" in text or "제어 정지" in text or "실제 정지" in text:
        return "actual_stop"
    return "normal"


def render_panel(curr_bgr, pred_label, confidence, event_label, scene_summary, reason, infer_ms, top_candidates, row, summary_payload):
    h, w = curr_bgr.shape[:2]
    panel_w = 520
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = curr_bgr
    canvas[:, w:] = (18, 18, 18)

    planning = row.get("planning_summary") or {}
    control = row.get("control_summary") or {}
    obstacle = row.get("obstacle_summary") or {}
    actual_motion = row.get("actual_motion_summary") or {}
    near_range = obstacle.get("near_raw_min_range_m")

    pil = PILImage.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    title_font = find_font(28)
    body_font = find_font(22)
    small_font = find_font(18)
    headline_font = find_font(30)
    reason_font = find_font(24)

    x0 = w + 20
    y = 24
    draw.text((x0, y), "student xai thought", fill=(255, 255, 255), font=title_font)
    y += 46

    driving_mode = str(summary_payload.get("driving_mode_ko") or "unknown")
    badge_rgb = driving_mode_color(driving_mode)
    badge_h = 42
    badge_box = (x0, y, x0 + 220, y + badge_h)
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(badge_box, radius=10, fill=badge_rgb)
    else:
        draw.rectangle(badge_box, fill=badge_rgb)
    draw.text((x0 + 12, y + 6), driving_mode, fill=(15, 15, 15), font=body_font)
    y += badge_h + 16

    draw.text((x0, y), "latest reason", fill=(255, 255, 255), font=small_font)
    y += 24
    for line in wrap_text(reason, width=22):
        draw.text((x0, y), line, fill=(255, 245, 210), font=reason_font)
        y += 32
    y += 8

    draw.text((x0, y), "scene: {}".format(scene_summary), fill=(220, 220, 220), font=body_font)
    y += 34

    lines = [
        "event: {}".format(event_label),
        "대표 객체: {} ({:.2f})".format(pred_label, confidence),
        "장애물 해석: {}".format(summary_payload.get("obstacle_label_ko") or pred_label),
        "cmd motion: {}".format(control.get("motion_state") or "unknown"),
        "cmd v/w: {:.2f} / {:.2f}".format(
            float(control.get("linear_x_mps") or 0.0),
            float(control.get("angular_z_radps") or 0.0),
        ),
        "actual motion: {}".format(actual_motion.get("motion_ko") or "unknown"),
        "steer: {}".format(control.get("steering_direction") or "unknown"),
        "path_change: {}".format(planning.get("path_change_direction") or "unknown"),
        "avoid_dir: {}".format(
            planning.get("resolved_avoidance_direction")
            or planning.get("avoid_direction")
            or "unknown"
        ),
        "actual speed: {:.2f} m/s".format(float(actual_motion.get("linear_speed_mps") or 0.0)),
        "near_range: {}".format("n/a" if near_range is None else "{:.2f} m".format(float(near_range))),
        "후보: {}".format(format_top_candidates(top_candidates)),
        "planner: {}".format(planning.get("behavior_reason") or row.get("planner_reason") or "unknown"),
        "infer: {:.1f} ms".format(infer_ms),
        "q: quit",
    ]
    for idx, line in enumerate(lines):
        font = body_font if idx < 4 else small_font
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
        self.stop_hits_topic = get_private_param(
            "stop_hits_topic", "/planning/near_field_stop_hits", explicit_args
        )
        self.odom_topic = get_private_param("odom_topic", "/lio_localizer/odometry/optimization", explicit_args)
        self.emergency_stop_topic = get_private_param(
            "emergency_stop_topic", "/planning/emergency_stop", explicit_args
        )
        self.astar_path_blocked_topic = get_private_param(
            "astar_path_blocked_topic", "/astar/path_blocked", explicit_args
        )
        self.global_path_topic = get_private_param("global_path_topic", "/astar/path", explicit_args)
        self.output_topic = get_private_param("output_topic", "/xai/vlm_log", explicit_args)
        self.overlay_topic = get_private_param("overlay_topic", "/student_xai/rich_overlay", explicit_args)
        legacy_model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "outdoor_sign_vehicle_rich_full"
            / "student_baseline.joblib"
        )
        self.model_path = Path(
            get_private_param(
                "model_path",
                default_model_path(legacy_model_path),
                explicit_args,
            )
        ).expanduser().resolve()
        self.model_info = load_model_info(self.model_path)
        self.model_fields = model_log_fields(self.model_info, self.model_path)
        self.flow_image_side_px = int(get_private_param("flow_image_side_px", 320, explicit_args))
        self.flow_motion_threshold = float(get_private_param("flow_motion_threshold", 1.5, explicit_args))
        self.display_window = _coerce_bool(get_private_param("display_window", True, explicit_args))
        self.max_image_age_s = float(get_private_param("max_image_age_s", 0.30, explicit_args))
        self.max_planner_age_s = float(get_private_param("max_planner_age_s", 0.80, explicit_args))
        self.max_pointcloud_age_s = float(get_private_param("max_pointcloud_age_s", 0.60, explicit_args))
        self.event_hold_s = float(get_private_param("event_hold_s", 1.5, explicit_args))
        self.normal_status_period_s = float(get_private_param("normal_status_period_s", 1.0, explicit_args))
        self.render_latest_on_image = _coerce_bool(get_private_param("render_latest_on_image", True, explicit_args))
        self.log_every_event = _coerce_bool(get_private_param("log_every_event", True, explicit_args))
        self.web_stream = _coerce_bool(get_private_param("web_stream", True, explicit_args))
        self.web_stream_host = str(get_private_param("web_stream_host", "127.0.0.1", explicit_args))
        self.web_stream_port = int(get_private_param("web_stream_port", 8090, explicit_args))
        self.web_stream_path = str(get_private_param("web_stream_path", "/stream.mjpg", explicit_args))
        self.web_streamer = None

        bundle = joblib.load(str(self.model_path))
        self.model = bundle["model"]
        self.vectorizer = bundle["vectorizer"]
        self.label_encoder = bundle["label_encoder"]
        self.image_size = int(bundle["image_size"])

        self.bridge = CvBridge()
        self.frames = deque(maxlen=3)
        self.odom_window = deque(maxlen=20)
        self.latest_planner = None
        self.latest_cloud = None
        self.latest_stop_hits = None
        self.latest_emergency_stop = None
        self.latest_astar_path_blocked = None
        self.latest_global_goal = None
        self.overlay_bgr = None
        self.latest_render_state = None
        self.last_event_display_stamp = None
        self.last_normal_status_stamp = 0.0
        self.frame_index = -1
        self.current_summary_payload = None
        self.current_summary_key = None
        self.last_logged_key = None
        self.safety_mode_latched = False
        self.current_avoidance_direction = None

        self.message_pub = rospy.Publisher(self.output_topic, String, queue_size=20)
        self.overlay_pub = rospy.Publisher(self.overlay_topic, Image, queue_size=2)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self._on_image, queue_size=1)
        self.event_sub = rospy.Subscriber(self.event_topic, String, self._on_event, queue_size=20)
        self.planner_sub = rospy.Subscriber(self.planner_topic, String, self._on_planner, queue_size=20)
        self.point_cloud_sub = rospy.Subscriber(self.point_cloud_topic, PointCloud2, self._on_point_cloud, queue_size=2)
        self.stop_hits_sub = rospy.Subscriber(self.stop_hits_topic, PointCloud2, self._on_stop_hits, queue_size=2)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._on_odom, queue_size=20)
        self.emergency_stop_sub = rospy.Subscriber(self.emergency_stop_topic, Bool, self._on_emergency_stop, queue_size=20)
        self.astar_path_blocked_sub = rospy.Subscriber(
            self.astar_path_blocked_topic, Bool, self._on_astar_path_blocked, queue_size=20
        )
        self.global_path_sub = rospy.Subscriber(self.global_path_topic, NavPath, self._on_global_path, queue_size=2)
        if self.web_stream:
            self.web_streamer = MjpegStreamer(
                host=self.web_stream_host,
                port=self.web_stream_port,
                stream_path=self.web_stream_path,
                title="student_xai_rich_realtime",
            )
            if self.web_streamer.start():
                rospy.loginfo("student_xai_rich web stream started | url=%s", self.web_streamer.stream_url)
            else:
                rospy.logwarn(
                    "student_xai_rich web stream 시작 실패 | host=%s port=%s",
                    self.web_stream_host,
                    self.web_stream_port,
                )
                self.web_streamer = None
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo(
            "student_xai_rich_node started | image=%s event=%s planner=%s point_cloud=%s stop_hits=%s emergency=%s astar_blocked=%s global_path=%s odom=%s output=%s overlay=%s model=%s display_window=%s web_stream=%s",
            self.image_topic,
            self.event_topic,
            self.planner_topic,
            self.point_cloud_topic,
            self.stop_hits_topic,
            self.emergency_stop_topic,
            self.astar_path_blocked_topic,
            self.global_path_topic,
            self.odom_topic,
            self.output_topic,
            self.overlay_topic,
            self.model_path,
            self.display_window,
            self.web_streamer.stream_url if self.web_streamer is not None else "off",
        )

    def _on_shutdown(self):
        if self.web_streamer is not None:
            self.web_streamer.stop()

    def _on_image(self, msg):
        self.frame_index += 1
        image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        image_stamp = float(msg.header.stamp.to_sec())
        self.frames.append(
            {
                "stamp": image_stamp,
                "frame_index": self.frame_index,
                "image_bgr": image_bgr,
                "header": msg.header,
            }
        )
        if not self.render_latest_on_image:
            if self.overlay_bgr is not None:
                overlay_msg = self.bridge.cv2_to_imgmsg(self.overlay_bgr, encoding="bgr8")
                overlay_msg.header = msg.header
                self.overlay_pub.publish(overlay_msg)
            return

        recent_event = (
            self.latest_render_state is not None
            and self.last_event_display_stamp is not None
            and abs(image_stamp - float(self.last_event_display_stamp)) <= self.event_hold_s
        )
        if recent_event:
            self._publish_overlay(image_bgr, msg.header, self.latest_render_state)
            return

        self._publish_normal_status(image_bgr, msg.header, image_stamp)

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

    def _on_stop_hits(self, msg):
        self.latest_stop_hits = summarize_stop_hits_cloud(msg, 2500)

    def _on_emergency_stop(self, msg):
        self.latest_emergency_stop = {
            "stamp": rospy.Time.now().to_sec(),
            "value": bool(getattr(msg, "data", False)),
        }

    def _on_astar_path_blocked(self, msg):
        self.latest_astar_path_blocked = {
            "stamp": rospy.Time.now().to_sec(),
            "value": bool(getattr(msg, "data", False)),
        }

    def _on_global_path(self, msg):
        if not msg.poses:
            return
        goal = msg.poses[-1].pose.position
        self.latest_global_goal = {
            "stamp": float(msg.header.stamp.to_sec()),
            "x": float(goal.x),
            "y": float(goal.y),
            "points": int(len(msg.poses)),
        }

    def _on_odom(self, msg):
        pose = msg.pose.pose
        self.odom_window.append(
            {
                "stamp": float(msg.header.stamp.to_sec()),
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "yaw": quaternion_to_yaw(pose.orientation),
            }
        )

    def _distance_to_goal(self):
        if self.latest_global_goal is None or not self.odom_window:
            return None
        odom = self.odom_window[-1]
        return math.hypot(
            float(odom["x"]) - float(self.latest_global_goal["x"]),
            float(odom["y"]) - float(self.latest_global_goal["y"]),
        )

    def _add_goal_distance(self, planning):
        planning = dict(planning or {})
        distance_to_goal = self._distance_to_goal()
        if distance_to_goal is not None:
            planning["distance_to_goal_m"] = distance_to_goal
            planning["global_goal_points"] = int((self.latest_global_goal or {}).get("points") or 0)
        return planning

    def _publish_overlay(self, image_bgr, header, render_state):
        self.overlay_bgr = render_panel(
            curr_bgr=image_bgr,
            pred_label=render_state["pred_label"],
            confidence=render_state["confidence"],
            event_label=render_state["event_label"],
            scene_summary=render_state["scene_summary"],
            reason=render_state["reason"],
            infer_ms=render_state["infer_ms"],
            top_candidates=render_state["top_candidates"],
            row=render_state["row"],
            summary_payload=render_state["summary_payload"],
        )
        overlay_msg = self.bridge.cv2_to_imgmsg(self.overlay_bgr, encoding="bgr8")
        overlay_msg.header = header
        self.overlay_pub.publish(overlay_msg)
        if self.web_streamer is not None:
            self.web_streamer.update(self.overlay_bgr)

        if self.display_window:
            cv2.imshow("student_xai_rich_realtime", self.overlay_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                rospy.signal_shutdown("user requested quit")

    def _publish_normal_status(self, image_bgr, header, image_stamp):
        planner_snapshot = self.latest_planner["data"] if self.latest_planner is not None else {}
        actual_motion = summarize_actual_motion(self.odom_window)
        normal_planning = self._add_goal_distance(planning_summary(planner_snapshot, {}))
        normal_planning.update(
            {
                "path_blocked": False,
                "path_change_changed": False,
                "path_change_direction": "unknown",
                "avoid_direction": "",
                "resolved_avoidance_direction": "",
            }
        )
        row = {
            "source_bag_stem": "record_real_realtime",
            "event_label": "normal_route",
            "planner_reason": planner_reason(planner_snapshot, {}),
            "motion_state": motion_state(planner_snapshot, {}),
            "path_blocked": False,
            "motion_summary": {},
            "obstacle_summary": {
                "near_raw_min_range_m": None,
                "near_raw_centroid_xyz": {},
            },
            "stop_hits_summary": self.latest_stop_hits or {},
            "control_summary": control_summary(planner_snapshot, {}),
            "planning_summary": normal_planning,
            "pointcloud_summary": {},
            "actual_motion_summary": actual_motion,
            "emergency_summary": {
                "emergency_stop_active": bool(self.latest_emergency_stop["value"]) if self.latest_emergency_stop else False,
                "safety_mode_latched": self.safety_mode_latched,
                "astar_path_blocked": bool(self.latest_astar_path_blocked["value"]) if self.latest_astar_path_blocked else False,
            },
        }
        scene_summary, reason, aux = infer_driving_message("장애물 없음", row, actual_motion)
        summary_payload = {
            "driving_mode_ko": aux.get("driving_mode"),
            "scene_summary_ko": scene_summary,
            "driving_reason_ko": reason,
            "obstacle_label_ko": aux.get("obstacle_label"),
            "updated_event_label": row["event_label"],
        }
        render_state = {
            "pred_label": "장애물 없음",
            "confidence": 0.0,
            "event_label": row["event_label"],
            "scene_summary": scene_summary,
            "reason": reason,
            "infer_ms": 0.0,
            "top_candidates": [],
            "row": row,
            "summary_payload": summary_payload,
        }
        self._publish_overlay(image_bgr, header, render_state)

        if image_stamp - self.last_normal_status_stamp < self.normal_status_period_s:
            return
        self.last_normal_status_stamp = image_stamp
        payload = {
            **self.model_fields,
            "frame_index": int(self.frame_index),
            "stamp": image_stamp,
            "event_label": row["event_label"],
            "prediction": aux["driving_mode"],
            "primary_object_ko": "장애물 없음",
            "confidence": 0.0,
            "top_candidates": [],
            "motion_state": row["motion_state"],
            "actual_motion": actual_motion,
            "steering_direction": aux["steering_direction"],
            "cmd_linear_x_mps": float(row["control_summary"].get("linear_x_mps") or 0.0),
            "cmd_angular_z_radps": float(row["control_summary"].get("angular_z_radps") or 0.0),
            "path_change_direction": aux["path_change_direction"],
            "avoidance_direction": aux["avoidance_direction"],
            "distance_to_goal_m": aux["distance_to_goal_m"],
            "planner_reason": row["planner_reason"],
            "motion_summary": {},
            "driving_mode_ko": aux["driving_mode"],
            "obstacle_label_ko": aux["obstacle_label"],
            "command_actual_mismatch": aux["command_actual_mismatch"],
            "control_stop": aux["control_stop"],
            "emergency_stop_active": aux["emergency_stop_active"],
            "safety_mode_latched": aux["safety_mode_latched"],
            "astar_path_blocked": aux["astar_path_blocked"],
            "scene_summary_ko": scene_summary,
            "driving_reason_ko": reason,
            "explanation": reason,
            "infer_ms": 0.0,
        }
        self.message_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        rospy.loginfo(
            "[STUDENT-XAI-RICH] frame=%d | event=%s | mode=%s | object=%s | confidence=%.2f | scene=%s | final=%s",
            int(self.frame_index),
            row["event_label"],
            aux["driving_mode"],
            "장애물 없음",
            0.0,
            scene_summary,
            reason,
        )

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
        stop_hits_summary = self.latest_stop_hits or {
            "frame_id": None,
            "stamp": None,
            "point_count": 0,
            "min_range_m": None,
            "min_x_m": None,
        }
        event_decision = event_data.get("decision") or {}
        planner_decision = planner_snapshot.get("decision") or {}
        event_emergency_value = (event_decision.get("emergency_stop") or {}).get("value")
        planner_emergency_value = (planner_decision.get("emergency_stop") or {}).get("value")
        event_path_blocked_value = (event_decision.get("path_blocked") or {}).get("value")
        planner_path_blocked_value = (planner_decision.get("path_blocked") or {}).get("value")
        latest_emergency_value = (
            self.latest_emergency_stop["value"] if self.latest_emergency_stop else None
        )
        latest_astar_path_blocked_value = (
            self.latest_astar_path_blocked["value"] if self.latest_astar_path_blocked else None
        )

        emergency_stop_active = next(
            (
                bool(value)
                for value in (
                    event_emergency_value,
                    planner_emergency_value,
                    latest_emergency_value,
                )
                if value is not None
            ),
            False,
        )
        astar_path_blocked = next(
            (
                bool(value)
                for value in (
                    event_path_blocked_value,
                    planner_path_blocked_value,
                    latest_astar_path_blocked_value,
                )
                if value is not None
            ),
            False,
        )
        row = {
            "source_bag_stem": "record_real_realtime",
            "event_label": str(event_data.get("event_label") or "unknown"),
            "planner_reason": planner_reason(planner_snapshot, event_data),
            "motion_state": motion_state(planner_snapshot, event_data),
            "path_blocked": bool((((event_data.get("decision") or {}).get("path_blocked") or {}).get("value"))),
            "motion_summary": motion_summary,
            "obstacle_summary": obstacle_summary(event_data),
            "stop_hits_summary": stop_hits_summary,
            "control_summary": control_summary(planner_snapshot, event_data),
            "planning_summary": self._add_goal_distance(planning_summary(planner_snapshot, event_data)),
            "pointcloud_summary": pointcloud_summary,
        }
        actual_motion = summarize_actual_motion(self.odom_window)
        control = row["control_summary"]
        cmd_linear = float(control.get("linear_x_mps") or 0.0)
        cmd_angular = float(control.get("angular_z_radps") or 0.0)
        commanded_move = abs(cmd_linear) > 0.03 or abs(cmd_angular) > 0.08
        actual_speed = float(actual_motion.get("linear_speed_mps") or 0.0)
        actual_yaw_rate = float(actual_motion.get("yaw_rate_radps") or 0.0)
        actual_move = actual_speed > 0.015 or abs(actual_yaw_rate) > 0.03
        behavior_stop = bool(row["planning_summary"].get("behavior_stop"))
        planning = row["planning_summary"]
        planner_action_taken = str(planning.get("planner_action_taken") or "")

        if planner_action_taken == "follow_avoidance":
            obstacle_direction = direction_from_obstacle_side(row["obstacle_summary"])
            if obstacle_direction in ("left", "right"):
                avoidance_direction = obstacle_direction
            elif self.current_avoidance_direction in ("left", "right"):
                avoidance_direction = self.current_avoidance_direction
            else:
                avoidance_direction = choose_avoidance_direction(
                    planning,
                    control,
                    row["obstacle_summary"],
                )
            if avoidance_direction in ("left", "right"):
                self.current_avoidance_direction = avoidance_direction
            if self.current_avoidance_direction in ("left", "right"):
                planning["resolved_avoidance_direction"] = self.current_avoidance_direction
        else:
            self.current_avoidance_direction = None
            planning["resolved_avoidance_direction"] = ""

        if emergency_stop_active or behavior_stop:
            self.safety_mode_latched = True
        elif actual_move:
            self.safety_mode_latched = False

        emergency_summary = {
            "emergency_stop_active": emergency_stop_active,
            "safety_mode_latched": self.safety_mode_latched,
            "astar_path_blocked": astar_path_blocked,
        }
        row["emergency_summary"] = emergency_summary
        row["actual_motion_summary"] = actual_motion

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

        scene_summary, reason, aux = infer_driving_message(pred_label, row, actual_motion)
        current_summary_key = (
            str(aux.get("driving_mode") or ""),
            str(scene_summary or ""),
            str(reason or ""),
            str(aux.get("obstacle_label") or ""),
        )
        new_summary_payload = {
            "driving_mode_ko": aux.get("driving_mode"),
            "scene_summary_ko": scene_summary,
            "driving_reason_ko": reason,
            "obstacle_label_ko": aux.get("obstacle_label"),
            "updated_event_label": row["event_label"],
        }
        current_family = driving_mode_family(aux.get("driving_mode"))
        previous_family = driving_mode_family(
            None if self.current_summary_payload is None else self.current_summary_payload.get("driving_mode_ko")
        )

        should_refresh_summary = False
        if self.current_summary_payload is None:
            should_refresh_summary = True
        elif current_family != previous_family:
            should_refresh_summary = True
        elif current_family in ("avoid", "blocked", "arrival", "actual_stop") and current_summary_key != self.current_summary_key:
            should_refresh_summary = True

        if should_refresh_summary:
            self.current_summary_key = current_summary_key
            self.current_summary_payload = new_summary_payload

        payload = {
            **self.model_fields,
            "frame_index": int(center["frame_index"]),
            "stamp": center["stamp"],
            "event_label": row["event_label"],
            "prediction": aux["driving_mode"],
            "primary_object_ko": pred_label,
            "confidence": confidence,
            "top_candidates": top_candidates,
            "motion_state": row["motion_state"],
            "actual_motion": actual_motion,
            "steering_direction": aux["steering_direction"],
            "cmd_linear_x_mps": float((row.get("control_summary") or {}).get("linear_x_mps") or 0.0),
            "cmd_angular_z_radps": float((row.get("control_summary") or {}).get("angular_z_radps") or 0.0),
            "path_change_direction": aux["path_change_direction"],
            "avoidance_direction": aux["avoidance_direction"],
            "distance_to_goal_m": aux["distance_to_goal_m"],
            "planner_reason": row["planner_reason"],
            "motion_summary": motion_summary,
            "driving_mode_ko": aux["driving_mode"],
            "obstacle_label_ko": aux["obstacle_label"],
            "command_actual_mismatch": aux["command_actual_mismatch"],
            "control_stop": aux["control_stop"],
            "emergency_stop_active": aux["emergency_stop_active"],
            "safety_mode_latched": aux["safety_mode_latched"],
            "astar_path_blocked": aux["astar_path_blocked"],
            "scene_summary_ko": self.current_summary_payload["scene_summary_ko"],
            "driving_reason_ko": self.current_summary_payload["driving_reason_ko"],
            "explanation": self.current_summary_payload["driving_reason_ko"],
            "infer_ms": infer_ms,
        }
        self.message_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        if self.log_every_event or self.current_summary_key != self.last_logged_key:
            self.last_logged_key = self.current_summary_key
            rospy.loginfo(
                "[STUDENT-XAI-RICH] frame=%d | event=%s | mode=%s | object=%s | confidence=%.2f | scene=%s | final=%s | infer=%.1fms",
                int(center["frame_index"]),
                row["event_label"],
                self.current_summary_payload["driving_mode_ko"],
                pred_label,
                confidence,
                self.current_summary_payload["scene_summary_ko"],
                self.current_summary_payload["driving_reason_ko"],
                infer_ms,
            )

        self.latest_render_state = {
            "pred_label": pred_label,
            "confidence": confidence,
            "event_label": row["event_label"],
            "scene_summary": self.current_summary_payload["scene_summary_ko"],
            "reason": self.current_summary_payload["driving_reason_ko"],
            "infer_ms": infer_ms,
            "top_candidates": top_candidates,
            "row": row,
            "summary_payload": self.current_summary_payload,
        }
        self.last_event_display_stamp = float(center["stamp"])
        self._publish_overlay(curr_bgr, center["header"], self.latest_render_state)


def main():
    rospy.init_node("student_xai_rich_node")
    StudentXAIRichNode()
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
from pathlib import Path

import cv2
import numpy as np


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_image_feature(path, image_size):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return np.zeros((image_size * image_size,), dtype=np.float32)
    return load_image_feature_from_gray(image, image_size)


def load_image_feature_from_gray(image, image_size):
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    feature = image.astype(np.float32).reshape(-1) / 255.0
    return feature


def load_image_feature_from_bgr(image_bgr, image_size):
    if image_bgr is None:
        return np.zeros((image_size * image_size,), dtype=np.float32)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return load_image_feature_from_gray(image, image_size)


def build_context_feature(row):
    obstacle = row.get("obstacle_summary") or {}
    centroid = obstacle.get("near_raw_centroid_xyz") or {}
    motion = row.get("motion_summary") or {}
    prev_to_curr = motion.get("prev_to_curr") or {}
    curr_to_next = motion.get("curr_to_next") or {}

    feature = {
        "event_label={}".format(row.get("event_label") or "unknown"): 1.0,
        "motion_state={}".format(row.get("motion_state") or "unknown"): 1.0,
        "planner_reason={}".format(row.get("planner_reason") or "unknown"): 1.0,
        "path_blocked": float(bool(row.get("path_blocked"))),
        "near_raw_points": float(obstacle.get("near_raw_points") or 0.0),
        "near_raw_min_range_m": float(obstacle.get("near_raw_min_range_m") or 0.0),
        "near_raw_min_x_m": float(obstacle.get("near_raw_min_x_m") or 0.0),
        "near_raw_centroid_x": float(centroid.get("x") or 0.0),
        "near_raw_centroid_y": float(centroid.get("y") or 0.0),
        "near_raw_centroid_z": float(centroid.get("z") or 0.0),
        "source_bag={}".format(row.get("source_bag_stem") or row.get("source_bag") or "unknown"): 1.0,
        "dominant_motion={}".format(motion.get("dominant_motion_ko") or "unknown"): 1.0,
        "ego_motion={}".format(motion.get("ego_motion_ko") or "unknown"): 1.0,
        "scene_state={}".format(motion.get("scene_state_ko") or "unknown"): 1.0,
        "prev_to_curr_mean_magnitude": float(prev_to_curr.get("mean_magnitude") or 0.0),
        "prev_to_curr_moving_ratio": float(prev_to_curr.get("moving_ratio") or 0.0),
        "prev_to_curr_center_moving_ratio": float(prev_to_curr.get("center_moving_ratio") or 0.0),
        "prev_to_curr_mean_dx": float(prev_to_curr.get("mean_dx") or 0.0),
        "prev_to_curr_mean_dy": float(prev_to_curr.get("mean_dy") or 0.0),
        "curr_to_next_mean_magnitude": float(curr_to_next.get("mean_magnitude") or 0.0),
        "curr_to_next_moving_ratio": float(curr_to_next.get("moving_ratio") or 0.0),
        "curr_to_next_center_moving_ratio": float(curr_to_next.get("center_moving_ratio") or 0.0),
        "curr_to_next_mean_dx": float(curr_to_next.get("mean_dx") or 0.0),
        "curr_to_next_mean_dy": float(curr_to_next.get("mean_dy") or 0.0),
    }
    return feature

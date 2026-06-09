#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import Counter


ALLOWED_LABELS_KO = [
    "사람",
    "차량",
    "주차금지 표지판",
    "정지 표지판",
    "안전 표지판",
    "안전봉",
    "주차콘",
    "쓰레기통",
    "나무",
    "벽",
    "기타 장애물",
]

EN_TO_KO = {
    "person": "사람",
    "vehicle": "차량",
    "car": "차량",
    "sedan": "차량",
    "suv": "차량",
    "van": "차량",
    "bus": "차량",
    "truck": "차량",
    "pickup truck": "차량",
    "no parking sign": "주차금지 표지판",
    "no-parking sign": "주차금지 표지판",
    "parking prohibition sign": "주차금지 표지판",
    "stop sign": "정지 표지판",
    "safety sign": "안전 표지판",
    "warning sign": "안전 표지판",
    "caution sign": "안전 표지판",
    "construction sign": "안전 표지판",
    "traffic sign": "안전 표지판",
    "signboard": "안전 표지판",
    "sign": "안전 표지판",
    "traffic cone": "주차콘",
    "cone": "주차콘",
    "bollard": "안전봉",
    "safety bollard": "안전봉",
    "post": "안전봉",
    "trash can": "쓰레기통",
    "garbage can": "쓰레기통",
    "bin": "쓰레기통",
    "tree": "나무",
    "wall": "벽",
    "fence": "벽",
    "building wall": "벽",
    "obstacle": "기타 장애물",
    "unknown obstacle": "기타 장애물",
}

KO_ALIASES = {
    "자동차": "차량",
    "승용차": "차량",
    "승합차": "차량",
    "트럭": "차량",
    "버스": "차량",
    "차": "차량",
    "차량": "차량",
    "주차 금지 표지판": "주차금지 표지판",
    "주차금지표지판": "주차금지 표지판",
    "주차금지 표지판": "주차금지 표지판",
    "정지표지판": "정지 표지판",
    "정지 표지판": "정지 표지판",
    "주의 표지판": "안전 표지판",
    "안내 표지판": "안전 표지판",
    "공사 표지판": "안전 표지판",
    "통제 표지판": "안전 표지판",
    "안전 표지판": "안전 표지판",
    "입간판": "안전 표지판",
    "표지판": "안전 표지판",
    "라바콘": "주차콘",
    "주차 콘": "주차콘",
    "주차콘": "주차콘",
    "콘": "주차콘",
    "볼라드": "안전봉",
    "안전봉": "안전봉",
    "쓰레기통": "쓰레기통",
    "나무": "나무",
    "벽": "벽",
    "담장": "벽",
    "기타 장애물": "기타 장애물",
    "장애물": "기타 장애물",
}

PROTECTED_LABELS_KO = set(ALLOWED_LABELS_KO) - {"기타 장애물"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize teacher annotations into trainable labels."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--metadata-path", default="")
    parser.add_argument("--annotation-path", default="")
    parser.add_argument("--output-path", default="")
    parser.add_argument("--min-class-count", type=int, default=3)
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_label(row):
    parsed = row.get("teacher_output_json") or {}
    raw = str(row.get("teacher_output_raw") or "").strip()

    primary_ko = str(parsed.get("primary_object_ko") or "").strip()
    if primary_ko in KO_ALIASES:
        return KO_ALIASES[primary_ko], "json_primary_object_ko_alias"
    if primary_ko in ALLOWED_LABELS_KO:
        return primary_ko, "json_primary_object_ko"

    primary_en = str(parsed.get("primary_object_en") or "").strip().lower()
    if primary_en in EN_TO_KO:
        return EN_TO_KO[primary_en], "json_primary_object_en"

    raw_lower = raw.lower()
    for label in sorted(ALLOWED_LABELS_KO, key=len, reverse=True):
        if label in raw:
            return label, "raw_substring_ko"
    for alias, label_ko in sorted(KO_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in raw:
            return label_ko, "raw_substring_ko_alias"
    for label_en, label_ko in sorted(EN_TO_KO.items(), key=lambda item: len(item[0]), reverse=True):
        if label_en in raw_lower:
            return label_ko, "raw_substring_en"

    if ("없음" in raw) or ("대표의 것" in raw) or ("판단" in raw) or not raw:
        return "기타 장애물", "fallback_other_obstacle"

    return "기타 장애물", "fallback_other_obstacle"


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    metadata_path = (
        Path(args.metadata_path).expanduser().resolve()
        if args.metadata_path
        else dataset_dir / "metadata" / "teacher_dataset.jsonl"
    )
    annotation_path = (
        Path(args.annotation_path).expanduser().resolve()
        if args.annotation_path
        else dataset_dir / "annotations" / "teacher_labels.jsonl"
    )
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else dataset_dir / "metadata" / "prepared_teacher_labels.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_rows = {str(row.get("sample_id")): row for row in read_jsonl(metadata_path)}
    annotation_rows = read_jsonl(annotation_path)

    prepared_rows = []
    raw_counter = Counter()
    source_counter = Counter()

    for ann in annotation_rows:
        sample_id = str(ann.get("sample_id"))
        meta = metadata_rows.get(sample_id)
        if meta is None:
            continue
        label_ko, source = normalize_label(ann)
        raw_counter[label_ko] += 1
        source_counter[source] += 1
        prepared_rows.append(
            {
                "sample_id": sample_id,
                "image_path": meta.get("image_path"),
                "temporal_image_paths": meta.get("temporal_image_paths") or [],
                "source_bag": meta.get("source_bag"),
                "source_bag_stem": meta.get("source_bag_stem"),
                "motion_summary": meta.get("motion_summary") or {},
                "actual_motion_summary": meta.get("actual_motion_summary") or {},
                "pointcloud_path": meta.get("pointcloud_path"),
                "event_label": meta.get("event_label"),
                "planner_reason": meta.get("planner_reason"),
                "motion_state": meta.get("motion_state"),
                "path_blocked": bool(meta.get("path_blocked")),
                "emergency_summary": meta.get("emergency_summary") or {},
                "stop_hits_summary": meta.get("stop_hits_summary") or {},
                "control_summary": meta.get("control_summary") or {},
                "planning_summary": meta.get("planning_summary") or {},
                "pointcloud_summary": meta.get("pointcloud_summary") or {},
                "obstacle_summary": meta.get("obstacle_summary") or {},
                "teacher_prompt_ko": meta.get("teacher_prompt_ko"),
                "teacher_prompt_camera_only_ko": meta.get("teacher_prompt_camera_only_ko"),
                "teacher_output_raw": ann.get("teacher_output_raw"),
                "teacher_output_json": ann.get("teacher_output_json"),
                "scene_domain_ko": str((ann.get("teacher_output_json") or {}).get("scene_domain_ko") or "불명"),
                "label_ko_raw": label_ko,
                "label_source": source,
            }
        )

    collapsed_counter = Counter()
    for row in prepared_rows:
        label_ko = row["label_ko_raw"]
        if label_ko not in PROTECTED_LABELS_KO and raw_counter[label_ko] < int(args.min_class_count):
            row["label_ko"] = "기타 장애물"
            row["label_collapsed"] = True
        else:
            row["label_ko"] = label_ko
            row["label_collapsed"] = False
        collapsed_counter[row["label_ko"]] += 1

    with open(output_path, "w", encoding="utf-8") as handle:
        for row in prepared_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("prepared_rows={}".format(len(prepared_rows)))
    print("label_distribution_raw={}".format(dict(raw_counter.most_common())))
    print("label_distribution_final={}".format(dict(collapsed_counter.most_common())))
    print("label_source_distribution={}".format(dict(source_counter.most_common())))
    print("output={}".format(output_path))


if __name__ == "__main__":
    main()

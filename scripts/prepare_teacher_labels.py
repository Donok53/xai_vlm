#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import Counter


ALLOWED_LABELS_KO = [
    "정지 표지판",
    "주차 미터기",
    "손가방",
    "자전거",
    "자동차",
    "오토바이",
    "신호등",
    "소화전",
    "벤치",
    "고양이",
    "사람",
    "트럭",
    "기차",
    "가방",
    "우산",
    "캐리어",
    "와인잔",
    "시계",
    "병",
    "컵",
    "공",
    "개",
    "책",
    "벽",
]

EN_TO_KO = {
    "person": "사람",
    "bicycle": "자전거",
    "car": "자동차",
    "motorcycle": "오토바이",
    "train": "기차",
    "truck": "트럭",
    "traffic light": "신호등",
    "fire hydrant": "소화전",
    "stop sign": "정지 표지판",
    "parking meter": "주차 미터기",
    "bench": "벤치",
    "cat": "고양이",
    "dog": "개",
    "backpack": "가방",
    "umbrella": "우산",
    "handbag": "손가방",
    "suitcase": "캐리어",
    "sports ball": "공",
    "bottle": "병",
    "cup": "컵",
    "wine glass": "와인잔",
    "book": "책",
    "clock": "시계",
    "wall": "벽",
}


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
    if primary_ko in ALLOWED_LABELS_KO:
        return primary_ko, "json_primary_object_ko"

    primary_en = str(parsed.get("primary_object_en") or "").strip().lower()
    if primary_en in EN_TO_KO:
        return EN_TO_KO[primary_en], "json_primary_object_en"

    raw_lower = raw.lower()
    for label in sorted(ALLOWED_LABELS_KO, key=len, reverse=True):
        if label in raw:
            return label, "raw_substring_ko"
    for label_en, label_ko in sorted(EN_TO_KO.items(), key=lambda item: len(item[0]), reverse=True):
        if label_en in raw_lower:
            return label_ko, "raw_substring_en"

    if ("없음" in raw) or ("대표의 것" in raw) or ("판단" in raw) or not raw:
        return "벽", "fallback_wall"

    return "벽", "fallback_wall"


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
                "pointcloud_path": meta.get("pointcloud_path"),
                "event_label": meta.get("event_label"),
                "planner_reason": meta.get("planner_reason"),
                "motion_state": meta.get("motion_state"),
                "path_blocked": bool(meta.get("path_blocked")),
                "pointcloud_summary": meta.get("pointcloud_summary") or {},
                "obstacle_summary": meta.get("obstacle_summary") or {},
                "teacher_prompt_ko": meta.get("teacher_prompt_ko"),
                "teacher_prompt_camera_only_ko": meta.get("teacher_prompt_camera_only_ko"),
                "teacher_output_raw": ann.get("teacher_output_raw"),
                "teacher_output_json": ann.get("teacher_output_json"),
                "label_ko_raw": label_ko,
                "label_source": source,
            }
        )

    collapsed_counter = Counter()
    for row in prepared_rows:
        label_ko = row["label_ko_raw"]
        if raw_counter[label_ko] < int(args.min_class_count):
            row["label_ko"] = "벽"
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

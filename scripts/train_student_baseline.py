#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a lightweight sklearn baseline student model."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--prepared-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
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


def load_image_feature(path, image_size):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return np.zeros((image_size * image_size,), dtype=np.float32)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    feature = image.astype(np.float32).reshape(-1) / 255.0
    return feature


def build_context_feature(row):
    obstacle = row.get("obstacle_summary") or {}
    centroid = obstacle.get("near_raw_centroid_xyz") or {}
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
    }
    return feature


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    prepared_path = (
        Path(args.prepared_path).expanduser().resolve()
        if args.prepared_path
        else dataset_dir / "metadata" / "prepared_teacher_labels.jsonl"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else dataset_dir / "student_baseline"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(prepared_path)
    if not rows:
        raise RuntimeError("prepared dataset이 비어 있습니다: {}".format(prepared_path))

    image_features = []
    context_dicts = []
    labels = []
    sample_ids = []
    for row in rows:
        label = str(row.get("label_ko") or "벽")
        image_path = dataset_dir / str(row["image_path"])
        image_features.append(load_image_feature(image_path, int(args.image_size)))
        context_dicts.append(build_context_feature(row))
        labels.append(label)
        sample_ids.append(str(row.get("sample_id")))

    image_matrix = np.stack(image_features, axis=0)
    vectorizer = DictVectorizer(sparse=False)
    context_matrix = vectorizer.fit_transform(context_dicts).astype(np.float32)
    X = np.concatenate([image_matrix, context_matrix], axis=1)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    stratify = y if len(set(y.tolist())) > 1 and min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        sample_ids,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        stratify=stratify,
    )

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        multi_class="auto",
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_.tolist(),
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_.tolist(),
        zero_division=0,
    )

    bundle = {
        "model": model,
        "vectorizer": vectorizer,
        "label_encoder": label_encoder,
        "image_size": int(args.image_size),
        "feature_dim": int(X.shape[1]),
        "classes": label_encoder.classes_.tolist(),
    }
    model_path = output_dir / "student_baseline.joblib"
    joblib.dump(bundle, model_path)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_rows": len(rows),
                "num_train": len(ids_train),
                "num_test": len(ids_test),
                "classes": label_encoder.classes_.tolist(),
                "report": report,
                "test_sample_ids": ids_test,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("trained_rows={}".format(len(rows)))
    print("classes={}".format(label_encoder.classes_.tolist()))
    print("feature_dim={}".format(X.shape[1]))
    print(report_text)
    print("model={}".format(model_path))
    print("metrics={}".format(metrics_path))


if __name__ == "__main__":
    main()

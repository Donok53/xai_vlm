#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from student_baseline_common import build_context_feature, load_image_feature, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the student XAI baseline with MLflow logging.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--prepared-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--experiment-name", default="xai_student_training")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--dataset-version", default="local")
    parser.add_argument("--registered-model-name", default="xai_student_model")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-register", action="store_true")
    return parser.parse_args()


def train_student_model(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    prepared_path = (
        Path(args.prepared_path).expanduser().resolve()
        if args.prepared_path
        else dataset_dir / "metadata" / "prepared_teacher_labels.jsonl"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else dataset_dir / "student_baseline_mlflow"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(prepared_path)
    if not rows:
        raise RuntimeError("prepared dataset is empty: {}".format(prepared_path))

    X, y, label_encoder, vectorizer, sample_ids = _build_training_matrix(
        rows=rows,
        dataset_dir=dataset_dir,
        image_size=int(args.image_size),
    )
    X_train, X_test, y_train, y_test, ids_train, ids_test, effective_test_size = _split_dataset(
        X=X,
        y=y,
        sample_ids=sample_ids,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
    )

    model = LogisticRegression(
        max_iter=int(args.max_iter),
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    all_labels = list(range(len(label_encoder.classes_)))
    report = classification_report(
        y_test,
        y_pred,
        labels=all_labels,
        target_names=label_encoder.classes_.tolist(),
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        labels=all_labels,
        target_names=label_encoder.classes_.tolist(),
        zero_division=0,
    )
    metrics = _extract_metrics(report)
    bundle = {
        "model": model,
        "vectorizer": vectorizer,
        "label_encoder": label_encoder,
        "image_size": int(args.image_size),
        "feature_dim": int(X.shape[1]),
        "classes": label_encoder.classes_.tolist(),
        "feature_spec_version": "camera_only_v2",
    }

    model_path = output_dir / "student_baseline.joblib"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "classification_report.txt"
    joblib.dump(bundle, model_path)
    metrics_payload = {
        "num_rows": len(rows),
        "num_train": len(ids_train),
        "num_test": len(ids_test),
        "effective_test_size": effective_test_size,
        "classes": label_encoder.classes_.tolist(),
        "metrics": metrics,
        "report": report,
        "test_sample_ids": ids_test,
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(report_text, encoding="utf-8")

    run_id = ""
    if not args.no_mlflow:
        run_id = _log_to_mlflow(
            args=args,
            metrics=metrics,
            model=model,
            model_path=model_path,
            metrics_path=metrics_path,
            report_path=report_path,
            output_dir=output_dir,
        )

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "metrics": metrics,
        "run_id": run_id,
    }


def _build_training_matrix(
    *,
    rows: list[dict[str, Any]],
    dataset_dir: Path,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, DictVectorizer, list[str]]:
    image_features = []
    context_dicts = []
    labels = []
    sample_ids = []
    for row in rows:
        label = str(row.get("label_ko") or "벽")
        image_path = dataset_dir / str(row["image_path"])
        image_features.append(load_image_feature(image_path, image_size))
        context_dicts.append(build_context_feature(row))
        labels.append(label)
        sample_ids.append(str(row.get("sample_id") or len(sample_ids)))

    image_matrix = np.stack(image_features, axis=0)
    vectorizer = DictVectorizer(sparse=False)
    context_matrix = vectorizer.fit_transform(context_dicts).astype(np.float32)
    X = np.concatenate([image_matrix, context_matrix], axis=1)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    return X, y, label_encoder, vectorizer, sample_ids


def _split_dataset(
    *,
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: list[str],
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], float]:
    num_rows = len(y)
    num_classes = len(set(y.tolist()))
    effective_test_size = float(test_size)
    if num_rows > 1 and int(np.ceil(effective_test_size * num_rows)) < max(1, num_classes):
        effective_test_size = min(0.5, float(max(1, num_classes)) / float(num_rows))

    stratify = y if len(set(y.tolist())) > 1 and min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        sample_ids,
        test_size=effective_test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test, ids_train, ids_test, effective_test_size


def _extract_metrics(report: dict[str, Any]) -> dict[str, float]:
    accuracy = report.get("accuracy")
    if accuracy is None:
        accuracy = (report.get("micro avg") or {}).get("f1-score", 0.0)
    return {
        "accuracy": float(accuracy or 0.0),
        "macro_f1": float((report.get("macro avg") or {}).get("f1-score", 0.0)),
        "weighted_f1": float((report.get("weighted avg") or {}).get("f1-score", 0.0)),
    }


def _log_to_mlflow(
    *,
    args: argparse.Namespace,
    metrics: dict[str, float],
    model: LogisticRegression,
    model_path: Path,
    metrics_path: Path,
    report_path: Path,
    output_dir: Path,
) -> str:
    import mlflow
    import mlflow.sklearn

    mlflow.set_experiment(args.experiment_name)
    run_name = args.run_name or "student-baseline-{}".format(args.dataset_version)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "dataset_version": args.dataset_version,
                "dataset_name": args.dataset_name or Path(args.dataset_dir).name,
                "test_size": float(args.test_size),
                "random_state": int(args.random_state),
                "max_iter": int(args.max_iter),
                "image_size": int(args.image_size),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "run_type": "student_training",
                "dataset_name": args.dataset_name or Path(args.dataset_dir).name,
                "git_commit": _git_commit(),
            }
        )
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifacts(str(output_dir), artifact_path="student_baseline_bundle")
        if not args.no_register:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="sklearn_model",
                registered_model_name=args.registered_model_name,
            )
        return run.info.run_id


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
        ).strip()
    except Exception:
        return ""


def main() -> None:
    result = train_student_model(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json

import cv2
import numpy as np

from ml.train_student_mlflow import train_student_model


def test_train_student_model_smoke_without_mlflow(tmp_path):
    dataset_dir = tmp_path / "dataset"
    image_dir = dataset_dir / "images"
    metadata_dir = dataset_dir / "metadata"
    image_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    rows = []
    for index in range(8):
        label = "사람" if index % 2 == 0 else "벽"
        image_path = image_dir / "sample_{:03d}.jpg".format(index)
        image = np.full((24, 24, 3), 40 + index * 20, dtype=np.uint8)
        cv2.imwrite(str(image_path), image)
        rows.append(
            {
                "sample_id": "sample_{:03d}".format(index),
                "image_path": "images/{}".format(image_path.name),
                "label_ko": label,
                "event_label": "normal_route",
                "motion_state": "forward",
                "planner_reason": "test",
            }
        )

    prepared_path = metadata_dir / "prepared_teacher_labels.jsonl"
    prepared_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = train_student_model(
        argparse.Namespace(
            dataset_dir=str(dataset_dir),
            prepared_path=str(prepared_path),
            output_dir=str(tmp_path / "out"),
            image_size=12,
            test_size=0.25,
            random_state=7,
            max_iter=200,
            experiment_name="test",
            run_name="",
            dataset_name="smoke",
            dataset_version="test",
            registered_model_name="xai_student_model",
            no_mlflow=True,
            no_register=True,
        )
    )

    assert result["metrics"]["accuracy"] >= 0.0
    assert (tmp_path / "out" / "student_baseline.joblib").exists()
    assert (tmp_path / "out" / "metrics.json").exists()
    assert (tmp_path / "out" / "classification_report.txt").exists()

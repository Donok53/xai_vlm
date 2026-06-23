#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_DIR = REPO_ROOT / "models" / "current"
CURRENT_MODEL_PATH = CURRENT_DIR / "student_baseline.joblib"
CURRENT_INFO_PATH = CURRENT_DIR / "model_info.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote an MLflow student model artifact to models/current.")
    parser.add_argument("--run-id", default="", help="MLflow run id containing the model artifact.")
    parser.add_argument(
        "--artifact-path",
        default="model/student_baseline.joblib",
        help="Artifact path inside the MLflow run.",
    )
    parser.add_argument("--source-model-path", default="", help="Use a local model file instead of MLflow download.")
    parser.add_argument("--model-name", default="xai_student_model")
    parser.add_argument("--version", required=True)
    parser.add_argument("--accuracy", type=float, default=None)
    parser.add_argument("--macro-f1", type=float, default=None)
    parser.add_argument("--status", default="champion")
    return parser.parse_args()


def promote_model(args: argparse.Namespace) -> dict[str, object]:
    source_path = _resolve_source_model(args)
    if not source_path.exists():
        raise FileNotFoundError("model artifact not found: {}".format(source_path))

    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if CURRENT_MODEL_PATH.exists():
        backup_path = CURRENT_DIR / "student_baseline.previous.joblib"
        shutil.copy2(CURRENT_MODEL_PATH, backup_path)
    shutil.copy2(source_path, CURRENT_MODEL_PATH)

    info = {
        "model_name": args.model_name,
        "version": args.version,
        "run_id": args.run_id,
        "accuracy": args.accuracy,
        "macro_f1": args.macro_f1,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "status": args.status,
        "model_path": "models/current/student_baseline.joblib",
        "previous_model_path": str(backup_path.relative_to(REPO_ROOT)) if backup_path else "",
    }
    CURRENT_INFO_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info


def _resolve_source_model(args: argparse.Namespace) -> Path:
    if args.source_model_path:
        return Path(args.source_model_path).expanduser().resolve()
    if not args.run_id:
        raise ValueError("--run-id or --source-model-path is required")

    import mlflow.artifacts

    downloaded = mlflow.artifacts.download_artifacts(
        run_id=args.run_id,
        artifact_path=args.artifact_path,
    )
    return Path(downloaded).expanduser().resolve()


def main() -> None:
    info = promote_model(parse_args())
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

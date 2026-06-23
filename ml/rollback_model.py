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
DEFAULT_PREVIOUS_MODEL_PATH = CURRENT_DIR / "student_baseline.previous.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollback the active student model in models/current.")
    parser.add_argument("--source-model-path", default=str(DEFAULT_PREVIOUS_MODEL_PATH))
    parser.add_argument("--model-name", default="xai_student_model")
    parser.add_argument("--version", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--accuracy", type=float, default=None)
    parser.add_argument("--macro-f1", type=float, default=None)
    return parser.parse_args()


def rollback_model(args: argparse.Namespace) -> dict[str, object]:
    source_path = Path(args.source_model_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError("rollback source model not found: {}".format(source_path))

    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, CURRENT_MODEL_PATH)
    info = {
        "model_name": args.model_name,
        "version": args.version,
        "run_id": args.run_id,
        "accuracy": args.accuracy,
        "macro_f1": args.macro_f1,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "status": "rollback",
        "model_path": "models/current/student_baseline.joblib",
        "rollback_source_path": str(source_path),
    }
    CURRENT_INFO_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info


def main() -> None:
    info = rollback_model(parse_args())
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

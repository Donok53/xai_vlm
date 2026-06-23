#!/usr/bin/env python3
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CURRENT_MODEL_PATH = REPO_ROOT / "models" / "current" / "student_baseline.joblib"
CURRENT_MODEL_INFO_PATH = REPO_ROOT / "models" / "current" / "model_info.json"


def default_model_path(fallback_path):
    env_model_path = os.getenv("MODEL_PATH")
    if env_model_path:
        return str(Path(env_model_path).expanduser().resolve())

    model_info = load_model_info()
    info_model_path = _path_from_model_info(model_info)
    if info_model_path is not None and info_model_path.exists():
        return str(info_model_path)

    if CURRENT_MODEL_PATH.exists():
        return str(CURRENT_MODEL_PATH)

    return str(Path(fallback_path).expanduser().resolve())


def load_model_info(model_path=None):
    candidates = []
    env_info_path = os.getenv("MODEL_INFO_PATH")
    if env_info_path:
        candidates.append(Path(env_info_path).expanduser().resolve())
    if model_path:
        candidates.append(Path(model_path).expanduser().resolve().parent / "model_info.json")
    candidates.append(CURRENT_MODEL_INFO_PATH)

    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            continue

    return {
        "model_name": "xai_student_model",
        "version": "unknown",
        "run_id": "",
        "status": "untracked",
    }


def model_log_fields(model_info, model_path):
    return {
        "model_name": str(model_info.get("model_name") or "xai_student_model"),
        "model_version": str(model_info.get("version") or "unknown"),
        "run_id": str(model_info.get("run_id") or ""),
        "model_status": str(model_info.get("status") or ""),
        "model_path": str(Path(model_path).expanduser().resolve()),
    }


def _path_from_model_info(model_info):
    raw_model_path = model_info.get("model_path") if isinstance(model_info, dict) else None
    if not raw_model_path:
        return None
    path = Path(str(raw_model_path)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()

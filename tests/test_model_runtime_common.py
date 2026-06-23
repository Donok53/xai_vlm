import json

from scripts.model_runtime_common import default_model_path, load_model_info, model_log_fields


def test_default_model_path_prefers_model_path_env(tmp_path, monkeypatch):
    model_path = tmp_path / "student_baseline.joblib"
    model_path.write_bytes(b"placeholder")
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    assert default_model_path(tmp_path / "fallback.joblib") == str(model_path.resolve())


def test_model_log_fields_uses_registry_metadata(tmp_path):
    model_path = tmp_path / "student_baseline.joblib"
    model_path.write_bytes(b"placeholder")
    info_path = tmp_path / "model_info.json"
    info_path.write_text(
        json.dumps(
            {
                "model_name": "xai_student_model",
                "version": "v2",
                "run_id": "run-2",
                "status": "champion",
            }
        ),
        encoding="utf-8",
    )

    info = load_model_info(model_path)
    fields = model_log_fields(info, model_path)

    assert fields["model_name"] == "xai_student_model"
    assert fields["model_version"] == "v2"
    assert fields["run_id"] == "run-2"
    assert fields["model_status"] == "champion"

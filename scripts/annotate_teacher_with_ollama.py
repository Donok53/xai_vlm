#!/usr/bin/env python3
import argparse
import base64
import json
from pathlib import Path
from urllib import request

ALLOWED_LABELS_KO = [
    "사람",
    "자전거",
    "자동차",
    "오토바이",
    "기차",
    "트럭",
    "신호등",
    "소화전",
    "정지 표지판",
    "주차 미터기",
    "벤치",
    "고양이",
    "개",
    "가방",
    "우산",
    "손가방",
    "캐리어",
    "공",
    "병",
    "컵",
    "와인잔",
    "책",
    "시계",
    "벽",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate exported teacher dataset with an Ollama VLM."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--metadata-path", default="")
    parser.add_argument("--output-path", default="")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--model", default="qwen2.5vl:3b")
    parser.add_argument(
        "--prompt-mode",
        choices=["metadata", "class_only"],
        default="class_only",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
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


def extract_json_object(text):
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        fragment = raw[start : end + 1]
        try:
            return json.loads(fragment)
        except Exception:
            return None
    return None


def encode_image_base64(path):
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def build_class_only_prompt(row):
    planner_reason = str(row.get("planner_reason") or "unknown")
    motion_state = str(row.get("motion_state") or "unknown")
    event_label = str(row.get("event_label") or "unknown")
    obstacle = row.get("obstacle_summary") or {}
    near_range = obstacle.get("near_raw_min_range_m")

    return "\n".join(
        [
            "너는 자율주행용 오프라인 teacher 모델이다.",
            "이미지에서 planner/LiDAR와 가장 관련 있는 대표 객체를 허용 라벨 중 정확히 하나만 선택하라.",
            "허용 라벨에 없거나 확실하지 않으면 반드시 '벽'을 선택하라.",
            "",
            "주행 문맥:",
            "- event_label: {}".format(event_label),
            "- planner_reason: {}".format(planner_reason),
            "- motion_state: {}".format(motion_state),
            "- near_raw_min_range_m: {}".format(near_range),
            "",
            "허용 라벨:",
            ", ".join(ALLOWED_LABELS_KO),
            "",
            "규칙:",
            "- 설명 문장은 쓰지 마라.",
            "- 반드시 JSON만 출력하라.",
            '- 형식: {"primary_object_ko":"허용라벨중하나","confidence":0.0}',
        ]
    )


def choose_prompt(row, prompt_mode):
    if prompt_mode == "metadata":
        return str(row.get("teacher_prompt_ko") or "")
    return build_class_only_prompt(row)


def ollama_chat(endpoint, model, prompt, image_path, temperature):
    image_base64 = encode_image_base64(image_path)
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64],
            }
        ],
        "options": {
            "temperature": float(temperature),
        },
    }
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=180.0) as response:
        data = json.loads(response.read().decode("utf-8"))
    message = (data.get("message") or {}).get("content") or ""
    return {
        "raw_response": message,
        "parsed_json": extract_json_object(message),
        "api_payload": data,
    }


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    metadata_path = (
        Path(args.metadata_path).expanduser().resolve()
        if args.metadata_path
        else dataset_dir / "metadata" / "teacher_dataset.jsonl"
    )
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else dataset_dir / "annotations" / "teacher_labels.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(metadata_path)
    done_ids = set()
    if output_path.exists() and not args.overwrite:
        for row in read_jsonl(output_path):
            done_ids.add(str(row.get("sample_id")))

    written = 0
    with open(output_path, "a" if output_path.exists() and not args.overwrite else "w", encoding="utf-8") as handle:
        for row in rows:
            sample_id = str(row.get("sample_id"))
            if sample_id in done_ids:
                continue

            image_path = dataset_dir / str(row["image_path"])
            prompt = choose_prompt(row, args.prompt_mode)
            result = ollama_chat(
                args.endpoint,
                args.model,
                prompt,
                image_path,
                args.temperature,
            )
            out_row = {
                "sample_id": sample_id,
                "model": args.model,
                "prompt_mode": args.prompt_mode,
                "image_path": row.get("image_path"),
                "event_label": row.get("event_label"),
                "teacher_prompt_used": prompt,
                "teacher_output_raw": result["raw_response"],
                "teacher_output_json": result["parsed_json"],
            }
            handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1
            print("annotated {} ({})".format(sample_id, args.model))
            if args.limit > 0 and written >= int(args.limit):
                break

    print("written_annotations={}".format(written))
    print("output={}".format(output_path))


if __name__ == "__main__":
    main()

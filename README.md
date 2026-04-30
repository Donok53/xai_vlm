# VLM Teacher Distill

`xai_autonomy_driving_explainer`의 런타임 경로는 그대로 두고,
오프라인에서만 VLM을 teacher로 써서 dataset을 만들기 위한 **독립 프로젝트**다.

핵심 아이디어:

- 실행 시점: `YOLO + LiDAR + planner + template`
- 오프라인 학습 시점: `VLM teacher`
- 목표: VLM이 추론에 직접 들어오지 않아도, 더 안정적인 semantic label과 설명 템플릿을 만들기

## 현재 포함된 기능

1. `scripts/export_teacher_dataset.py`
   - bag에서 `/xai/event_log`, `/xai/planner_snapshot`, 카메라 이미지, point cloud를 묶어
     teacher dataset을 만든다.
   - event 중심 샘플을 뽑는다.
   - 이미지와 downsampled point cloud를 함께 저장한다.
   - VLM용 prompt 초안도 metadata에 같이 넣는다.

2. `scripts/annotate_teacher_with_ollama.py`
   - export한 dataset을 읽고, 이미지 + planner/LiDAR 문맥을 VLM에 보내
     teacher annotation을 JSONL로 저장한다.
   - 런타임이 아니라 오프라인 데이터 생성용이다.

3. `scripts/prepare_teacher_labels.py`
   - `teacher_output_json`이 비어 있어도 raw 응답에서 허용 라벨을 최대한 복구한다.
   - 너무 드문 클래스는 기본적으로 `벽`으로 접어 baseline 학습이 가능하게 만든다.

4. `scripts/train_student_baseline.py`
   - 이미지 grayscale 특징 + planner/LiDAR 문맥 특징으로
     가벼운 `sklearn` baseline student를 학습한다.

5. `scripts/export_camera_only_teacher_dataset.py`
   - XAI 토픽 없이 카메라 이미지만으로 teacher dataset을 만든다.
   - `prev/current/next` 3프레임을 같이 저장하고 optical flow 기반 움직임 요약을 metadata에 넣는다.
   - camera-only teacher가 “왜 그런 주행을 했는지”를 시각 단서만으로 답하게 만들기 위한 경로다.

## 구조

```text
xai_autonomy_vlm_teacher_distill/
├── README.md
├── requirements.txt
├── data/
│   └── .gitkeep
└── scripts/
    ├── annotate_teacher_with_ollama.py
    ├── export_camera_only_teacher_dataset.py
    └── export_teacher_dataset.py
    ├── prepare_teacher_labels.py
    └── train_student_baseline.py
```

## 권장 실행 환경

ROS bag를 직접 읽으므로 `/usr/bin/python3` 기준이 가장 안전하다.

```bash
cd ~/code/xai_autonomy_vlm_teacher_distill
source /opt/ros/noetic/setup.bash
```

## 1. Teacher Dataset Export

기본 예시:

```bash
/usr/bin/python3 scripts/export_teacher_dataset.py \
  --bag /home/byeongjae/bagfiles/record_real_20260422_180049.bag \
  --output-dir /home/byeongjae/code/xai_autonomy_vlm_teacher_distill/data/record_real_teacher
```

생성 결과:

- `images/*.jpg`
- `pointcloud/*.npz`
- `metadata/teacher_dataset.jsonl`

`teacher_dataset.jsonl` 한 줄에는 대략 이런 정보가 들어간다.

- `sample_id`
- `stamp`
- `event_label`
- `planner_reason`
- `motion_state`
- `path_blocked`
- `image_path`
- `pointcloud_path`
- `teacher_prompt_ko`
- 원본 `event_log`, `planner_snapshot`

### Camera-only Temporal Export

XAI 토픽 없이 카메라만으로 teacher dataset을 만들려면:

```bash
/usr/bin/python3 scripts/export_camera_only_teacher_dataset.py \
  --bag /home/byeongjae/bagfiles/record_real_20260422_180049.bag \
  --output-dir /home/byeongjae/code/xai_autonomy_vlm_teacher_distill/data/record_real_camera_only \
  --sample-every-n 8
```

여러 bag를 하나의 학습셋으로 묶을 수도 있다:

```bash
/usr/bin/python3 scripts/export_camera_only_teacher_dataset.py \
  --bag /home/byeongjae/bagfiles/1.made_map/camera_left.bag /home/byeongjae/bagfiles/1.made_map/camera_right.bag \
  --output-dir /home/byeongjae/code/xai_autonomy_vlm_teacher_distill/data/made_map_camera_lr \
  --sample-every-n 24
```

생성 결과:

- `images/sample_xxxxx_prev.jpg`
- `images/sample_xxxxx_current.jpg`
- `images/sample_xxxxx_next.jpg`
- `metadata/teacher_dataset.jsonl`

여기에는:

- `temporal_image_paths`
- `motion_summary`
- `teacher_prompt_camera_only_ko`
- `source_bag`
- `source_bag_stem`

가 함께 들어간다.

## 2. Ollama Teacher Annotation

예시:

```bash
/usr/bin/python3 - <<'PY'
print("권장 teacher 모델:", "qwen2.5vl:32b-q4_K_M")
print("필요시 먼저 실행: ollama pull qwen2.5vl:32b-q4_K_M")
PY
```

```bash
/usr/bin/python3 scripts/annotate_teacher_with_ollama.py \
  --dataset-dir /home/byeongjae/code/xai_autonomy_vlm_teacher_distill/data/record_real_teacher \
  --model qwen2.5vl:32b-q4_K_M \
  --prompt-mode class_only
```

출력:

- `annotations/teacher_labels.jsonl`

각 라인은:

- `sample_id`
- `model`
- `prompt_mode`
- `teacher_output_raw`
- `teacher_output_json`

형태로 저장된다.

camera-only temporal teacher 예시:

```bash
/usr/bin/python3 scripts/annotate_teacher_with_ollama.py \
  --dataset-dir /home/byeongjae/code/xai_autonomy_vlm_teacher_distill/data/record_real_camera_only \
  --model qwen2.5vl:32b-q4_K_M \
  --prompt-mode camera_reason_temporal \
  --prewarm \
  --timeout-s 1800 \
  --max-image-side-px 256 \
  --jpeg-quality 60 \
  --num-predict 96 \
  --num-ctx 384
```

권장:

- teacher 품질이 중요하므로 기본 teacher는 `qwen2.5vl:32b-q4_K_M`
- 프롬프트는 `class_only`
  - 허용 클래스 중 하나만 선택하게 해서 student용 라벨 품질을 우선 높인다.
- 현재 장비처럼 GPU 메모리가 충분하면 `32b`를 우선 시도하고, 속도가 너무 느릴 때만 `7b`로 내린다.

## 3. Teacher 라벨 정규화

현재 작은 로컬 VLM은 JSON을 안정적으로 주지 않을 수 있으므로,
raw 응답에서 허용 라벨을 다시 뽑는 단계가 필요하다.

```bash
/usr/bin/python3 scripts/prepare_teacher_labels.py \
  --dataset-dir /home/byeongjae/code/xai_autonomy_vlm_teacher_distill/data/record_real_teacher
```

출력:

- `metadata/prepared_teacher_labels.jsonl`

## 4. Baseline Student 학습

```bash
/usr/bin/python3 scripts/train_student_baseline.py \
  --dataset-dir /home/byeongjae/code/xai_autonomy_vlm_teacher_distill/data/record_real_teacher
```

## 5. One-shot Camera-only Pipeline

라벨링부터 student 학습까지 한 번에 돌리려면:

```bash
/usr/bin/python3 scripts/run_camera_only_pipeline.py \
  --bag /home/byeongjae/bagfiles/1.made_map/camera_left.bag /home/byeongjae/bagfiles/1.made_map/camera_right.bag \
  --output-dir /home/byeongjae/code/xai_autonomy_vlm_teacher_distill/data/made_map_camera_lr_full \
  --sample-every-n 24 \
  --model qwen2.5vl:32b-q4_K_M \
  --prompt-mode camera_reason_temporal \
  --timeout-s 1800 \
  --retries 1 \
  --prewarm \
  --max-image-side-px 256 \
  --annotate-jpeg-quality 60 \
  --num-predict 96 \
  --num-ctx 384 \
  --overwrite
```

빠른 점검용으로 일부 샘플만 먼저 돌릴 수도 있다:

```bash
/usr/bin/python3 scripts/run_camera_only_pipeline.py \
  --bag /home/byeongjae/bagfiles/1.made_map/camera_left.bag /home/byeongjae/bagfiles/1.made_map/camera_right.bag \
  --output-dir /home/byeongjae/code/xai_autonomy_vlm_teacher_distill/data/made_map_camera_lr_quick \
  --sample-every-n 24 \
  --max-samples 120 \
  --annotate-limit 120 \
  --model qwen2.5vl:32b-q4_K_M \
  --prompt-mode camera_reason_temporal \
  --timeout-s 1800 \
  --retries 1 \
  --prewarm \
  --max-image-side-px 256 \
  --annotate-jpeg-quality 60 \
  --num-predict 96 \
  --num-ctx 384 \
  --overwrite
```

## 6. Student 추론 시각화

학습된 student가 카메라 장면을 보고 어떤 대표 객체와 주행 이유를 떠올렸는지 영상으로 보려면:

```bash
/usr/bin/python3 scripts/infer_student_camera_only_visual.py \
  --dataset-dir /home/byeongjae/code/xai_autonomy_vlm_teacher_distill/data/made_map_camera_lr_full \
  --show-teacher
```

결과:

- `student_inference/student_camera_reason.mp4`

window로 바로 보고 싶으면:

```bash
/usr/bin/python3 scripts/infer_student_camera_only_visual.py \
  --dataset-dir /home/byeongjae/code/xai_autonomy_vlm_teacher_distill/data/made_map_camera_lr_full \
  --show-teacher \
  --display-window
```

## 7. Bag Replay 실시간 추론 보기

학습된 student가 bag를 따라가며 현재 카메라 장면을 어떻게 해석하는지 실시간처럼 보려면:

```bash
/usr/bin/python3 scripts/replay_student_camera_only_realtime.py \
  --bag /home/byeongjae/bagfiles/1.made_map/camera_left.bag \
  --sample-every-n 8
```

두 배 빠르게 보고 싶으면:

```bash
/usr/bin/python3 scripts/replay_student_camera_only_realtime.py \
  --bag /home/byeongjae/bagfiles/1.made_map/camera_left.bag \
  --sample-every-n 8 \
  --playback-rate 2.0
```

창은 띄우지 않고 결과만 mp4로 남기려면:

```bash
/usr/bin/python3 scripts/replay_student_camera_only_realtime.py \
  --bag /home/byeongjae/bagfiles/1.made_map/camera_left.bag \
  --sample-every-n 8 \
  --no-display-window
```

출력:

- `student_baseline/student_baseline.joblib`
- `student_baseline/metrics.json`

## 의도한 다음 단계

이 프로젝트의 목표는 바로 student 학습까지 끝내는 것이 아니라,
우선 아래 파이프라인을 안정화하는 것이다.

1. bag -> teacher dataset export
2. dataset -> VLM teacher label
3. teacher label 정규화
4. 작은 student classifier / template selector 학습

## 참고

- 현재 class 해석은 `xai_autonomy_driving_explainer`에서 쓰는 whitelist 철학을 그대로 따른다.
- 런타임 project는 수정하지 않고, 여기서는 오프라인 teacher dataset만 다룬다.
- 이 폴더는 기존 `xai_autonomy_driving_explainer` repo 밖에 따로 둔 실험용 프로젝트다.

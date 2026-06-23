# XAI VLM Rich Student Runtime

ROS bag을 재생하면서 카메라 영상 위에 student XAI 설명을 overlay하고,
설명 JSON을 ROS topic으로 publish하는 runtime 프로젝트다.

## 1. 설치

새 PC 또는 새 환경에서는 ROS Noetic의 Python 환경을 기준으로 실행한다.
Conda `base`가 켜져 있어도 실행 명령은 `/usr/bin/python3`를 사용한다.

필수 요소:

- Ubuntu + ROS Noetic
- Python 3 패키지: `numpy`, `opencv-python`, `scikit-learn`, `Pillow`, `joblib`, `mlflow`
- ROS Python 패키지: `rospy`, `rosbag`, `cv_bridge`
- 배포 모델: `models/outdoor_sign_vehicle_rich_full/student_baseline.joblib`

설치 예시:

```bash
sudo apt update

sudo apt install -y \
  git \
  python3-pip \
  python3-opencv \
  python3-pil \
  python3-sklearn \
  python3-joblib \
  fonts-nanum \
  ros-noetic-ros-base \
  ros-noetic-rosbag \
  ros-noetic-rostopic \
  ros-noetic-cv-bridge \
  ros-noetic-rqt-image-view
```

저장소 clone:

```bash
mkdir -p ~/code
cd ~/code

git clone https://github.com/Donok53/xai_vlm.git xai_autonomy_vlm_teacher_distill
cd ~/code/xai_autonomy_vlm_teacher_distill
```

Python 의존성 설치:

```bash
/usr/bin/python3 -m pip install --user -r requirements.txt
```

ROS 환경 로드:

```bash
source /opt/ros/noetic/setup.bash
```

모델 파일 확인:

```bash
cat models/current/model_info.json
ls models/outdoor_sign_vehicle_rich_full/student_baseline.joblib
```

runtime은 모델 경로를 아래 순서로 찾는다.

1. `MODEL_PATH` 환경 변수
2. `MODEL_INFO_PATH` 또는 `models/current/model_info.json`의 `model_path`
3. `models/current/student_baseline.joblib`
4. 기존 bundled 모델 `models/outdoor_sign_vehicle_rich_full/student_baseline.joblib`

`/xai/vlm_log` payload에는 `model_name`, `model_version`, `run_id`, `prediction`, `confidence`, `explanation`이 포함된다.

## 2. MLflow 학습 및 모델 승격

teacher label 데이터로 student baseline을 재학습하고 MLflow에 parameter, metric, artifact, model registry 기록을 남긴다.

```bash
cd ~/code/xai_autonomy_vlm_teacher_distill

/usr/bin/python3 ml/train_student_mlflow.py \
  --dataset-dir data/record_real_teacher \
  --dataset-name record_real_teacher \
  --dataset-version v2 \
  --output-dir models/student_mlflow_v2 \
  --experiment-name xai_student_training \
  --registered-model-name xai_student_model
```

학습 결과는 `student_baseline.joblib`, `metrics.json`, `classification_report.txt`로 저장되고, MLflow run에는 `accuracy`, `macro_f1`, `weighted_f1`이 기록된다.

성능이 좋은 run을 현재 서비스 모델로 반영:

```bash
/usr/bin/python3 ml/promote_model.py \
  --run-id <MLFLOW_RUN_ID> \
  --artifact-path model/student_baseline.joblib \
  --version v2 \
  --accuracy 0.91 \
  --macro-f1 0.88
```

로컬 파일을 바로 champion으로 반영할 수도 있다.

```bash
/usr/bin/python3 ml/promote_model.py \
  --source-model-path models/student_mlflow_v2/student_baseline.joblib \
  --version v2 \
  --accuracy 0.91 \
  --macro-f1 0.88
```

필요하면 이전 모델로 되돌린다.

```bash
/usr/bin/python3 ml/rollback_model.py --version v1
```

## 3. Bagfile 토픽

rich runtime 노드는 아래 입력 토픽을 사용한다. bag에 모든 토픽이 있으면 가장 안정적으로 동작한다.

| Topic | Type | 설명 |
| --- | --- | --- |
| `/camera/color/image_raw` | `sensor_msgs/Image` | overlay를 만들 카메라 원본 영상 |
| `/xai/event_log` | `std_msgs/String` | 이벤트 중심 XAI JSON 로그. 회피, 정지, 상태 변화의 핵심 입력 |
| `/xai/planner_snapshot` | `std_msgs/String` | planner/control/planning 최신 snapshot JSON |
| `/planning/linefit_ground/non_ground_cloud` | `sensor_msgs/PointCloud2` | 비지면 point cloud. 장애물 문맥 보조 |
| `/planning/near_field_stop_hits` | `sensor_msgs/PointCloud2` | 근거리 정지 후보 point cloud |
| `/planning/emergency_stop` | `std_msgs/Bool` | 안전모드/긴급정지 상태 |
| `/astar/path_blocked` | `std_msgs/Bool` | A* 경로 차단 여부 |
| `/astar/path` | `nav_msgs/Path` | global path와 최종 목적지 위치 |
| `/lio_localizer/odometry/optimization` | `nav_msgs/Odometry` | 실제 이동/정지 판단용 odom |

출력 토픽:

| Topic | Type | 설명 |
| --- | --- | --- |
| `/xai/vlm_log` | `std_msgs/String` | student XAI 설명 JSON |
| `/student_xai/rich_overlay` | `sensor_msgs/Image` | 카메라 영상 + 설명 패널 overlay |

주요 판정 기준:

- 안전모드 정지는 경로 차단/회피보다 우선한다.
- 목적지 도착은 `local_path` 잔여 길이와 제어 정지 상태를 함께 보고 판단한다.
- 좌/우 회피는 stale `path_change`가 아니라 LiDAR obstacle 위치를 우선 사용한다.
- 이벤트가 없는 동안에도 정상 경로 상태를 주기적으로 출력한다.

## 4. 실행 명령어

아래 예시는 실외 주행 bag `record_real_20260604_142137.bag` 기준이다.

터미널 1: ROS master

```bash
source /opt/ros/noetic/setup.bash
roscore
```

터미널 2: rich student XAI 노드

```bash
cd ~/code/xai_autonomy_vlm_teacher_distill
source /opt/ros/noetic/setup.bash

/usr/bin/python3 scripts/ros_student_xai_rich_node.py \
  _display_window:=true \
  _render_latest_on_image:=true \
  _max_image_age_s:=0.6 \
  _max_planner_age_s:=1.2 \
  _max_pointcloud_age_s:=1.0 \
  _normal_status_period_s:=0.5
```

터미널 3: bag 재생

```bash
source /opt/ros/noetic/setup.bash

rosbag play --clock \
  "/home/byeongjae/bagfiles/7차 주행_실외주행/record_real_20260604_142137.bag"
```

특정 구간만 빠르게 확인:

```bash
rosbag play --clock --start=58 --duration=18 \
  "/home/byeongjae/bagfiles/7차 주행_실외주행/record_real_20260604_142137.bag"
```

```bash
rosbag play --clock --start=178 --duration=15 \
  "/home/byeongjae/bagfiles/7차 주행_실외주행/record_real_20260604_142137.bag"
```

설명 JSON 확인:

```bash
source /opt/ros/noetic/setup.bash
rostopic echo /xai/vlm_log
```

overlay topic 확인:

```bash
source /opt/ros/noetic/setup.bash
rqt_image_view /student_xai/rich_overlay
```

웹 화면 확인:

```bash
http://127.0.0.1:8090/
```

대시보드의 `실시간 VLM 화면` 패널은 기본으로 아래 스트림을 표시한다.

```bash
http://127.0.0.1:8090/stream.mjpg
```

rich student XAI 노드는 `cv2.imshow()`에 표시하는 overlay frame을 같은 주소로도 내보낸다.
필요하면 실행 옵션으로 주소를 바꿀 수 있다.

```bash
/usr/bin/python3 scripts/ros_student_xai_rich_node.py \
  _web_stream:=true \
  _web_stream_host:=127.0.0.1 \
  _web_stream_port:=8090 \
  _web_stream_path:=/stream.mjpg
```

결과 토픽까지 새 bag으로 저장하려면, 노드를 실행한 상태에서 먼저 record를 시작한다.

```bash
source /opt/ros/noetic/setup.bash

rosbag record --lz4 -a \
  -O /home/byeongjae/bagfiles/xai_results/record_real_20260604_142137_with_student_xai_rich.bag
```

그 다음 원본 bag을 재생한다.

```bash
source /opt/ros/noetic/setup.bash

rosbag play --clock \
  "/home/byeongjae/bagfiles/7차 주행_실외주행/record_real_20260604_142137.bag"
```

재생이 끝나면 `rosbag record` 터미널에서 `Ctrl+C`로 종료한다.

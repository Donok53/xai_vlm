# record_real_20260502 Student XAI Bag 생성 정리

## 목적

기존 주행 rosbag을 재생하면서 `xai_autonomy_vlm_teacher_distill`의 rich student XAI 노드를 함께 실행하고, 원본 토픽에 student XAI 결과 토픽을 추가로 담은 새 bag 파일을 생성했다.

중요한 점은 원본 bag을 직접 수정한 것이 아니라, 원본 bag을 `rosbag play`로 재생한 뒤 현재 ROS graph에 publish되는 토픽들을 새 bag으로 다시 녹화했다는 점이다.

## 입력 Bag

```bash
/home/byeongjae/bagfiles/3차 실내 주행/record_real_20260502_145846.bag
```

원본 bag 요약:

```text
size: 2.7 GB
duration: 61s
important topics:
  /camera/color/image_raw
  /xai/event_log
  /xai/planner_snapshot
  /planning/linefit_ground/non_ground_cloud
  /planning/near_field_stop_hits
  /planning/emergency_stop
  /astar/path_blocked
  /lio_localizer/odometry/optimization
```

## 사용한 Node

프로젝트:

```bash
/home/byeongjae/code/xai_autonomy_vlm_teacher_distill
```

실행 노드:

```bash
scripts/ros_student_xai_rich_node.py
```

사용 모델:

```bash
/home/byeongjae/code/xai_autonomy_vlm_teacher_distill/models/indoor3_emergency_rich_full/student_baseline.joblib
```

이 모델은 현재 rich node의 기본 배포 모델이며, 목적지 도착과 안전모드/실제 정지 문맥을 반영한다.

## 재현 명령어

터미널 1:

```bash
source /opt/ros/noetic/setup.bash
roscore
```

터미널 2:

```bash
cd /home/byeongjae/code/xai_autonomy_vlm_teacher_distill
source /opt/ros/noetic/setup.bash

/usr/bin/python3 scripts/ros_student_xai_rich_node.py \
  _display_window:=false \
  _max_image_age_s:=0.6 \
  _max_planner_age_s:=1.2 \
  _max_pointcloud_age_s:=1.0
```

터미널 3:

```bash
source /opt/ros/noetic/setup.bash

rosbag record --lz4 -a \
  -O /home/byeongjae/bagfiles/xai_results/record_real_20260502_145846_with_student_xai_rich.bag
```

터미널 4:

```bash
source /opt/ros/noetic/setup.bash

rosbag play --clock \
  "/home/byeongjae/bagfiles/3차 실내 주행/record_real_20260502_145846.bag"
```

녹화 종료 시에는 `rosbag play`가 끝난 뒤 `rosbag record`를 `Ctrl+C`로 정상 종료해야 `.bag.active`가 아닌 정상 `.bag` 파일로 닫힌다.

## 생성된 Bag

```bash
/home/byeongjae/bagfiles/xai_results/record_real_20260502_145846_with_student_xai_rich.bag
```

생성 결과:

```text
size: 3.2 GB
duration: 64s
messages: 66948
compression: lz4
```

로그 디렉터리:

```bash
/home/byeongjae/bagfiles/xai_results/logs_record_real_20260502_145846
```

## 추가된 토픽

원본 bag과 비교했을 때 새 bag에 추가된 토픽은 다음 5개다.

| Topic | Type | Count | Meaning |
| --- | --- | ---: | --- |
| `/student_xai/rich_overlay` | `sensor_msgs/Image` | 973 | 카메라 프레임 위에 student XAI 판단 패널을 붙인 영상 결과 |
| `/student_xai/rich_reason` | `std_msgs/String` | 93 | student XAI 설명 JSON |
| `/clock` | `rosgraph_msgs/Clock` | 6008 | `rosbag play --clock`로 생긴 ROS sim time |
| `/rosout` | `rosgraph_msgs/Log` | 65 | 실행 중 ROS log |
| `/rosout_agg` | `rosgraph_msgs/Log` | 62 | aggregate ROS log |

핵심 결과물은 `/student_xai/rich_overlay`와 `/student_xai/rich_reason` 두 개다. `/clock`, `/rosout`, `/rosout_agg`는 재생 및 노드 실행 과정에서 생긴 부가 토픽이다.

## Student XAI 결과 검증

`/student_xai/rich_reason`의 `driving_mode_ko` 분포:

| Mode | Count |
| --- | ---: |
| `정상 경로` | 59 |
| `경로 차단` | 19 |
| `우측 회피` | 5 |
| `목적지 도착` | 4 |
| `실제 정지` | 3 |
| `좌측 회피` | 3 |

확인된 주요 문구:

```text
목적지에 도착해 정지한 상태로 본다.
주행 명령은 있으나 실제 이동이 거의 없어 안전모드나 수동 정지 상태로 본다.
전방 우측에 장애물이 인지되어 경로가 막혀 정지 또는 재계획을 진행하고 있다고 본다.
전방 우측의 사람을 피해 좌측 회피 경로로 주행을 진행하고 있다고 본다.
현재 정상 경로를 따라가며 우측 조향으로 주행을 보정하고 있다고 본다.
전방 중앙의 사람을 피해 우측 회피 경로로 주행을 진행하고 있다고 본다.
```

## 원본 Bag과의 차이

새 bag은 원본 bag을 다시 record해서 만든 것이므로 공통 토픽의 메시지 수가 몇 개씩 다를 수 있다. 예를 들어:

```text
/camera/color/image_raw: 929 -> 915
/xai/planner_snapshot: 305 -> 301
/xai/event_log: 94 -> 94
```

`/xai/event_log`는 원본과 같은 94개가 들어갔다.

## 재생 확인

새 bag을 재생:

```bash
source /opt/ros/noetic/setup.bash

rosbag play /home/byeongjae/bagfiles/xai_results/record_real_20260502_145846_with_student_xai_rich.bag
```

overlay 영상 확인:

```bash
source /opt/ros/noetic/setup.bash
rqt_image_view /student_xai/rich_overlay
```

설명 JSON 확인:

```bash
source /opt/ros/noetic/setup.bash
rostopic echo /student_xai/rich_reason
```

## Safety Latched + Arrival Hold 수정본

초기 safety priority 수정본에서는 `emergency_stop_active` 자체를 너무 넓게 안전모드로 해석할 수 있었다. 이후 `cmd_vel`과 odom의 불일치를 우선 기준으로 바꿨지만, 초반 실제 안전모드가 command 자체를 막는 구간에서는 `cmd_vel=0`, `odom=0`이라 `정상 경로` 또는 `제어 정지`로 보일 수 있었다.

최종 판정 기준:

- `cmd_vel`이 0에 가깝지 않은데 odom 기준 실제 이동이 거의 0이면 `안전모드 정지`로 출력한다.
- `emergency_stop` 또는 planner `behavior_stop`이 한 번 들어오면 실제 움직임이 시작될 때까지 safety 상태를 유지한다.
- safety 상태가 유지되는 동안에는 `cmd_vel=0`, `odom=0`이어도 `안전모드 정지`로 출력한다.
- safety 상태가 아닌데 `cmd_vel=0`, `odom=0`, active route가 있으면 `출발 대기`로 출력한다.
- `안전모드 정지`, `출발 대기`, `제어 정지`는 `path_blocked`보다 먼저 표시한다.
- global path 정보가 아직 없는 초반 상태를 `0m, 0 points`로 해석해서 `목적지 도착`으로 오판하지 않도록, path 길이와 포인트 수가 실제로 들어온 경우에만 도착 판정을 한다.
- 도착 직후 global path 길이와 point 수가 약간 흔들려도 `목적지 도착`이 유지되도록 arrival hold 범위를 둔다.
- 영상 패널에 `cmd v/w`를 추가해서 control 명령과 실제 odom 정지 여부를 같이 볼 수 있게 했다.

최종 수정 후 생성한 새 bag:

```bash
/home/byeongjae/bagfiles/xai_results/record_real_20260502_145846_with_student_xai_rich_safety_latched_arrival_hold.bag
```

최종 수정 후 생성한 mp4:

```bash
/home/byeongjae/bagfiles/xai_results/student_xai_rich_overlay_20260502_safety_latched_arrival_hold.mp4
```

최종 수정 후 `/student_xai/rich_reason`의 `driving_mode_ko` 분포:

| Mode | Count |
| --- | ---: |
| `정상 경로` | 58 |
| `경로 차단` | 13 |
| `안전모드 정지` | 12 |
| `우측 회피` | 5 |
| `좌측 회피` | 3 |
| `목적지 도착` | 2 |

확인 예시:

```text
frame=46  mode=안전모드 정지  cmd=(0.000, 0.000)  odom_speed=0.00085  latched=True
frame=58  mode=안전모드 정지  cmd=(0.000, 0.000)  odom_speed=0.00144  latched=True
frame=144 mode=안전모드 정지  cmd=(0.139, 0.179)  odom_speed=0.00510
frame=851 mode=목적지 도착    cmd=(0.000, 0.000)  odom_speed=0.00656
```

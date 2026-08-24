# 0. Overall workflow (수정 및 보완 버전)

1. **프로세스 관리 및 IPC 통신 (`main.py`):**
   * [main.py](file:///D:/School/4-3/pupil/pupil_src/main.py)는 전체 시스템의 진입점(entry point)입니다. `main.py`는 모든 플러그인을 한군데에서 생성하고 관리하는 대신, `multiprocessing.Process`를 사용하여 개별 프로세스([eye.py](file:///D:/School/4-3/pupil/pupil_src/launchables/eye.py)와 [world.py](file:///D:/School/4-3/pupil/pupil_src/launchables/world.py))를 격리시켜 띄웁니다. 
   * 각 프로세스는 자신만의 로컬 플러그인 리스트(`Plugin_List`)를 독립적으로 관리하며, 프로세스 간 통신은 ZeroMQ IPC(Inter-Process Communication) 메시지(Notification 및 데이터 퍼블리셔)를 통해 이루어집니다.

2. **로컬 플러그인 관리 및 실행 (`plugin.py`):**
   * [plugin.py](file:///D:/School/4-3/pupil/pupil_src/shared_modules/plugin.py)의 `Plugin_List`는 각 프로세스 내부에서 로컬 플러그인들을 인스턴스화하고, `.order` 속성을 기준으로 정렬하여 실행 순서를 제어합니다. 매 프레임 루프마다 순서대로 `recent_events`와 `gl_display`를 호출합니다.

3. **눈 영상 및 동공 검출 (`eye.py`):**
   * [eye.py](file:///D:/School/4-3/pupil/pupil_src/launchables/eye.py) 프로세스는 눈 카메라에서 비디오 프레임을 받아옵니다. 활성화된 동공 검출 플러그인(예: `Detector2DPlugin` 등)이 프레임 내의 동공 윤곽을 검출하고 타원 피팅(ellipse fit)을 수행하여 `pupil_datum`을 생성합니다. 이 데이터는 ZeroMQ IPC를 통해 퍼블리싱(전송)됩니다.

4. **Gaze 데이터 매핑 및 추론 (`gazer_base.py` & `gazer_2d.py`):**
   * [world.py](file:///D:/School/4-3/pupil/pupil_src/launchables/world.py) 프로세스에 등록된 `Pupil_Data_Relay` 플러그인이 IPC를 통해 전달된 `pupil_datum` 스트림을 수신합니다.
   * 수신한 동공 데이터를 실시간으로 활성화된 `Gazer` 플러그인(예: `Gazer2D`)으로 보내고, Gazer는 사전에 매핑 및 보정된 보정 모델(Calibration Model)을 사용하여 2D 화면 좌표나 3D 시선(gaze) 데이터로 변환/추론합니다.

5. **시선 시각화 및 월드 프레임 매칭 (`world.py`):**
   * [world.py](file:///D:/School/4-3/pupil/pupil_src/launchables/world.py) 프로세스는 주 월드 카메라(장면 카메라) 영상을 획득하고 관리하는 프로세스입니다. 
   * **영상 프레임 획득:** `UVC_Source`나 `File_Source` 같은 비디오 소스 플러그인이 실행되어 월드 카메라 영상 프레임을 가져와 `events["frame"]`에 담아둡니다.
   * **플러그인 실행 루프:** 매 프레임마다 활성화된 플러그인 리스트(`g_pool.plugins`)를 루프 돌며 `recent_events(events)` 메소드를 실행합니다. 이때 순서에 따라 `Pupil_Data_Relay`가 Gazer 모델로 추론한 시선 좌표를 `events["gaze"]`에 추가하고, `Display_Recent_Gaze`가 이를 가져와 월드 프레임 위에 원형(circle)으로 시각화 오버레이 렌더링을 처리합니다.
   * **최종 데이터 배포:** 매칭 완료된 gaze 데이터와 분석 결과는 ZeroMQ IPC를 통해 실시간 방송(Publish)되어 다른 프로세스로 배포되거나 `Recorder` 플러그인을 통해 디스크에 최종 기록됩니다.

6. **보정 마커 생성 및 관리 (`calibration_choreography`):**
   * `calibration_choreography` 폴더 내의 플러그인들은 보정(Calibration)을 위한 화면 좌표계 상의 마커를 띄우고, 마커가 떠 있는 타겟 좌표(`ref_list`)와 사용자의 눈동자 위치 데이터(`pupil_list`)를 동시에 수집하는 주체입니다.

---

### 질문 답변: Gazer는 pupil에서 gaze도 추론하고 해당 데이터와 marker의 GT가 맞는지도 검증하는건가?

**네, 맞습니다!** Gazer 모델([gazer_base.py](./pupil_src/shared_modules/gaze_mapping/gazer_base.py))은 동작하는 모드에 따라 다음과 같이 시선 추론과 오차 검증(Accuracy Test)을 모두 처리합니다.

1. **보정 단계 (Calibration):**
   * 사용자가 마커를 쳐다보며 데이터 수집이 끝나면, 수집된 마커의 화면 상 실제 물리적 위치(Ground Truth, GT)와 매칭된 시선의 동공 좌표(`calib_data`)를 활용해 매핑 함수(예: 2차 다항식 회귀 모델)를 피팅(`model.fit(X, Y)`)시킵니다.

2. **일반 시선 매핑 단계 (Gaze Mapping):**
   * 피팅이 끝나 실시간 시선 매핑이 시작되면, 들어오는 실시간 동공 특징 좌표 `X`를 받아 학습된 보정 계수를 곱해 화면상의 추론 시선 좌표 `Y`를 계산(`model.predict(X)`)합니다.

3. **검증 단계 (Validation / Accuracy Test):**
   * 보정 완료 후 검증 모드(Accuracy Test / 'T' 버튼 클릭)가 켜지면 Gazer는 실시간으로 시선을 **추론(Prediction)**합니다. 
   * 동시에 검증용 화면 마커가 뿌려지는 좌표(GT)와 Gazer가 예측한 시선 좌표(Prediction)의 차이를 비교하여 픽셀 오차 및 각도 오차(Root Mean Squared Error, RMSE)를 연산하여 보정의 정확도를 수치화합니다.

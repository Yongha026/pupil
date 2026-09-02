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

---

# 6. pye3d Pupil Detection (공식 문서 재정리 및 오개념/질문 해설)

> 공식 문서: [Pupil Labs pye3d Developer Docs](https://docs.pupil-labs.com/core/developer/pye3d/)  
> 관련 코드: [`Pye3DPlugin`](file:///D:/School/4-3/pupil/pupil_src/shared_modules/pupil_detector_plugins/pye3d_plugin.py#L53) ([pye3d_plugin.py](file:///D:/School/4-3/pupil/pupil_src/shared_modules/pupil_detector_plugins/pye3d_plugin.py))

`pye3d`는 단일 눈 카메라(Single camera) 영상에서 적외선 반사광(Glint) 없이 기하학적 3D 안구 모델(Model-based approach)을 피팅하여 **3D 안구 위치(Eyeball center)**, **3D 시선 벡터(Gaze direction)**, **실제 동공 크기(Pupil radius)**를 추정하는 알고리즘입니다.

---

### 핵심 1: 헤드셋 미끄러짐(Slippage)과 3개의 타임스케일(Timescales)

눈을 추적하는 동안 말하기, 표정 변화, 착용 흔들림 등으로 인해 **헤드셋 미끄러짐(Headset slippage)**이 불가피하게 발생합니다.  
카메라와 안구 중심 간의 상대적 위치는 고정되어 있지 않으므로, `pye3d`는 3가지 타임스케일 모델을 병렬로 운용하여 **최신성(Recentness)**과 **안정성(Stability)**의 균형을 맞춥니다.

1. **Ultra-long-term 모델 (1~5분 단위):**
   * 가장 보수적인 지지 데이터(Support set) 유지 전략을 사용합니다.
   * 장기적인 평균 안구 위치(Longtime average eyeball position)를 계산합니다.
   * 새로운 데이터가 부족할 때 짧은 타임스케일 모델 피팅이 발산하지 않도록 **사전 기준점(Weak prior)** 역할을 제공합니다.

2. **Long-term 모델 (5~25초 단위):**
   * 최근 5~25초 동안의 관측치를 주로 유지하여 안구의 중기적 위치 변화를 추적합니다.
   * **동공 반경(Pupil radius)**을 산출하는 기준 안구 위치로 사용됩니다 (안정성과 최신성의 절충안).

3. **Short-term 모델 (1초 미만, 최근 10개 고신뢰도 관측치):**
   * 프레임이 들어올 때마다 즉각 갱신되는 초단기 모델입니다.
   * **원시 시선 방향(Raw gaze direction)** 추정에 사용됩니다.
   * 단일 프레임 2D 검출 노이즈를 부드럽게 걸러주는 **Gaze-direction aware filter** 역할을 합니다.

---

### 핵심 2: 오개념 교정 및 질문 상세 해설 (Q&A)

#### Q1. "새로운 pupil observation 들어오지 않으면 supporting(gaze 추론?) 금지"의 의미는?
* **작성하신 메모:**  
  `새로운 pupil observation 들어오지 않으면 supporting(gaze 추론?) 금지`
* **정확한 의미:**  
  원문의 **"Support"**는 시선 추론(Supporting)을 뜻하는 것이 아니라, 수학/통계학에서 모델을 피팅하기 위해 메모리에 들고 있는 **'지지 데이터셋(Support set / Supporting observations)'**을 의미합니다.
  * 원문: *"supporting pupil observations are retained unless equivalent newer ones become available and older ones are deleted only when not reducing the total amount of observations below a pre-defined threshold."*
  * **올바른 해석:** Ultra-long-term 모델은 기존에 피팅에 사용하던 과거 관측치(supporting pupil observations)를 최대한 버리지 않고 유지(retain)합니다. 같은 각도/영역(spatial bin)에 해당하는 새로운 동공 관측치가 들어올 때만 기존 데이터를 교체하고, 전체 관측치 개수가 정해진 임계치 밑으로 떨어지지 않을 때만 오래된 데이터를 삭제하여 **피팅용 데이터 풀(Support)을 보수적으로 유지**한다는 뜻입니다. (시선 추론을 금지한다는 말이 아닙니다.)

---

#### Q2. "Corneal interface의 refraction 효과 = glint?"
* **작성하신 메모:**  
  `Corneal interface의 refraction효과(=glint?)에 따라 수정.`
* **정확한 의미:**  
  **Refraction(굴절)은 Glint(반사광)가 전혀 아닙니다.**
  * **Glint (각막 반사광):** 안경이나 각막 표면에서 적외선 LED 불빛이 반사되어 카메라에 하얗게 맺히는 반사점(reflection)입니다. 전통적인 눈 추적기는 이 글린트를 기준으로 쓰지만, `pye3d`는 **Glint-free(글린트 불필요)** 방식입니다.
  * **Refraction (각막 굴절):** 안구 앞쪽의 각막(Cornea)은 볼록렌즈 형태의 투명한 굴절체입니다. 실제 안구 속 동공(Anatomical pupil)에서 나온 빛이 각막-공기 경계면을 통과할 때 **빛이 굴절(Bending of light rays)**되어 카메라에는 왜곡된 **가상 동공(Virtual pupil)**으로 맺히게 됩니다.
  * 따라서 2D 카메라 영상에서 보이는 동공 타원을 역투영하면 실제 동공 위치 및 시선 각도와 물리적인 오차가 발생하므로, 각막의 굴절률과 곡률을 고려한 **광학 굴절 역보정(Refraction correction)**을 거쳐야만 실제 3D 시선 벡터가 나옵니다.

---

#### Q3. "Absolute scale 에러 vs Relative pupil size" 문장의 의미는?
* **작성하신 메모:**  
  `Absolute scale에서 생긴 에러(동공 위치 잘못 측정해서)는 relative pupil size(동공 위치 추론과는 무관)보다 별로 안 중요하다.(이 문장 자체가 이해 안 감)`
* **상세 해설:**  
  원문: *"errors in absolute scale (changed by fluctuations in eyeball-position estimates) are less detrimental than errors in relative pupil sizes (less sensitive to the eyeball-position estimate)."*
  
  1. **절대 크기(Absolute scale) vs 상대 크기(Relative size):**
     * **절대 크기:** "동공 지름이 정확히 몇 mm인가?" (예: 3.45mm vs 3.60mm)
     * **상대 크기 변화:** "이전보다 동공이 몇 % 커졌는가/작아졌는가?" (예: 빛/인지부하 자극에 의해 +10% 팽창)
  2. **인지과학/동공측정학(Pupillometry)에서의 목적:**
     * 실제 안구 연구나 인지과학 실험에서는 동공이 정확히 3.45mm인지 3.50mm인지(절대 수치)는 크게 중요하지 않습니다. 중요한 것은 **자극에 반응하여 시간에 따라 동공이 얼마나 팽창/수축하는지(상대적 변화율)**입니다.
  3. **왜 Long-term 모델을 쓰는가?**
     * 3D 기하학에서 동공의 실제 물리적 크기($mm$)는 카메라와 안구 중심 사이의 거리($d$)에 비례하여 계산됩니다 ($r \propto d$).
     * 만약 매 프레임마다 안구 거리 추정치($d$)가 미세하게 덜덜 떨리면(Fluctuation), 실제 동공 크기는 가만히 있는데도 계산된 동공 크기가 요동치게 되어 **상대적 동공 변화율(Relative size) 분석이 완전히 망가집니다.**
     * 따라서 동공 크기를 잴 때는 5~25초 동안 안정적으로 스무딩된 **Long-term 모델의 안구 위치**를 사용하여 거리 측정 노이즈로 인한 인위적인 동공 크기 떨림을 방지합니다.

---

#### Q4. "Short-term 모델은 왜 recentness(최신성)가 중요한가?"
* **작성하신 메모:**  
  `Slippage 때문에 눈 위치 바뀌면 에러. Short-term 모델의 recentness가 중요하다.(왜 recentness?)`
* **상세 해설:**  
  * 사람의 시선(Gaze direction)은 1초에도 여러 번 수백 밀리초(ms) 단위로 급격하게 바뀝니다 (Saccade, Fixation).
  * 또한 안면 근육이 움직여 헤드셋이 미세하게 틀어지면(Slippage), 카메라가 바라보는 안구 각도 기준점이 즉각 변합니다.
  * 만약 시선 계산에 10~30초 전 과거 데이터를 참조한다면, 헤드셋이 이미 미끄러진 후에도 과거 위치를 기준으로 시선을 쏘게 되어 **지속적이고 영구적인 시선 각도 편향(Systematic bias/offset)**이 발생합니다.
  * 따라서 시선 방향 계산은 방금 막 발생한 헤드셋 미끄러짐을 즉각 반영할 수 있도록 **가장 최신 데이터(Recentness, 최근 1초 미만 10개 프레임)**를 우선해야 합니다.
  * 동시에 단 1프레임만 쓰지 않고 최근 10프레임을 쓰는 이유는, 2D 동공 윤곽 검출 시 발생하는 프레임 단위의 깜빡임/노이즈를 안정적으로 걸러주는 필터(Gaze-direction aware filter) 역할을 하기 위함입니다.

---

#### Q5. "총 3번 보정하는 것인가?"
* **작성하신 메모:**  
  `(그럼 pye3d, refraction correction 두 개 보정본 받고 또 보정해서 총 3번 보정하는건가?)`
* **정확한 구조:**  
  **아닙니다! 총 3번 보정하는 것이 아니라, `pye3d` 내부에서 딱 2단계(2 stages)로 순차 실행됩니다.**
  
  ```mermaid
  graph LR
      A[2D Pupil Observation] --> B[Stage 1: Raw 3D Eye Model]
      B -->|Raw gaze & pupil size| C[Stage 2: Refraction Correction]
      D[Long-term Eyeball Position] --> C
      C --> E[Final Corrected 3D Gaze & Pupil Datum]
  ```

  1. **1단계 (Raw Estimate 도출):**
     * 각막 굴절을 배제한 단순 기하학적 3D 눈 모델 피팅을 통해 초기 원시값(Raw gaze direction, Raw pupil size)과 안구 위치를 계산합니다.
  2. **2단계 (Refraction Correction 수행):**
     * 1단계에서 얻은 원시 시선/동공 크기와 Long-term 안구 위치를 각막 굴절 보정 함수(Snell's law 물리 수식)에 통과시킵니다.
     * 각막 경계면에서의 굴절 왜곡을 역계산하여 **최종 보정된 3D 시선 벡터와 동공 크기**를 단 한 번 산출해냅니다.
  * 즉, `pye3d` 외부에서 추가 보정을 계속 덧붙이는 것이 아니라 `pye3d` 내부 파이프라인 자체가 **[1단계: Raw 추정] $\rightarrow$ [2단계: 굴절 물리 보정]** 2단계로 완결되는 구조입니다.

---

### 요약 비교 표

| 모델 구분 | 시간 범위 (Time scale) | 데이터 유지 전략 (Support Building) | 주 활용 목적 (Downstream Purpose) |
| :--- | :--- | :--- | :--- |
| **Ultra-long-term** | 1 ~ 5분 | 가장 보수적, 공간 빈(Bin) 교체 및 임계치 이하 삭제 방지 | 장기 평균 안구 위치 산출, 단기 모델 피팅의 **약한 사전확률(Weak prior)** 기준점 |
| **Long-term** | 5 ~ 25초 | 최근 수십 초 데이터 위주로 유지 (안정성-최신성 절충) | 안구 거리 안정화를 통한 **동공 반경(Pupil radius)** 측정 기준 |
| **Short-term** | < 1초 (최근 10 프레임) | 최신 고신뢰도 관측치 10개 실시간 교체 (최신성 극대화) | 헤드셋 슬립 즉시 반영 및 2D 노이즈를 스무딩한 **시선 방향(Raw gaze direction)** 도출 |

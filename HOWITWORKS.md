> **⚠️ Personal study by Yongha Chun⚠️**  
Might be bullshit
# 0. Overall workflow
    1. ./pupil_src/main.py 에서 pupil 관리, eye detection 등 plugin들 인스턴스화 후 IPC(Inter-Process Comm.)로 플러그인들 리스트 줄세워놓고 관리
    2. plugin.py에서 플러드인들 줄 관리, 실행
    3. eye.py에서 pupil detector 활용, eye video에서 동공 검출 (+ellpse fit)
    4. gazer에서 pupil data로 gaze data 추론 (RITnet 버전은 Gazer2D만 사용 - 2d이미지에서 뽑은 data만 사용하니)
    5. world.py에서 gaze data와 world camera 맞춰서 화면에 어디인지(??)
    6. calibration_choreography 폴더에서 칼리 위한 마커 생성, gaze 데이터와 맞는지는 gazer.py에서 
    # TODO:  Gazer는 pupil에서 gaze도 추론하고 해당 데이터와 marker의 GT가 맞는지도 검증하는건가??? 정확한 코드 보기

# 1. Pupil detection - RITnet 버전(develop branch)
**[detector_2d_plugin.py](./pupil_src/shared_modules/pupil_detector_plugins/detector_2d_plugin.py)**
```python
class Detector2DPlugin():
# ...
    model_name = "densenet"
    model_path = "./best_model.pkl"
    self.model.load_state_dict(torch.load(model_path)) #로 load
    
    def detect_RITnet(self, frame, *args, **kwargs):
        # self.get_img로 Gamma, CLAHE, norm -> torch.Tensor 후 model(data)로 추론.
        utils.get_predictions(output) # max()로 seg class index남기기
        # 3번(pupil)만 남기고 ellipse fit후 datum(ellipse params output)

### detector_2d_plugin.py의 self.detect가 C++로 pupil detect하는애
```

**[detector_base_plugin.py](./pupil_src/shared_modules/pupil_detector_plugins/detector_base_plugin.py)**

```python
class PupilDetectorPlugin(Plugin):
    # abstractmethod인 pupil_detector pupil에서 import해옴
```
**[plugin.py](./pupil_src/shared_modules/plugin.py)**
```python
class Plugin:
    # 둘 다 부모클래스로 메소드 다 pass해서 껍데기만 쓰네? abstractmethod랑 뭐가 다르지
    # ABC는 껍데기만 강제당해서 함수 넣기 불가. 부모 클래스는 함수 만들어서 상속 가능.
    # 애네는 그냥 어플리케이션 같은데...?   
class Plugin_List:
    def import_runtime_plugins(plugin_dir):
    # 폴더 보면서 플러그인 가능성 있는 애들(.py, __init__있는 폴더) 싹다 임포트.
```
**[pupil_detector_plugins.__init__.py](./pupil_src/shared_modules/pupil_detector_plugins/__init__.py)**
```python
def available_detector_plugins() -> T.List[T.Type[PupilDetectorPlugin]]:
    # 가능한 detector들 list로 보내줌. pye3d_plugin도 가능하면 import.
```
***이걸 실제로 불러오는 애는 [eye.py](./pupil_src/launchables/eye.py).***
```python
def eye(*args, **kwargs):
    
    """
    Eye video 읽고 pupil 검출 : 눈 보는 카메라
    출력:
        pupil.<eye_id>      : Pupil data
        frame.eye.<eye_id>  : Eye frames
    """
    with Is_Alive_Manager(*args, **kwargs):
        def load_runtime_pupil_detection_plugins():
            for plugin in import_runtime_plugins(plugins_path):
                # ...
                yield plugin
        available_detectors = available_detector_plugins()
        plugins = (
            # ...
            + available_detectors   # 가능한 detector 리스트로 추가.
            + #...)

        # Event loop L630
        window_should_close = False
        while not window_should_close:
            if notify_sub.new_data:
                if subject.startswith("eye_process.should_stop"):
                    # ...
                elif (
                    subject.startswith("start_eye_plugin")
                    and notification["target"] == g_pool.process)
                ): # L693
                    try:
                        g_pool.plugins.add( # plugin.py L420에 class Plugin_List 메소드 add(self, new_plugin_cls, args={}):
                            g_pool.plugin_by_name[notification["name"]], # ex. notification["name"] = Detector2DPlugin
                            notification.get("args",{})
                        )
                    except KeyError as err:
                        pass
            
            event = {} # L711
            for plugin in g_pool.plugins:
                plugin.recent_events(event) # detector_base_plugin.py에서 detection_result = self.detect_RITnet로 결과 저장.
            
            frame event.get("frame") 
            if frame:
                # ...
                for result in event.get(EVENT_KEY, ()): # L775 
                    pupil_socket.send(result) # detect해서 ellipse fit한 datum 전송
                    # pupil_socket = zmq_tools.Msg_Streamer @L100
        
```

# 2. Pupil Data 어떻게 활용?
zmq: Application. 여긴 내가 안 봐도 될듯 한디...


# 3. Gaze Mapping

[gazer_base.py](./pupil_src/shared_modules/gaze_mapping/gazer_base.py)
```python
class GazerBase(abd.ABC, Plugin):
    def predict(self, matched_pupil_data: T.Iterator[T.List["Pupil"]])s -> T.Iterator["Gaze"]:
        pass
    def filter_pupil_data(self, pupil_data: T.Iterable, confidence_threshold: T.Optional[float]=None) -> T.Iterable:
        # Confidence 기준으로 pupil data 필터링.
        return pupil_data

    def __init__(
            self, g_pool, *, calib_data, params, raise_calibration_error
    ):
        # calib_data 받아서 fit하는 부분. Target 5개 / 9개 이용해서 calib_data 생성.

    def map_pupil_to_gaze(self, pupil_data, sort_by_creation_time):
        pupil_data = self.filter_pupil_data(pupil_data)
        if sort_by_creation_time:
            pupil_data.sort(key=lambda p: p["timestamp"])
        
        matches = (self.matcher.on_pupil_datum(datum) for datum in pupil_data)
        matches = itertools.chain.from_iterable(matches)

        yield from self.predict(matches)
        # = for i item in iterable: yield item 
        # (iterative한거 풀어서 다 리턴해줌)
```

[gazer_2d.py](./pupil_src/shared_modules/gaze_mapping/gazer_2d.py)  
Model2D_Monocualr, Model2D_Binocualr?  
Gazer_2d는 2d pupil detection data만 활용해서 gaze 추정.
```python
class Gazer2D(GazerBase):
    # def _init_left_model, _init_right_model은 Model2D_Monocular return
    # def _init_binocular_model은 Model2D_Binocular return.
    def _extract_pupil_features(self, pupil_data) -> np.ndarray:
        pupil_features = np.array([p["norm_pos"] for p in pupil_data])
        assert pupil_features.shape == (len(pupil_data), _REFERENCE_FEATURE_COUNT) # _REF=2(x,y) => (len(pupil_data), 2)
        return pupil_features
    def predict(self, matched_pupil_data:T.Iterator[T.List["Pupil"]]) -> T.Iterator["Gaze"]:
        for pupil_match in matched_pupil_data:
            num_matched = len(pupil_match)
            
            if num_matched == 2:
                if self.binocular_model.is_fitted:
                    right = self._extract_pupil_features([pupil_match[0]])
                    left = self._extract_pupil_features([pupil_match[1]])
                    gaze_positions = self.binocular_model.predict(X).tolist()
                    ...
                
            elif num_matched == 1:
                if pupil_match[0]["id"] == 0:
                    if self.right_model.is_fitted:
                        gaze_positions = self.right_model.predict(X).tolist()
                elif pupil_match[0]["id"] == 1:
                    if self.left_model.is_fitted;
                        gaze_positions = self.left_model.predict(X).tolist()
        # R,L이면 monoculr, 둘 다면 binocular model로 predict 후 gaze positions 계산.
            
            for gaze_pos in gaze_positions:
                gaze_datum ={
                    ..., "norm_pos" : gaze_pos, ... 
                } # topic(L?R?Bino?), norm_pos, confidence, timestep, base_data(pupil_match)
```
Matching은 [matching.py](./pupil_src/shared_modules/gaze_mapping/matching.py)에서.
```python
class RealtimeMatcher:
    def on_pupil_datum(self, p) -> T.Iterator:
        # Confidence로 필터링하고 시간 맞춰서 가능하면 Binocular, 안되면 Monocular data로 p yield.
        # Left, right 시간 맞춰서 matched data임.

```
[pupil_data_relay.py](./pupil_src/shared_modules/pupil_data_relay.py)  
```python
class Pupil_Data_Relay(System_Plugin_Base):
    def recent_events(self, events):
        while self.pupil_sum.new_data:
            gazer = self.g_pool.active_gaze_mapping_plugin
            for gaze_datum in gazer.map_pupil_to_gaze([pupil_datum]):
                self.gaze_pub.send(gaze_datum)
                recent_gaze_data.append(gaze_datum) # pupil ellipse data받아서 gaze data로 바꿔서 전송.
```

# 4. Calibration
[base_plugin.py](./pupil_src/shared_modules/calibration_choreography/base_plugin.py)
```python
class CalibrationChoreographyPlugin(Plugin):
    def on_choreography_successfull(self, mode:ChoreographyMode, pupil_list, ref_list):
        if mode == ChoreographyMode.CALIBRATION: 
            calib_data = {"ref_list": ref_list, "pupil_list": pupil_list})
            self._start_pluin(self.selected_gazer_class, calib_data = calib_data) 
            # plugin_name 만들어서 notify_all(@ plugin.py Plugin class)로 쏴주면 plugin_list에 추가돼서 실행.
```
[marker_sindow_controller.py](./pupil_src/shared_modules/calibration_choreography/controller/marker_window_controller.py)
```python
_MARKER_CIRCLE_RGB_FEEDBACK_INVALID = (0.8, 0.0, 0.0) # 등으로 마커 크기, 색 등 지정.
```
[screen_marker_plugin.py](./pupil_src/shared_modules/calibration_choreography/screen_marker_plugin.py) 여기서 타겟 5개 생성.
```python
class ScreenMarkerChoreographyPlugin(MonitorSelectionMixin, CalibrationChoreographyPlugin):
    def __init__(self,...,marker_scale):
        self.__marker_window = MarkerWindowController(marker_scale=marker_scale) # 로 마커 생성되는 창 관리.
```

# 5. main.py
[main.py](./pupil_src/main.py)
```python
# 여기서는 world.py, eye.py call하기만 함.
def process_notification
    # L368
    if "notify.eye_process.should_start" in topic:
        eye_id = notification["eye_id"]
        Process(taget = eye, name=f"eye{eye_id}", args=...).start()
    # L404
    if "notify.world_process.should_start" in topic:
        Process(target=world, name="world", args = ...)
    
"""
from multiprocessing import Process, ...
Process(target = func, args = {}).start하면 멀티프로세스로 함수 실행.
"""
```
[world.py](./pupil_src/launchables/world.py)
```python
def world():
    """
    world video 읽고 plugin 호출 : 바깥 보는 카메라
    출력: 
        gaze: Gaze mapping으로 얻은 gaze data
    """
def world(...):
    events = {}
    events["dt"] = get_dt() # loop중 timestep 가져오기
    
    for p in g_pool.plugins:
        p.recent_events(events) # 각 플러그인들에게 데이터 주고 해야할 일 시키기 = detect pupil, predict gaze...
    
    del events["pupil"]
    del events["gaze"] # 처리 및 전송 완료한 데이터 삭제 
```
---
# 99. Gemini 정리
### Brief Flow of the Total System

graph TD
    A[Camera Feed / Video Source] $\rightarrow$ |Capture Frame| B(Eye Process: eye.py)  
    B $\rightarrow$ |Calls recent_events| C(DetectorBasePlugin)  
    C $\rightarrow$ |Calls detect| D[Your Custom UNet Inference in detector_2d_plugin.py]  
    D $\rightarrow$ |Returns Pupil Datum| C  
    C $\rightarrow$ |IPC Publish pupil topic| E[ZeroMQ IPC Backbone]  
    E $\rightarrow$ |IPC Subscribe pupil topic| F(World Process: world.py via Pupil_Data_Relay)  
    F $\rightarrow$ |Maps pupil data| G[Active Gaze Mapper eg. Gazer2D]  
    G $\rightarrow$ |Outputs Gaze Coordinates| H[ZMQ IPC gaze topic & GUI Renderer]  

#### Step-by-Step System Flow:

  1. Frame Capture: The eye camera capture plugin grabs a new video frame and passes it to the event dictionary in
  the eye.py:55 process.
  2. Model Inference: detector_2d_plugin.py:38 processes the frame with your custom UNet model, extracting
  coordinates of the pupil center, diameter, and confidence, returning a standardized dictionary called a pupil_datum.
  3. Data Streaming: The eye.py:55 process streams this pupil_datum over ZeroMQ IPC under the pupil topic.
  4. Relaying: The world.py:18 process receives the stream via pupil_data_relay.py:16.
  5. Gaze Estimation: The relay feeds the data into the active gaze mapping plugin (e.g. Gazer2D), which applies the
  calibration math to project the raw pupil coordinates into gaze coordinates in the world coordinate space.
  6. Consumer Distribution: The resulting gaze coordinates are broadcasted under the gaze topic for real-time
  visualization overlays, network APIs, or disk recordings.


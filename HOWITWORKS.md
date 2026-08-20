from plugin import System_Plugin_Basefrom plugin import Pluginfrom glfw import window_should_closefrom pupil_detector_plugins import available_detector_plugins> **⚠️ Personal study by Yongha Chun⚠️**  
Might be bullshit


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
        # calib_data 받아서 fit하는 부분. 초반 몇 프레임 받아서 calibration 하는 부분 맞는지?
        # TODO: 칼리브레이션 어떻게 하는건지 보기.

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
                    # TODO: pupil_match는 뭐가 매칭된거지

```
[pupil_data_relay.py](./pupil_src/shared_modules/pupil_data_relay.py)  
```python
class Pupil_Data_Relay(System_Plugin_Base):
    def recent_events(self, events):
        while self.pupil_sum.new_data:
            gazer = self.g_pool.active_gaze_mapping_plugin
            for gaze_datum in gazer.map_pupil_to_gaze([pupil_datum]):
                self.gaze_pub.send(gaze_datum)
                recent_gaze_data.append(gaze_datum) # pupil ellipse data받아서 gaze data로 바꿔서 전송해주는 부분.
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
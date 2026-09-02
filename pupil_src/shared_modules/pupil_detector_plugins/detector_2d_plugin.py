"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
DETECT_MODEL = "pmrnet" # adgbc, nn_ritnet, nn_unext, 2dcpp, ritnet, mambaliteunet, rollingunet, ulvmunet, ukan, pmrnet

import logging
import numpy as np
import os
import time


import glfw
from gl_utils import (
    GLFWErrorReporting,
    adjust_gl_view,
    basic_gl_setup,
    clear_gl_screen,
    make_coord_system_norm_based,
    make_coord_system_pixel_based,
)
from pupil_detectors import Detector2D, DetectorBase, Roi
from pyglui import ui
from pyglui.cygl.utils import draw_gl_texture

GLFWErrorReporting.set_default()

from methods import normalize
from plugin import Plugin

from . import color_scheme
from .detector_base_plugin import PupilDetectorPlugin
from .visualizer_2d import draw_pupil_outline
from pupil_detector_plugins import adgbc, nn_ritnet, nn_unext, mambaliteunet, rollingunet, ulvmunet, ukan, pmrnet
from draw_ellipse import fit_ellipse
from CheckEllipse import computeEllipseConfidence
import cv2
import torch
import PIL
from pupil_detector_plugins.utils import get_predictions
from pupil_detector_plugins.models import model_dict
import torchvision
import time

COLOR_MAX = 255
COLOR_CAP = 256
EYE_CLASS = 1
IMAGE_MOD = 16
BBOX_EXTRA_SPACE = 20
CLIP_LIMIT = 1.5
TILE_GRID_SIZE = 8
EYE_CLASS = 1
logger = logging.getLogger(__name__)

class Detector2DPlugin(PupilDetectorPlugin):
    pupil_detection_identifier = "2d"
    pupil_detection_method = "2d c++"

    label = "C++ 2d detector"
    icon_font = "pupil_icons"
    icon_chr = chr(0xEC18)
    order = 0.100


    @property
    def pretty_class_name(self):
        return "Pupil Detector 2D"

    @property
    def pupil_detector(self) -> DetectorBase:
        return self.detector_2d

    def __init__(
        self,
        g_pool=None,
        properties=None,
        detector_2d: Detector2D = None,
    ):
        super().__init__(g_pool=g_pool)
        self.detector_2d = detector_2d or Detector2D(properties or {})
        self.latency_log = []
        """
        기존 __init__에 model_path, device, preview 등을 인자로 추가.
        """

        self.plugin_dir = os.path.join(os.path.dirname(__file__), "model_ckpts")
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.models = {}

        # Set initial active_model from properties if saved, otherwise default to DETECT_MODEL
        initial_model = (properties or {}).get("active_model", DETECT_MODEL)
        self._active_model = initial_model
        self.set_active_model(initial_model)

        ################################################################################################

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.clahe = cv2.createCLAHE(
            clipLimit=CLIP_LIMIT, tileGridSize=(TILE_GRID_SIZE, TILE_GRID_SIZE)
        )

    @property
    def active_model(self):
        return self._active_model

    @active_model.setter
    def active_model(self, model_name):
        self.set_active_model(model_name)

    def set_active_model(self, model_name):
        global DETECT_MODEL
        DETECT_MODEL = model_name
        self._active_model = model_name
        logger.info(f"Active model set to: {model_name}")
        if model_name != "2dcpp" and model_name not in self.models:
            self.load_model(model_name)

    def load_model(self, model_name):
        if model_name == "2dcpp":
            return None

        if model_name in self.models:
            return self.models[model_name]

        model = None
        if model_name == "adgbc":
            # 1) AD-GBC Model
            try:
                model = adgbc.GBC_Rolling_Unet_L(num_classes=4, input_channels=1, deep_supervision=False).to(self.device)
                model_path = os.path.join(self.plugin_dir, "adgbc_nn_best.pth")
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    state_dict = checkpoint["network_weights"] if (isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
                    model.load_state_dict(state_dict)
                    model.eval()
                    self.models["adgbc"] = model
                    logger.info("Loaded AD-GBC weights successfully")
                else:
                    logger.warning(f"AD-GBC ckpt file not found at {model_path}")
            except Exception as e:
                logger.error(f"Failed to load AD-GBC: {e}")

        elif model_name == "ritnet":
            # 2) RITnet Model (Original Densenet)
            try:
                model = model_dict['densenet']().to(self.device)
                model_path = os.path.join(self.plugin_dir, "best_model.pkl")
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
                    model.eval()
                    self.models["ritnet"] = model
                    logger.info("Loaded RITnet (original) weights successfully")
                else:
                    logger.warning(f"RITnet original ckpt file not found at {model_path}")
            except Exception as e:
                logger.error(f"Failed to load RITnet: {e}")

        elif model_name == "nn_ritnet":
            # 3) nnRITnet Model
            try:
                model = nn_ritnet.DenseNet2D(in_channels=1, out_channels=4, dropout=True, prob=0.2, deep_supervision=False).to(self.device)
                model_path = os.path.join(self.plugin_dir, "ritnet_nn_best.pth")
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    state_dict = checkpoint["network_weights"] if (isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
                    model.load_state_dict(state_dict)
                    model.eval()
                    self.models["nn_ritnet"] = model
                    logger.info("Loaded nnRITnet weights successfully")
                else:
                    logger.warning(f"nnRITnet ckpt file not found at {model_path}")
            except Exception as e:
                logger.error(f"Failed to load nnRITnet: {e}")

        elif model_name == "nn_unext":
            # 4) UNeXt Model
            try:
                model = nn_unext.UNext(num_classes=4, input_channels=1, deep_supervision=False).to(self.device)
                model_path = os.path.join(self.plugin_dir, "unext_nn_best.pth")
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    state_dict = checkpoint["network_weights"] if (isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
                    model.load_state_dict(state_dict)
                    model.eval()
                    self.models["nn_unext"] = model
                    logger.info("Loaded UNeXt weights successfully")
                else:
                    logger.warning(f"UNeXt ckpt file not found at {model_path}")
            except Exception as e:
                logger.error(f"Failed to load UNeXt: {e}")

        elif model_name == "mambaliteunet":
            # 5) MambaLiteUNet Model
            try:
                model = mambaliteunet.MambaLiteUNet(num_classes=4, input_channels=1).to(self.device)
                model_path = os.path.join(self.plugin_dir, "mambaliteunet_nn_best.pth")
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    state_dict = checkpoint["network_weights"] if (isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
                    model.load_state_dict(state_dict)
                    model.eval()
                    self.models["mambaliteunet"] = model
                    logger.info("Loaded MambaLiteUNet weights successfully")
                else:
                    logger.warning(f"MambaLiteUNet ckpt file not found at {model_path}")
            except Exception as e:
                logger.error(f"Failed to load MambaLiteUNet: {e}")

        elif model_name == "rollingunet":
            # 6) Rolling_Unet_L Model
            try:
                model = rollingunet.Rolling_Unet_L(num_classes=4, input_channels=1, deep_supervision=False).to(self.device)
                model_path = os.path.join(self.plugin_dir, "rollingunet_nn_best.pth")
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    state_dict = checkpoint["network_weights"] if (
                                isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
                    model.load_state_dict(state_dict)
                    model.eval()
                    self.models["rollingunet"] = model
                    logger.info("Loaded RollingUNet_L weights successfully")
                else:
                    logger.warning(f"RollingUNet_L ckpt file not found at {model_path}")
            except Exception as e:
                logger.error(f"Failed to load RollingUNet_L: {e}")

        elif model_name == "ulvmunet":
            # 7) UltraLight_VM_UNet Model
            try:
                model = ulvmunet.UltraLight_VM_UNet(num_classes=4, input_channels=1).to(self.device)
                model_path = os.path.join(self.plugin_dir, "ulvm_nn_best.pth")
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    state_dict = checkpoint["network_weights"] if (isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
                    model.load_state_dict(state_dict)
                    model.eval()
                    self.models["ulvmunet"] = model
                    logger.info("Loaded ULVMUNet weights successfully")
                else:
                    logger.warning(f"ULVMUNet ckpt file not found at {model_path}")
            except Exception as e:
                logger.error(f"Failed to load ULVMUNet: {e}")

        elif model_name == "ukan":
            # 8) UKAN
            try:
                model = ukan.UKAN(num_classes=4, input_channels=1, deep_supervision=False).to(self.device)
                model_path = os.path.join(self.plugin_dir, "ukan_nn_best.pth")
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    state_dict = checkpoint["network_weights"] if (isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
                    model.load_state_dict(state_dict)
                    model.eval()
                    self.models["ukan"] = model
                    logger.info("Loaded UKAN weights successfully")
                else:
                    logger.warning(f"UKAN ckpt file not found at {model_path}")
            except Exception as e:
                logger.error(f"Failed to load UKAN: {e}")

        elif model_name == "pmrnet":
            # 9) PMRNet Model
            try:
                try:
                    model = pmrnet.PMRNet(num_classes=4, in_channels=1).to(self.device)
                except TypeError:
                    model = pmrnet.PMRNet(num_classes=4, input_channels=1).to(self.device)
                model_path = os.path.join(self.plugin_dir, "pmr_nn_best.pth")
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    state_dict = checkpoint["network_weights"] if (isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
                    model.load_state_dict(state_dict)
                    model.eval()
                    self.models["pmrnet"] = model
                    logger.info("Loaded PMRNet weights successfully")
                else:
                    logger.warning(f"PMRNet ckpt file not found at {model_path}")
            except Exception as e:
                logger.error(f"Failed to load PMRNet: {e}")

        return model

    def get_init_dict(self):
        init_dict = super().get_init_dict()
        init_dict["properties"] = self.detector_2d.get_properties()
        init_dict["properties"]["active_model"] = self.active_model
        return init_dict

    def detect(self, frame, **kwargs):
        if not self.active_model == "2dcpp":
            return self.detect_MODEL(frame, **kwargs)
        else:
            # convert roi-plugin to detector roi
            roi = Roi(*self.g_pool.roi.bounds)

            debug_img = frame.bgr if self.g_pool.display_mode == "algorithm" else None
            result = self.detector_2d.detect(
                gray_img=frame.gray,
                color_img=debug_img,
                roi=roi,
            )

            norm_pos = normalize(
                result["location"], (frame.width, frame.height), flip_y=True
            )

            # Create basic pupil datum
            datum = self.create_pupil_datum(
                norm_pos=norm_pos,
                diameter=result["diameter"],
                confidence=result["confidence"],
                timestamp=frame.timestamp,
            )

            # Fill out 2D model data
            datum["ellipse"] = {}
            datum["ellipse"]["axes"] = result["ellipse"]["axes"]
            datum["ellipse"]["angle"] = result["ellipse"]["angle"]
            datum["ellipse"]["center"] = result["ellipse"]["center"]

            return datum

    def detect_MODEL(self, frame, model_name=None, **kwargs):
        # Determine model to use
        model_name = model_name or self.active_model
        model = self.models.get(model_name)
        if model is None:
            logger.error(f"Model '{model_name}' is not loaded/available!")
            return {
                "location": (0.0, 0.0),
                "diameter": 0.0,
                "confidence": 0.0,
                "ellipse": {"axes": (0.0, 0.0), "angle": 0.0, "center": (0.0, 0.0)},
            }

        # 1) Frame preprocessing (robust check)
        if hasattr(frame, "gray"):
            gray = frame.gray
        elif isinstance(frame, np.ndarray):
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
        else:
            try:
                img = self.convert_mjpeg_to_numpy(frame)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                logger.error(f"Error converting MJPEGFrame: {e}")
                return None

        gray = gray.astype(np.uint8)

        # 2) Preprocess -> Tensor
        tensor = self.get_img(gray).unsqueeze(0).to(self.device)  # (1, 1, H, W)

        # 3) Inference
        with torch.no_grad():
            output = model(tensor)

        # 4) Label map extraction
        predict = get_predictions(output)  # (1, H, W)
        predict_2d = predict[0].cpu().numpy()  # (H, W)

        # 5) Pupil mask extraction (class 3)
        pupil_mask = np.zeros_like(predict_2d, dtype=np.uint8)
        pupil_mask[predict_2d == 3] = 255

        # 6) Find contours (using RETR_EXTERNAL and CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(
            pupil_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            result = {
                "location": (0.0, 0.0),
                "diameter": 0.0,
                "confidence": 0.0,
                "ellipse": {"axes": (0.0, 0.0), "angle": 0.0, "center": (0.0, 0.0)},
            }
        else:
            best_contour = max(contours, key=cv2.contourArea)
            if len(best_contour) < 5:
                result = {
                    "location": (0.0, 0.0),
                    "diameter": 0.0,
                    "confidence": 0.0,
                    "ellipse": {"axes": (0.0, 0.0), "angle": 0.0, "center": (0.0, 0.0)},
                }
            else:
                (cx, cy), (MA, ma), angle_deg = cv2.fitEllipse(best_contour)
                result = {
                    "location": (float(cx), float(cy)),
                    "diameter": float(MA),
                    "confidence": 1.0,
                    "ellipse": {
                        "axes": (float(MA), float(ma)),
                        "angle": float(angle_deg),
                        "center": (float(cx), float(cy)),
                    },
                }

        # 7) Create Pupil Labs datum
        norm_pos = normalize(result["location"], (frame.width, frame.height), flip_y=True)
        datum = self.create_pupil_datum(
            norm_pos=norm_pos,
            diameter=result["diameter"],
            confidence=result["confidence"],
            timestamp=frame.timestamp,
        )
        datum["ellipse"] = {
            "axes": result["ellipse"]["axes"],
            "angle": result["ellipse"]["angle"],
            "center": result["ellipse"]["center"],
        }

        return datum

    def cleanup(self):
        super().cleanup()


    def convert_mjpeg_to_numpy(self, frame):
        try:
            # frame.jpeg_buffer를 numpy 배열로 변환
            img_array = np.frombuffer(frame.jpeg_buffer, dtype=np.uint8)
            # OpenCV로 MJPEG 디코딩
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except AttributeError as e:
            raise AttributeError(f"frame 객체에서 jpeg_buffer를 찾을 수 없습니다: {e}")
        except Exception as e:
            raise RuntimeError(f"MJPEG 데이터를 numpy로 변환하는 중 오류 발생: {e}")

    def get_img(self, img: np.ndarray) -> torch.Tensor:
        """
        1) Gamma correction (0.8)
        2) CLAHE
        3) PIL 변환 -> transforms.ToTensor() + Normalize([0.5],[0.5])
        4) 바로 텐서로 리턴
        """
        # (H, W) = img.shape[:2]  # 필요시 사용

        # 1) gamma correction
        table = float(COLOR_MAX) * (np.linspace(0, 1, COLOR_CAP) ** 0.8)
        img_gamma = cv2.LUT(img.astype(np.uint8), table.astype(np.uint8))

        # 2) CLAHE
        img_clahe = self.clahe.apply(img_gamma)

        # 3) PIL 변환
        pil_img = PIL.Image.fromarray(img_clahe)

        # 4) ToTensor + Normalize([0.5],[0.5])
        #   (self.transform이 이미 transforms.Compose([...])로 정의되어 있다고 가정)
        tensor_img = self.transform(pil_img)
        # tensor_img: shape [C, H, W], dtype=torch.float32, 범위 ~ [-1,1]

        return tensor_img

    def find_bbox(self, img):
        """find the region most likely to be the eye and find its bbox

        Args:
            img: output from the eye segmentation
        """
        shape = img.shape

        bbox = {"x_min": shape[1], "x_max": 0, "y_min": shape[0], "y_max": 0}

        bboxs = []
        for c in range(shape[1]):
            check = False
            for r in range(shape[0]):
                if img[r, c] >= EYE_CLASS:
                    bbox["x_min"] = min(bbox["x_min"], c)
                    bbox["y_min"] = min(bbox["y_min"], r)
                    bbox["x_max"] = max(bbox["x_max"], c)
                    bbox["y_max"] = max(bbox["y_max"], r)
                    check = True

            if not check and bbox["x_max"] > 0:
                bboxs.append(bbox)
                bbox = {"x_min": shape[1], "x_max": 0, "y_min": shape[0], "y_max": 0}

        if len(bboxs) == 0:
            return {"x_min": 0, "x_max": shape[1], "y_min": 0, "y_max": shape[0]}

        # find the biggest region to be the bbox
        best_bbox = bboxs[0]
        for bbox in bboxs:
            area = (bbox["x_max"] - bbox["x_min"]) * (bbox["y_max"] - bbox["y_min"])

            best_area = (best_bbox["x_max"] - best_bbox["x_min"]) * (
                best_bbox["y_max"] - best_bbox["y_min"]
            )

            if area > best_area:
                best_bbox = dict(bbox)

        return dict(best_bbox)

    def extract_pupil(self, predict):
        """
            this function extract pupil from segmentation map,
            pupil result is used in the later gaze prediction process.
        """
        predict = np.array(predict)
        bbox = self.find_bbox(predict)
        if np.max(predict) > 0:
            predict = predict / np.max(predict)
        blank_img = np.zeros_like(predict)
        blank_img[
            bbox["y_min"] : bbox["y_max"], bbox["x_min"] : bbox["x_max"]
        ] = predict[bbox["y_min"] : bbox["y_max"], bbox["x_min"] : bbox["x_max"]]

        predict = blank_img

        low_pass_filter = predict < EYE_CLASS
        predict[low_pass_filter] = 0

        # if self.preview:
        #     cv2.imshow(name, predict)
        #     cv2.waitKey(30)

        predict = np.expand_dims(predict, axis=0)

        return predict




    def convert_to_builtin(self, obj):
        """
        재귀적으로 numpy.ndarray를 기본 Python list로 변환하는 함수.
        dict, list, tuple 내에 있는 numpy 배열도 변환합니다.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.convert_to_builtin(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_builtin(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_to_builtin(i) for i in obj)
        else:
            return obj

    #################################################################

    def unproject_single_observation(self, prediction, mask=None, threshold=0.5):
        # try:
        #     assert len(prediction.shape) == 2
        #     assert prediction.shape == self.image_shape
        # except(AssertionError):
        #     raise AssertionError(
        #         "Shape of the observation input has to be (image_height, image_width) specified in the initialization of object, or if default, (240,320)")

        # Fit an ellipse from the prediction map
        ellipse_info = fit_ellipse(prediction, mask=mask)
        ellipse_confidence = 0

        if ellipse_info is not None:
            rr, cc, centre, w, h, radian, ell = ellipse_info
            ellipse_confidence = computeEllipseConfidence(prediction, centre, w, h, radian)

            result = {
                'ellipse': {
                    'center': (float(centre[0]), float(centre[1])),
                    'axes': (float(w), float(h)),
                    'angle': float(np.degrees(radian)),  # 라디안을 각도로 변환
                },
                'diameter': float(h),
                'location': (float(centre[0]), float(centre[1])),
                'confidence': float(ellipse_confidence),
            }
        else:
            result = {
                'ellipse': {
                    'center': (0.0, 0.0),
                    'axes': (0.0, 0.0),
                    'angle': 0.0,
                },
                'diameter': 0.0,
                'location': (0.0, 0.0),
                'confidence': 0.0,
            }

        return result



    def init_ui(self):
        super().init_ui()
        self.menu.label = self.pretty_class_name
        self.menu_icon.label_font = "pupil_icons"
        info = ui.Info_Text(
            "Switch to the algorithm display mode to see a visualization of pupil detection parameters overlaid on the eye video. "
            + "Adjust the pupil intensity range so that the pupil is fully overlaid with blue. "
            + "Adjust the pupil min and pupil max ranges (red circles) so that the detected pupil size (green circle) is within the bounds."
        )
        self.menu.append(info)
        self.menu.append(
            ui.Selector(
                "active_model",
                self,
                label="Active Model",
                selection=["adgbc", "nn_ritnet", "nn_unext", "2dcpp", "ritnet", "mambaliteunet", "rollingunet", "ulvmunet", "ukan", "pmrnet"],
                setter=self.set_active_model,
            )
        )
        self.menu.append(
            ui.Slider(
                "intensity_range",
                self.pupil_detector_properties,
                label="Pupil intensity range",
                min=0,
                max=60,
                step=1,
            )
        )
        self.menu.append(
            ui.Slider(
                "pupil_size_min",
                self.pupil_detector_properties,
                label="Pupil min",
                min=1,
                max=250,
                step=1,
            )
        )
        self.menu.append(
            ui.Slider(
                "pupil_size_max",
                self.pupil_detector_properties,
                label="Pupil max",
                min=50,
                max=400,
                step=1,
            )
        )
        info = ui.Info_Text(
            "When using Neon in bright light, increasing the Canny Threshold can "
            "help reduce the effect of reflections in the eye image and improve pupil "
            "detection. The default value is 160."
        )
        self.menu.append(info)
        self.menu.append(
            ui.Slider(
                "canny_treshold",
                self.pupil_detector_properties,
                label="Canny Threshold",
                min=0,
                max=1000,
                step=1,
            )
        )
        self.menu.append(ui.Info_Text("Color Legend"))
        self.menu.append(
            ui.Color_Legend(color_scheme.PUPIL_ELLIPSE_2D.as_float, "2D pupil ellipse")
        )

    def gl_display(self):
        if self._recent_detection_result:
            draw_pupil_outline(
                self._recent_detection_result,
                color_rgb=color_scheme.PUPIL_ELLIPSE_2D.as_float,
            )

    def on_resolution_change(self, old_size, new_size):
        properties = self.pupil_detector.get_properties()
        properties["pupil_size_max"] *= new_size[0] / old_size[0]
        properties["pupil_size_min"] *= new_size[0] / old_size[0]
        self.pupil_detector.update_properties(properties)

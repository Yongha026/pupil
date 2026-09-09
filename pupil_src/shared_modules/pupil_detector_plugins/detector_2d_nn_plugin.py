import csv
import gc
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple
import cv2
import gl_utils
import glfw
import numpy as np
import PIL.Image
import torch
import torchvision

from methods import normalize
from pupil_detectors import Detector2D, DetectorBase, Roi
from pyglui import graph, ui
from pyglui.cygl.utils import RGBA, mix_smooth

from pupil_detector_plugins import (
    adgbc,
    color_scheme,
    mambaliteunet,
    nn_ritnet,
    nn_unext,
    pmrnet,
    rollingunet,
    ukan,
    ulvmunet,
)
from pupil_detector_plugins.detector_base_plugin import PupilDetectorPlugin
from pupil_detector_plugins.models import model_dict
from pupil_detector_plugins.visualizer_2d import draw_pupil_outline

logger = logging.getLogger(__name__)

COLOR_MAX = 255
COLOR_CAP = 256
CLIP_LIMIT = 1.5
TILE_GRID_SIZE = 8

AVAILABLE_MODELS: List[Tuple[str, str]] = [
    ("pmrnet", "PMRNet"),
    ("ritnet", "RITnet (Original)"),
    ("nn_ritnet", "nnRITnet"),
    ("nn_unext", "UNeXt"),
    ("mambaliteunet", "MambaLiteUNet"),
    ("rollingunet", "RollingUNet"),
    ("ulvmunet", "UltraLight-VMUNet"),
    ("ukan", "U-KAN"),
    ("adgbc", "AD-GBC"),
    ("adgbc_400","AD-GBC_400"),
    ("2dcpp", "Classic C++ (2D)"),
]

SMOOTHING_METHODS: List[Tuple[str, str]] = [
    ("ema", "EMA (Heuristic)"),
    ("one_euro", "One-Euro Filter"),
]


class OneEuroFilter:
    """
    1€ Filter: A simple speed-based low-pass filter for noisy input in interactive systems.
    Reference: Casiez, Roussel, Vogel (CHI 2012).
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.01,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(rate: float, cutoff: float) -> float:
        if rate <= 0.0 or cutoff <= 0.0:
            return 1.0
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        x = float(x)
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = float(timestamp) if timestamp is not None and timestamp > 0 else None
            return self.x_prev

        if timestamp is not None and timestamp > 0 and self.t_prev is not None and self.t_prev > 0:
            dt = timestamp - self.t_prev
            if dt > 1e-4:
                rate = 1.0 / dt
            else:
                rate = 60.0
            if dt > 1.0:
                self.reset()
                self.x_prev = x
                self.t_prev = float(timestamp)
                return self.x_prev
        else:
            rate = 60.0

        self.t_prev = float(timestamp) if timestamp is not None and timestamp > 0 else None

        # 1. Estimate and low-pass filter derivative (velocity)
        dx = (x - self.x_prev) * rate
        alpha_d = self._alpha(rate, self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev
        self.dx_prev = dx_hat

        # 2. Dynamically adapt cutoff frequency based on velocity
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(rate, cutoff)

        # 3. Filter signal
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = x_hat
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class EllipseOneEuroFilter:
    """
    Multi-dimensional 1€ filter for pupil ellipse parameters:
    (cx, cy, minor_diameter, major_diameter, angle_deg)
    with circular angle unwrapping.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ):
        self.f_cx = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.f_cy = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.f_minor = OneEuroFilter(min_cutoff=min_cutoff, beta=beta * 0.5, d_cutoff=d_cutoff)
        self.f_major = OneEuroFilter(min_cutoff=min_cutoff, beta=beta * 0.5, d_cutoff=d_cutoff)
        self.f_angle = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def filter(
        self,
        cx: float,
        cy: float,
        minor_d: float,
        major_d: float,
        angle_deg: float,
        timestamp: Optional[float] = None,
    ) -> Tuple[float, float, float, float, float]:
        cx_hat = self.f_cx.filter(cx, timestamp)
        cy_hat = self.f_cy.filter(cy, timestamp)
        minor_hat = self.f_minor.filter(minor_d, timestamp)
        major_hat = self.f_major.filter(major_d, timestamp)

        if self.f_angle.x_prev is None:
            angle_hat = self.f_angle.filter(angle_deg, timestamp) % 180.0
        else:
            prev_ang_base = self.f_angle.x_prev
            prev_ang_norm = prev_ang_base % 180.0
            diff_ang = (angle_deg - prev_ang_norm + 90.0) % 180.0 - 90.0
            unwrapped = prev_ang_base + diff_ang
            filtered_ang = self.f_angle.filter(unwrapped, timestamp)
            angle_hat = filtered_ang % 180.0
            if abs(self.f_angle.x_prev) > 3600.0:
                self.f_angle.x_prev = angle_hat

        return cx_hat, cy_hat, minor_hat, major_hat, angle_hat

    def reset(self):
        self.f_cx.reset()
        self.f_cy.reset()
        self.f_minor.reset()
        self.f_major.reset()
        self.f_angle.reset()


class nnUNetDetector2DPlugin(PupilDetectorPlugin):
    """
    Neural Network based 2D pupil detector supporting dynamic model selection
    across multiple deep learning architectures and classic C++ 2D.

    VRAM is managed strictly such that only the currently active model is held in GPU
    memory, freeing resources completely when switching to the C++ detector.
    """

    pupil_detection_identifier = "2d"
    pupil_detection_method = "2d c++"

    uniqueness = "by_class"
    icon_font = "pupil_icons"
    icon_chr = chr(0xEC18)

    label = "Neural Network 2D Detector"
    order = 0.100

    @property
    def pretty_class_name(self) -> str:
        return "Pupil Detector 2D (Neural Net)"

    @property
    def pupil_detector(self) -> DetectorBase:
        return self.__detector_2d

    def __init__(
        self,
        g_pool=None,
        active_model: str = "adgbc",
        confidence_threshold: float = 0.6,
        show_confidence_graph: bool = True,
        enable_smoothing: bool = True,
        smoothing_method: str = "one_euro",
        smooth_alpha: float = 0.4,
        one_euro_min_cutoff: float = 1.0,
        one_euro_beta: float = 0.01,
        properties: Optional[dict] = None,
        flip_vertically = False,
        flip_horizontally = False,
        **kwargs,
    ):
        super().__init__(g_pool=g_pool)
        self.__detector_2d = Detector2D(properties or {})
        self._stop_other_pupil_detectors()

        self.plugin_dir = os.path.dirname(__file__)
        self.ckpt_dir = os.path.join(self.plugin_dir, "model_ckpts")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inherit model from g_pool session if available
        if hasattr(self.g_pool, "pupil_detector_model") and self.g_pool.pupil_detector_model:
            active_model = self.g_pool.pupil_detector_model

        self.latency_log = []
        self.active_model = active_model
        self.confidence_threshold = float(confidence_threshold)
        self.show_confidence_graph = bool(show_confidence_graph)

        self._enable_smoothing = bool(enable_smoothing)
        if hasattr(self.g_pool, "pupil_detector_smoothing"):
            self._enable_smoothing = bool(self.g_pool.pupil_detector_smoothing)

        self._smoothing_method = str(smoothing_method)
        if hasattr(self.g_pool, "pupil_detector_smoothing_method"):
            self._smoothing_method = str(self.g_pool.pupil_detector_smoothing_method)

        self.smooth_alpha = float(smooth_alpha)
        self.one_euro_min_cutoff = float(one_euro_min_cutoff)
        self.one_euro_beta = float(one_euro_beta)

        self._prev_ellipse = None
        self._consecutive_jumps = 0
        self._one_euro_filter = EllipseOneEuroFilter(
            min_cutoff=self.one_euro_min_cutoff,
            beta=self.one_euro_beta,
        )

        self.conf_graph = None
        self.conf_grad = None
        self.conf_grad_limits = (0.0, 1.0)
        self._last_fb_size = None
        self._last_content_scale = None
        self.model = None

        self.model_keys = [k for k, _ in AVAILABLE_MODELS]
        self.model_labels = [label for _, label in AVAILABLE_MODELS]

        self.smoothing_method_keys = [k for k, _ in SMOOTHING_METHODS]
        self.smoothing_method_labels = [label for _, label in SMOOTHING_METHODS]

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.clahe = cv2.createCLAHE(
            clipLimit=CLIP_LIMIT, tileGridSize=(TILE_GRID_SIZE, TILE_GRID_SIZE)
        )

        self.flip_vertically = flip_vertically
        self.flip_horizontally = flip_horizontally

        # Initial single-model VRAM loading
        if self.active_model != "2dcpp":
            self.model = self._load_model(self.active_model)

    def _stop_other_pupil_detectors(self):
        plugin_list = getattr(self.g_pool, "plugins", None)
        if plugin_list is None:
            return

        for plugin in plugin_list:
            if (
                isinstance(plugin, PupilDetectorPlugin)
                and plugin is not self
                and getattr(plugin, "pupil_detection_identifier", "") == "2d"
            ):
                plugin.alive = False

        plugin_list.clean()

    def on_resolution_change(self, old_size, new_size):
        properties = self.pupil_detector.get_properties()
        properties["pupil_size_max"] *= new_size[0] / old_size[0]
        properties["pupil_size_min"] *= new_size[0] / old_size[0]
        self.pupil_detector.update_properties(properties)

    # -------------------------------------------------------------------------
    # VRAM & Single Model Management
    # -------------------------------------------------------------------------
    def _unload_current_model(self):
        """Unload active PyTorch model from VRAM and free GPU cache."""
        if self.model is not None:
            del self.model
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Unloaded previous model and cleared GPU cache.")

    def _load_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """Instantiate and load weights for the selected model onto self.device."""
        logger.info(f"Loading '{model_name}' onto {self.device}...")
        model = None

        try:
            if model_name == "pmrnet":
                ckpt_path = os.path.join(self.ckpt_dir, "pmr_nn_best.pth")
                model = pmrnet.PMRNet(num_classes=4, in_channels=1).to(self.device)
                self._load_state_dict(model, ckpt_path, "PMRNet")

            elif model_name == "ritnet":
                ckpt_path = os.path.join(self.ckpt_dir, "best_model.pkl")
                if not os.path.exists(ckpt_path):
                    ckpt_path = os.path.join(self.plugin_dir, "best_model.pkl")
                model = model_dict["densenet"]().to(self.device)
                if os.path.exists(ckpt_path):
                    weights = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                    model.load_state_dict(weights)
                    model.eval()
                    logger.info(f"Loaded RITnet (original) weights from {ckpt_path}")
                else:
                    logger.warning(f"RITnet ckpt not found at {ckpt_path}")

            elif model_name == "nn_ritnet":
                ckpt_path = os.path.join(self.ckpt_dir, "ritnet_nn_best.pth")
                model = nn_ritnet.DenseNet2D(
                    in_channels=1,
                    out_channels=4,
                    dropout=True,
                    prob=0.2,
                    deep_supervision=False,
                ).to(self.device)
                self._load_state_dict(model, ckpt_path, "nnRITnet")

            elif model_name == "nn_unext":
                ckpt_path = os.path.join(self.ckpt_dir, "unext_nn_best.pth")
                model = nn_unext.UNext(
                    num_classes=4, input_channels=1, deep_supervision=False
                ).to(self.device)
                self._load_state_dict(model, ckpt_path, "UNeXt")

            elif model_name == "mambaliteunet":
                ckpt_path = os.path.join(self.ckpt_dir, "mambaliteunet_nn_best.pth")
                model = mambaliteunet.MambaLiteUNet(
                    num_classes=4, input_channels=1
                ).to(self.device)
                self._load_state_dict(model, ckpt_path, "MambaLiteUNet")

            elif model_name == "rollingunet":
                ckpt_path = os.path.join(self.ckpt_dir, "rollingunet_nn_best.pth")
                model = rollingunet.Rolling_Unet_L(
                    num_classes=4, input_channels=1, deep_supervision=False
                ).to(self.device)
                self._load_state_dict(model, ckpt_path, "RollingUNet")

            elif model_name == "ulvmunet":
                ckpt_path = os.path.join(self.ckpt_dir, "ulvm_nn_best.pth")
                model = ulvmunet.UltraLight_VM_UNet(
                    num_classes=4, input_channels=1
                ).to(self.device)
                self._load_state_dict(model, ckpt_path, "UltraLight-VMUNet")

            elif model_name == "ukan":
                ckpt_path = os.path.join(self.ckpt_dir, "ukan_nn_best.pth")
                model = ukan.UKAN(
                    num_classes=4, input_channels=1, deep_supervision=False
                ).to(self.device)
                self._load_state_dict(model, ckpt_path, "U-KAN")

            elif model_name == "adgbc":
                ckpt_path = os.path.join(self.ckpt_dir, "adgbc_nn_best.pth")
                model = adgbc.GBC_Rolling_Unet_S(
                    num_classes=4, input_channels=1, deep_supervision=False
                ).to(self.device)
                self._load_state_dict(model, ckpt_path, "AD-GBC")

            elif model_name == "adgbc_400":
                ckpt_path = os.path.join(self.ckpt_dir, "adgbc_400_best.pth")
                model = adgbc.GBC_Rolling_Unet_S(
                    num_classes=4, input_channels=1, deep_supervision=False
                ).to(self.device)
                self._load_state_dict(model, ckpt_path, "AD-GBC_400")

            else:
                logger.warning(f"Unknown neural network model requested: {model_name}")

        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}", exc_info=True)
            return None

        return model

    def _load_state_dict(self, model: torch.nn.Module, ckpt_path: str, model_name_tag: str):
        if not os.path.exists(ckpt_path):
            logger.warning(f"{model_name_tag} checkpoint not found at {ckpt_path}")
            return
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = (
            checkpoint["network_weights"]
            if (isinstance(checkpoint, dict) and "network_weights" in checkpoint)
            else checkpoint
        )
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"Successfully loaded {model_name_tag} weights from {ckpt_path}")

    @property
    def enable_smoothing(self) -> bool:
        return self._enable_smoothing

    @enable_smoothing.setter
    def enable_smoothing(self, value: bool):
        self.set_smoothing(value, broadcast=True)

    def set_smoothing(self, value: bool, broadcast: bool = True):
        new_val = bool(value)
        if hasattr(self, "_enable_smoothing") and new_val == self._enable_smoothing and self._prev_ellipse is None:
            return
        self._enable_smoothing = new_val
        self._prev_ellipse = None
        self._consecutive_jumps = 0
        if hasattr(self, "_one_euro_filter"):
            self._one_euro_filter.reset()

        if hasattr(self.g_pool, "pupil_detector_smoothing"):
            self.g_pool.pupil_detector_smoothing = new_val

        if broadcast:
            self.notify_all({"subject": "pupil_detector.set_smoothing", "value": new_val})

    @property
    def smoothing_method(self) -> str:
        return self._smoothing_method

    @smoothing_method.setter
    def smoothing_method(self, value: str):
        self.set_smoothing_method(value, broadcast=True)

    def set_smoothing_method(self, method: str, broadcast: bool = True):
        method_str = str(method)
        if hasattr(self, "_smoothing_method") and method_str == self._smoothing_method and self._prev_ellipse is None:
            return
        self._smoothing_method = method_str
        self._prev_ellipse = None
        self._consecutive_jumps = 0
        if hasattr(self, "_one_euro_filter"):
            self._one_euro_filter.reset()

        if hasattr(self.g_pool, "pupil_detector_smoothing_method"):
            self.g_pool.pupil_detector_smoothing_method = method_str

        if broadcast:
            self.notify_all({"subject": "pupil_detector.set_smoothing_method", "method": method_str})

    def set_active_model(self, model_name: str, broadcast: bool = True):
        """
        Switch active model: Purge previous model from VRAM and load the selected one.
        """
        if model_name == self.active_model and (self.model is not None or model_name == "2dcpp"):
            return

        logger.info(f"Switching pupil detector model to '{model_name}'...")
        self._unload_current_model()
        self.active_model = model_name
        self._prev_ellipse = None
        self._consecutive_jumps = 0
        if hasattr(self, "_one_euro_filter"):
            self._one_euro_filter.reset()

        if hasattr(self.g_pool, "pupil_detector_model"):
            self.g_pool.pupil_detector_model = model_name

        if model_name != "2dcpp":
            self.model = self._load_model(model_name)

        if broadcast:
            self.notify_all({"subject": "pupil_detector.set_model", "model": model_name})

    def on_notify(self, notification):
        super().on_notify(notification)
        subject = notification.get("subject")
        if subject == "pupil_detector.set_model":
            model_name = notification.get("model")
            if model_name in self.model_keys:
                if model_name != self.active_model or (self.model is None and model_name != "2dcpp"):
                    logger.info(f"Received model switch notification to '{model_name}'")
                    self.set_active_model(model_name, broadcast=False)
        elif subject == "pupil_detector.set_smoothing":
            new_val = bool(notification.get("value", True))
            if new_val != self._enable_smoothing:
                logger.info(f"Received pupil detector smoothing change: {new_val}")
                self.set_smoothing(new_val, broadcast=False)
        elif subject == "pupil_detector.set_smoothing_method":
            new_method = str(notification.get("method", "ema"))
            if new_method != self._smoothing_method:
                logger.info(f"Received pupil detector smoothing method change: {new_method}")
                self.set_smoothing_method(new_method, broadcast=False)

    # -------------------------------------------------------------------------
    # Detection Loop
    # -------------------------------------------------------------------------
    def detect(self, frame, **kwargs) -> Dict:
        if self.active_model == "2dcpp":
            datum = self._detect_2dcpp(frame, **kwargs)
        else:
            datum = self._detect_nn(frame, **kwargs)

        raw_conf = float(datum.get("raw_confidence", datum.get("confidence", 0.0)))
        if self.conf_graph is not None:
            self.conf_graph.add(raw_conf)

        return datum

    def _detect_2dcpp(self, frame, **kwargs) -> Dict:
        t_detect_start = time.perf_counter()

        # ROI extraction timing (only measured for 2dcpp)
        t_roi_start = time.perf_counter()
        roi = Roi(*self.g_pool.roi.bounds)
        t_roi_end = time.perf_counter()
        roi_ms = (t_roi_end - t_roi_start) * 1000.0

        debug_img = frame.bgr if self.g_pool.display_mode == "algorithm" else None

        # 2D detection execution timing
        t_infer_start = time.perf_counter()
        result = self.__detector_2d.detect(
            gray_img=frame.gray,
            color_img=debug_img,
            roi=roi,
        )
        t_infer_end = time.perf_counter()
        infer_ms = (t_infer_end - t_infer_start) * 1000.0

        raw_conf = float(result.get("confidence", 0.0))
        if np.isnan(raw_conf) or np.isinf(raw_conf):
            raw_conf = 0.0
        raw_conf = max(0.0, min(1.0, raw_conf))

        if raw_conf < self.confidence_threshold:
            confidence = 0.0
        else:
            confidence = raw_conf

        norm_pos = normalize(
            result["location"], (frame.width, frame.height), flip_y=True
        )

        datum = self.create_pupil_datum(
            norm_pos=norm_pos,
            diameter=result["diameter"],
            confidence=confidence,
            timestamp=frame.timestamp,
        )
        datum["raw_confidence"] = raw_conf
        datum["ellipse"] = {
            "axes": result["ellipse"]["axes"],
            "angle": result["ellipse"]["angle"],
            "center": result["ellipse"]["center"],
        }

        # Capture ingestion latency from monotonic clock
        now_ts = self.g_pool.get_timestamp() if hasattr(self.g_pool, "get_timestamp") else time.time()
        capture_ts = getattr(frame, "timestamp", now_ts)
        ingest_ms = max(0.05, (now_ts - capture_ts) * 1000.0) if capture_ts > 0 else 1.0

        datum["waterfall_timing"] = {
            "frame_id": getattr(frame, "index", 0),
            "model": self.active_model,
            "process": getattr(self.g_pool, "process", getattr(self.g_pool, "name", "eye0")),
            "ingest_ms": ingest_ms,
            "roi_ms": roi_ms,
            "preprocess_ms": 0.0,
            "inference_ms": infer_ms,
            "ellipse_fit_ms": 0.0,
            "pye3d_ms": 0.0,
            "t_detect_start": t_detect_start,
            "t_detect_end": t_infer_end,
        }

        return datum

    def _detect_nn(self, frame, **kwargs) -> Dict:
        t_detect_start = time.perf_counter()

        if self.model is None:
            self.model = self._load_model(self.active_model)
            if self.model is None:
                return self._create_empty_datum(frame.timestamp, raw_confidence=0.0)

        # 1. Extract grayscale image & 2. Preprocess to normalized tensor
        t_prep_start = time.perf_counter()
        gray = self._extract_gray_image(frame)
        if gray is None:
            return self._create_empty_datum(frame.timestamp, raw_confidence=0.0)
        tensor = self.get_img(gray).unsqueeze(0).to(self.device)
        t_prep_end = time.perf_counter()

        # 3. Model inference
        t_infer_start = time.perf_counter()
        with torch.no_grad():
            output = self.model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_infer_end = time.perf_counter()

        # 4. Softmax and class prediction & 5. Contour & Ellipse fitting
        t_post_start = time.perf_counter()
        probs = torch.softmax(output, dim=1)  # (1, 4, H, W)
        pred = torch.argmax(probs, dim=1)[0].cpu().numpy()  # (H, W)

        # Class 3 represents the pupil
        pupil_pixels = (pred == 3)
        if np.any(pupil_pixels):
            pupil_probs = probs[0, 3].detach().cpu().numpy()[pupil_pixels]
            raw_conf = float(np.mean(pupil_probs))
        else:
            raw_conf = float(torch.max(probs[0, 3]).detach().cpu().item())

        if np.isnan(raw_conf) or np.isinf(raw_conf):
            raw_conf = 0.0
        raw_conf = max(0.0, min(1.0, raw_conf))

        pupil_mask = np.zeros_like(pred, dtype=np.uint8)
        pupil_mask[pupil_pixels] = 255

        if self._enable_smoothing:
            # 1. Anti-aliasing Gaussian blur & thresholding to smooth discrete pixel staircase
            pupil_mask = cv2.GaussianBlur(pupil_mask, (5, 5), 0)
            _, pupil_mask = cv2.threshold(pupil_mask, 127, 255, cv2.THRESH_BINARY)

        # Fit ellipse to pupil contour
        contours, _ = cv2.findContours(
            pupil_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            self._prev_ellipse = None
            if hasattr(self, "_one_euro_filter"):
                self._one_euro_filter.reset()
            return self._create_empty_datum(frame.timestamp, raw_confidence=raw_conf)

        best_contour = max(contours, key=cv2.contourArea)
        if len(best_contour) < 5:
            self._prev_ellipse = None
            if hasattr(self, "_one_euro_filter"):
                self._one_euro_filter.reset()
            return self._create_empty_datum(frame.timestamp, raw_confidence=raw_conf)

        ellipse = cv2.fitEllipse(best_contour)
        (cx, cy), (d1, d2), angle_deg = ellipse

        # Guarantee axes[0] is minor_diameter and axes[1] is major_diameter (axes[0] <= axes[1])
        if d1 > d2:
            minor_d = float(d2)
            major_d = float(d1)
            angle_deg = (angle_deg + 90.0) % 180.0
        else:
            minor_d = float(d1)
            major_d = float(d2)
            angle_deg = angle_deg % 180.0

        area = cv2.contourArea(best_contour)
        aspect_ratio = minor_d / (major_d + 1e-6)

        # 2. Blink & noise rejection filter (reject when eye is almost closed or area is tiny)
        if aspect_ratio < 0.20 or area < 15.0:
            self._prev_ellipse = None
            if hasattr(self, "_one_euro_filter"):
                self._one_euro_filter.reset()
            return self._create_empty_datum(frame.timestamp, raw_confidence=raw_conf)

        # 3. Temporal Outlier Gating (Jump Rejection) & Smoothing (EMA or One-Euro Filter)
        if self._enable_smoothing:
            if self._prev_ellipse is not None:
                p_c, p_ax, p_ang = self._prev_ellipse
                dist = np.sqrt((cx - p_c[0]) ** 2 + (cy - p_c[1]) ** 2)
                if dist > 40.0:
                    self._consecutive_jumps += 1
                    if self._consecutive_jumps < 5:
                        raw_conf = 0.0
                        cx, cy = p_c
                        minor_d, major_d = p_ax
                        angle_deg = p_ang
                    else:
                        self._consecutive_jumps = 0
                else:
                    self._consecutive_jumps = 0

                if self._smoothing_method == "one_euro":
                    cx, cy, minor_d, major_d, angle_deg = self._one_euro_filter.filter(
                        cx, cy, minor_d, major_d, angle_deg, frame.timestamp
                    )
                else:
                    a = self.smooth_alpha
                    cx = a * cx + (1.0 - a) * p_c[0]
                    cy = a * cy + (1.0 - a) * p_c[1]
                    minor_d = a * minor_d + (1.0 - a) * p_ax[0]
                    major_d = a * major_d + (1.0 - a) * p_ax[1]

                    # Continuous circular angle smoothing (mod 180 deg)
                    diff_ang = (angle_deg - p_ang + 90.0) % 180.0 - 90.0
                    angle_deg = (p_ang + a * diff_ang) % 180.0
            else:
                self._consecutive_jumps = 0
                if self._smoothing_method == "one_euro":
                    self._one_euro_filter.reset()
                    cx, cy, minor_d, major_d, angle_deg = self._one_euro_filter.filter(
                        cx, cy, minor_d, major_d, angle_deg, frame.timestamp
                    )

            self._prev_ellipse = ((cx, cy), (minor_d, major_d), angle_deg)
        else:
            self._prev_ellipse = None
            self._consecutive_jumps = 0
            if hasattr(self, "_one_euro_filter"):
                self._one_euro_filter.reset()

        if raw_conf < self.confidence_threshold:
            confidence = 0.0
        else:
            confidence = raw_conf

        t_post_end = time.perf_counter()

        prep_ms = (t_prep_end - t_prep_start) * 1000.0
        infer_ms = (t_infer_end - t_infer_start) * 1000.0
        post_ms = (t_post_end - t_post_start) * 1000.0

        # Estimate capture ingestion latency from monotonic clock
        now_ts = self.g_pool.get_timestamp() if hasattr(self.g_pool, "get_timestamp") else time.time()
        capture_ts = getattr(frame, "timestamp", now_ts)
        ingest_ms = max(0.05, (now_ts - capture_ts) * 1000.0) if capture_ts > 0 else 1.0

        result = {
            "location": (float(cx), float(cy)),
            "diameter": float(major_d),
            "confidence": float(confidence),
            "raw_confidence": float(raw_conf),
            "ellipse": {
                "axes": (float(minor_d), float(major_d)),
                "angle": float(angle_deg),
                "center": (float(cx), float(cy)),
            },
        }

        norm_pos = normalize(
            result["location"], (frame.width, frame.height), flip_y=True
        )

        datum = self.create_pupil_datum(
            norm_pos=norm_pos,
            diameter=result["diameter"],
            confidence=result["confidence"],
            timestamp=frame.timestamp,
        )
        datum["raw_confidence"] = result["raw_confidence"]
        datum["ellipse"] = result["ellipse"]

        # Attach sub-stage timings to datum for downstream cross-process tracing
        datum["waterfall_timing"] = {
            "frame_id": getattr(frame, "index", 0),
            "model": self.active_model,
            "process": getattr(self.g_pool, "process", getattr(self.g_pool, "name", "eye0")),
            "ingest_ms": ingest_ms,
            "roi_ms": 0.0,  # 0.0 for neural network models (no ROI cropping)
            "preprocess_ms": prep_ms,
            "inference_ms": infer_ms,
            "ellipse_fit_ms": post_ms,
            "pye3d_ms": 0.0,
            "t_detect_start": t_detect_start,
            "t_detect_end": t_post_end,
        }

        return datum

    def _extract_gray_image(self, frame) -> Optional[np.ndarray]:
        if hasattr(frame, "gray") and frame.gray is not None:
            return frame.gray.astype(np.uint8)
        elif isinstance(frame, np.ndarray):
            if len(frame.shape) == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            return frame.astype(np.uint8)
        else:
            try:
                img_array = np.frombuffer(frame.jpeg_buffer, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            except Exception as e:
                logger.error(f"Failed to extract gray image from frame: {e}")
                return None

    def get_img(self, img: np.ndarray) -> torch.Tensor:
        table = float(COLOR_MAX) * (np.linspace(0, 1, COLOR_CAP) ** 0.8)
        img_gamma = cv2.LUT(img.astype(np.uint8), table.astype(np.uint8))
        img_clahe = self.clahe.apply(img_gamma)
        if self.flip_vertically:
            img_flipud = np.flipud(img_clahe)
        else: img_flipud = img_clahe
        if self.flip_horizontally:
            img_fliplr = np.fliplr(img_flipud)
        else: img_fliplr = img_flipud
        pil_img = PIL.Image.fromarray(img_fliplr)
        return self.transform(pil_img)

    def _create_empty_datum(self, timestamp: float, raw_confidence: float = 0.0) -> Dict:
        datum = self.create_pupil_datum(
            norm_pos=(0.0, 0.0),
            diameter=0.0,
            confidence=0.0,
            timestamp=timestamp,
        )
        datum["raw_confidence"] = float(raw_confidence)
        datum["ellipse"] = {
            "axes": (0.0, 0.0),
            "angle": 0.0,
            "center": (0.0, 0.0),
        }
        return datum

    # -------------------------------------------------------------------------
    # UI & Visualization
    # -------------------------------------------------------------------------
    def init_ui(self):
        super().init_ui()
        self.menu.label = self.pretty_class_name
        self.menu_icon.label_font = "pupil_icons"

        # Model Selector dropdown
        self.menu.append(
            ui.Selector(
                "active_model",
                self,
                selection=self.model_keys,
                labels=self.model_labels,
                setter=self.set_active_model,
                getter=lambda: self.active_model,
                label="Model",
            )
        )

        # Confidence threshold slider
        self.menu.append(
            ui.Slider(
                "confidence_threshold",
                self,
                min=0.0,
                max=1.0,
                step=0.01,
                label="Confidence Threshold",
            )
        )


        self.menu.append(ui.Info_Text("Color Legend"))
        self.menu.append(
            ui.Color_Legend(color_scheme.PUPIL_ELLIPSE_2D.as_float, "2D pupil ellipse")
        )

        self.menu.append(
            ui.Switch(
                "show_confidence_graph",
                self,
                label="Show Confidence Graph",
            )
        )
        self.menu.append(
            ui.Switch(
                "enable_smoothing",
                self,
                label="Enable Smoothing",
            )
        )
        self.menu.append(
            ui.Selector(
                "smoothing_method",
                self,
                selection=self.smoothing_method_keys,
                labels=self.smoothing_method_labels,
                setter=self.set_smoothing_method,
                getter=lambda: self.smoothing_method,
                label="Smoothing Method",
            )
        )

        self.menu.append(
            ui.Switch(
                "flip_vertically",
                self,
                label="Flip Vertically",
            )
        )
        self.menu.append(
            ui.Switch(
                "flip_horizontally",
                self,
                label="Flip Horizontally",
            )
        )

        # Set up confidence performance graph (matching system_graphs.py)
        eye_id = getattr(self.g_pool, "eye_id", 0)
        self.conf_graph = graph.Bar_Graph(max_val=1.0)
        self.conf_graph.pos = (140, 50)
        self.conf_graph.update_rate = 1
        self.conf_graph.label = f"id{eye_id} conf: %0.2f"

        self.conf_grad = (
            RGBA(1.0, 0.0, 0.0, self.conf_graph.color[3]),
            self.conf_graph.color,
        )

        # Retrieve window handle from active context or g_pool
        window = glfw.get_current_context()
        if not window and hasattr(self.g_pool, "main_window"):
            window = self.g_pool.main_window

        if window is not None:
            self.on_window_resize(window)
        else:
            render_size = getattr(self.g_pool, "camera_render_size", (550, 500))
            w = max(int(render_size[0]), 100)
            h = max(int(render_size[1]), 100)
            self.conf_graph.scale = 1.0
            self.conf_graph.adjust_window_size(w, h)
            self._last_fb_size = (w, h)
            self._last_content_scale = 1.0

    def on_window_resize(self, window, *args):
        if self.conf_graph is not None and window is not None:
            try:
                fb_size = glfw.get_framebuffer_size(window)
                content_scale = gl_utils.get_content_scale(window)
                w = max(int(fb_size[0]), 100)
                h = max(int(fb_size[1]), 100)
                self.conf_graph.scale = content_scale
                self.conf_graph.adjust_window_size(w, h)
                self._last_fb_size = (w, h)
                self._last_content_scale = content_scale
            except Exception as e:
                logger.debug(f"Failed to adjust confidence graph size on resize: {e}")

    def deinit_ui(self):
        self.conf_graph = None
        self.conf_grad = None
        super().deinit_ui()

    def gl_display(self):
        if self._recent_detection_result:
            draw_pupil_outline(
                self._recent_detection_result,
                color_rgb=color_scheme.PUPIL_ELLIPSE_2D.as_float,
            )

        if self.show_confidence_graph and self.conf_graph is not None:
            window = glfw.get_current_context()
            if not window and hasattr(self.g_pool, "main_window"):
                window = self.g_pool.main_window

            if window is not None:
                try:
                    fb_size = glfw.get_framebuffer_size(window)
                    content_scale = gl_utils.get_content_scale(window)
                    if fb_size != self._last_fb_size or content_scale != self._last_content_scale:
                        w = max(int(fb_size[0]), 100)
                        h = max(int(fb_size[1]), 100)
                        self.conf_graph.scale = content_scale
                        self.conf_graph.adjust_window_size(w, h)
                        self._last_fb_size = (w, h)
                        self._last_content_scale = content_scale
                except Exception:
                    pass
            elif self._last_fb_size is None:
                render_size = getattr(self.g_pool, "camera_render_size", (550, 500))
                w = max(int(render_size[0]), 100)
                h = max(int(render_size[1]), 100)
                self.conf_graph.scale = 1.0
                self.conf_graph.adjust_window_size(w, h)
                self._last_fb_size = (w, h)
                self._last_content_scale = 1.0

            # Guard: ensure win_width and win_height are positive to prevent GL_INVALID_VALUE in glOrtho
            if getattr(self.conf_graph, "win_width", 0) > 0 and getattr(self.conf_graph, "win_height", 0) > 0:
                if self.conf_grad is not None:
                    self.conf_graph.color = mix_smooth(
                        self.conf_grad[0],
                        self.conf_grad[1],
                        self.conf_graph.avg,
                        self.conf_grad_limits[0],
                        self.conf_grad_limits[1],
                    )
                self.conf_graph.draw()

    # -------------------------------------------------------------------------
    # Persistence & Cleanup
    # -------------------------------------------------------------------------
    def get_init_dict(self) -> Dict:
        d = super().get_init_dict()
        d["active_model"] = self.active_model
        d["confidence_threshold"] = self.confidence_threshold
        d["show_confidence_graph"] = self.show_confidence_graph
        d["enable_smoothing"] = self.enable_smoothing
        d["smoothing_method"] = self.smoothing_method
        d["smooth_alpha"] = self.smooth_alpha
        d["one_euro_min_cutoff"] = self.one_euro_min_cutoff
        d["one_euro_beta"] = self.one_euro_beta
        d["properties"] = self.__detector_2d.get_properties()
        d["flip_vertically"] = self.flip_vertically
        d["flip_horizontally"] = self.flip_horizontally
        return d

    def cleanup(self):
        self._unload_current_model()
        self.conf_graph = None
        self.conf_grad = None
        super().cleanup()
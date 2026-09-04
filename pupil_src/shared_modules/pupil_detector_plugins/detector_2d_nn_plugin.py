import csv
import gc
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision

from methods import normalize
from pupil_detectors import Detector2D, DetectorBase, Roi
from pyglui import ui

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
    ("2dcpp", "Classic C++ (2D)"),
]


class nnUNetDetector2DPlugin(PupilDetectorPlugin):
    """
    Neural Network based 2D pupil detector supporting dynamic model selection
    across multiple deep learning architectures and classic C++ 2D.

    VRAM is managed strictly such that only the currently active model is held in GPU
    memory, freeing resources completely when switching to the C++ detector.
    """

    uniqueness = "by_class"
    icon_font = "pupil_icons"
    icon_chr = chr(0xEC18)

    label = "Neural Network 2D Detector"
    identifier = "2d"
    order = 0.9

    @property
    def pretty_class_name(self) -> str:
        return "Pupil Detector 2D (Neural Net)"

    @property
    def pupil_detector(self) -> DetectorBase:
        return self.__detector_2d

    def __init__(
        self,
        g_pool=None,
        active_model: str = "pmrnet",
        confidence_threshold: float = 0.6,
        **properties,
    ):
        super().__init__(g_pool=g_pool)
        self.__detector_2d = Detector2D({})
        self._stop_other_pupil_detectors()

        self.plugin_dir = os.path.dirname(__file__)
        self.ckpt_dir = os.path.join(self.plugin_dir, "model_ckpts")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.latency_log = []
        self.active_model = active_model
        self.confidence_threshold = float(confidence_threshold)
        self.model = None

        self.model_keys = [k for k, _ in AVAILABLE_MODELS]
        self.model_labels = [label for _, label in AVAILABLE_MODELS]

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.clahe = cv2.createCLAHE(
            clipLimit=CLIP_LIMIT, tileGridSize=(TILE_GRID_SIZE, TILE_GRID_SIZE)
        )

        # Initial single-model VRAM loading
        if self.active_model != "2dcpp":
            self.model = self._load_model(self.active_model)

    def _stop_other_pupil_detectors(self):
        plugin_list = getattr(self.g_pool, "plugins", None)
        if plugin_list is None:
            return

        for plugin in plugin_list:
            if isinstance(plugin, PupilDetectorPlugin) and plugin is not self:
                plugin.alive = False

        plugin_list.clean()

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
                model = adgbc.GBC_Rolling_Unet_L(
                    num_classes=4, input_channels=1, deep_supervision=False
                ).to(self.device)
                self._load_state_dict(model, ckpt_path, "AD-GBC")

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

    def set_active_model(self, model_name: str):
        """
        Switch active model: Purge previous model from VRAM and load the selected one.
        """
        if model_name == self.active_model and (self.model is not None or model_name == "2dcpp"):
            return

        logger.info(f"Switching pupil detector model to '{model_name}'...")
        self._unload_current_model()
        self.active_model = model_name

        if model_name != "2dcpp":
            self.model = self._load_model(model_name)

    # -------------------------------------------------------------------------
    # Detection Loop
    # -------------------------------------------------------------------------
    def detect(self, frame, **kwargs) -> Dict:
        if self.active_model == "2dcpp":
            return self._detect_2dcpp(frame, **kwargs)
        else:
            return self._detect_nn(frame, **kwargs)

    def _detect_2dcpp(self, frame, **kwargs) -> Dict:
        roi = Roi(*self.g_pool.roi.bounds)
        debug_img = frame.bgr if self.g_pool.display_mode == "algorithm" else None

        result = self.__detector_2d.detect(
            gray_img=frame.gray,
            color_img=debug_img,
            roi=roi,
        )

        confidence = float(result.get("confidence", 0.0))
        if confidence < self.confidence_threshold:
            confidence = 0.0

        norm_pos = normalize(
            result["location"], (frame.width, frame.height), flip_y=True
        )

        datum = self.create_pupil_datum(
            norm_pos=norm_pos,
            diameter=result["diameter"],
            confidence=confidence,
            timestamp=frame.timestamp,
        )
        datum["ellipse"] = {
            "axes": result["ellipse"]["axes"],
            "angle": result["ellipse"]["angle"],
            "center": result["ellipse"]["center"],
        }
        return datum

    def _detect_nn(self, frame, **kwargs) -> Dict:
        if self.model is None:
            self.model = self._load_model(self.active_model)
            if self.model is None:
                return self._create_empty_datum(frame.timestamp)

        # 1. Extract grayscale image
        gray = self._extract_gray_image(frame)
        if gray is None:
            return self._create_empty_datum(frame.timestamp)

        # 2. Preprocess to normalized tensor
        tensor = self.get_img(gray).unsqueeze(0).to(self.device)

        # 3. Model inference
        with torch.no_grad():
            output = self.model(tensor)

        # 4. Softmax and class prediction
        probs = torch.softmax(output, dim=1)  # (1, 4, H, W)
        pred = torch.argmax(probs, dim=1)[0].cpu().numpy()  # (H, W)

        # Class 3 represents the pupil
        pupil_mask = np.zeros_like(pred, dtype=np.uint8)
        pupil_mask[pred == 3] = 255

        # 5. Fit ellipse to pupil contour
        contours, _ = cv2.findContours(
            pupil_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return self._create_empty_datum(frame.timestamp)

        best_contour = max(contours, key=cv2.contourArea)
        if len(best_contour) < 5:
            return self._create_empty_datum(frame.timestamp)

        (cx, cy), (MA, ma), angle_deg = cv2.fitEllipse(best_contour)

        # 6. Mean Softmax Probability over pupil mask
        pupil_pixels = (pred == 3)
        if np.any(pupil_pixels):
            pupil_probs = probs[0, 3].cpu().numpy()[pupil_pixels]
            mean_conf = float(np.mean(pupil_probs))
        else:
            mean_conf = 0.0

        if mean_conf < self.confidence_threshold:
            confidence = 0.0
        else:
            confidence = mean_conf

        result = {
            "location": (float(cx), float(cy)),
            "diameter": float(MA),
            "confidence": float(confidence),
            "ellipse": {
                "axes": (float(MA), float(ma)),
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
        datum["ellipse"] = result["ellipse"]
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
        pil_img = PIL.Image.fromarray(img_clahe)
        return self.transform(pil_img)

    def _create_empty_datum(self, timestamp: float) -> Dict:
        datum = self.create_pupil_datum(
            norm_pos=(0.0, 0.0),
            diameter=0.0,
            confidence=0.0,
            timestamp=timestamp,
        )
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

    def gl_display(self):
        if self._recent_detection_result:
            draw_pupil_outline(
                self._recent_detection_result,
                color_rgb=color_scheme.PUPIL_ELLIPSE_2D.as_float,
            )

    # -------------------------------------------------------------------------
    # Persistence & Cleanup
    # -------------------------------------------------------------------------
    def get_init_dict(self) -> Dict:
        d = super().get_init_dict()
        d["active_model"] = self.active_model
        d["confidence_threshold"] = self.confidence_threshold
        return d

    def cleanup(self):
        self._unload_current_model()
        super().cleanup()
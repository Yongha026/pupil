"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import csv
from datetime import datetime
import logging
import os
import traceback
import typing as T

import numpy as np
import scipy.spatial
from calibration_choreography import (
    ChoreographyAction,
    ChoreographyMode,
    ChoreographyNotification,
)
from gaze_mapping import gazer_classes_by_class_name, registered_gazer_classes
from gaze_mapping.notifications import (
    CalibrationResultNotification,
    CalibrationSetupNotification,
)
from gaze_mapping.utils import closest_matches_monocular
from plugin import Plugin


logger = logging.getLogger(__name__)


class CalculationResult(T.NamedTuple):
    result: float
    num_used: int
    num_total: int


class CorrelatedAndCoordinateTransformedResult(T.NamedTuple):
    """Holds result from correlating reference and gaze data and their respective
    transformations into norm, image, and camera coordinate systems.
    """

    norm_space: np.ndarray  # shape: 2*n, 2
    image_space: np.ndarray  # shape: 2*n, 2
    camera_space: np.ndarray  # shape: 2*n, 3

    @staticmethod
    def empty() -> "CorrelatedAndCoordinateTransformedResult":
        return CorrelatedAndCoordinateTransformedResult(
            norm_space=np.ndarray([]),
            image_space=np.ndarray([]),
            camera_space=np.ndarray([]),
        )

    @property
    def is_valid(self) -> bool:
        if len(self.norm_space.shape) != 2:
            return False
        # TODO: Make validity check exhaustive
        return True


class CorrelationError(ValueError):
    pass


class AccuracyPrecisionResult(T.NamedTuple):
    accuracy: CalculationResult
    precision: CalculationResult
    error_lines: np.ndarray
    correlation: CorrelatedAndCoordinateTransformedResult

    @staticmethod
    def failed() -> "AccuracyPrecisionResult":
        return AccuracyPrecisionResult(
            accuracy=CalculationResult(0.0, 0, 0),
            precision=CalculationResult(0.0, 0, 0),
            error_lines=np.array([]),
            correlation=CorrelatedAndCoordinateTransformedResult.empty(),
        )

    @property
    def is_valid(self) -> bool:
        if not self.correlation.is_valid:
            return False
        # TODO: Make validity check exhaustive
        return True


class ValidationInput:
    def __init__(self):
        self.clear()

    @property
    def gazer_class(self) -> T.Optional[T.Any]:
        return self.__gazer_class

    @property
    def gazer_params(self) -> T.Optional[T.Any]:
        return self.__gazer_params

    @property
    def gazer_class_name(self) -> T.Optional[str]:
        return self.__gazer_class.__name__ if self.__gazer_class is not None else None

    @property
    def pupil_list(self) -> T.Optional[T.Any]:
        return self.__pupil_list

    @property
    def ref_list(self) -> T.Optional[T.Any]:
        return self.__ref_list

    @property
    def is_complete(self) -> bool:
        return None not in (
            self.pupil_list,
            self.ref_list,
            self.gazer_class,
            self.gazer_params,
        )

    def clear(self):
        self.__pupil_list = None
        self.__ref_list = None
        self.__gazer_class = None
        self.__gazer_params = None
        self.gaze_list = None
        self.is_calibration = True

    def update(
        self,
        gazer_class_name: str,
        gazer_params=...,
        pupil_list=...,
        ref_list=...,
        gaze_list=None,
        is_calibration=...,
    ):
        if (
            self.gazer_class_name is not None
            and self.gazer_class_name != gazer_class_name
        ):
            logger.debug(
                f'Overwriting gazer_class_name from "{self.gazer_class_name}" to '
                f'"{gazer_class_name}" and resetting the input.'
            )
            self.clear()

        self.__gazer_class = self.__gazer_class_from_name(gazer_class_name)

        if gazer_params is not ...:
            self.__gazer_params = gazer_params

        if pupil_list is not ...:
            self.__pupil_list = pupil_list

        if ref_list is not ...:
            self.__ref_list = ref_list

        if gaze_list is not None:
            self.gaze_list = gaze_list

        if is_calibration is not ...:
            self.is_calibration = is_calibration

    @staticmethod
    def __gazer_class_from_name(gazer_class_name: str) -> T.Optional[T.Any]:
        gazers_by_name = gazer_classes_by_class_name(registered_gazer_classes())

        try:
            gazer_cls = gazers_by_name[gazer_class_name]
        except KeyError:
            logger.error(f'Unknown gazer "{gazer_class_name}"')
            return None

        return gazer_cls


class Accuracy_Visualizer(Plugin):
    """Calibrate using a marker on your screen
    We use a ring detector that moves across the screen to 9 sites
    Points are collected at sites not between
    """

    order = 0.8
    icon_chr = chr(0xEC11)
    icon_font = "pupil_icons"

    def __init__(
        self,
        g_pool,
        outlier_threshold=2.5,
        vis_mapping_error=True,
        vis_calibration_area=True,
    ):
        super().__init__(g_pool)
        self.vis_mapping_error = vis_mapping_error
        self.vis_calibration_area = vis_calibration_area
        self.calibration_area = None
        self.accuracy = None
        self.precision = None
        self.error_lines = None

        self.recent_input = ValidationInput()

        # .5 degrees, used to remove outliers from precision calculation
        self.succession_threshold = np.cos(np.deg2rad(0.5))
        self._outlier_threshold = outlier_threshold  # in degrees

        # Logging dir
        root_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        self.val_dir = os.environ.get(
            "PUPIL_VALIDATION_DIR", os.path.join(root_dir, "val_results")
        )
        os.makedirs(self.val_dir, exist_ok=True)

        date_str = datetime.now().strftime("%y_%m_%d-%H-%M")
        filename = f"val_results_{date_str}.csv"
        self.csv_path = os.path.join(self.val_dir, filename)

        self.export_raw_data =True

    def init_ui(self):
        from pyglui import ui

        self.add_menu()
        self.menu.label = "Accuracy Visualizer"

        mapping_error_help = """The mapping error (orange line) is the angular
                             distance between mapped pupil positions (red) and
                             their corresponding reference points (blue).
                             """.replace(
            "\n", " "
        ).replace(
            "  ", ""
        )

        calib_area_help = """The calibration area (green) is defined as the
                          convex hull of the reference points that were used
                          for calibration. 2D mapping looses accuracy outside
                          of this area. It is recommended to calibrate a big
                          portion of the subject's field of view.
                          """.replace(
            "\n", " "
        ).replace(
            "  ", ""
        )
        self.menu.append(ui.Info_Text(calib_area_help))
        self.menu.append(
            ui.Switch("vis_mapping_error", self, label="Visualize mapping error")
        )

        self.menu.append(ui.Info_Text(mapping_error_help))
        self.menu.append(
            ui.Switch("vis_calibration_area", self, label="Visualize calibration area")
        )

        general_help = """Measure gaze mapping accuracy and precision using samples
                          that were collected during calibration. The outlier threshold
                          discards samples with high angular errors.""".replace(
            "\n", " "
        ).replace(
            "  ", ""
        )
        self.menu.append(ui.Info_Text(general_help))

        # self.menu.append(ui.Info_Text(''))
        self.menu.append(
            ui.Text_Input(
                "outlier_threshold", self, label="Outlier Threshold [degrees]"
            )
        )

        accuracy_help = """Accuracy is calculated as the average angular
                        offset (distance) (in degrees of visual angle)
                        between fixation locations and the corresponding
                        locations of the fixation targets.""".replace(
            "\n", " "
        ).replace(
            "  ", ""
        )

        precision_help = """Precision is calculated as the Root Mean Square (RMS)
                            of the angular distance (in degrees of visual angle)
                            between successive samples during a fixation.""".replace(
            "\n", " "
        ).replace(
            "  ", ""
        )

        def ignore(_):
            pass

        self.menu.append(ui.Info_Text(accuracy_help))
        self.menu.append(
            ui.Text_Input(
                "accuracy",
                self,
                "Angular Accuracy",
                setter=ignore,
                getter=lambda: f"{self.accuracy.result:.3f} deg. Samples used: "
                f"{self.accuracy.num_used} / {self.accuracy.num_total}"
                if self.accuracy is not None
                else "Not available",
            )
        )
        self.menu.append(ui.Info_Text(precision_help))
        self.menu.append(
            ui.Text_Input(
                "precision",
                self,
                "Angular Precision",
                setter=ignore,
                getter=lambda: f"{self.precision.result:.3f} deg. Samples used: "
                f"{self.precision.num_used} / {self.precision.num_total}"
                if self.precision is not None
                else "Not available",
            )
        )
        self.menu.append(
            ui.Button("Export Validation Results to CSV", self._export_validation_to_csv)
        )

        self.menu.append(ui.Info_Text(
            "Raw Data Export: saves pupil_positions.csv, gaze_positions.csv, "
            "evaluation.csv and export_info.csv to val_results/raw_data/ on every "
            "calibration and validation run."
        ))
        self.menu.append(
            ui.Switch("export_raw_data", self, label="Export Raw Data on Calib/Test")
        )
        self.menu.append(
            ui.Button("Export Raw Data Now", self._export_raw_data_now)
        )

    def deinit_ui(self):
        self.remove_menu()

    @property
    def outlier_threshold(self):
        return self._outlier_threshold

    @outlier_threshold.setter
    def outlier_threshold(self, value):
        self._outlier_threshold = value
        self.notify_all(
            {"subject": "accuracy_visualizer.outlier_threshold_changed", "delay": 0.5}
        )

    def on_notify(self, notification):
        if self.__handle_calibration_setup_notification(notification):
            return

        if self.__handle_calibration_result_notification(notification):
            return

        if self.__handle_validation_data_notification(notification):
            return

        if notification["subject"] == "accuracy_visualizer.outlier_threshold_changed":
            if self.recent_input.is_complete:
                self.recalculate()

    def __handle_calibration_setup_notification(self, note_dict: dict) -> bool:
        try:
            note = CalibrationSetupNotification.from_dict(note_dict)
        except ValueError:
            return False

        self.recent_input.update(
            gazer_class_name=note.gazer_class_name,
            pupil_list=note.calib_data["pupil_list"],
            ref_list=note.calib_data["ref_list"],
        )
        return True

    def __handle_calibration_result_notification(self, note_dict: dict) -> bool:
        try:
            note = CalibrationResultNotification.from_dict(note_dict)
        except ValueError:
            return False

        self.recent_input.update(
            gazer_class_name=note.gazer_class_name,
            gazer_params=note.params,
            is_calibration=True,
        )

        self.recalculate()
        if self.export_raw_data:
            self._export_raw_session_data("calibration")
        return True

    def __handle_validation_data_notification(self, note_dict: dict) -> bool:
        try:
            note = ChoreographyNotification.from_dict(note_dict)
            assert note.mode == ChoreographyMode.VALIDATION
            assert note.action == ChoreographyAction.DATA
        except (AssertionError, ValueError):
            return False

        self.recent_input.clear()
        self.recent_input.update(
            gazer_class_name=note_dict["gazer_class_name"],
            gazer_params=note_dict["gazer_params"],
            pupil_list=note_dict["pupil_list"],
            ref_list=note_dict["ref_list"],
            gaze_list=note_dict.get("gaze_list"),
            is_calibration=False,
        )

        self.recalculate()
        self._export_validation_to_csv()
        if self.export_raw_data:
            self._export_raw_session_data("validation")
        return True

    def recalculate(self):
        NOT_ENOUGH_DATA_COLLECTED_ERR_MSG = (
            "Did not collect enough data to estimate gaze mapping accuracy."
        )

        if not self.recent_input.is_complete:
            logger.warning(NOT_ENOUGH_DATA_COLLECTED_ERR_MSG)
            return

        results = self.calc_acc_prec_errlines(
            gazer_class=self.recent_input.gazer_class,
            g_pool=self.g_pool,
            gazer_params=self.recent_input.gazer_params,
            pupil_list=self.recent_input.pupil_list,
            ref_list=self.recent_input.ref_list,
            intrinsics=self.g_pool.capture.intrinsics,
            outlier_threshold=self.outlier_threshold,
            succession_threshold=self.succession_threshold,
            gaze_list=getattr(self.recent_input, "gaze_list", None),
        )

        if not results.is_valid:
            logger.warning(NOT_ENOUGH_DATA_COLLECTED_ERR_MSG)
            return

        if np.isnan(results.accuracy.result):
            self.accuracy = None
            logger.warning(
                "Not enough data available for angular accuracy calculation."
            )
        else:
            self.accuracy = results.accuracy
            logger.info(f"Angular accuracy: {results.accuracy.result:.3f} degrees")

        if np.isnan(results.precision.result):
            self.precision = None
            logger.warning(
                "Not enough data available for angular precision calculation."
            )
        else:
            self.precision = results.precision
            logger.info(f"Angular precision: {results.precision.result:.3f} degrees")

        self.error_lines = results.error_lines
        if getattr(self.recent_input, "is_calibration", True):
            ref_locations = results.correlation.norm_space[1::2, :]
            if len(ref_locations) >= 3:
                try:
                    # requires at least 3 points
                    hull = scipy.spatial.ConvexHull(ref_locations)
                    self.calibration_area = hull.points[hull.vertices, :]
                except scipy.spatial.qhull.QhullError:
                    logger.warning("Calibration area could not be calculated")
                    logger.debug(traceback.format_exc())

    def _export_validation_to_csv(self):
        """Appends validation results to 'validation_results_YY_MM_DD_HH-MM-SS.csv' in val_results directory."""
        if not self.recent_input.is_complete or (self.accuracy is None and self.precision is None):
            logger.warning("No completed validation results available to export to CSV.")
            return

        try:
            file_exists = os.path.exists(self.csv_path)

            fieldnames = [
                "date",
                "time",
                "model",
                "gazer_class",
                "accuracy_deg",
                "accuracy_used_samples",
                "accuracy_total_samples",
                "precision_deg",
                "precision_used_samples",
                "precision_total_samples",
                "outlier_threshold_deg",
                "pupil_positions_count",
                "ref_points_count",
                "status",
            ]

            now = datetime.now()
            model_name = getattr(self.g_pool, "pupil_detector_model", "unknown")
            gazer_name = getattr(self.recent_input, "gazer_class_name", "unknown")

            acc_val = (
                round(float(self.accuracy.result), 4)
                if self.accuracy and not np.isnan(self.accuracy.result)
                else ""
            )
            acc_used = self.accuracy.num_used if self.accuracy else 0
            acc_total = self.accuracy.num_total if self.accuracy else 0

            prec_val = (
                round(float(self.precision.result), 4)
                if self.precision and not np.isnan(self.precision.result)
                else ""
            )
            prec_used = self.precision.num_used if self.precision else 0
            prec_total = self.precision.num_total if self.precision else 0

            status = (
                "success"
                if (self.accuracy is not None and self.precision is not None)
                else "failed"
            )

            row = {
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "model": model_name,
                "gazer_class": gazer_name,
                "accuracy_deg": acc_val,
                "accuracy_used_samples": acc_used,
                "accuracy_total_samples": acc_total,
                "precision_deg": prec_val,
                "precision_used_samples": prec_used,
                "precision_total_samples": prec_total,
                "outlier_threshold_deg": self.outlier_threshold,
                "pupil_positions_count": len(
                    getattr(self.recent_input, "pupil_list", []) or []
                ),
                "ref_points_count": len(
                    getattr(self.recent_input, "ref_list", []) or []
                ),
                "status": status,
            }

            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

            logger.info(
                f"Validation result appended to {self.csv_path}: "
                f"Accuracy={acc_val} deg, Precision={prec_val} deg (Model: {model_name})"
            )
        except Exception as e:
            logger.error(f"Failed to export validation result to CSV: {e}")

    def _export_raw_data_now(self):
        """Called from the UI 'Export Raw Data Now' button."""
        if not self.recent_input.is_complete:
            logger.warning("No completed calibration/validation data available to export.")
            return
        session_type = "calibration" if getattr(self.recent_input, "is_calibration", True) else "validation"
        self._export_raw_session_data(session_type)

    def _export_raw_session_data(self, session_type: str):
        """
        Export raw pupil & gaze data for one calibration or validation session.

        Creates val_results/raw_data/<session_type>_<model>_<YYYY-MM-DD_HH-MM-SS>/
        containing:
            pupil_positions.csv  - Pupil_Positions_Exporter schema + raw diagnostic cols
            gaze_positions.csv   - Gaze_Positions_Exporter schema
            evaluation.csv       - per-sample target vs gaze angular error
            export_info.csv      - metadata
        """
        import csv as _csv
        import traceback as _tb
        from datetime import datetime as _dt
        if not self.recent_input.is_complete:
            logger.warning("_export_raw_session_data: no complete data available.")
            return

        try:
            model_name = getattr(self.g_pool, "pupil_detector_model", "unknown")
            gazer_name = getattr(self.recent_input, "gazer_class_name", "unknown")
            timestamp_str = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
            folder_name = f"{session_type}_{model_name}_{timestamp_str}"
            out_dir = os.path.join(self.val_dir, "raw_data", folder_name)
            os.makedirs(out_dir, exist_ok=True)

            pupil_list = self.recent_input.pupil_list or []
            ref_list = self.recent_input.ref_list or []
            gazer_class = self.recent_input.gazer_class
            gazer_params = self.recent_input.gazer_params

            # Build gaze_pos from gazer or from cached gaze_list
            gaze_list_cached = getattr(self.recent_input, "gaze_list", None)
            if gaze_list_cached:
                gaze_pos = list(gaze_list_cached)
            elif gazer_class is not None and gazer_params is not None:
                try:
                    gazer = gazer_class(self.g_pool, params=gazer_params, register_as_active=False)
                    gaze_pos = list(gazer.map_pupil_to_gaze(pupil_list))
                except Exception as exc:
                    logger.warning(f"Could not map gaze for raw export: {exc}")
                    gaze_pos = []
            else:
                gaze_pos = []

            # ── 1. pupil_positions.csv ─────────────────────────────────
            pupil_fields = [
                "pupil_timestamp", "world_index", "eye_id", "confidence",
                "norm_pos_x", "norm_pos_y", "diameter", "method",
                "ellipse_center_x", "ellipse_center_y", "ellipse_axis_a",
                "ellipse_axis_b", "ellipse_angle",
                # "diameter_3d", "model_confidence", "model_id",
                # "sphere_center_x", "sphere_center_y", "sphere_center_z", "sphere_radius",
                # "circle_3d_center_x", "circle_3d_center_y", "circle_3d_center_z",
                # "circle_3d_normal_x", "circle_3d_normal_y", "circle_3d_normal_z",
                # "circle_3d_radius", "theta", "phi",
                # "projected_sphere_center_x", "projected_sphere_center_y",
                # "projected_sphere_axis_a", "projected_sphere_axis_b",
                # "projected_sphere_angle",
                "raw_center_x", "raw_center_y", "raw_axis_a", "raw_axis_b",
                "raw_angle", "raw_confidence", "pixel_jitter",
            ]
            pupil_path = os.path.join(out_dir, "pupil_positions.csv")
            with open(pupil_path, "w", newline="", encoding="utf-8") as f:
                writer = _csv.DictWriter(f, fieldnames=pupil_fields, extrasaction="ignore")
                writer.writeheader()
                for p in pupil_list:
                    ellipse = p.get("ellipse") or {}
                    e_center = ellipse.get("center", (None, None))
                    e_axes = ellipse.get("axes", (None, None))
                    raw_el = p.get("raw_ellipse") or {}
                    raw_center = raw_el.get("center", (None, None))
                    raw_axes = raw_el.get("axes", (None, None))
                    norm = p.get("norm_pos", (None, None))
                    # sphere = p.get("sphere") or {}
                    # c3d = p.get("circle_3d") or {}
                    # psphere = p.get("projected_sphere") or {}
                    row = {
                        "pupil_timestamp": str(p.get("timestamp")),
                        "world_index": "",
                        "eye_id": p.get("id"),
                        "confidence": p.get("confidence"),
                        "norm_pos_x": norm[0] if norm else None,
                        "norm_pos_y": norm[1] if norm else None,
                        "diameter": p.get("diameter"),
                        "method": p.get("method"),
                        "ellipse_center_x": e_center[0],
                        "ellipse_center_y": e_center[1],
                        "ellipse_axis_a": e_axes[0],
                        "ellipse_axis_b": e_axes[1],
                        "ellipse_angle": ellipse.get("angle"),
                        # "diameter_3d": p.get("diameter_3d"),
                        # "model_confidence": p.get("model_confidence"),
                        # "model_id": p.get("model_id"),
                        # "sphere_center_x": sphere.get("center", [None, None, None])[0],
                        # "sphere_center_y": sphere.get("center", [None, None, None])[1],
                        # "sphere_center_z": sphere.get("center", [None, None, None])[2],
                        # "sphere_radius": sphere.get("radius"),
                        # "circle_3d_center_x": c3d.get("center", [None, None, None])[0],
                        # "circle_3d_center_y": c3d.get("center", [None, None, None])[1],
                        # "circle_3d_center_z": c3d.get("center", [None, None, None])[2],
                        # "circle_3d_normal_x": c3d.get("normal", [None, None, None])[0],
                        # "circle_3d_normal_y": c3d.get("normal", [None, None, None])[1],
                        # "circle_3d_normal_z": c3d.get("normal", [None, None, None])[2],
                        # "circle_3d_radius": c3d.get("radius"),
                        # "theta": p.get("theta"),
                        # "phi": p.get("phi"),
                        # "projected_sphere_center_x": psphere.get("center", [None, None])[0],
                        # "projected_sphere_center_y": psphere.get("center", [None, None])[1],
                        # "projected_sphere_axis_a": psphere.get("axes", [None, None])[0],
                        # "projected_sphere_axis_b": psphere.get("axes", [None, None])[1],
                        # "projected_sphere_angle": psphere.get("angle"),
                        "raw_center_x": raw_center[0],
                        "raw_center_y": raw_center[1],
                        "raw_axis_a": raw_axes[0],
                        "raw_axis_b": raw_axes[1],
                        "raw_angle": raw_el.get("angle"),
                        "raw_confidence": p.get("raw_confidence"),
                        "pixel_jitter": p.get("pixel_jitter"),
                    }
                    writer.writerow(row)
            logger.info(f"Raw export: wrote {len(pupil_list)} pupil rows -> {pupil_path}")

            # ── 2. gaze_positions.csv ──────────────────────────────────
            gaze_fields = [
                "gaze_timestamp", "world_index", "confidence",
                "norm_pos_x", "norm_pos_y", "base_data",
                # "gaze_point_3d_x", "gaze_point_3d_y", "gaze_point_3d_z",
                # "eye_center0_3d_x", "eye_center0_3d_y", "eye_center0_3d_z",
                # "gaze_normal0_x", "gaze_normal0_y", "gaze_normal0_z",
                # "eye_center1_3d_x", "eye_center1_3d_y", "eye_center1_3d_z",
                # "gaze_normal1_x", "gaze_normal1_y", "gaze_normal1_z",
            ]
            gaze_path = os.path.join(out_dir, "gaze_positions.csv")
            with open(gaze_path, "w", newline="", encoding="utf-8") as f:
                writer = _csv.DictWriter(f, fieldnames=gaze_fields, extrasaction="ignore")
                writer.writeheader()
                for g in gaze_pos:
                    norm = g.get("norm_pos", (None, None))
                    # gp3d = g.get("gaze_point_3d") or [None, None, None]
                    base_data = g.get("base_data")
                    if base_data:
                        base_data = " ".join(
                            f"{b['timestamp']}-{b['id']}" for b in base_data
                        )
                    # ec3d = g.get("eye_centers_3d") or {}
                    # gn3d = g.get("gaze_normals_3d") or {}
                    # ec0 = ec3d.get("0", ec3d.get(0, [None, None, None]))
                    # ec1 = ec3d.get("1", ec3d.get(1, [None, None, None]))
                    # gn0 = gn3d.get("0", gn3d.get(0, [None, None, None]))
                    # gn1 = gn3d.get("1", gn3d.get(1, [None, None, None]))
                    # if not ec3d and g.get("eye_center_3d"):
                    #     try:
                    #         eye_id = str(g["base_data"][0]["id"])
                    #     except (KeyError, IndexError, TypeError):
                    #         eye_id = "0"
                    #     if eye_id == "0":
                    #         ec0 = g["eye_center_3d"]
                    #         gn0 = g.get("gaze_normal_3d", [None, None, None])
                    #     else:
                    #         ec1 = g["eye_center_3d"]
                    #         gn1 = g.get("gaze_normal_3d", [None, None, None])
                    row = {
                        "gaze_timestamp": str(g.get("timestamp")),
                        "world_index": "",
                        "confidence": g.get("confidence"),
                        "norm_pos_x": norm[0] if norm else None,
                        "norm_pos_y": norm[1] if norm else None,
                        "base_data": base_data,
                        # "gaze_point_3d_x": gp3d[0],
                        # "gaze_point_3d_y": gp3d[1],
                        # "gaze_point_3d_z": gp3d[2],
                        # "eye_center0_3d_x": ec0[0], "eye_center0_3d_y": ec0[1],
                        # "eye_center0_3d_z": ec0[2],
                        # "gaze_normal0_x": gn0[0], "gaze_normal0_y": gn0[1],
                        # "gaze_normal0_z": gn0[2],
                        # "eye_center1_3d_x": ec1[0], "eye_center1_3d_y": ec1[1],
                        # "eye_center1_3d_z": ec1[2],
                        # "gaze_normal1_x": gn1[0], "gaze_normal1_y": gn1[1],
                        # "gaze_normal1_z": gn1[2],
                    }
                    writer.writerow(row)
            logger.info(f"Raw export: wrote {len(gaze_pos)} gaze rows -> {gaze_path}")

            # ── 3. evaluation.csv ──────────────────────────────────────
            eval_path = os.path.join(out_dir, "evaluation.csv")
            if ref_list and gaze_pos:
                correlated = closest_matches_monocular(gaze_pos, ref_list)
                try:
                    intrinsics = self.g_pool.capture.intrinsics
                    cr = Accuracy_Visualizer._coordinate_transform_ref_in_norm_space(
                        correlated, intrinsics
                    )
                    cam_space = cr.camera_space.reshape(-1, 6)
                    dot_products = np.einsum(
                        "ij,ij->i", cam_space[:, :3], cam_space[:, 3:]
                    ).clip(-1.0, 1.0)
                    angular_errors = np.rad2deg(np.arccos(dot_products))
                except Exception as exc:
                    logger.warning(
                        f"Could not compute angular errors for evaluation.csv: {exc}"
                    )
                    correlated = []
                    angular_errors = np.array([])

                eval_fields = [
                    "sample_idx", "session_type", "model", "gazer_class",
                    "ref_norm_x", "ref_norm_y", "gaze_norm_x", "gaze_norm_y",
                    "angular_error_deg", "is_outlier",
                    "pupil_timestamp", "pupil_confidence",
                    "ellipse_cx", "ellipse_cy",
                    "raw_cx", "raw_cy", "pixel_jitter",
                ]
                with open(eval_path, "w", newline="", encoding="utf-8") as f:
                    writer = _csv.DictWriter(
                        f, fieldnames=eval_fields, extrasaction="ignore"
                    )
                    writer.writeheader()
                    outlier_thresh = float(getattr(self, "_outlier_threshold", 2.5))
                    for idx, match in enumerate(correlated):
                        gaze = match["ref"]
                        ref = match["pupil"]
                        ang_err = (
                            float(angular_errors[idx])
                            if idx < len(angular_errors)
                            else None
                        )
                        base_data = gaze.get("base_data") or []
                        pupil_datum = base_data[0] if base_data else gaze
                        pupil_ts = pupil_datum.get("timestamp")
                        pupil_conf = pupil_datum.get("confidence")
                        ell = pupil_datum.get("ellipse") or {}
                        e_ctr = ell.get("center", (None, None))
                        raw_ell = pupil_datum.get("raw_ellipse") or {}
                        r_ctr = raw_ell.get("center", (None, None))
                        pjitter = pupil_datum.get("pixel_jitter")
                        ref_norm = ref.get("norm_pos", (None, None))
                        gaze_norm = gaze.get("norm_pos", (None, None))
                        row = {
                            "sample_idx": idx,
                            "session_type": session_type,
                            "model": model_name,
                            "gazer_class": gazer_name,
                            "ref_norm_x": ref_norm[0],
                            "ref_norm_y": ref_norm[1],
                            "gaze_norm_x": gaze_norm[0],
                            "gaze_norm_y": gaze_norm[1],
                            "angular_error_deg": (
                                round(ang_err, 5) if ang_err is not None else None
                            ),
                            "is_outlier": (
                                    ang_err is not None and ang_err > outlier_thresh
                            ),
                            "pupil_timestamp": pupil_ts,
                            "pupil_confidence": pupil_conf,
                            "ellipse_cx": e_ctr[0],
                            "ellipse_cy": e_ctr[1],
                            "raw_cx": r_ctr[0],
                            "raw_cy": r_ctr[1],
                            "pixel_jitter": pjitter,
                        }
                        writer.writerow(row)
                logger.info(
                    f"Raw export: wrote {len(correlated)} eval rows -> {eval_path}"
                )
            else:
                logger.info("Raw export: skipping evaluation.csv (no ref or gaze data)")

            # ── 4. export_info.csv ────────────────────────────────────
            info_path = os.path.join(out_dir, "export_info.csv")
            import math as _math
            acc_val = (
                round(float(self.accuracy.result), 5)
                if self.accuracy and not _math.isnan(self.accuracy.result)
                else None
            )
            prec_val = (
                round(float(self.precision.result), 5)
                if self.precision and not _math.isnan(self.precision.result)
                else None
            )
            with open(info_path, "w", newline="", encoding="utf-8") as f:
                writer = _csv.writer(f)
                writer.writerow(["key", "value"])
                writer.writerow(["session_type", session_type])
                writer.writerow(["model", model_name])
                writer.writerow(["gazer_class", gazer_name])
                writer.writerow(["timestamp", timestamp_str])
                writer.writerow(["accuracy_deg", acc_val])
                writer.writerow(["precision_deg", prec_val])
                writer.writerow(["outlier_threshold_deg", self._outlier_threshold])
                writer.writerow(["pupil_count", len(pupil_list)])
                writer.writerow(["ref_count", len(ref_list)])
                writer.writerow(["gaze_count", len(gaze_pos)])

            logger.info(f"Raw session data exported to: {out_dir}")

        except Exception as exc:
            logger.error(f"_export_raw_session_data failed: {exc}")
            logger.debug(traceback.format_exc())

    @staticmethod
    def calc_acc_prec_errlines(
        g_pool,
        gazer_class,
        gazer_params,
        pupil_list,
        ref_list,
        intrinsics,
        outlier_threshold,
        succession_threshold=np.cos(np.deg2rad(0.5)),
        gaze_list=None,
    ) -> AccuracyPrecisionResult:
        if gaze_list:
            gaze_pos = list(gaze_list)
        else:
            gazer = gazer_class(g_pool, params=gazer_params, register_as_active=False)
            gaze_pos = list(gazer.map_pupil_to_gaze(pupil_list))
        ref_pos = ref_list

        try:
            correlation_result = Accuracy_Visualizer.correlate_and_coordinate_transform(
                gaze_pos, ref_pos, intrinsics
            )
            error_lines = correlation_result.norm_space.reshape(-1, 4)
            undistorted_3d = correlation_result.camera_space
        except CorrelationError:
            return AccuracyPrecisionResult.failed()

        # Accuracy is calculated as the average angular
        # offset (distance) (in degrees of visual angle)
        # between fixations locations and the corresponding
        # locations of the fixation targets.

        # Cosine distance of A and B: (A @ B) / (||A|| * ||B||)
        # No need to calculate norms, since A and B are normalized in our case.
        # np.einsum('ij,ij->i', A, B) equivalent to np.diagonal(A @ B.T) but faster.
        angular_err = np.einsum(
            "ij,ij->i", undistorted_3d[::2, :], undistorted_3d[1::2, :]
        )

        # Good values are close to 1. since cos(0) == 1.
        # Therefore we look for values greater than cos(outlier_threshold)
        selected_indices = angular_err > np.cos(np.deg2rad(outlier_threshold))
        selected_samples = angular_err[selected_indices]
        num_used, num_total = selected_samples.shape[0], angular_err.shape[0]

        error_lines = error_lines[selected_indices].reshape(
            -1, 2
        )  # shape: num_used x 2
        if num_used > 0:
            accuracy = np.rad2deg(np.arccos(selected_samples.clip(-1.0, 1.0).mean()))
        else:
            accuracy = float("nan")
        accuracy_result = CalculationResult(accuracy, num_used, num_total)

        # lets calculate precision:  (RMS of distance of succesive samples.)
        # This is a little rough as we do not compensate headmovements in this test.

        # Precision is calculated as the Root Mean Square (RMS)
        # of the angular distance (in degrees of visual angle)
        # between successive samples during a fixation
        undistorted_3d.shape = -1, 6  # shape: n x 6
        succesive_distances_gaze = np.einsum(
            "ij,ij->i", undistorted_3d[:-1, :3], undistorted_3d[1:, :3]
        )
        succesive_distances_ref = np.einsum(
            "ij,ij->i", undistorted_3d[:-1, 3:], undistorted_3d[1:, 3:]
        )

        # if the ref distance is to big we must have moved to a new fixation or there is
        # headmovement, if the gaze dis is to big we can assume human error
        # both times gaze data is not valid for this mesurement
        selected_indices = np.logical_and(
            succesive_distances_gaze > succession_threshold,
            succesive_distances_ref > succession_threshold,
        )
        succesive_distances = succesive_distances_gaze[selected_indices]
        num_used, num_total = (
            succesive_distances.shape[0],
            succesive_distances_gaze.shape[0],
        )
        if num_used > 0:
            precision = np.sqrt(
                np.mean(np.rad2deg(np.arccos(succesive_distances.clip(-1.0, 1.0))) ** 2)
            )
        else:
            precision = float("nan")
        precision_result = CalculationResult(precision, num_used, num_total)

        return AccuracyPrecisionResult(
            accuracy_result, precision_result, error_lines, correlation_result
        )

    @staticmethod
    def correlate_and_coordinate_transform(
        gaze_pos, ref_pos, intrinsics
    ) -> CorrelatedAndCoordinateTransformedResult:
        # reuse closest_matches_monocular to correlate one label to each prediction
        # correlated['ref']: prediction, correlated['pupil']: label location
        # NOTE the switch of the ref and pupil keys! This effects mostly hmd data.
        correlated = closest_matches_monocular(gaze_pos, ref_pos)
        # [[pred.x, pred.y, label.x, label.y], ...], shape: n x 4
        if not correlated:
            raise CorrelationError("No correlation possible")

        try:
            return Accuracy_Visualizer._coordinate_transform_ref_in_norm_space(
                correlated, intrinsics
            )
        except KeyError as err:
            if "norm_pos" in err.args:
                return Accuracy_Visualizer._coordinate_transform_ref_in_camera_space(
                    correlated, intrinsics
                )
            else:
                raise

    @staticmethod
    def _coordinate_transform_ref_in_norm_space(
        correlated, intrinsics
    ) -> CorrelatedAndCoordinateTransformedResult:
        width, height = intrinsics.resolution
        locations_norm = np.array(
            [(*e["ref"]["norm_pos"], *e["pupil"]["norm_pos"]) for e in correlated]
        )
        locations_image = locations_norm.copy()  # n x 4
        locations_image[:, ::2] *= width
        locations_image[:, 1::2] = (1.0 - locations_image[:, 1::2]) * height
        locations_image.shape = -1, 2
        locations_norm.shape = -1, 2
        locations_camera = intrinsics.unprojectPoints(locations_image, normalize=True)
        return CorrelatedAndCoordinateTransformedResult(
            locations_norm, locations_image, locations_camera
        )

    @staticmethod
    def _coordinate_transform_ref_in_camera_space(
        correlated, intrinsics
    ) -> CorrelatedAndCoordinateTransformedResult:
        width, height = intrinsics.resolution
        locations_mixed = np.array(
            # NOTE: This looks incorrect, but is actually correct. The switch comes from
            # using closest_matches_monocular() above with switched arguments.
            [(*e["ref"]["norm_pos"], *e["pupil"]["mm_pos"]) for e in correlated]
        )  # n x 5
        pupil_norm = locations_mixed[:, 0:2]  # n x 2
        pupil_image = pupil_norm.copy()
        pupil_image[:, 0] *= width
        pupil_image[:, 1] = (1.0 - pupil_image[:, 1]) * height
        pupil_camera = intrinsics.unprojectPoints(pupil_image, normalize=True)  # n x 3

        ref_camera = locations_mixed[:, 2:5]  # n x 3
        ref_camera /= np.linalg.norm(ref_camera, axis=1, keepdims=True)
        ref_image = intrinsics.projectPoints(ref_camera)  # n x 2
        ref_norm = ref_image.copy()
        ref_norm[:, 0] /= width
        ref_norm[:, 1] = 1.0 - (ref_norm[:, 1] / height)

        locations_norm = np.hstack([pupil_norm, ref_norm])  # n x 4
        locations_norm.shape = -1, 2

        locations_image = np.hstack([pupil_image, ref_image])  # n x 4
        locations_image.shape = -1, 2

        locations_camera = np.hstack([pupil_camera, ref_camera])  # n x 6
        locations_camera.shape = -1, 3

        return CorrelatedAndCoordinateTransformedResult(
            locations_norm, locations_image, locations_camera
        )

    def gl_display(self):
        import OpenGL.GL as gl
        from pyglui.cygl.utils import RGBA, draw_points_norm, draw_polyline_norm

        if self.vis_mapping_error and self.error_lines is not None:
            draw_polyline_norm(
                self.error_lines, color=RGBA(1.0, 0.5, 0.0, 0.5), line_type=gl.GL_LINES
            )
            draw_points_norm(
                self.error_lines[1::2], size=3, color=RGBA(0.0, 0.5, 0.5, 0.5)
            )
            draw_points_norm(
                self.error_lines[0::2], size=3, color=RGBA(0.5, 0.0, 0.0, 0.5)
            )
        if self.vis_calibration_area and self.calibration_area is not None:
            draw_polyline_norm(
                self.calibration_area,
                thickness=2.0,
                color=RGBA(0.663, 0.863, 0.463, 0.8),
                line_type=gl.GL_LINE_LOOP,
            )

    def get_init_dict(self):
        return {
            "outlier_threshold": self.outlier_threshold,
            "vis_mapping_error": self.vis_mapping_error,
            "vis_calibration_area": self.vis_calibration_area,
        }
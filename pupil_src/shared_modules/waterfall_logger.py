"""
Pupil Labs Waterfall Pipeline Latency Logger
=============================================
Instruments and logs per-frame end-to-end latency breakdowns from camera capture
through pupil detection, 3D gaze estimation, IPC transport, and display rendering.
"""

import csv
from datetime import datetime
import logging
import os
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Global thread-safe singleton
_lock = threading.Lock()
_waterfall_logger_instance: Optional["WaterfallLogger"] = None


class WaterfallLogger:
    def __init__(self, log_dir: Optional[str] = None):
        if log_dir is None:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            log_dir = os.path.join(root_dir, "logged_latencies")

        os.makedirs(log_dir, exist_ok=True)
        session_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = os.environ.get(
            "PUPIL_WATERFALL_CSV",
            os.path.join(log_dir, f"waterfall_{session_time_str}.csv"),
        )
        os.environ["PUPIL_WATERFALL_CSV"] = self.log_path

        self.buffer = []
        self.frame_count = 0
        self.header_written = os.path.exists(self.log_path)
        # @pupil_data_relay.py에서 모든 데이터 취합 후 waterfall logger로 전송.
        # 아직 render는 하지 않았으니 해당 데이터 없으면(첫 번째 loop) 0.0으로 fallback.
        self.fieldnames = [
            "frame_id",
            "process",                  # eye0 / eye1
            "model",                    # 추론 모델(2dcpp, pmrnet, ...)
            "phase",                    # 콜드스타트(boot) / 안정후 loop(loop)
            "ingest_ms",                # eye 카메라 -> 프로세스 전송시간 @detector_2d_nn_plugin. 현재시간 - frame에 찍혀있는 timestamp
            "roi_ms",                   # 2dcpp 사용시 ROI 설정시간(딥러닝 모델 사용시 X)  @detector_2d_nn_plugin.py
            "preprocess_ms",            # gamma LUT, CLAHE 등    @detector_2d_nn_plugin.py
            "inference_ms",             # 동공 세그멘트 시간    @detector_2d_nn_plugin.py
            "ellipse_fit_ms",           # 추론 후 동공 타원 피팅 시간  @detector_2d_nn_plugin.py
            "pye3d_ms",                 # pye3d(3d 안구 모델) 추론시간 => 3D 안구 중심, 3d gaze 벡터  @pye3d_plugin.py
            "ipc_transport_ms",         # eye.py, world.py 데이터 주고받은 시간(pupil data)  @eye.py send() - @pupil_data_relay.py recv()
            "gaze_mapping_ms",          # 3d -> 2d gaze로 gaze mapping   @puipl_data_relay.py
            "render_ms",                # world.py 화면 렌더링 시간    @world.py
            "buffer_swap_ms",           # openGL 프레임버퍼  @world.py
            "total_system_latency_ms",  # 전체 레이턴시 @pupil_data_relay.py
            "t_start",                  # 해당 프레임 처리 시작시간    @detector_2d_nn_plugin.py
            "t_end",                    # 해당 프레임 처리 종료시간    @detector_2d_nn_plugin.py
        ]

    def log_frame_trace(self, trace_data: Dict):
        """Buffer a completed frame's sequential pipeline stages."""
        with _lock:
            self.frame_count += 1
            phase = "boot" if self.frame_count == 1 else "loop"
            trace_data["phase"] = trace_data.get("phase", phase)
            self.buffer.append(trace_data)

            # Flush periodically to keep memory minimal
            if len(self.buffer) >= 50:
                self.flush()

    def flush(self):
        """Write buffered traces to CSV file."""
        if not self.buffer:
            return

        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            file_exists = os.path.exists(self.log_path)
            with open(self.log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                if not file_exists:
                    writer.writeheader()
                for row in self.buffer:
                    filtered_row = {k: row.get(k, 0.0) for k in self.fieldnames}
                    writer.writerow(filtered_row)
            logger.debug(f"Flushed {len(self.buffer)} waterfall records to {self.log_path}")
            self.buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush waterfall log: {e}")


def get_waterfall_logger() -> WaterfallLogger:
    global _waterfall_logger_instance
    with _lock:
        if _waterfall_logger_instance is None:
            _waterfall_logger_instance = WaterfallLogger()
        return _waterfall_logger_instance

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
        self.fieldnames = [
            "frame_id",
            "process",
            "model",
            "phase",
            "ingest_ms",
            "roi_ms",
            "preprocess_ms",
            "inference_ms",
            "ellipse_fit_ms",
            "pye3d_ms",
            "ipc_transport_ms",
            "gaze_mapping_ms",
            "render_ms",
            "buffer_swap_ms",
            "total_system_latency_ms",
            "t_start",
            "t_end",
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
